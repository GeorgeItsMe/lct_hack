"""Scenario recognition on the polygon, scored against the known truth.

Two kinds of output, as the dispatcher receives them:

* triage of an alarm that has just fired (fire: real fire, hot work or sensor noise;
  access: intrusion, work under a permit or work without one), decided a few minutes after
  the first record from what is known by then;
* forecasts under the 24-hour rule (one warning per object or unit and kind per day):
  a fire after hours of heating, a pump or fan failure after days of wear, a gas alarm
  after the ventilation stops.

Models are trained on January-April of several polygon worlds, thresholds are chosen on
May (highest recall with precision >= 0.7, otherwise the best F1) and June of other worlds,
never seen in training, is scored once. Every world keeps the same real journal as
background; only the injected incidents differ.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from moscollector import daily_rule
from moscollector.polygon import Catalog, World

MIN = pd.Timedelta(minutes=1)
HOUR = pd.Timedelta(hours=1)
TRAIN = (pd.Timestamp("2026-01-08"), pd.Timestamp("2026-05-01"))
SELECT = (pd.Timestamp("2026-05-01"), pd.Timestamp("2026-06-01"))
TEST = (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-01"))
FAULT_STATES = ("Неисправен", "Отключено устройство", "Обесточен")


def real_events(processed: Path, catalog: Catalog, cache: Path) -> pd.DataFrame:
    """Real records of the scenario groups (gas excluded: it is used hourly)."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        channels = catalog.channels[~catalog.channels.group.eq("gas")]
        con = duckdb.connect()
        frame = con.execute(
            "SELECT event_id, channel_id, ts, value, numeric_value, alarm FROM read_parquet(?) "
            "WHERE list_contains(?, channel_id)",
            [str(processed / "events-2026.parquet"), [int(c) for c in channels.channel_id]],
        ).df()
        con.close()
        frame.to_parquet(cache, index=False)
    return pd.read_parquet(cache)


def real_gas_hourly(processed: Path, catalog: Catalog, cache: Path) -> pd.DataFrame:
    if not cache.exists():
        con = duckdb.connect()
        frame = con.execute(
            "SELECT channel_id, date_trunc('hour', ts) AS hour, "
            "max(CASE WHEN numeric_value BETWEEN 0 AND 100 THEN numeric_value END) AS gas, "
            "bool_or(value = 'Обнаружен газ' AND alarm) AS gas_alarm "
            "FROM read_parquet(?) WHERE list_contains(?, channel_id) GROUP BY 1, 2",
            [
                str(processed / "events-2026.parquet"),
                [int(c) for c in catalog.channels.loc[catalog.channels.group.eq("gas"), "channel_id"]],
            ],
        ).df()
        con.close()
        frame.to_parquet(cache, index=False)
    return pd.read_parquet(cache)


class WorldData:
    """Real background plus one world's injected records, registries and truth."""

    def __init__(self, world: World, catalog: Catalog, real: pd.DataFrame, gas: pd.DataFrame):
        info = catalog.channels[["channel_id", "object_id", "group", "sensor_type", "pk"]]
        injected = world.events.merge(info, on="channel_id")
        background = real.merge(info, on="channel_id")
        background["incident_id"] = None
        keep = np.ones(len(background), dtype=bool)
        for s in world.suppressed.itertuples():
            keep &= ~(
                background.channel_id.eq(s.channel_id) & background.ts.ge(s.start) & background.ts.lt(s.end)
            ).to_numpy()
        events = pd.concat([background[keep], injected[~injected.group.eq("gas")]], ignore_index=True)
        self.events = events.sort_values(["ts", "event_id"], kind="stable").reset_index(drop=True)
        self.gas = self._gas(world, gas, injected)
        self.world, self.catalog = world, catalog
        self.truth = world.truth
        self.permits = world.permits
        incident_kind = world.truth.set_index("incident_id").scenario
        self.events["scenario"] = self.events.incident_id.map(incident_kind)

    @staticmethod
    def _gas(world, gas, injected):
        frame = gas.copy()
        for s in world.suppressed.itertuples():
            frame = frame[
                ~(
                    frame.channel_id.eq(s.channel_id)
                    & frame.hour.ge(s.start.floor("h"))
                    & frame.hour.lt(s.end)
                )
            ]
        own = injected[injected.group.eq("gas")].assign(hour=lambda f: f.ts.dt.floor("h"))
        if len(own):
            add = own.groupby(["channel_id", "hour"]).agg(
                gas=("numeric_value", "max"),
                gas_alarm=("value", lambda v: bool((v == "Обнаружен газ").any())),
            )
            frame = pd.concat([frame, add.reset_index()], ignore_index=True)
        return frame

    def permit_flags(self, objects, times, margin=15 * MIN):
        """For each (object, moment): a permit covers it; a hot-work permit covers it."""
        objects = np.asarray(objects)
        t = pd.to_datetime(pd.Series(times)).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        covered = np.zeros(len(t), dtype=bool)
        hot = np.zeros(len(t), dtype=bool)
        m = int(margin.total_seconds()) * 10**9
        for (obj, is_hot), p in self.permits.assign(
            is_hot=self.permits.work_type.eq("Огневые работы")
        ).groupby(["object_id", "is_hot"]):
            idx = np.where(objects == obj)[0]
            if not len(idx):
                continue
            p = p.sort_values("start")
            starts = p.start.to_numpy(dtype="datetime64[ns]").astype(np.int64) - m
            ends = np.maximum.accumulate(p.end.to_numpy(dtype="datetime64[ns]").astype(np.int64) + m)
            j = np.searchsorted(starts, t[idx], side="right") - 1
            inside = (j >= 0) & (ends[np.maximum(j, 0)] >= t[idx])
            covered[idx] |= inside
            if is_hot:
                hot[idx] |= inside
        return covered, hot


def series(frame):
    """channel -> (times ns, values) for forward-filled readings."""
    out = {}
    for channel, part in frame.groupby("channel_id"):
        out[int(channel)] = (
            part.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64),
            part.numeric_value.to_numpy(),
        )
    return out


def alarm_groups(alarms, window=10 * MIN):
    """Alarm records of one object within ``window`` of the first form one group."""
    alarms = alarms.sort_values(["object_id", "ts"])
    group_id = np.empty(len(alarms), dtype=np.int64)
    start, last_object, gid = None, None, -1
    for i, (obj, ts) in enumerate(zip(alarms.object_id.to_numpy(), alarms.ts.to_numpy(), strict=True)):
        if obj != last_object or ts - start > np.timedelta64(int(window.total_seconds()), "s"):
            gid += 1
            start, last_object = ts, obj
        group_id[i] = gid
    alarms = alarms.assign(group=group_id)
    return alarms


def temperature_store(data):
    temps = data.events[data.events.group.eq("temp") & data.events.numeric_value.notna()]
    return series(temps), data.catalog.channels[data.catalog.channels.group.eq("temp")].groupby(
        "object_id"
    ).channel_id.apply(list).to_dict()


def temperature_features(store, temp_channels, objects, times):
    """Largest rise over 1, 3, 6, 12, 24 h and channels heating by 4+ degrees in 6 h."""
    objects = np.asarray(objects)
    t = pd.to_datetime(pd.Series(times)).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    keys = ("rise1", "rise3", "rise6", "rise12", "rise24")
    out = {k: np.full(len(t), np.nan) for k in ("temp_max", *keys)}
    heating = np.zeros(len(t), dtype=np.int64)
    for obj in np.unique(objects):
        idx = np.where(objects == obj)[0]
        at = t[idx]
        for ch in temp_channels.get(int(obj), []):
            stamps, values = store.get(int(ch), (None, None))
            if stamps is None:
                continue

            def value(x, stamps=stamps, values=values):
                j = np.searchsorted(stamps, x, side="right") - 1
                return np.where(j >= 0, values[np.maximum(j, 0)], np.nan)

            now = value(at)
            out["temp_max"][idx] = np.fmax(out["temp_max"][idx], now)
            for h in (1, 3, 6, 12, 24):
                rise = now - value(at - h * daily_rule.HOUR_NS)
                out[f"rise{h}"][idx] = np.fmax(out[f"rise{h}"][idx], rise)
                if h == 6:
                    heating[idx] += np.nan_to_num(rise) >= 4
    return pd.DataFrame({**out, "heating_channels": heating})


# ---------------------------------------------------------------- fire: triage of an alarm


FIRE_TRIAGE_FEATURES = [
    "hour",
    "weekend",
    "night",
    "channels_10m",
    "still_active",
    "first_seconds",
    "heat_alarm",
    "permit",
    "hot_work",
    "faults_nearby",
    "channel_fault_24h",
    "groups_7d",
    "temp_max",
    "rise1",
    "rise3",
    "rise6",
    "rise12",
    "rise24",
    "heating_channels",
]


def fire_triage_units(data):
    e = data.events
    alarms = e[
        e.alarm
        & (
            (e.group.eq("smoke") & e.value.eq("Обнаружен дым"))
            | (e.group.eq("heat") & e.value.eq("Не замкнут"))
        )
    ].copy()
    nxt = e.groupby("channel_id").ts.shift(-1)
    alarms["clear"] = nxt.loc[alarms.index]
    alarms = alarm_groups(alarms)
    faults = e[e.alarm & e.value.isin(FAULT_STATES)]
    fault_times = (
        faults.groupby("object_id")
        .ts.apply(lambda s: s.to_numpy(dtype="datetime64[ns]").astype(np.int64))
        .to_dict()
    )
    channel_faults = (
        faults.groupby("channel_id")
        .ts.apply(lambda s: s.to_numpy(dtype="datetime64[ns]").astype(np.int64))
        .to_dict()
    )
    rows = []
    starts_by_object = {}
    for _, g in alarms.groupby("group", sort=False):
        start = g.ts.iloc[0]
        decide = start + 10 * MIN
        g = g[g.ts.le(decide)]
        obj = int(g.object_id.iloc[0])
        clear = g.clear.fillna(pd.Timestamp.max)
        first_clear = min(clear.iloc[0], decide)
        s_ns = start.value
        ft = fault_times.get(obj, np.array([], dtype=np.int64))
        nearby = int(((ft >= s_ns - 15 * 60 * 10**9) & (ft <= decide.value)).sum())
        cf = channel_faults.get(int(g.channel_id.iloc[0]), np.array([], dtype=np.int64))
        previous = starts_by_object.setdefault(obj, [])
        rows.append(
            {
                "object_id": obj,
                "start": start,
                "hour": start.hour,
                "weekend": int(start.dayofweek >= 5),
                "night": int(start.hour >= 22 or start.hour < 6),
                "channels_10m": g.channel_id.nunique(),
                "still_active": int((clear > decide).sum()),
                "first_seconds": (first_clear - start).total_seconds(),
                "heat_alarm": int(g.group.eq("heat").any()),
                "faults_nearby": nearby,
                "channel_fault_24h": int(((cf >= s_ns - 24 * 3600 * 10**9) & (cf < s_ns)).sum() > 0),
                "groups_7d": sum(1 for p in previous if (start - p) <= pd.Timedelta(days=7)),
                "real_fire": int(g.scenario.eq("real_fire").any()),
                "hot_work_truth": int(g.scenario.eq("hot_work_smoke").any()),
                "incident_id": next((i for i in g.incident_id if isinstance(i, str)), None),
            }
        )
        previous.append(start)
    units = pd.DataFrame(rows)
    units["permit"], units["hot_work"] = data.permit_flags(units.object_id.to_numpy(), units.start.to_numpy())
    units["permit"] = units.permit.astype(int)
    units["hot_work"] = units.hot_work.astype(int)
    store, temp_channels = temperature_store(data)
    units = pd.concat(
        [units, temperature_features(store, temp_channels, units.object_id, units.start)], axis=1
    )
    return units


# ---------------------------------------------------------------- fire: 24-hour forecast


FIRE_FORECAST_FEATURES = [
    "hour",
    "temp_max",
    "rise1",
    "rise3",
    "rise6",
    "rise12",
    "rise24",
    "heating_channels",
    "smoke_24h",
    "permit",
    "hot_work",
]


def fire_forecast_units(data, begin, end):
    store, temp_channels = temperature_store(data)
    # Every object with smoke detectors: a fire where no temperature sensor stands is a miss.
    smoke_objects = set(data.catalog.channels.loc[data.catalog.channels.group.eq("smoke"), "object_id"])
    objects = sorted(set(temp_channels) | smoke_objects)
    hours = pd.date_range(begin, end - HOUR, freq="h")
    grid = pd.DataFrame([(o, h) for o in objects for h in hours], columns=["object_id", "as_of"])
    feats = temperature_features(store, temp_channels, grid.object_id, grid.as_of)
    grid = pd.concat([grid, feats], axis=1)
    grid["hour"] = grid.as_of.dt.hour
    smoke = data.events[
        data.events.alarm & data.events.group.eq("smoke") & data.events.value.eq("Обнаружен дым")
    ]
    counts = []
    for obj, part in grid.groupby("object_id", sort=False):
        times = np.sort(
            smoke.loc[smoke.object_id.eq(obj), "ts"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        )
        t = part.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        counts.append(
            pd.Series(
                np.searchsorted(times, t) - np.searchsorted(times, t - 24 * daily_rule.HOUR_NS),
                index=part.index,
            )
        )
    grid["smoke_24h"] = pd.concat(counts)
    permit, hot = data.permit_flags(grid.object_id.to_numpy(), grid.as_of.to_numpy(), margin=pd.Timedelta(0))
    grid["permit"], grid["hot_work"] = permit.astype(int), hot.astype(int)
    fires = data.truth[data.truth.scenario.eq("real_fire")][["object_id", "start"]].rename(
        columns={"start": "start_ts"}
    )
    return grid, fires


# ---------------------------------------------------------------- shared model and selection


def fit(frame, features, label, seed=7):
    model = CatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=6,
        random_seed=seed,
        thread_count=8,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(frame[features], frame[label])
    return model


def precision_recall(score, label, threshold):
    flagged = score >= threshold
    tp = int((flagged & (label == 1)).sum())
    alerts = int(flagged.sum())
    positives = int((label == 1).sum())
    precision = tp / alerts if alerts else 0.0
    recall = tp / positives if positives else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "flagged": alerts,
        "true_flagged": tp,
        "positives": positives,
    }


def select_unit_threshold(score, label, minimum=3):
    grid = np.unique(np.quantile(score, np.linspace(0.5, 0.9999, 400)))
    results = [precision_recall(score, label, t) for t in grid]
    results = [r for r in results if r["flagged"] >= minimum]
    reaching = [r for r in results if r["precision"] >= daily_rule.TARGET_PRECISION]
    if reaching:
        best = max(reaching, key=lambda r: (r["recall"], r["precision"]))
        best["rule"] = "max_recall_at_precision_0.7"
    else:
        best = max(results, key=lambda r: (r["f1"], r["precision"]))
        best["rule"] = "best_f1_precision_0.7_unreachable"
    return best


def within(frame, column, period):
    return frame[frame[column].ge(period[0]) & frame[column].lt(period[1])]
