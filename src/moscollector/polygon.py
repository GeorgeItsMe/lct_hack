"""Polygon: the real 2026 journal with registries and incidents whose truth is known.

The real journal has no confirmed incidents, permits, hot-work schedules or equipment
passports, so nothing proves that the service recognises a real fire or intrusion. The
polygon keeps every real record as background and adds what the customer's systems would
hold, generated from the journal's own statistics (organizers allowed plausible simulation):

* registries: work permits derived from real access activity (about one visit in ten has
  no registered permit), hot-work permits, equipment passports and an emergency repair log;
* incidents written as journal records on real channels of real objects, in the value
  formats those channels use: intrusions (with an occasional probe the day before), hot-work
  smoke, real fires preceded by heating, pump and fan wear ending in failure (a failed pump
  floods the pit, a failed fan lets gas build up), and loop or controller cascades.

Every injected record carries its incident id; ``truth`` lists each incident with its time,
channels and kind. Rates below are the team's assumptions, not customer statistics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from moscollector.paths import PROCESSED

PERIOD = (pd.Timestamp("2026-01-08"), pd.Timestamp("2026-06-30 23:00"))
GROUPS = {
    "door": ["КД Дверь", "КД Люк", "КД АВ", "9-секционный люк"],
    "motion": ["Датчик движения"],
    "smoke": ["Датчик дыма"],
    "heat": ["Тепловой датчик"],
    "temp": ["Датчик температуры"],
    "gas": ["Газовый датчик"],
    "pump": ["Состояние насоса"],
    "fan": ["Состояние вентилятора"],
    "guard": ["Состояние охраны"],
}
# Team assumptions for the whole park of 78 objects.
RATES = {
    "intrusions_per_week": 1.5,
    "intrusion_probe_share": 0.3,
    "intrusion_off_hours_share": 0.7,
    "real_fires_per_week": 0.5,
    "fire_during_hot_work_share": 0.2,
    "hot_works_per_week": 4,
    "permit_missing_share": 0.1,
    "pump_failures_per_half_year": 10,
    "fan_failures_per_half_year": 10,
    "cascades_per_week": 2,
    "smoulder_share": 0.3,
    "heavy_hot_work_share": 0.2,
    "hot_work_unregistered_share": 0.0,
}
# Robustness check: the same incidents, made harder to tell from routine work.
HARD_RATES = {
    "intrusion_off_hours_share": 0.4,
    "permit_missing_share": 0.2,
    "hot_work_unregistered_share": 0.3,
    "fire_during_hot_work_share": 0.35,
    "smoulder_share": 0.5,
    "heavy_hot_work_share": 0.4,
}
PUMP_RESOURCE_HOURS = 20_000  # assumption: typical drainage pump resource
FAN_RESOURCE_HOURS = 30_000  # assumption: typical tunnel fan resource
WEEKS = (PERIOD[1] - PERIOD[0]) / pd.Timedelta(weeks=1)
EVENT_ID_BASE = 9_000_000_000_000
MIN = pd.Timedelta(minutes=1)
HOUR = pd.Timedelta(hours=1)
PK = re.compile(r"ПК\s*(\d+(?:[.,]\d+)?)")


@dataclass
class World:
    seed: int
    rates: dict
    events: pd.DataFrame  # injected journal records with incident_id
    suppressed: pd.DataFrame  # channel_id, start, end: real records replaced by a scenario
    truth: pd.DataFrame
    permits: pd.DataFrame
    passports: pd.DataFrame
    repairs: pd.DataFrame


class Catalog:
    """Active channels of each group per object, with picket positions parsed from names."""

    def __init__(self, processed=PROCESSED):
        channels = pd.read_parquet(processed / "channels.parquet")
        con = duckdb.connect()
        active = con.execute(
            "SELECT channel_id, count(*) AS n FROM read_parquet(?) GROUP BY 1 ORDER BY 1",
            [str(processed / "events-2026.parquet")],
        ).df()
        con.close()
        channels = channels.merge(active, on="channel_id")
        channels["group"] = None
        for group, types in GROUPS.items():
            channels.loc[channels.sensor_type.isin(types), "group"] = group
        channels["pk"] = (
            channels.sensor_name.str.extract(PK, expand=False).str.replace(",", ".").astype(float)
        )
        self.channels = channels[channels.group.notna()].reset_index(drop=True)

    def of(self, obj, group):
        return self.channels[self.channels.object_id.eq(obj) & self.channels.group.eq(group)]

    def objects_with(self, *groups):
        sets = [set(self.channels.loc[self.channels.group.eq(g), "object_id"]) for g in groups]
        return sorted(set.intersection(*sets))


def nearest(frame, pk, rng, k=1):
    """k channels closest by picket to pk (random order when pickets are unknown)."""
    if frame.empty:
        return frame
    if pk is None or np.isnan(pk) or frame.pk.isna().all():
        return frame.sample(min(k, len(frame)), random_state=int(rng.integers(1 << 31)))
    order = (frame.pk - pk).abs().fillna(1e9).sort_values().index
    return frame.loc[order[:k]]


class Writer:
    def __init__(self):
        self.rows, self.truth, self.suppressed = [], [], []

    def add(self, incident, channel, ts, value, alarm, numeric=None):
        self.rows.append((int(channel), pd.Timestamp(ts).floor("s"), value, numeric, bool(alarm), incident))

    def numeric(self, incident, channel, ts, number, decimals=0):
        text = f"{number:.{decimals}f}"
        self.add(incident, channel, ts, text, False, round(float(number), decimals))

    def incident(self, **row):
        row["incident_id"] = f"INC-{len(self.truth) + 1:05d}"
        self.truth.append(row)
        return row["incident_id"]

    def frames(self):
        events = pd.DataFrame(
            self.rows, columns=["channel_id", "ts", "value", "numeric_value", "alarm", "incident_id"]
        )
        events = events.sort_values(["ts", "channel_id"], kind="stable").reset_index(drop=True)
        events.insert(0, "event_id", EVENT_ID_BASE + np.arange(len(events)))
        suppressed = pd.DataFrame(self.suppressed, columns=["channel_id", "start", "end", "incident_id"])
        return events, pd.DataFrame(self.truth), suppressed


def planned(rate):
    """Fixed incident counts from the agreed rates; only places and times are random."""
    return int(round(rate * WEEKS))


def random_moment(rng, off_hours):
    """A moment inside the period: off hours (night 22-06 or weekend) or a weekday day."""
    while True:
        day = PERIOD[0] + pd.Timedelta(days=int(rng.integers((PERIOD[1] - PERIOD[0]).days)))
        if off_hours:
            if rng.random() < 0.3:
                if day.dayofweek < 5:
                    continue
                t = day + pd.Timedelta(minutes=int(rng.integers(24 * 60)))
            else:
                t = day + pd.Timedelta(hours=22) + pd.Timedelta(minutes=int(rng.integers(8 * 60)))
        else:
            if day.dayofweek >= 5:
                continue
            t = day + pd.Timedelta(hours=9) + pd.Timedelta(minutes=int(rng.integers(8 * 60)))
        if t < PERIOD[1]:
            return t


def build_permits(episodes, catalog, rng, rates):
    """One permit per real visit (access episodes less than 2 h apart), some unregistered.

    Returns the registry and every hot work, including hot works done without a permit.
    """
    access = episodes[episodes.kind.eq("access")].sort_values(["object_id", "start_ts"])
    rows = []
    for obj, part in access.groupby("object_id"):
        visit_start = visit_end = None
        for r in part.itertuples():
            end = r.end_ts if pd.notna(r.end_ts) else r.start_ts + 10 * MIN
            if visit_start is not None and r.start_ts - visit_end <= 2 * HOUR:
                visit_end = max(visit_end, end)
                continue
            if visit_start is not None:
                rows.append((obj, visit_start, visit_end))
            visit_start, visit_end = r.start_ts, end
        if visit_start is not None:
            rows.append((obj, visit_start, visit_end))
    visits = pd.DataFrame(rows, columns=["object_id", "visit_start", "visit_end"])
    registered = rng.random(len(visits)) >= rates["permit_missing_share"]
    visits["registered"] = registered
    permits = visits[registered].copy()
    permits["start"] = permits.visit_start - pd.to_timedelta(rng.uniform(15, 45, len(permits)), unit="min")
    permits["end"] = permits.visit_end + pd.to_timedelta(rng.uniform(30, 90, len(permits)), unit="min")
    permits["work_type"] = np.where(rng.random(len(permits)) < 0.65, "Осмотр", "Ремонт")
    # Hot works: weekday daytime permits on objects with smoke detectors.
    smoke_objects = set(catalog.objects_with("smoke"))
    eligible = permits[
        permits.object_id.isin(smoke_objects)
        & permits.start.dt.dayofweek.lt(5)
        & permits.start.dt.hour.between(7, 16)
        & (permits.end - permits.start).ge(90 * MIN)
    ].index
    count = min(len(eligible), planned(rates["hot_works_per_week"]))
    permits.loc[rng.choice(eligible, count, replace=False), "work_type"] = "Огневые работы"
    permits = permits.sort_values("start").reset_index(drop=True)
    permits["permit_id"] = [f"НД-{i + 1:05d}" for i in range(len(permits))]
    permits["brigade"] = [f"Бригада {int(b)}" for b in rng.integers(1, 13, len(permits))]
    permits = permits[["permit_id", "object_id", "start", "end", "work_type", "brigade"]]
    hot = permits[permits.work_type.eq("Огневые работы")].copy()
    hot["registered"] = rng.random(len(hot)) >= rates["hot_work_unregistered_share"]
    registry = permits[~permits.permit_id.isin(hot.loc[~hot.registered, "permit_id"])].reset_index(drop=True)
    return registry, hot.reset_index(drop=True)


def permit_at(permits, obj, t, margin=pd.Timedelta(0)):
    rows = permits[permits.object_id.eq(obj) & (permits.start - margin).le(t) & (permits.end + margin).ge(t)]
    return rows.iloc[0] if len(rows) else None


def inject_intrusions(w, catalog, permits, rng, rates):
    objects = catalog.objects_with("door", "motion")
    for _ in range(planned(rates["intrusions_per_week"])):
        obj = int(rng.choice(objects))
        off_hours = rng.random() < rates["intrusion_off_hours_share"]
        t0 = random_moment(rng, off_hours)
        while permit_at(permits, obj, t0, 2 * HOUR) is not None:
            t0 = random_moment(rng, off_hours)
        doors = catalog.of(obj, "door")
        hatches = doors[doors.sensor_type.isin(["КД Люк", "КД АВ", "9-секционный люк"])]
        door = (hatches if len(hatches) else doors).sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
        motions = nearest(catalog.of(obj, "motion"), door.pk, rng, int(rng.integers(1, 4)))
        incident = w.incident(
            scenario="intrusion",
            object_id=obj,
            start=t0,
            channel_ids=[int(door.channel_id)],
            off_hours=off_hours,
        )
        w.add(incident, door.channel_id, t0, "Не замкнут", True)
        w.add(incident, door.channel_id, t0 + pd.Timedelta(seconds=rng.uniform(60, 900)), "Норма", False)
        t = t0 + pd.Timedelta(seconds=rng.uniform(30, 240))
        for m in motions.itertuples():
            w.add(incident, m.channel_id, t, "Обнаружено движение", True)
            w.add(incident, m.channel_id, t + pd.Timedelta(seconds=rng.uniform(5, 60)), "Движения нет", False)
            w.truth[-1]["channel_ids"].append(int(m.channel_id))
            t += pd.Timedelta(seconds=rng.uniform(60, 300))
        if rng.random() < rates["intrusion_probe_share"]:
            tp = t0 - pd.Timedelta(hours=rng.uniform(6, 24))
            probe = w.incident(
                scenario="intrusion_probe",
                object_id=obj,
                start=tp,
                channel_ids=[int(door.channel_id)],
                off_hours=bool(tp.hour >= 22 or tp.hour < 6 or tp.dayofweek >= 5),
            )
            w.add(probe, door.channel_id, tp, "Не замкнут", True)
            w.add(probe, door.channel_id, tp + pd.Timedelta(seconds=rng.uniform(5, 40)), "Норма", False)


def temperature_baseline(temps, channel, t):
    past = temps[temps.channel_id.eq(channel) & temps.ts.lt(t)]
    if len(past):
        return float(past.numeric_value.iloc[-1])
    values = temps.loc[temps.channel_id.eq(channel), "numeric_value"]
    return float(values.median()) if len(values) else 18.0


def write_temperature(w, incident, channel, points):
    """Integer readings, written when the value moves by 2 degrees (as the real sensors do)."""
    last = None
    for ts, value in points:
        value = int(round(value))
        if last is None or abs(value - last) >= 2:
            w.numeric(incident, channel, ts, value)
            last = value


def inject_hot_works(w, catalog, hot_works, temps, rng, rates):
    for p in hot_works.itertuples():
        smoke = catalog.of(p.object_id, "smoke")
        place = smoke.sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
        channels = nearest(smoke, place.pk, rng, 2 if rng.random() < 0.4 else 1)
        incident = w.incident(
            scenario="hot_work_smoke",
            object_id=int(p.object_id),
            start=p.start,
            channel_ids=[int(c) for c in channels.channel_id],
            permit_id=p.permit_id if p.registered else None,
            registered=bool(p.registered),
        )
        window = (p.end - p.start).total_seconds() - 3600
        bursts = sorted(
            p.start + pd.Timedelta(seconds=1800 + rng.uniform(0, window)) for _ in range(rng.integers(2, 5))
        )
        w.truth[-1]["start"] = bursts[0]
        temp = nearest(catalog.of(p.object_id, "temp"), place.pk, rng)
        heavy = rng.random() < rates["heavy_hot_work_share"]  # long cutting: smoke holds 10-30 minutes
        w.truth[-1]["heavy"] = bool(heavy)
        for b in bursts:
            for c in channels.itertuples():
                start = b + pd.Timedelta(seconds=rng.uniform(0, 90))
                hold = rng.uniform(600, 1800) if heavy else rng.uniform(60, 600)
                w.add(incident, c.channel_id, start, "Обнаружен дым", True)
                w.add(incident, c.channel_id, start + pd.Timedelta(seconds=hold), "Дыма нет", False)
            close = len(temp) and (
                pd.isna(place.pk) or pd.isna(temp.pk.iloc[0]) or abs(temp.pk.iloc[0] - place.pk) <= 30
            )
            if close:
                channel = int(temp.channel_id.iloc[0])
                base = temperature_baseline(temps, channel, b)
                bump = rng.uniform(2, 6)
                write_temperature(
                    w,
                    incident,
                    channel,
                    [
                        (b + pd.Timedelta(minutes=rng.uniform(2, 8)), base + bump),
                        (b + pd.Timedelta(minutes=rng.uniform(25, 60)), base),
                    ],
                )


def inject_fires(w, catalog, hot_works, temps, rng, rates):
    """A fire on any object with smoke detectors; heating beforehand shows only where a
    temperature sensor stands nearby. A third smoulder: short on-off smoke on one detector
    for 20-60 minutes before it spreads."""
    objects = catalog.objects_with("smoke")
    hot = hot_works
    for _ in range(planned(rates["real_fires_per_week"])):
        during = rng.random() < rates["fire_during_hot_work_share"] and len(hot)
        if during:
            p = hot.iloc[int(rng.integers(len(hot)))]
            obj = int(p.object_id)
            tf = p.start + (p.end - p.start) * rng.uniform(0.3, 0.9)
        else:
            obj = int(rng.choice(objects))
            tf = PERIOD[0] + (PERIOD[1] - PERIOD[0]) * rng.uniform(0.02, 0.99)
        smoke_all = catalog.of(obj, "smoke")
        origin = smoke_all.sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
        smoke = nearest(smoke_all, origin.pk, rng, int(rng.integers(2, 6)))
        temp = nearest(catalog.of(obj, "temp"), origin.pk, rng)
        if (
            len(temp)
            and pd.notna(temp.pk.iloc[0])
            and pd.notna(origin.pk)
            and abs(temp.pk.iloc[0] - origin.pk) > 30
        ):
            temp = temp.iloc[0:0]
        heat = nearest(catalog.of(obj, "heat"), origin.pk, rng)
        precursor = pd.Timedelta(hours=rng.uniform(2, 12))
        smoulder = rng.random() < rates["smoulder_share"]
        incident = w.incident(
            scenario="real_fire",
            object_id=obj,
            start=tf,
            precursor_start=tf - precursor if len(temp) else tf,
            channel_ids=[*map(int, temp.channel_id), *map(int, smoke.channel_id)],
            during_hot_work=bool(during),
            smoulder=bool(smoulder),
            heating_visible=bool(len(temp)),
        )
        out = tf + pd.Timedelta(minutes=rng.uniform(40, 150))
        spread = tf
        if smoulder:
            first = int(smoke.channel_id.iloc[0])
            t = tf
            spread = tf + pd.Timedelta(minutes=rng.uniform(20, 60))
            while t < spread:
                w.add(incident, first, t, "Обнаружен дым", True)
                t += pd.Timedelta(minutes=rng.uniform(1, 4))
                w.add(incident, first, min(t, spread), "Дыма нет", False)
                t += pd.Timedelta(minutes=rng.uniform(3, 10))
            out = max(out, spread + pd.Timedelta(minutes=30))
        if len(temp):
            channel = int(temp.channel_id.iloc[0])
            slope = rng.uniform(1, 3)
            base = temperature_baseline(temps, channel, tf - precursor)
            hours = precursor / HOUR
            top = base + slope * hours
            peak = top + rng.uniform(15, 40)
            points = [(tf - precursor + HOUR * h, base + slope * h) for h in np.arange(0, hours, 0.25)]
            points += [
                (spread + pd.Timedelta(minutes=m), top + (peak - top) * m / 30) for m in range(5, 31, 5)
            ]
            points += [(out + HOUR * h, peak - (peak - base) * h / 6) for h in np.arange(0, 6.25, 0.5)]
            write_temperature(w, incident, channel, points)
        t = spread
        for c in smoke.itertuples():
            w.add(incident, c.channel_id, t, "Обнаружен дым", True)
            w.add(incident, c.channel_id, out + pd.Timedelta(minutes=rng.uniform(0, 20)), "Дыма нет", False)
            t += pd.Timedelta(minutes=rng.uniform(2, 8))
        if len(heat):
            channel = int(heat.channel_id.iloc[0])
            w.add(incident, channel, spread + pd.Timedelta(minutes=rng.uniform(5, 15)), "Не замкнут", True)
            w.add(incident, channel, out + pd.Timedelta(minutes=rng.uniform(10, 40)), "Норма", False)


def equipment_history(processed, channels):
    con = duckdb.connect()
    frame = con.execute(
        # event_id breaks ties of one channel within a second, so every run reads the same order.
        "SELECT channel_id, ts, value, alarm FROM read_parquet(?) WHERE list_contains(?, channel_id) "
        "ORDER BY channel_id, ts, event_id",
        [str(processed / "events-2026.parquet"), [int(c) for c in channels]],
    ).df()
    con.close()
    return frame


def build_passports(catalog, rng):
    units = catalog.channels[catalog.channels.group.isin(["pump", "fan"])].copy()
    units["unit"] = np.where(units.group.eq("pump"), "Насос", "Вентилятор")
    years = rng.uniform(2, 18, len(units))
    start = pd.Timestamp("2026-01-01")
    units["installed"] = (start - pd.to_timedelta(years * 365.25, unit="D")).normalize()
    units["last_maintenance"] = (
        start - pd.to_timedelta(rng.uniform(30, 365, len(units)), unit="D")
    ).normalize()
    units["resource_hours"] = np.where(units.group.eq("pump"), PUMP_RESOURCE_HOURS, FAN_RESOURCE_HOURS)
    return units[
        [
            "channel_id",
            "object_id",
            "group",
            "unit",
            "sensor_name",
            "installed",
            "last_maintenance",
            "resource_hours",
        ]
    ].reset_index(drop=True)


def inject_wear(w, catalog, passports, history, gases, rng, group, rates):
    """Wear for days, then failure; a failed pump floods its pit, a failed fan lets gas rise."""
    on = history[history.value.eq("Включен")]
    rate = on.groupby("channel_id").size() / ((PERIOD[1] - PERIOD[0]).days + 7)
    # Units that actually run: a pump at least twice a day, a fan at least once a week.
    units = passports[
        passports.group.eq(group)
        & passports.channel_id.isin(rate[rate >= (2 if group == "pump" else 1 / 7)].index)
    ]
    age = (pd.Timestamp("2026-01-01") - units.installed).dt.days.to_numpy(dtype=float)
    weights = age**2 / (age**2).sum()
    key = "pump_failures_per_half_year" if group == "pump" else "fan_failures_per_half_year"
    count = min(len(units), int(round(rates[key] * WEEKS / 26)))
    for i in rng.choice(len(units), count, replace=False, p=weights):
        unit = units.iloc[i]
        channel = int(unit.channel_id)
        tf = PERIOD[0] + (PERIOD[1] - PERIOD[0]) * rng.uniform(0.1, 0.97)
        wear = pd.Timedelta(days=rng.uniform(3, 10))
        repaired = tf + pd.Timedelta(hours=rng.uniform(12, 48))
        incident = w.incident(
            scenario=f"{group}_failure",
            object_id=int(unit.object_id),
            start=tf,
            precursor_start=tf - wear,
            end=repaired,
            channel_ids=[channel],
        )
        w.suppressed.append((channel, tf - wear, repaired, incident))
        own = on[on.channel_id.eq(channel)].ts
        gaps = own.diff().dt.total_seconds().dropna()
        gap = float(np.clip(gaps.median() if len(gaps) else 6000, 300, 6 * 3600))
        runs = history[history.channel_id.eq(channel)]
        after_on = runs.ts.shift(-1) - runs.ts
        run = (
            float(np.clip(after_on[runs.value.eq("Включен")].dt.total_seconds().median(), 15, 1800))
            if len(own)
            else 120.0
        )
        t = tf - wear
        while t < tf:
            progress = (t - (tf - wear)) / wear
            t += pd.Timedelta(seconds=rng.exponential(gap / (1 + 2 * progress**1.5)))
            if t >= tf:
                break
            short = group == "fan" and rng.random() < 0.6 * progress
            duration = rng.uniform(5, 40) if short else run * (1 - 0.5 * progress) * rng.lognormal(0, 0.2)
            w.add(incident, channel, t, "Включен", False)
            w.add(incident, channel, t + pd.Timedelta(seconds=duration), "Выключен", False)
            if progress > 0.4 and rng.random() < 0.08 * progress:
                state = "Обесточен" if rng.random() < 0.5 else "Неисправен"
                ft = t + pd.Timedelta(seconds=duration + rng.uniform(10, 120))
                w.add(incident, channel, ft, state, True)
                w.add(incident, channel, ft + pd.Timedelta(seconds=rng.uniform(30, 300)), "Выключен", False)
        w.add(incident, channel, tf, "Неисправен", True)
        if group == "pump":
            flood = tf + pd.Timedelta(hours=rng.uniform(2, 8))
            if flood < repaired:
                w.add(incident, channel, flood, "Затоплен", True)
        w.add(incident, channel, repaired, "Норма", False)
        if group == "fan":
            inject_gas(w, catalog, unit, tf, repaired, gases, rng)


def inject_gas(w, catalog, unit, tf, repaired, gases, rng):
    sensors = nearest(
        catalog.of(unit.object_id, "gas"),
        catalog.channels.set_index("channel_id").pk.get(unit.channel_id),
        rng,
        int(rng.integers(1, 3)),
    )
    if sensors.empty:
        return
    rise = rng.uniform(0.05, 0.15)
    begin = tf + pd.Timedelta(hours=rng.uniform(1, 4))
    end = repaired + 2 * HOUR
    incident = w.incident(
        scenario="gas_rise",
        object_id=int(unit.object_id),
        start=pd.NaT,  # the first gas alarm, if the level reaches it before the repair
        precursor_start=begin,
        end=end,
        channel_ids=[int(c) for c in sensors.channel_id],
        cause=f"Отказ вентилятора {unit.sensor_name}",
    )
    for s in sensors.itertuples():
        base = gases.get(int(s.channel_id), 0.01)
        w.suppressed.append((int(s.channel_id), begin, end, incident))
        t, alarmed, peak = begin, False, base
        while t < end:
            if t < repaired:
                value = base + rise * ((t - begin) / HOUR) * rng.uniform(0.9, 1.1)
                peak = value
            else:
                value = max(base, peak * (1 - ((t - repaired) / (2 * HOUR))))
            w.numeric(incident, s.channel_id, t, value, 2)
            if value >= 1.0 and not alarmed:
                w.add(incident, s.channel_id, t, "Обнаружен газ", True)
                alarmed = True
                first = w.truth[-1]["start"]
                if pd.isna(first) or t < first:
                    w.truth[-1]["start"] = t
            if alarmed and value < 0.3:
                w.add(incident, s.channel_id, t, "Норма", False)
                alarmed = False
            t += pd.Timedelta(minutes=10)


def inject_cascades(w, catalog, rng, rates):
    objects = catalog.channels.groupby("object_id").size()
    objects = objects[objects >= 10].index
    for _ in range(planned(rates["cascades_per_week"])):
        obj = int(rng.choice(objects))
        pool = catalog.channels[catalog.channels.object_id.eq(obj)]
        centre = pool.sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
        group = nearest(pool, centre.pk, rng, int(min(len(pool), rng.integers(5, 41))))
        t0 = PERIOD[0] + (PERIOD[1] - PERIOD[0]) * rng.uniform(0.02, 0.99)
        restored = t0 + pd.Timedelta(hours=rng.uniform(0.5, 6))
        incident = w.incident(
            scenario="cascade",
            object_id=obj,
            start=t0,
            end=restored,
            channel_ids=[int(c) for c in group.channel_id],
        )
        for c in group.itertuples():
            state = "Отключено устройство" if rng.random() < 0.3 else "Неисправен"
            w.add(incident, c.channel_id, t0 + pd.Timedelta(seconds=rng.uniform(0, 600)), state, True)
            w.add(
                incident, c.channel_id, restored + pd.Timedelta(seconds=rng.uniform(0, 300)), "Норма", False
            )


def episodes_source(processed=PROCESSED):
    """Episode file the worlds are built from: the hourly 2026 build when present."""
    full = processed / "episodes-all-2026.parquet"
    return full if full.exists() else processed / "episodes.parquet"


def input_files(processed=PROCESSED):
    """Every prepared file a polygon run reads, for the report's reproducibility record."""
    return {
        "channels": processed / "channels.parquet",
        "objects": processed / "objects.parquet",
        "events_2026": processed / "events-2026.parquet",
        "episodes": episodes_source(processed),
    }


def generate(seed, processed=PROCESSED, catalog=None, rates=None):
    rates = {**RATES, **(rates or {})}
    rng = np.random.default_rng(seed)
    catalog = catalog or Catalog(processed)
    episodes = pd.read_parquet(episodes_source(processed))
    permits, hot_works = build_permits(episodes, catalog, rng, rates)
    con = duckdb.connect()
    temps = con.execute(
        "SELECT channel_id, ts, numeric_value FROM read_parquet(?) WHERE list_contains(?, channel_id) "
        "AND numeric_value IS NOT NULL ORDER BY ts, channel_id, event_id",
        [
            str(processed / "events-2026.parquet"),
            [int(c) for c in catalog.channels.loc[catalog.channels.group.eq("temp"), "channel_id"]],
        ],
    ).df()
    gases = con.execute(
        "SELECT channel_id, median(numeric_value) AS base FROM read_parquet(?) WHERE list_contains(?, channel_id) "
        "AND numeric_value BETWEEN 0 AND 0.5 GROUP BY 1 ORDER BY 1",
        [
            str(processed / "events-2026.parquet"),
            [int(c) for c in catalog.channels.loc[catalog.channels.group.eq("gas"), "channel_id"]],
        ],
    ).df()
    con.close()
    gases = dict(zip(gases.channel_id.astype(int), gases.base, strict=True))
    passports = build_passports(catalog, rng)
    history = equipment_history(processed, passports.channel_id)
    w = Writer()
    inject_intrusions(w, catalog, permits, rng, rates)
    inject_hot_works(w, catalog, hot_works, temps, rng, rates)
    inject_fires(w, catalog, hot_works, temps, rng, rates)
    inject_wear(w, catalog, passports, history, gases, rng, "pump", rates)
    inject_wear(w, catalog, passports, history, gases, rng, "fan", rates)
    inject_cascades(w, catalog, rng, rates)
    events, truth, suppressed = w.frames()
    repairs = truth[truth.scenario.isin(["pump_failure", "fan_failure"])].assign(
        channel_id=lambda f: f.channel_ids.str[0], work="Аварийный ремонт"
    )[["incident_id", "channel_id", "object_id", "start", "end", "work"]]
    return World(seed, rates, events, suppressed, truth, permits, passports, repairs)
