"""Polygon units for access triage, pump and fan wear, gas and cascades.

See ``polygon_scenarios`` for the protocol; fire triage and the fire forecast live there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from moscollector import daily_rule
from moscollector.polygon_scenarios import FAULT_STATES, HOUR, MIN, alarm_groups

# ---------------------------------------------------------------- access: triage of an alarm

ACCESS_TRIAGE_FEATURES = [
    "hour",
    "weekend",
    "night",
    "permit",
    "guard_armed",
    "door_alarms_5m",
    "motion_alarms_5m",
    "channels_5m",
    "motion_after_door",
    "activity_1h",
    "groups_7d",
    "off_hours_groups_24h",
]


def stamps(frame):
    return np.sort(frame.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64))


def access_triage_units(data):
    """An access alarm, decided five minutes after its first record."""
    e = data.events
    door_motion = e[e.group.isin(["door", "motion"])]
    alarms = door_motion[
        door_motion.alarm & door_motion.value.isin(["Не замкнут", "Обнаружено движение"])
    ].copy()
    alarms = alarm_groups(alarms)
    quiet = door_motion[~door_motion.alarm]
    quiet_by_object = {k: stamps(g) for k, g in quiet.groupby("object_id")}
    guard = e[e.group.eq("guard") & e.value.isin(["На охране", "Снято с охраны"])].sort_values("ts")
    guard_by_object = {
        k: (g.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64), g.value.eq("На охране").to_numpy())
        for k, g in guard.groupby("object_id")
    }
    rows, history = [], {}
    for _, g in alarms.groupby("group", sort=False):
        start = g.ts.iloc[0]
        decide = start + 5 * MIN
        g = g[g.ts.le(decide)]
        obj = int(g.object_id.iloc[0])
        s = start.value
        q = quiet_by_object.get(obj, np.array([], dtype=np.int64))
        armed = -1
        if obj in guard_by_object:
            times, states = guard_by_object[obj]
            j = np.searchsorted(times, s, side="right") - 1
            armed = int(states[j]) if j >= 0 else -1
        doors = g[g.group.eq("door")]
        motions = g[g.group.eq("motion")]
        before = history.setdefault(obj, [])
        off_hours = bool(start.hour >= 22 or start.hour < 6 or start.dayofweek >= 5)
        rows.append(
            {
                "object_id": obj,
                "start": start,
                "hour": start.hour,
                "weekend": int(start.dayofweek >= 5),
                "night": int(start.hour >= 22 or start.hour < 6),
                "guard_armed": armed,
                "door_alarms_5m": len(doors),
                "motion_alarms_5m": len(motions),
                "channels_5m": g.channel_id.nunique(),
                "motion_after_door": int(len(doors) > 0 and bool((motions.ts > doors.ts.iloc[0]).any())),
                "activity_1h": int(np.searchsorted(q, s) - np.searchsorted(q, s - 3600 * 10**9)),
                "groups_7d": sum(1 for b, _ in before if start - b <= pd.Timedelta(days=7)),
                "off_hours_groups_24h": sum(
                    1 for b, off in before if off and start - b <= pd.Timedelta(hours=24)
                ),
                "intrusion": int(g.scenario.eq("intrusion").any()),
                "probe": int(g.scenario.eq("intrusion_probe").any()),
                "incident_id": next((i for i in g.incident_id if isinstance(i, str)), None),
            }
        )
        before.append((start, off_hours))
    units = pd.DataFrame(rows)
    permit, _ = data.permit_flags(units.object_id.to_numpy(), units.start.to_numpy())
    units["permit"] = permit.astype(int)
    return units


# ---------------------------------------------------------------- pumps and fans: 24-hour failure forecast

EQUIPMENT_FEATURES = [
    "pump",
    "on_24h",
    "on_72h",
    "on_baseline",
    "on_ratio",
    "run_24h",
    "run_ratio",
    "faults_24h",
    "faults_72h",
    "short_24h",
    "short_share",
    "short_share_change",
    "age_years",
    "days_since_service",
]


def _window(x, hours):
    c = np.concatenate([[0], np.cumsum(x)])
    idx = np.arange(1, len(x) + 1)
    return c[idx] - c[np.maximum(idx - hours, 0)]


def equipment_units(data, begin, end, units=None):
    """Hourly state of each pump and fan; the label is a failure within the next 24 hours.

    Hours when the unit is already in a fault state are left out: there is nothing to predict.
    """
    e = data.events[data.events.group.isin(["pump", "fan"])]
    passports = data.world.passports.set_index("channel_id")
    units = passports.index if units is None else units
    hours = pd.date_range(begin - pd.Timedelta(days=16), end - HOUR, freq="h")
    h0, n = hours[0].value, len(hours)
    day = np.arange(n) // 24
    failures = data.truth[data.truth.scenario.isin(["pump_failure", "fan_failure"])].assign(
        unit=lambda f: f.channel_ids.str[0]
    )
    frames = []
    for channel, part in e[e.channel_id.isin(units)].groupby("channel_id"):
        part = part.sort_values("ts")
        t = part.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        slot = (t - h0) // daily_rule.HOUR_NS
        ok = (slot >= 0) & (slot < n)
        on = part.value.eq("Включен").to_numpy()
        fault = part.alarm.to_numpy() & part.value.isin(FAULT_STATES).to_numpy()
        duration = np.append(np.diff(t), 0) / 1e9
        on_n = np.bincount(slot[ok & on], minlength=n).astype(float)
        run = np.bincount(slot[ok & on], weights=np.minimum(duration[ok & on], 3600), minlength=n)
        faults = np.bincount(slot[ok & fault], minlength=n).astype(float)
        # A start that stops within a minute: a motor or a protection relay giving up.
        short = np.bincount(slot[ok & on & (duration < 60)], minlength=n).astype(float)
        ends = hours.values.astype("datetime64[ns]").astype(np.int64) + daily_rule.HOUR_NS
        last = np.searchsorted(t, ends, side="right") - 1
        failed_now = np.where(last >= 0, fault[np.maximum(last, 0)], False)
        on24 = _window(on_n, 24)
        daily = pd.Series(on_n).groupby(day).sum()
        baseline = daily.shift(1).rolling(14, min_periods=3).median().reindex(day).to_numpy()
        run_day = pd.Series(run).groupby(day).sum() / daily.replace(0, np.nan)
        run_base = run_day.shift(1).rolling(14, min_periods=3).median().reindex(day).to_numpy()
        run24 = _window(run, 24) / np.maximum(on24, 1)
        short24 = _window(short, 24)
        share = short24 / np.maximum(on24, 1)
        share_day = pd.Series(short).groupby(day).sum() / daily.replace(0, np.nan)
        share_base = share_day.shift(1).rolling(14, min_periods=3).median().reindex(day).to_numpy()
        frame = pd.DataFrame(
            {
                "unit": int(channel),
                "as_of": hours + HOUR,  # the hour [h, h+1) is known at its end
                "on_24h": on24,
                "on_72h": _window(on_n, 72),
                "on_baseline": baseline,
                "on_ratio": on24 / np.maximum(np.nan_to_num(baseline, nan=0.0), 0.5),
                "run_24h": run24,
                "run_ratio": run24 / np.where(np.isnan(run_base) | (run_base <= 0), np.nan, run_base),
                "faults_24h": _window(faults, 24),
                "faults_72h": _window(faults, 72),
                "short_24h": short24,
                "short_share": share,
                "short_share_change": share - np.nan_to_num(share_base, nan=0.0),
                "failed_now": failed_now,
            }
        )
        frame = frame[frame.as_of.ge(begin) & frame.as_of.lt(end) & ~frame.failed_now].copy()
        p = passports.loc[channel]
        at = frame.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        own = failures[failures.unit.eq(int(channel))]
        repaired = np.sort(own.end.dropna().to_numpy(dtype="datetime64[ns]").astype(np.int64))
        j = np.searchsorted(repaired, at, side="right") - 1
        service = np.where(
            j >= 0, repaired[np.maximum(j, 0)] if len(repaired) else 0, p.last_maintenance.value
        )
        starts = np.sort(own.start.to_numpy(dtype="datetime64[ns]").astype(np.int64))
        frame["pump"] = int(p.group == "pump")
        frame["object_id"] = int(p.object_id)
        frame["age_years"] = (at - p.installed.value) / (365.25 * 86400e9)
        frame["days_since_service"] = (at - service) / 86400e9
        frame["target"] = (
            (np.searchsorted(starts, at + 24 * daily_rule.HOUR_NS) - np.searchsorted(starts, at)) > 0
        ).astype(int)
        frames.append(frame.drop(columns="failed_now"))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- gas: 24-hour alarm forecast

GAS_FEATURES = ["hour", "gas_now", "gas_3h", "gas_rise6", "gas_rise24", "fans_failed", "fan_faults_24h"]


def gas_units(data, begin, end):
    """Object-hours with gas sensors; the label is a gas alarm (real or injected) within 24 h."""
    gas = data.gas.merge(data.catalog.channels[["channel_id", "object_id"]], on="channel_id")
    hourly = (
        gas.groupby(["object_id", "hour"]).agg(gas=("gas", "max"), alarm=("gas_alarm", "max")).reset_index()
    )
    hours = pd.date_range(begin - pd.Timedelta(days=2), end - HOUR, freq="h")
    fans = data.events[data.events.group.eq("fan")].sort_values("ts")
    frames, alarms = [], []
    for obj, part in hourly.groupby("object_id"):
        s = part.set_index("hour").reindex(hours)
        level = s.gas.ffill()
        alarm = s.alarm.astype("boolean").fillna(False).to_numpy(dtype=bool)
        frame = pd.DataFrame(
            {
                "object_id": int(obj),
                "as_of": hours + HOUR,
                "gas_now": level.to_numpy(),
                "gas_3h": level.rolling(3, min_periods=1).max().to_numpy(),
                "gas_rise6": (level - level.shift(6)).to_numpy(),
                "gas_rise24": (level - level.shift(24)).to_numpy(),
            }
        )
        starts = hours[alarm & ~np.concatenate([[False], alarm[:-1]])]
        at = frame.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        failed = np.zeros(len(frame))
        fan_faults = np.zeros(len(frame))
        for _, f in fans[fans.object_id.eq(obj)].groupby("channel_id"):
            t = f.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
            bad = f.alarm.to_numpy() & f.value.isin(FAULT_STATES).to_numpy()
            j = np.searchsorted(t, at, side="right") - 1
            failed += np.where(j >= 0, bad[np.maximum(j, 0)], False)
            bt = t[bad]
            fan_faults += np.searchsorted(bt, at) - np.searchsorted(bt, at - 24 * daily_rule.HOUR_NS)
        frame["fans_failed"] = failed
        frame["fan_faults_24h"] = fan_faults
        frame["hour"] = frame.as_of.dt.hour
        st = starts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        frame["target"] = (
            (np.searchsorted(st, at + 24 * daily_rule.HOUR_NS) - np.searchsorted(st, at)) > 0
        ).astype(int)
        frames.append(frame[frame.as_of.ge(begin) & frame.as_of.lt(end)])
        alarms.append(pd.DataFrame({"object_id": int(obj), "start_ts": starts}))
    return pd.concat(frames, ignore_index=True), pd.concat(alarms, ignore_index=True)


# ---------------------------------------------------------------- cascades: rule


def cascades(data, channels=5, window=10 * MIN):
    """Five or more channels of one object entering a fault state within 10 minutes."""
    faults = data.events[data.events.alarm & data.events.value.isin(FAULT_STATES)].sort_values(
        ["object_id", "ts"]
    )
    found = []
    span = np.timedelta64(int(window.total_seconds()), "s")
    for obj, part in faults.groupby("object_id"):
        t = part.ts.to_numpy()
        ch = part.channel_id.to_numpy()
        inc = part.incident_id.to_numpy()
        i = 0
        while i < len(t):
            j = np.searchsorted(t, t[i] + span, side="right")
            if len(set(ch[i:j])) >= channels:
                ids = [x for x in inc[i:j] if isinstance(x, str)]
                found.append(
                    {
                        "object_id": int(obj),
                        "start": pd.Timestamp(t[i]),
                        "channels": len(set(ch[i:j])),
                        "incident_id": ids[0] if ids else None,
                    }
                )
                i = j
            else:
                i += 1
    return pd.DataFrame(found)
