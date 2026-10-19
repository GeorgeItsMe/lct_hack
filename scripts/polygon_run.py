"""Polygon run: generate worlds, train scenario models, score June against the truth.

Training worlds 1-8 (January-April to train, May to choose thresholds). Test worlds 101-110
score June once; incidents there were never seen in training. Robustness worlds 201-210 make
the same incidents harder to tell from routine work (``HARD_RATES``) and are scored with the
models and thresholds chosen on the ordinary worlds. Every world keeps the real 2026 journal
as background.

Writes ``artifacts/polygon_report.json`` and, for the service, the report, a June feed of
world 101 and that world's registries, truth and injected records into
``vercel_runtime/artifacts/polygon``. Needs the full prepared 2026 data.

    python scripts/polygon_run.py [--reuse] [--refresh KIND ...]
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version

import numpy as np
import pandas as pd

from moscollector import daily_rule
from moscollector import polygon_scenarios as ps
from moscollector import polygon_units as pu
from moscollector.paths import BUNDLE, PROCESSED, ROOT
from moscollector.polygon import HARD_RATES, RATES, Catalog, generate, input_files

TRAIN_WORLDS = range(1, 9)
TEST_WORLDS = range(101, 111)
HARD_WORLDS = range(201, 211)
FEED_WORLD = 101
CACHE = ROOT / "data" / "polygon"
OUT = BUNDLE / "artifacts" / "polygon"
REPORT = ROOT / "artifacts" / "polygon_report.json"
NAMES = (
    "fire_triage",
    "access_triage",
    "fire_forecast",
    "fire_starts",
    "equipment",
    "gas",
    "gas_starts",
    "cascades",
    "truth",
)
START = time.time()


def log(*args):
    print(f"[{time.time() - START:6.0f}s]", *args, flush=True)


def running_units(catalog):
    """Pumps that start at least twice a day and fans at least once a week (as in the generator)."""
    real = ps.real_events(PROCESSED, catalog, CACHE / "real_events.parquet")
    on = real[real.value.eq("Включен")].groupby("channel_id").size() / 181
    units = catalog.channels[catalog.channels.group.isin(["pump", "fan"])].set_index("channel_id")
    return [c for c, g in units.group.items() if on.get(c, 0) >= (2 if g == "pump" else 1 / 7)]


def keyed(frame, key="object_id"):
    """Worlds share object ids: give every world its own ids for per-object rules."""
    return frame.assign(**{key: frame.world * 1_000_000 + frame[key]})


def label_window(frame, starts, key="object_id"):
    """1 when an incident of that object starts within 24 hours after the row's moment."""
    out = np.zeros(len(frame), dtype=int)
    by = {
        k: np.sort(g.start_ts.to_numpy(dtype="datetime64[ns]").astype(np.int64))
        for k, g in starts.groupby(key)
    }
    for k, part in frame.groupby(key):
        st = by.get(k)
        if st is None:
            continue
        t = part.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        out[part.index] = (np.searchsorted(st, t + 24 * daily_rule.HOUR_NS) - np.searchsorted(st, t)) > 0
    return out


class Context:
    def __init__(self):
        self.catalog = Catalog()
        self.real = ps.real_events(PROCESSED, self.catalog, CACHE / "real_events.parquet")
        self.gas = ps.real_gas_hourly(PROCESSED, self.catalog, CACHE / "real_gas_hourly.parquet")
        self.units = running_units(self.catalog)


def collect(ctx, seeds, rates=None, keys=None):
    keys = set(keys or NAMES)
    store = {k: [] for k in NAMES if k in keys}
    for seed in seeds:
        world = generate(seed, catalog=ctx.catalog, rates=rates)
        data = ps.WorldData(world, ctx.catalog, ctx.real, ctx.gas)
        begin, end = (ps.TRAIN[0], ps.SELECT[1]) if seed < 100 else ps.TEST
        tag = {"world": seed}
        if "truth" in keys:
            store["truth"].append(world.truth.assign(**tag))
        if "fire_triage" in keys:
            store["fire_triage"].append(ps.fire_triage_units(data).assign(**tag))
        if "access_triage" in keys:
            store["access_triage"].append(pu.access_triage_units(data).assign(**tag))
        if "fire_forecast" in keys:
            grid, fires = ps.fire_forecast_units(data, begin, end)
            grid["target"] = label_window(grid, fires)
            store["fire_forecast"].append(grid.assign(**tag))
            store["fire_starts"].append(fires.assign(**tag))
        if "equipment" in keys:
            equipment = pu.equipment_units(data, begin, end, ctx.units)
            if seed < 100:
                # Training needs every failure hour, the hard cases around wear and a sample of calm hours.
                hard = equipment.target.eq(1) | equipment.on_ratio.gt(1.5) | equipment.faults_72h.gt(0)
                hard |= equipment.short_share_change.gt(0.2)
                calm = np.random.default_rng(seed).random(len(equipment)) < 0.03
                equipment = equipment[hard | calm | equipment.as_of.ge(ps.SELECT[0])]
            store["equipment"].append(equipment.assign(**tag))
        if "gas" in keys:
            gas_grid, gas_starts = pu.gas_units(data, begin, end)
            store["gas"].append(gas_grid.assign(**tag))
            store["gas_starts"].append(gas_starts.assign(**tag))
        if "cascades" in keys:
            store["cascades"].append(pu.cascades(data).assign(**tag))
        if seed == FEED_WORLD:
            export_world(world)
        log("world", seed, {k: len(v[-1]) for k, v in store.items()})
    return {k: pd.concat(v, ignore_index=True) for k, v in store.items()}


def cached(ctx, prefix, seeds, rates, reuse, refresh):
    files = {n: CACHE / f"{prefix}-{n}.parquet" for n in NAMES}
    if reuse and all(f.exists() for f in files.values()):
        data = {n: pd.read_parquet(f) for n, f in files.items()}
        if not refresh:
            return data
        fresh = collect(ctx, seeds, rates, refresh)
    else:
        data, fresh = {}, collect(ctx, seeds, rates)
    data.update(fresh)
    for n, frame in fresh.items():
        if "channel_ids" in frame:
            frame = frame.assign(channel_ids=frame.channel_ids.map(list))
        frame.to_parquet(files[n], index=False)
    return data


def in_period(frame, column, period):
    return frame[frame[column].ge(period[0]) & frame[column].lt(period[1])]


def per_day(count, worlds):
    return count / (len(worlds) * 30)


def importance(model, features):
    pairs = zip(features, map(float, model.get_feature_importance()), strict=True)
    return dict(sorted(pairs, key=lambda x: -x[1])[:6])


# ---------------------------------------------------------------- triage


def train_triage(units, features, label):
    train = in_period(units[units.world.lt(100)], "start", ps.TRAIN)
    select = in_period(units[units.world.lt(100)], "start", ps.SELECT)
    model = ps.fit(train, features, label)
    choice = ps.select_unit_threshold(model.predict_proba(select[features])[:, 1], select[label].to_numpy())
    return model, choice


def apply_triage(model, choice, units, features, label, scenario, truth, worlds):
    test = in_period(units[units.world.isin(list(worlds))], "start", ps.TEST).copy()
    test["score"] = model.predict_proba(test[features])[:, 1]
    test["flagged"] = test.score >= choice["threshold"]
    incidents = in_period(
        truth[truth.world.isin(list(worlds)) & truth.scenario.eq(scenario)], "start", ps.TEST
    )
    caught_ids = set(test.loc[test.flagged & test[label].eq(1), "incident_id"].dropna())
    caught = incidents.incident_id.isin(caught_ids)
    false = int((test.flagged & test[label].eq(0)).sum())
    result = {
        "threshold": choice["threshold"],
        "selection": {k: choice[k] for k in ("rule", "precision", "recall", "flagged", "positives")},
        "incidents": len(incidents),
        "caught": int(caught.sum()),
        "recall": float(caught.mean()) if len(incidents) else None,
        "escalated": int(test.flagged.sum()),
        "false_escalations": false,
        "precision": float((test.flagged & test[label].eq(1)).sum() / max(test.flagged.sum(), 1)),
        "alarms_per_day": per_day(len(test), worlds),
        "escalated_per_day": per_day(int(test.flagged.sum()), worlds),
        "false_per_day": per_day(false, worlds),
        "importance": importance(model, features),
    }
    return test, result


# ---------------------------------------------------------------- forecasts


def scored(model, frame, features, key):
    pred = frame[[key, "as_of", "world"]].copy()
    pred["probability"] = model.predict_proba(frame[features])[:, 1]
    return pred


def as_rule(pred, key):
    return keyed(pred, key).drop(columns="world").rename(columns={key: "object_id"})


def starts_of(starts, worlds, period, key):
    part = in_period(starts[starts.world.isin(list(worlds))], "start_ts", period)
    return keyed(part, key).drop(columns="world").rename(columns={key: "object_id"})


def train_forecast(grid, starts, features, key="object_id"):
    train = in_period(grid[grid.world.lt(100)], "as_of", ps.TRAIN)
    select = in_period(grid[grid.world.lt(100)], "as_of", ps.SELECT)
    model = ps.fit(train, features, "target")
    choice = daily_rule.select_threshold(
        as_rule(scored(model, select, features, key), key),
        starts_of(starts, TRAIN_WORLDS, ps.SELECT, key),
        points=150,
    )
    return model, choice


def apply_forecast(model, choice, grid, starts, features, worlds, key="object_id", subsets=None):
    test = in_period(grid[grid.world.isin(list(worlds))], "as_of", ps.TEST)
    pred = scored(model, test, features, key)
    rule = as_rule(pred, key)
    result = daily_rule.metrics(rule, starts_of(starts, worlds, ps.TEST, key), choice["threshold"])
    keep = (
        "threshold",
        "precision",
        "recall",
        "alerts",
        "true_alerts",
        "incident_days",
        "caught_incident_days",
        "median_lead_hours",
    )
    out = {k: result[k] for k in keep}
    out["alerts_per_day"] = per_day(result["alerts"], worlds)
    out["false_alerts_per_day"] = per_day(result["false_alerts"], worlds)
    out["selection"] = {k: choice[k] for k in ("rule", "precision", "recall", "alerts")}
    out["importance"] = importance(model, features)
    for name, subset in (subsets or {}).items():
        part = (
            keyed(subset[subset.world.isin(list(worlds))], key)
            .drop(columns="world")
            .rename(columns={key: "object_id"})
        )
        part = in_period(part, "start_ts", ps.TEST)
        sub = daily_rule.metrics(rule, part, choice["threshold"])
        out[name] = {
            k: sub[k] for k in ("incident_days", "caught_incident_days", "recall", "median_lead_hours")
        }
    return pred, out


# ---------------------------------------------------------------- evaluation


def heated_fires(t):
    fires = t[t.scenario.eq("real_fire") & t.heating_visible.astype(bool)]
    return fires.rename(columns={"start": "start_ts"})[["object_id", "start_ts", "world"]]


def unit_failures(t, scenario):
    f = t[t.scenario.eq(scenario)].assign(unit=lambda x: x.channel_ids.str[0])
    return f.rename(columns={"start": "start_ts"})[["unit", "start_ts", "world"]]


def fan_gas(t):
    g = t[t.scenario.eq("gas_rise") & t.start.notna()]
    return g.rename(columns={"start": "start_ts"})[["object_id", "start_ts", "world"]]


def evaluate(base, hard):
    runs = (("scenarios", base, TEST_WORLDS), ("hard", hard, HARD_WORLDS))
    report, effect, feed = {"scenarios": {}, "hard": {}}, {}, {}

    model, choice = train_triage(base["fire_triage"], ps.FIRE_TRIAGE_FEATURES, "real_fire")
    for name, data, worlds in runs:
        test, result = apply_triage(
            model,
            choice,
            data["fire_triage"],
            ps.FIRE_TRIAGE_FEATURES,
            "real_fire",
            "real_fire",
            data["truth"],
            worlds,
        )
        hot = test.hot_work_truth.eq(1) & test.real_fire.eq(0)
        result["hot_work_alarms_per_day"] = per_day(int(hot.sum()), worlds)
        result["hot_work_escalated"] = int((hot & test.flagged).sum())
        result["hot_work_with_permit"] = float(test.loc[hot, "hot_work"].mean()) if hot.any() else None
        report[name]["fire_triage"] = result
        if name == "scenarios":
            feed["fire_triage"] = test[test.world.eq(FEED_WORLD)]
            labelled = hot & test.hot_work.eq(1) & ~test.flagged
            effect["fire"] = {
                "alarms_per_day": result["alarms_per_day"],
                "raised_per_day": result["escalated_per_day"],
                "hot_work_per_day": per_day(int(labelled.sum()), worlds),
                "low_priority_per_day": per_day(int((~test.flagged & ~labelled).sum()), worlds),
            }
    log(
        "fire triage",
        {n: report[n]["fire_triage"]["caught"] for n in report},
        "of",
        {n: report[n]["fire_triage"]["incidents"] for n in report},
    )

    model, choice = train_triage(base["access_triage"], pu.ACCESS_TRIAGE_FEATURES, "intrusion")
    for name, data, worlds in runs:
        test, result = apply_triage(
            model,
            choice,
            data["access_triage"],
            pu.ACCESS_TRIAGE_FEATURES,
            "intrusion",
            "intrusion",
            data["truth"],
            worlds,
        )
        routine = test.intrusion.eq(0) & ~test.flagged
        result["permitted_per_day"] = per_day(int((routine & test.permit.eq(1)).sum()), worlds)
        result["no_permit_per_day"] = per_day(int((routine & test.permit.eq(0)).sum()), worlds)
        report[name]["access_triage"] = result
        if name == "scenarios":
            feed["access_triage"] = test[test.world.eq(FEED_WORLD)]
            effect["access"] = {
                "alarms_per_day": result["alarms_per_day"],
                "trips_per_day": result["escalated_per_day"],
                "permitted_per_day": result["permitted_per_day"],
                "calls_per_day": result["no_permit_per_day"],
            }
    log(
        "access triage",
        {n: report[n]["access_triage"]["caught"] for n in report},
        "of",
        {n: report[n]["access_triage"]["incidents"] for n in report},
    )

    model, choice = train_forecast(base["fire_forecast"], base["fire_starts"], ps.FIRE_FORECAST_FEATURES)
    for name, data, worlds in runs:
        pred, result = apply_forecast(
            model,
            choice,
            data["fire_forecast"],
            data["fire_starts"],
            ps.FIRE_FORECAST_FEATURES,
            worlds,
            subsets={"with_visible_heating": heated_fires(data["truth"])},
        )
        report[name]["fire_forecast"] = result
        if name == "scenarios":
            feed["fire_forecast"] = (pred[pred.world.eq(FEED_WORLD)], choice["threshold"])
    log("fire forecast", report["scenarios"]["fire_forecast"])

    for group, scenario in (("pump", "pump_failure"), ("fan", "fan_failure")):
        flag = int(group == "pump")
        grid = base["equipment"][base["equipment"].pump.eq(flag)]
        model, choice = train_forecast(
            grid, unit_failures(base["truth"], scenario), pu.EQUIPMENT_FEATURES, key="unit"
        )
        for name, data, worlds in runs:
            grid = data["equipment"][data["equipment"].pump.eq(flag)]
            pred, result = apply_forecast(
                model,
                choice,
                grid,
                unit_failures(data["truth"], scenario),
                pu.EQUIPMENT_FEATURES,
                worlds,
                key="unit",
            )
            report[name][f"{group}_forecast"] = result
            if name == "scenarios":
                feed[f"{group}_forecast"] = (pred[pred.world.eq(FEED_WORLD)], choice["threshold"])
        log(group, report["scenarios"][f"{group}_forecast"])

    model, choice = train_forecast(base["gas"], base["gas_starts"], pu.GAS_FEATURES)
    for name, data, worlds in runs:
        pred, result = apply_forecast(
            model,
            choice,
            data["gas"],
            data["gas_starts"],
            pu.GAS_FEATURES,
            worlds,
            subsets={"fan_caused": fan_gas(data["truth"])},
        )
        report[name]["gas_forecast"] = result
        if name == "scenarios":
            feed["gas_forecast"] = (pred[pred.world.eq(FEED_WORLD)], choice["threshold"])
    log("gas", report["scenarios"]["gas_forecast"])

    for name, data, worlds in runs:
        t = data["truth"]
        injected = in_period(t[t.world.isin(list(worlds)) & t.scenario.eq("cascade")], "start", ps.TEST)
        found = in_period(data["cascades"][data["cascades"].world.isin(list(worlds))], "start", ps.TEST)
        report[name]["cascades"] = {
            "incidents": len(injected),
            "caught": int(injected.incident_id.isin(set(found.incident_id.dropna())).sum()),
            "detections_per_day": per_day(len(found), worlds),
            "background_detections_per_day": per_day(int(found.incident_id.isna().sum()), worlds),
        }
        if name == "scenarios":
            feed["cascades"] = found[found.world.eq(FEED_WORLD)]
    forecasts = ("fire_forecast", "pump_forecast", "fan_forecast", "gas_forecast")
    effect["forecasts_per_day"] = {k: report["scenarios"][k]["alerts_per_day"] for k in forecasts}
    report["effect"] = effect
    return report, feed


# ---------------------------------------------------------------- feed and dataset of the demo world


def export_world(world):
    """Registries, truth and injected records of the demo world, for the service and download."""
    OUT.mkdir(parents=True, exist_ok=True)
    world.permits.to_csv(OUT / "permits.csv", index=False, encoding="utf-8")
    world.passports.to_csv(OUT / "passports.csv", index=False, encoding="utf-8")
    world.repairs.to_csv(OUT / "repairs.csv", index=False, encoding="utf-8")
    truth = world.truth.assign(channel_ids=world.truth.channel_ids.map(lambda c: " ".join(map(str, c))))
    truth.to_csv(OUT / "truth.csv", index=False, encoding="utf-8")
    world.events.to_parquet(OUT / "injected.parquet", index=False)
    world.suppressed.to_parquet(OUT / "suppressed.parquet", index=False)


SCENARIO_LABELS = {
    "real_fire": "Возгорание",
    "intrusion": "Проникновение",
    "hot_work_smoke": "Огневые работы",
    "pump_failure": "Отказ насоса",
    "fan_failure": "Отказ вентилятора",
    "gas_rise": "Рост газа",
    "cascade": "Сбой шлейфа или контроллера",
}
WARNING_TEXT = {
    "real_fire": "Предупреждение о риске пожара",
    "pump_failure": "Предупреждение об отказе насоса",
    "fan_failure": "Предупреждение об отказе вентилятора",
    "gas_rise": "Предупреждение о газе",
}


def issued(pred, threshold, key):
    return daily_rule.schedule(pred.rename(columns={key: "object_id"}), threshold)


def lead_before(schedule, key, start):
    times = schedule.get(int(key), np.array([], dtype=np.int64))
    s = pd.Timestamp(start).value
    before = times[(times <= s) & (times > s - 24 * daily_rule.HOUR_NS)]
    return (s - before.min()) / daily_rule.HOUR_NS if len(before) else None


def hours(value):
    return f"{value:.1f}".replace(".", ",")


class Feed:
    def __init__(self, catalog):
        self.objects = (
            pd.read_parquet(PROCESSED / "objects.parquet").set_index("object_id").object_name.to_dict()
        )
        self.channels = catalog.channels.set_index("channel_id")
        self.rows = []

    def add(self, when, obj, scenario, truth, system, outcome, lead=None, incident=None, kind="incident"):
        self.rows.append(
            {
                "time": pd.Timestamp(when).isoformat(),
                "object_id": int(obj),
                "object_name": self.objects.get(int(obj), str(obj)),
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
                "truth": truth,
                "system": system,
                "outcome": outcome,
                "lead_hours": None if lead is None else round(float(lead), 1),
                "incident_id": incident,
                "kind": kind,
            }
        )


def build_feed(inputs, truth, catalog):
    feed = Feed(catalog)
    t = in_period(truth[truth.world.eq(FEED_WORLD)], "start", ps.TEST)
    fire_tri, access_tri = inputs["fire_triage"], inputs["access_triage"]
    fire_warn = issued(*inputs["fire_forecast"], "object_id")
    for inc in t.itertuples():
        if inc.scenario == "real_fire":
            raised = bool(fire_tri[fire_tri.incident_id.eq(inc.incident_id)].flagged.any())
            lead = lead_before(fire_warn, inc.object_id, inc.start)
            parts = [f"предупреждение о риске пожара за {hours(lead)} ч"] if lead is not None else []
            parts.append("тревога поднята как возгорание" if raised else "тревога не поднята")
            how = (
                "во время огневых работ"
                if inc.during_hot_work
                else ("тлеющее" if inc.smoulder else "быстрое")
            )
            heat = "нагрев виден" if inc.heating_visible else "датчика температуры рядом нет"
            system = "; ".join(parts)
            feed.add(
                inc.start,
                inc.object_id,
                inc.scenario,
                f"Возгорание: {how}, {heat}",
                system[0].upper() + system[1:],
                "поймано" if raised or lead is not None else "пропущено",
                lead,
                inc.incident_id,
            )
        elif inc.scenario == "intrusion":
            raised = bool(access_tri[access_tri.incident_id.eq(inc.incident_id)].flagged.any())
            when = "в нерабочее время" if inc.off_hours else "днём в будни"
            feed.add(
                inc.start,
                inc.object_id,
                inc.scenario,
                f"Проникновение {when}, наряда нет",
                "Признаки проникновения — выезд" if raised else "Не поднята: похоже на обычную работу",
                "поймано" if raised else "пропущено",
                None,
                inc.incident_id,
            )
        elif inc.scenario == "hot_work_smoke":
            groups = fire_tri[fire_tri.incident_id.eq(inc.incident_id)]
            if groups.empty:
                continue
            raised = bool(groups.flagged.any())
            if raised:
                system = "Поднята как возгорание"
            elif groups.hot_work.any():
                system = f"Огневые работы по наряду {inc.permit_id} — позвонить бригаде"
            else:
                system = "Короткое срабатывание, низкий приоритет"
            truth_text = "Сварка по наряду" if inc.registered else "Сварка без оформленного наряда"
            feed.add(
                inc.start,
                inc.object_id,
                inc.scenario,
                truth_text,
                system,
                "ложный подъём" if raised else "верно",
                None,
                inc.incident_id,
            )
        elif inc.scenario in ("pump_failure", "fan_failure"):
            group = inc.scenario.split("_")[0]
            unit = int(inc.channel_ids[0])
            lead = lead_before(issued(*inputs[f"{group}_forecast"], "unit"), unit, inc.start)
            label = "насоса" if group == "pump" else "вентилятора"
            unit_name = str(feed.channels.sensor_name.get(unit, unit)).strip()
            system = (
                f"Предупреждение об отказе {label} за {hours(lead)} ч"
                if lead is not None
                else "Отказ не предсказан"
            )
            feed.add(
                inc.start,
                inc.object_id,
                inc.scenario,
                f"Отказ {label} {unit_name} после износа",
                system,
                "поймано" if lead is not None else "пропущено",
                lead,
                inc.incident_id,
            )
        elif inc.scenario == "gas_rise" and pd.notna(inc.start):
            lead = lead_before(issued(*inputs["gas_forecast"], "object_id"), inc.object_id, inc.start)
            system = (
                f"Предупреждение о газе за {hours(lead)} ч"
                if lead is not None
                else "Газовая тревога не предсказана"
            )
            feed.add(
                inc.start,
                inc.object_id,
                inc.scenario,
                f"Газовая тревога: {str(inc.cause).strip()}",
                system,
                "поймано" if lead is not None else "пропущено",
                lead,
                inc.incident_id,
            )
        elif inc.scenario == "cascade":
            hit = inputs["cascades"][inputs["cascades"].incident_id.eq(inc.incident_id)]
            system = (
                f"Сбой шлейфа или контроллера: {int(hit.channels.iloc[0])} каналов, не авария"
                if len(hit)
                else "Не распознан"
            )
            feed.add(
                inc.start,
                inc.object_id,
                inc.scenario,
                f"Сбой {len(inc.channel_ids)} каналов",
                system,
                "поймано" if len(hit) else "пропущено",
                None,
                inc.incident_id,
            )
    # What the system raised although nothing happened.
    for frame, scenario, label in (
        (fire_tri, "real_fire", "Поднята как возгорание"),
        (access_tri, "intrusion", "Признаки проникновения — выезд"),
    ):
        for r in frame[frame.flagged & frame[scenario].eq(0)].itertuples():
            feed.add(
                r.start,
                r.object_id,
                scenario,
                "Реального инцидента нет",
                label,
                "ложный подъём",
                kind="false",
            )
    for name, key, scenario in (
        ("fire_forecast", "object_id", "real_fire"),
        ("pump_forecast", "unit", "pump_failure"),
        ("fan_forecast", "unit", "fan_failure"),
        ("gas_forecast", "object_id", "gas_rise"),
    ):
        events = t[t.scenario.eq(scenario) & t.start.notna()]
        own = {}
        for s in events.itertuples():
            k = int(s.channel_ids[0]) if key == "unit" else int(s.object_id)
            own.setdefault(k, []).append(pd.Timestamp(s.start).value)
        for k, times in issued(*inputs[name], key).items():
            for w in times:
                if any(0 <= s - w < 24 * daily_rule.HOUR_NS for s in own.get(int(k), [])):
                    continue
                obj = int(k) if key == "object_id" else int(feed.channels.object_id.get(int(k), 0))
                feed.add(
                    pd.Timestamp(int(w)),
                    obj,
                    scenario,
                    "В следующие сутки ничего не произошло",
                    WARNING_TEXT[scenario],
                    "ложное предупреждение",
                    kind="false",
                )
    return sorted(feed.rows, key=lambda r: r["time"])


def provenance():
    """Library versions and input hashes: two runs with equal records must give equal reports."""
    digest = {}
    for name, path in input_files().items():
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        digest[name] = {"file": path.name, "sha256": h.hexdigest()}
    packages = ("numpy", "pandas", "pyarrow", "duckdb", "catboost")
    return {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "packages": {p: version(p) for p in packages},
        "inputs": digest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true", help="reuse collected units from data/polygon")
    parser.add_argument("--refresh", nargs="*", default=[], help="unit kinds to collect again")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    ctx = Context()
    base = cached(ctx, "units", [*TRAIN_WORLDS, *TEST_WORLDS], None, args.reuse, args.refresh)
    hard = cached(ctx, "hard", HARD_WORLDS, HARD_RATES, args.reuse, args.refresh)
    report, inputs = evaluate(base, hard)
    report["design"] = {
        "train_worlds": list(TRAIN_WORLDS),
        "test_worlds": list(TEST_WORLDS),
        "hard_worlds": list(HARD_WORLDS),
        "train": [str(x) for x in ps.TRAIN],
        "select": [str(x) for x in ps.SELECT],
        "test": [str(x) for x in ps.TEST],
        "feed_world": FEED_WORLD,
        "rates": RATES,
        "hard_rates": HARD_RATES,
    }
    report["provenance"] = provenance()
    feed = build_feed(inputs, base["truth"], ctx.catalog)
    text = json.dumps(report, ensure_ascii=False, indent=2, default=float)
    REPORT.write_text(text, encoding="utf-8", newline="\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(text, encoding="utf-8", newline="\n")
    (OUT / "feed.json").write_text(
        json.dumps(feed, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    log("written", REPORT, "feed rows", len(feed))


if __name__ == "__main__":
    main()
