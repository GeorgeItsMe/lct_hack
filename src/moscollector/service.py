"""Read-only access to reproducible analytical artifacts and model evidence."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from moscollector.domain import (
    KIND_LABELS,
    RECOMMENDATIONS,
    RISK_LEVELS,
    data_recommendations,
    feature_label,
    is_fault_state,
    risk_level,
)
from moscollector.inference import CATEGORICAL, model_input
from moscollector.paths import ARTIFACTS, PROCESSED


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if value is pd.NaT:
        return None
    return value


LEVEL_ORDER = ["low", "watch", "high", "critical"]


def prediction_id(obj, kind, as_of):
    identity = f"{obj}:{kind}:{pd.Timestamp(as_of).isoformat()}"
    return "PR-" + hashlib.sha256(identity.encode()).hexdigest()[:14].upper()


class AnalyticsService:
    def __init__(self):
        self.objects = pd.read_parquet(PROCESSED / "objects.parquet")
        self.channels = pd.read_parquet(PROCESSED / "channels.parquet")
        self.predictions = pd.read_parquet(
            ARTIFACTS / "predictions" / "all.parquet", filters=[("as_of", ">=", pd.Timestamp("2026-01-01"))]
        )
        self.features = pd.read_parquet(
            PROCESSED / "features.parquet", filters=[("as_of", ">=", pd.Timestamp("2026-01-01"))]
        )
        self.episodes = pd.read_parquet(
            PROCESSED / "episodes.parquet", filters=[("start_ts", ">=", pd.Timestamp("2026-01-01"))]
        )
        self.hourly = pd.read_parquet(PROCESSED / "hourly-2026.parquet")
        self.models = {}
        self.meta = {}
        for kind in sorted(self.predictions.kind.unique()):
            self.meta[kind] = json.loads((ARTIFACTS / "models" / f"{kind}.json").read_text(encoding="utf-8"))
            model = CatBoostClassifier()
            model.load_model(str(ARTIFACTS / "models" / f"{kind}.cbm"))
            self.models[kind] = model
        self.report = json.loads((ARTIFACTS / "evaluation_report.json").read_text(encoding="utf-8"))
        self.uncertainty = None
        uncertainty_path = ARTIFACTS / "uncertainty_report.json"
        if uncertainty_path.exists():
            sensitivity = json.loads(uncertainty_path.read_text(encoding="utf-8"))
            report_hash = hashlib.sha256((ARTIFACTS / "evaluation_report.json").read_bytes()).hexdigest()
            if sensitivity.get("input_sha256", {}).get("evaluation_report") == report_hash:
                self.uncertainty = sensitivity
        self.catalog_audit = json.loads((ARTIFACTS / "catalog_audit.json").read_text(encoding="utf-8"))
        self.feature_audit = json.loads((ARTIFACTS / "feature_audit.json").read_text(encoding="utf-8"))
        self.audits = [
            json.loads(p.read_text(encoding="utf-8")) for p in sorted(ARTIFACTS.glob("audit-*.json"))
        ]
        self.times = pd.DatetimeIndex(
            sorted(self.predictions.loc[self.predictions.eligible, "as_of"].unique())
        )
        self.object_map = self.objects.set_index("object_id").to_dict("index")
        self.version = hashlib.sha256((ARTIFACTS / "evaluation_report.json").read_bytes()).hexdigest()[:10]

    def resolve_time(self, requested=None):
        if requested is None:
            requested = "2026-06-15T12:00:00"
        timestamp = pd.Timestamp(requested)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("Europe/Moscow").tz_localize(None)
        index = self.times.searchsorted(timestamp, side="right") - 1
        if index < 0 or timestamp > self.times.max() + pd.Timedelta(hours=3):
            raise ValueError("Время вне доступного архива")
        return self.times[max(0, index)]

    def thresholds(self, overrides=None):
        defaults = {kind: m["threshold"] for kind, m in self.meta.items()}
        return {**defaults, **(overrides or {})}

    def forecast_rows(self, as_of=None, overrides=None):
        t = self.resolve_time(as_of)
        data = self.predictions[self.predictions.as_of.eq(t)].copy()
        threshold = self.thresholds(overrides)
        rows = []
        for r in data.itertuples():
            obj = self.object_map[int(r.object_id)]
            p = float(r.probability)
            cut = threshold[r.kind]
            above = p >= cut
            level = risk_level(p, cut)
            rows.append(
                {
                    "id": prediction_id(r.object_id, r.kind, t),
                    "object_id": int(r.object_id),
                    "object_name": obj["object_name"],
                    "parent_id": int(obj["parent_id"]) if pd.notna(obj["parent_id"]) else None,
                    "kind": r.kind,
                    "kind_label": KIND_LABELS[r.kind],
                    "probability": p,
                    "threshold": cut,
                    "risk": level,
                    "risk_label": RISK_LEVELS[level],
                    "above_threshold": above,
                    "as_of": t.isoformat(),
                    "horizon_hours": 24,
                    "valid_until": (t + pd.Timedelta(hours=24)).isoformat(),
                    "label_type": "proxy_sensor_episode",
                    "split": r.split,
                    "recommendation": RECOMMENDATIONS[r.kind][0],
                    "model_version": self.version,
                }
            )
        rows.sort(
            key=lambda r: (
                not r["above_threshold"],
                -r["probability"] / (r["threshold"] or 0.001),
                -r["probability"],
            )
        )
        return rows

    def overview(self, as_of=None, overrides=None):
        t = self.resolve_time(as_of)
        forecasts = self.forecast_rows(t, overrides)
        high = [r for r in forecasts if r["above_threshold"]]
        object_ids = {r["object_id"] for r in forecasts}
        last_day = self.hourly[self.hourly.hour.ge(t - pd.Timedelta(hours=24)) & self.hourly.hour.lt(t)]
        spark = (
            last_day.groupby("hour")[["events", "alarms", "fault_reports", "unknown_reports"]]
            .sum()
            .reset_index()
        )
        return clean(
            {
                "as_of": t,
                "timezone": "Europe/Moscow",
                "mode": "historical_replay",
                "horizon_hours": 24,
                "model_version": self.version,
                "stats": {
                    "objects": len(object_ids),
                    "catalog_objects": len(self.objects),
                    "channels": len(self.channels),
                    "warnings": len(high),
                    "objects_at_risk": len({r["object_id"] for r in high}),
                    "events_24h": int(last_day.events.sum()),
                    "alarms_24h": int(last_day.alarms.sum()),
                },
                "forecasts": forecasts,
                "activity": spark.to_dict("records"),
                "kinds": [
                    {
                        "id": k,
                        "label": KIND_LABELS[k],
                        "count": sum(r["kind"] == k for r in high),
                        "threshold": self.thresholds(overrides)[k],
                    }
                    for k in self.meta
                ],
            }
        )

    def topology(self, as_of=None, overrides=None):
        forecasts = self.forecast_rows(as_of, overrides)
        by_object = {}
        for row in forecasts:
            by_object.setdefault(row["object_id"], []).append(row)
        nodes = []
        for r in self.objects.itertuples():
            risks = by_object.get(int(r.object_id), [])
            parent = int(r.parent_id) if pd.notna(r.parent_id) else None
            nodes.append(
                {
                    "id": int(r.object_id),
                    "parent_id": parent,
                    "name": r.object_name,
                    "level": int(r.level),
                    "kind": r.object_kind,
                    "channels": int(self.channels.object_id.eq(r.object_id).sum()),
                    "risk": max((x["risk"] for x in risks), key=LEVEL_ORDER.index) if risks else "unknown",
                    "warnings": sum(x["above_threshold"] for x in risks),
                    "probability": max((x["probability"] for x in risks), default=None),
                    "forecast_id": risks[0]["id"] if risks else None,
                }
            )
        return {
            "coordinate_system": "schematic",
            "geographic": False,
            "nodes": nodes,
            "description": "Схема иерархии объектов из справочника. Географические координаты не предоставлены.",
        }

    def explain_features(self, features, kind, model_version="legacy"):
        if len(features) != 1:
            raise KeyError("Снимок признаков прогноза не найден")
        from moscollector.model_registry import load_bundle

        head = load_bundle(model_version).get(kind) if model_version != "legacy" else None
        if head is not None:
            meta, shap = head.meta, head.shap(features)[0]
        else:
            meta = self.meta[kind]
            x = model_input(features, meta["features"])
            shap = self.models[kind].get_feature_importance(
                Pool(x, cat_features=CATEGORICAL), type="ShapValues"
            )[0]
        contributions = []
        for name, contribution in zip(meta["features"], shap[:-1], strict=True):
            value = features[name].iloc[0]
            contributions.append(
                {
                    "feature": name,
                    "label": feature_label(name),
                    "value": clean(value),
                    "contribution": float(contribution * meta["calibration"]["slope"]),
                }
            )
        contributions.sort(key=lambda v: abs(v["contribution"]), reverse=True)
        return contributions[:8]

    def detail(self, obj: int, kind: str, as_of=None, overrides=None):
        t = self.resolve_time(as_of)
        forecasts = self.forecast_rows(t, overrides)
        row = next((x for x in forecasts if x["object_id"] == obj and x["kind"] == kind), None)
        if row is None:
            raise KeyError("Прогноз не найден")
        features = self.features[self.features.object_id.eq(obj) & self.features.as_of.eq(t)]
        contributions = self.explain_features(features, kind)
        meta = self.meta[kind]
        past = self.predictions[
            self.predictions.object_id.eq(obj)
            & self.predictions.kind.eq(kind)
            & self.predictions.as_of.le(t)
            & self.predictions.as_of.ge(t - pd.Timedelta(hours=72))
        ]
        measurements = self.hourly[
            self.hourly.object_id.eq(obj)
            & self.hourly.hour.lt(t)
            & self.hourly.hour.ge(t - pd.Timedelta(hours=72))
        ]
        catalog = self.channels[self.channels.object_id.eq(obj)]
        source_events = self.source_events(obj, t)
        details = data_recommendations(kind, source_events)
        return clean(
            {
                **row,
                "explanation": contributions[:8],
                "explanation_unit": "log_odds",
                "recommendations": [d["text"] for d in details],
                "recommendation_details": details,
                "verification": self.verification(obj, kind, t, forecasts, source_events),
                "calendar": self.calendar(obj, kind, t),
                "trend": past[["as_of", "probability"]].to_dict("records"),
                "signals": measurements[
                    [
                        "hour",
                        "alarms",
                        "fault_reports",
                        "power_reports",
                        "unknown_reports",
                        "temperature",
                        "smoke_reports",
                        "flood_reports",
                        "pump_switches",
                    ]
                ].to_dict("records"),
                "channels": catalog.head(200).to_dict("records"),
                "channel_count": len(catalog),
                "source_events": source_events,
                "calibration_status": meta["calibration"]["status"],
            }
        )

    def verification(self, obj, kind, t, forecasts, source_events):
        """What the dispatcher can check before deciding (step 4 of the section 12 scenario)."""
        alarms = [e for e in source_events if e["alarm"]]
        by_sensor = {}
        for e in alarms:
            by_sensor.setdefault(e["sensor_type"], set()).add(e["channel_id"])
        faults = {e["channel_id"] for e in source_events if is_fault_state(e["value"])}
        related = [
            {
                "kind": r["kind"],
                "kind_label": r["kind_label"],
                "probability": r["probability"],
                "risk": r["risk"],
            }
            for r in forecasts
            if r["object_id"] == obj and r["kind"] != kind
        ]
        return {
            "alarm_messages": len(alarms),
            "alarm_sensors": [
                {"sensor_type": k, "channels": len(v)}
                for k, v in sorted(by_sensor.items(), key=lambda item: -len(item[1]))
            ][:6],
            "fault_channels": len(faults),
            "related_forecasts": related,
            "external_sources": [
                {"id": "cameras", "label": "Видеонаблюдение", "status": "not_connected"},
                {"id": "planned_works", "label": "Реестр плановых работ", "status": "not_connected"},
                {"id": "access_permits", "label": "Допуски в коллектор", "status": "not_connected"},
            ],
            "window_hours": 24,
        }

    def calendar(self, obj, kind, t):
        """Calendar context from episodes that started strictly before the forecast moment."""
        past = self.episodes[self.episodes.kind.eq(kind) & self.episodes.start_ts.lt(t)]
        weekday_names = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
        weekday = int(t.dayofweek)
        share = None
        if len(past) >= 30:
            share = float(past.start_ts.dt.dayofweek.eq(weekday).mean())
        own = past[past.object_id.eq(obj)]
        return clean(
            {
                "weekday": weekday_names[weekday],
                "weekend": weekday >= 5,
                "month": int(t.month),
                "hour": int(t.hour),
                "weekday_share": share,
                "weekday_ratio": share * 7 if share is not None else None,
                "history_from": past.start_ts.min() if len(past) else None,
                "kind_episodes": len(past),
                "object_episodes_30d": int(own.start_ts.ge(t - pd.Timedelta(days=30)).sum()),
                "object_last_episode": own.start_ts.max() if len(own) else None,
            }
        )

    @lru_cache(maxsize=64)  # noqa: B019 — one service singleton, bounded lifetime cache
    def channel_states(self, as_of):
        """Latest state of every channel in the 24 hours before the moment, fault-like only."""
        t = pd.Timestamp(as_of)
        con = duckdb.connect()
        con.execute("SET threads=2")
        con.read_parquet(str(PROCESSED / "events-2026.parquet")).create_view("e")
        con.read_parquet(str(PROCESSED / "channels.parquet")).create_view("c")
        frame = con.execute(
            """WITH w AS (
                 SELECT e.channel_id, e.ts, e.value, e.alarm, e.event_id,
                        row_number() OVER (PARTITION BY e.channel_id ORDER BY e.ts DESC, e.event_id DESC) AS rn,
                        count(*) FILTER (WHERE e.value IN ('Неисправен','Обесточен','Отключено устройство',
                            'Питание от батарей','Батарея разряжена') OR starts_with(e.value,'01.01.1970'))
                            OVER (PARTITION BY e.channel_id) AS fault_messages
                 FROM e WHERE e.ts < ? AND e.ts >= ?)
               SELECT w.channel_id, w.ts, w.value, w.alarm, w.fault_messages,
                      c.object_id, c.sensor_type, c.sensor_name
               FROM w JOIN c USING(channel_id)
               WHERE w.rn = 1 AND w.fault_messages > 0""",
            [t.to_pydatetime(), (t - pd.Timedelta(hours=24)).to_pydatetime()],
        ).df()
        con.close()
        frame["current_fault"] = frame.value.map(is_fault_state)
        return frame

    def equipment(self, as_of=None, overrides=None):
        """Equipment condition per object for technical staff: fault forecast plus recorded faults."""
        t = self.resolve_time(as_of)
        forecasts = {r["object_id"]: r for r in self.forecast_rows(t, overrides) if r["kind"] == "fault"}
        states = self.channel_states(t.isoformat())
        rows = []
        for obj, forecast in forecasts.items():
            own = states[states.object_id.eq(obj)].sort_values(["current_fault", "ts"], ascending=False)
            channels = [
                {
                    "channel_id": int(c.channel_id),
                    "sensor_type": c.sensor_type,
                    "sensor_name": c.sensor_name,
                    "value": c.value,
                    "ts": c.ts,
                    "current_fault": bool(c.current_fault),
                    "fault_messages": int(c.fault_messages),
                }
                for c in own.head(12).itertuples()
            ]
            events = [{**c, "alarm": True} for c in channels if c["current_fault"]]
            rows.append(
                {
                    "object_id": obj,
                    "object_name": forecast["object_name"],
                    "parent_id": forecast["parent_id"],
                    "forecast_id": forecast["id"],
                    "probability": forecast["probability"],
                    "threshold": forecast["threshold"],
                    "risk": forecast["risk"],
                    "risk_label": forecast["risk_label"],
                    "above_threshold": forecast["above_threshold"],
                    "faulty_now": int(own.current_fault.sum()),
                    "channels_with_faults_24h": len(own),
                    "fault_messages_24h": int(own.fault_messages.sum()),
                    "channels": channels,
                    "recommendations": [
                        r for r in data_recommendations("fault", events) if r["source"] == "signals"
                    ],
                }
            )
        rows.sort(key=lambda r: (-LEVEL_ORDER.index(r["risk"]), -r["faulty_now"], -r["probability"]))
        return clean(
            {
                "as_of": t,
                "objects": rows,
                "totals": {
                    "objects": len(rows),
                    "at_risk": sum(r["above_threshold"] for r in rows),
                    "faulty_now": sum(r["faulty_now"] for r in rows),
                    "objects_with_faults": sum(r["channels_with_faults_24h"] > 0 for r in rows),
                },
                "note": "За 24 часа до момента прогноза",
            }
        )

    def threshold_preview(self, kind, threshold, as_of=None, overrides=None):
        """Effect of a threshold on the policy period (16–31 May), never on the final test."""
        from moscollector.alert_policy import alert_metrics

        if kind not in self.meta:
            raise KeyError("Тип риска не найден")
        policy = self.predictions[self.predictions.kind.eq(kind) & self.predictions.split.eq("policy")][
            ["object_id", "as_of", "probability"]
        ]
        episodes = self.episodes[self.episodes.kind.eq(kind)]
        current = self.thresholds(overrides)[kind]
        t = self.resolve_time(as_of)
        window = self.predictions[
            self.predictions.kind.eq(kind)
            & self.predictions.eligible
            & self.predictions.as_of.gt(t - pd.Timedelta(days=7))
            & self.predictions.as_of.le(t)
        ]
        days = max(1, window.as_of.nunique() * 3 / 24)

        def load(value):
            # Same 24h per-object pause as the warning policy, so this is dispatcher workload.
            issued = alert_metrics(window[["object_id", "as_of", "probability"]], episodes, value)["alerts"]
            return {
                "warnings_now": int(window[window.as_of.eq(t)].probability.ge(value).sum()),
                "warnings_per_day_7d": float(issued / days),
            }

        return clean(
            {
                "kind": kind,
                "period": "policy",
                "period_label": "Период настройки: 16–31 мая 2026",
                "current": {
                    "threshold": current,
                    **alert_metrics(policy, episodes, current),
                    **load(current),
                },
                "proposed": {
                    "threshold": threshold,
                    **alert_metrics(policy, episodes, threshold),
                    **load(threshold),
                },
                "model_threshold": self.meta[kind]["threshold"],
                "curve": [
                    {k: c[k] for k in ("threshold", "precision", "recall", "f1", "alerts")}
                    for c in self.meta[kind].get("policy_curve", [])
                ],
                "as_of": t,
            }
        )

    def summary(self, as_of=None, overrides=None, parent=None):
        """Unit head overview: risk by operational node and type, warning dynamics, seasonality."""
        t = self.resolve_time(as_of)
        forecasts = self.forecast_rows(t, overrides)
        if parent is not None:
            forecasts = [r for r in forecasts if r["parent_id"] == parent]
        nodes = []
        for node in self.objects[self.objects.level.eq(2)].itertuples():
            rows = [r for r in forecasts if r["parent_id"] == int(node.object_id)]
            if parent is not None and int(node.object_id) != parent:
                continue
            nodes.append(
                {
                    "id": int(node.object_id),
                    "name": node.object_name,
                    "objects": len({r["object_id"] for r in rows}),
                    "warnings": sum(r["above_threshold"] for r in rows),
                    "critical": sum(r["risk"] == "critical" for r in rows),
                    "levels": {level: sum(r["risk"] == level for r in rows) for level in LEVEL_ORDER},
                    "by_kind": {
                        k: sum(r["above_threshold"] and r["kind"] == k for r in rows) for k in self.meta
                    },
                    "max_probability": max((r["probability"] for r in rows), default=None),
                }
            )
        nodes.sort(key=lambda n: (-n["critical"], -n["warnings"], n["name"]))
        threshold = self.thresholds(overrides)
        recent = self.predictions[
            self.predictions.eligible
            & self.predictions.as_of.gt(t - pd.Timedelta(days=7))
            & self.predictions.as_of.le(t)
        ].copy()
        if parent is not None:
            children = set(self.objects[self.objects.parent_id.eq(parent)].object_id)
            recent = recent[recent.object_id.isin(children)]
        recent["warning"] = recent.probability.ge(recent.kind.map(threshold))
        trend = (
            recent.pivot_table(index="as_of", columns="kind", values="warning", aggfunc="sum", fill_value=0)
            .reset_index()
            .to_dict("records")
        )
        seasonality = []
        for audit in self.audits:
            monthly = {}
            for day in audit.get("daily", []):
                if not day.get("day"):
                    continue
                month = int(str(day["day"])[5:7])
                monthly[month] = monthly.get(month, 0) + int(day.get("alarms") or 0)
            seasonality.extend(
                {"year": audit["year"], "month": m, "alarms": v, "quarantined": audit["quarantined"]}
                for m, v in sorted(monthly.items())
            )
        quality = {
            kind: {
                "precision": m["test"]["alerts"]["precision"],
                "recall": m["test"]["alerts"]["recall"],
                "f1": m["test"]["alerts"]["f1"],
                "baseline_f1": m["test"]["baseline_alerts"]["f1"],
                "eligible_episodes": m["test"]["alerts"]["eligible_episodes"],
                "enabled": threshold[kind] <= 1,
            }
            for kind, m in self.report["models"].items()
        }
        return clean(
            {
                "as_of": t,
                "totals": {
                    "forecasts": len(forecasts),
                    "warnings": sum(r["above_threshold"] for r in forecasts),
                    "critical": sum(r["risk"] == "critical" for r in forecasts),
                    "objects_at_risk": len({r["object_id"] for r in forecasts if r["above_threshold"]}),
                },
                "kinds": [
                    {
                        "id": k,
                        "label": KIND_LABELS[k],
                        "warnings": sum(r["above_threshold"] and r["kind"] == k for r in forecasts),
                        "critical": sum(r["risk"] == "critical" and r["kind"] == k for r in forecasts),
                        "threshold": threshold[k],
                    }
                    for k in self.meta
                ],
                "nodes": nodes,
                "trend": trend,
                "seasonality": seasonality,
                "quality": quality,
            }
        )

    @lru_cache(maxsize=256)  # noqa: B019 — one service singleton, bounded lifetime cache
    def source_events(self, obj, as_of):
        t = pd.Timestamp(as_of)
        con = duckdb.connect()
        con.execute("SET threads=2")
        con.read_parquet(str(PROCESSED / "events-2026.parquet")).create_view("e")
        con.read_parquet(str(PROCESSED / "channels.parquet")).create_view("c")
        frame = con.execute(
            """SELECT e.event_id,e.channel_id,e.ts,e.value,e.alarm,c.sensor_type,c.sensor_name
            FROM e JOIN c USING(channel_id) WHERE c.object_id=? AND e.ts<? AND e.ts>=?
            ORDER BY e.alarm DESC,e.ts DESC,e.event_id DESC LIMIT 40""",
            [obj, t.to_pydatetime(), (t - pd.Timedelta(hours=24)).to_pydatetime()],
        ).df()
        con.close()
        return clean(frame.to_dict("records"))

    def retrospective(self, obj, kind, as_of):
        t = self.resolve_time(as_of)
        ep = self.episodes[
            self.episodes.object_id.eq(obj)
            & self.episodes.kind.eq(kind)
            & self.episodes.start_ts.ge(t)
            & self.episodes.start_ts.lt(t + pd.Timedelta(hours=24))
        ]
        return clean(
            {
                "as_of": t,
                "mode": "retrospective_only",
                "occurred": len(ep) > 0,
                "episodes": ep.to_dict("records"),
                "note": "Будущие события показаны только для проверки по архиву; модель их не использовала.",
            }
        )

    def quality(self):
        yearly = []
        for audit in self.audits:
            y = audit["year"]
            s = audit["summary"]
            episode_path = ARTIFACTS / f"episode-audit-{y}.json"
            episode_audit = (
                json.loads(episode_path.read_text(encoding="utf-8")) if episode_path.exists() else {}
            )
            yearly.append(
                {
                    "year": y,
                    **s,
                    "quarantined": audit["quarantined"],
                    "used_for_model": y in self.feature_audit["years"],
                    "ambiguities": episode_audit.get("ambiguous_channel_seconds"),
                    "collapsed_rows": episode_audit.get("collapsed_rows"),
                }
            )
        daily = next(a["daily"] for a in self.audits if a["year"] == 2026)
        by_day = {r["day"]: r for r in daily}
        daily = [
            by_day.get(
                t.strftime("%Y-%m-%d"),
                {"day": t.strftime("%Y-%m-%d"), "rows": 0, "alarms": 0, "channels": 0, "source_gap": True},
            )
            for t in pd.date_range(min(by_day), max(by_day), freq="D")
        ]
        return {
            "total_rows": sum(a["summary"]["rows"] for a in self.audits),
            "years": yearly,
            "catalog": self.catalog_audit,
            "features": self.feature_audit,
            "daily_2026": daily,
            "rules": [
                {
                    "title": "Конфликтующие состояния",
                    "detail": "Несколько разных значений канала в одну секунду отмечаются как неизвестное состояние.",
                },
                {
                    "title": "Длительная неисправность",
                    "detail": "Цель прогноза — новый эпизод состояния «Неисправен» длительностью от одного часа.",
                },
                {
                    "title": "Каскадные сообщения",
                    "detail": "Начала эпизодов на одном объекте объединяются в фиксированное окно 10 минут.",
                },
                {
                    "title": "Разрывы выгрузки",
                    "detail": "Из оценки исключаются окна с неполной предысторией или ненаблюдаемым будущим.",
                },
                {
                    "title": "2021 год",
                    "detail": "Исключён из обучения: миграция мониторинга указана в командном брифе.",
                },
            ],
        }
