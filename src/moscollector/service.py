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

from moscollector.domain import KIND_LABELS, RECOMMENDATIONS, feature_label
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
        self.audits = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(ARTIFACTS.glob("audit-*.json"))]
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
                    "risk": "high" if above else "watch" if p >= cut * 0.5 else "low",
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
                    "risk": "high"
                    if any(x["above_threshold"] for x in risks)
                    else "low"
                    if risks
                    else "unknown",
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
        return clean(
            {
                **row,
                "explanation": contributions[:8],
                "explanation_unit": "log_odds",
                "recommendations": RECOMMENDATIONS[kind],
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
                "source_events": self.source_events(obj, t),
                "calibration_status": meta["calibration"]["status"],
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
            episode_audit = json.loads(episode_path.read_text(encoding="utf-8")) if episode_path.exists() else {}
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
