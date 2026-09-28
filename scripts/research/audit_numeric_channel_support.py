"""Describe unused channel-level numeric variation on a fixed training interval.

No episode labels, test periods, fitted models or event-quality claims are used.
Canonicalization and numeric ranges match the original hourly features.
"""

import json
from pathlib import Path

import duckdb

from moscollector.prepare import sha256, write_json

ROOT = Path("artifacts/research-v40-support")
REPORT = Path("artifacts/numeric_channel_support_audit.json")
START, END = "2024-07-01", "2025-01-01"


def run():
    sources = [
        Path("data/processed/events-2024.parquet"),
        Path("data/processed/channels.parquet"),
        Path("data/processed/hourly-2024.parquet"),
    ]
    plan = {
        "scope": "Descriptive numeric-input support only; no targets or model selection.",
        "period": [START, END],
        "families": ["Датчик температуры", "Газовый датчик"],
        "numeric_ranges": {"Датчик температуры": [-50, 100], "Газовый датчик": [0, 100]},
        "source_hashes": {str(p): sha256(p) for p in sources},
        "code_hashes": {str(p): sha256(p) for p in (Path(__file__), Path("src/moscollector/features.py"))},
    }
    path = ROOT / "plan.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("Numeric support audit inputs or code changed")
    else:
        write_json(path, plan)
    c = duckdb.connect()
    c.execute("SET threads=2")
    c.execute("SET memory_limit='2GB'")
    c.execute("SET temp_directory='tmp/numeric-support'")
    c.read_parquet(str(sources[0])).create_view("events")
    c.read_parquet(str(sources[1])).create_view("channels")
    c.read_parquet(str(sources[2])).create_view("original_hourly")
    c.execute(
        """CREATE TABLE canonical AS
        SELECT e.channel_id,e.ts,c.object_id,c.sensor_type,
          CASE WHEN count(DISTINCT e.value)=1 THEN min(e.numeric_value) END AS x,
          count(DISTINCT e.value)>1 AS ambiguous, count(*) AS source_rows
        FROM events e JOIN channels c USING(channel_id)
        WHERE e.ts>=?::TIMESTAMP AND e.ts<?::TIMESTAMP AND e.alarm IS NOT NULL
          AND c.sensor_type IN ('Датчик температуры','Газовый датчик')
        GROUP BY e.channel_id,e.ts,c.object_id,c.sensor_type""",
        [START, END],
    )
    c.execute("""CREATE TABLE valid AS SELECT * FROM canonical
        WHERE (sensor_type='Датчик температуры' AND x BETWEEN -50 AND 100)
           OR (sensor_type='Газовый датчик' AND x BETWEEN 0 AND 100)""")
    c.execute("""CREATE TABLE channel_hours AS
        SELECT sensor_type,object_id,channel_id,date_trunc('hour',ts) AS hour,
          count(*) AS n,sum(x) AS total,avg(x) AS mean,min(x) AS lo,max(x) AS hi
        FROM valid GROUP BY 1,2,3,4""")
    c.execute("""CREATE TABLE object_hours AS
        SELECT sensor_type,object_id,hour,sum(n) AS n,sum(total)/sum(n) AS mean,
          max(hi) AS hi,count(*) AS channels,max(hi-lo) AS max_within_channel_span
        FROM channel_hours GROUP BY 1,2,3""")
    # Reconstruct the old aggregates before interpreting new information.
    parity = c.execute("""SELECT o.sensor_type,count(*) AS compared_hours,
        count_if(CASE WHEN o.sensor_type='Датчик температуры'
          THEN h.temperature IS NULL OR h.temperature_max IS NULL
          ELSE h.gas IS NULL END) AS missing_original,
        max(abs(o.mean-CASE WHEN o.sensor_type='Датчик температуры'
          THEN h.temperature ELSE h.gas END)) AS max_mean_difference,
        max(CASE WHEN o.sensor_type='Датчик температуры'
          THEN abs(o.hi-h.temperature_max) ELSE 0 END) AS max_maximum_difference
        FROM object_hours o LEFT JOIN original_hourly h USING(object_id,hour)
        GROUP BY 1 ORDER BY 1""").fetchdf()
    if (
        len(parity) != 2
        or parity.missing_original.sum() != 0
        or (parity.max_mean_difference > 1e-8).any()
        or (parity.max_maximum_difference > 1e-8).any()
    ):
        raise ValueError("Numeric reconstruction differs from original hourly inputs")

    def records(query):
        return json.loads(c.execute(query).fetchdf().to_json(orient="records", date_format="iso"))

    report = {
        **plan,
        "plan_sha256": sha256(path),
        "canonical": records("""SELECT sensor_type,sum(source_rows) AS source_rows,
            count(*) AS canonical_rows,count_if(ambiguous) AS conflicting_rows,
            count_if(x IS NOT NULL) AS numeric_rows,count(DISTINCT channel_id) AS channels,
            count(DISTINCT object_id) AS objects FROM canonical GROUP BY 1 ORDER BY 1"""),
        "valid": records("""SELECT sensor_type,count(*) AS rows,
            count(DISTINCT channel_id) AS channels,count(DISTINCT object_id) AS objects,
            min(x) AS minimum,max(x) AS maximum,count(DISTINCT x) AS distinct_values
            FROM valid GROUP BY 1 ORDER BY 1"""),
        "channel_hours": records("""SELECT sensor_type,count(*) AS observed_channel_hours,
            count_if(n>=2) AS hours_with_multiple_observations,
            count_if(hi>lo) AS hours_with_changing_values,
            quantile_cont(hi-lo,[0.5,0.9,0.99]) AS within_hour_span_quantiles
            FROM channel_hours GROUP BY 1 ORDER BY 1"""),
        "object_hours": records("""SELECT sensor_type,count(*) AS observed_object_hours,
            count_if(channels>=2) AS hours_with_multiple_channels,
            count_if(max_within_channel_span>0) AS hours_with_changing_channel,
            quantile_cont(channels,[0.5,0.9,0.99]) AS channel_count_quantiles
            FROM object_hours GROUP BY 1 ORDER BY 1"""),
        "adjacent_channel_hours": records("""WITH ordered AS (
            SELECT *,lag(hour) OVER w AS previous_hour,lag(mean) OVER w AS previous_mean
            FROM channel_hours WINDOW w AS (PARTITION BY channel_id ORDER BY hour))
            SELECT sensor_type,count(*) AS consecutive_observed_hour_pairs,
              count_if(mean<>previous_mean) AS changed_hourly_means,
              quantile_cont(abs(mean-previous_mean),[0.5,0.9,0.99]) AS absolute_change_quantiles
            FROM ordered WHERE hour-previous_hour=INTERVAL '1 hour'
            GROUP BY 1 ORDER BY 1"""),
        "hourly_parity": json.loads(parity.to_json(orient="records")),
        "limitations": [
            "Observed hours do not imply continuous telemetry coverage.",
            "Sensor family does not establish shared physical units or gas species.",
            "Variation is not evidence of predictive power or a precursor to incidents.",
            "No incident cohort is filtered, no June data or labels are read.",
        ],
        "models_trained": 0,
        "event_metrics_evaluated": False,
        "serving_changed": False,
        "goal_achieved": False,
    }
    c.close()
    write_json(REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run()
