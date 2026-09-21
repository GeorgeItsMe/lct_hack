export type Kind = "fault" | "fire" | "flood" | "access";
export type Page =
  | "overview"
  | "map"
  | "journal"
  | "decisions"
  | "evaluation"
  | "quality"
  | "settings";
export type User = {
  id: number;
  name: string;
  role: "dispatcher" | "analyst" | "admin";
  demo_mode: boolean;
};
export type Decision = {
  id: number;
  action: string;
  reason: string;
  comment: string;
};
export type Forecast = {
  id: string;
  object_id: number;
  object_name: string;
  parent_id: number | null;
  kind: Kind;
  kind_label: string;
  probability: number;
  threshold: number;
  risk: "high" | "watch" | "low";
  above_threshold: boolean;
  as_of: string;
  horizon_hours: number;
  valid_until: string;
  recommendation: string;
  model_version: string;
  split: string;
  decision?: Decision | null;
};
export type Overview = {
  as_of: string;
  model_version: string;
  stats: {
    objects: number;
    catalog_objects: number;
    channels: number;
    warnings: number;
    objects_at_risk: number;
    events_24h: number;
    alarms_24h: number;
  };
  forecasts: Forecast[];
  activity: {
    hour: string;
    events: number;
    alarms: number;
    fault_reports: number;
    unknown_reports: number;
  }[];
  kinds: { id: Kind; label: string; count: number; threshold: number }[];
};
export type Topology = {
  nodes: {
    id: number;
    parent_id: number | null;
    name: string;
    level: number;
    kind: string;
    channels: number;
    risk: string;
    probability: number | null;
    forecast_id: string | null;
  }[];
  description: string;
};
export type Detail = Forecast & {
  explanation: {
    feature: string;
    label: string;
    value: string | number | null;
    contribution: number;
  }[];
  recommendations: string[];
  trend: { as_of: string; probability: number }[];
  signals: {
    hour: string;
    alarms: number;
    fault_reports: number;
    power_reports: number;
    unknown_reports: number;
    temperature: number | null;
    smoke_reports: number;
    flood_reports: number;
    pump_switches: number;
  }[];
  channels: { channel_id: number; sensor_type: string; sensor_name: string }[];
  channel_count: number;
  source_events: {
    event_id: number;
    channel_id: number;
    ts: string;
    value: string;
    alarm: boolean;
    sensor_type: string;
    sensor_name: string;
  }[];
  calibration_status: string;
};
export type AlertMetrics = {
  precision: number;
  recall: number;
  f1: number;
  true_alerts: number;
  false_alerts: number;
  alerts: number;
  eligible_episodes: number;
  missed_episodes: number;
  false_alerts_per_object_day: number;
  median_lead_hours: number | null;
  forecast_opportunities: number;
};
export type RowMetrics = {
  rows: number;
  positives: number;
  prevalence: number;
  pr_auc: number;
  roc_auc: number;
  brier: number;
  calibration_curve: { predicted: number; observed: number; count: number }[];
};
export type Evaluation = {
  operational_quality?: {
    model_version: string;
    kind: Kind;
    target_summary?: string;
    below_target?: string;
    additional_periods?: boolean;
    rows: {
      period: string;
      precision: number;
      recall: number;
      f1: number;
      eligible_episodes: number;
    }[];
    f1_gain_interval: { low: number; high: number };
  } | null;
  research?: {
    deployed_changed: boolean;
    models: Partial<
      Record<
        Kind,
        {
          selected: string;
          uncertainty?: {
            by_parent: {
              percentile_95: {
                f1_difference: { low: number | null; high: number | null };
              };
            };
          };
          rows: {
            period: string;
            reference: number;
            context: number | null;
            regularized: number;
            blend: number;
            eligible_episodes: number;
          }[];
        }
      >
    >;
  } | null;
  splits: Record<string, [string, string]>;
  horizon_hours: number;
  purge_hours: number;
  uncertainty?: {
    models: Record<
      Kind,
      {
        status: string;
        by_parent?: {
          clusters: number;
          clusters_with_true_alerts: number;
          replicates: number;
          percentile_95: Record<
            "f1" | "baseline_f1" | "f1_difference",
            {
              low: number | null;
              high: number | null;
              valid_replicates: number;
            }
          >;
        };
      }
    >;
  } | null;
  models: Record<
    Kind,
    {
      threshold: number;
      training_rows: number;
      training_positive_rows: number;
      best_iteration: number;
      feature_importance: { feature: string; importance: number }[];
      test: {
        alerts: AlertMetrics;
        rows: RowMetrics;
        baseline_alerts: AlertMetrics;
        baseline_rows: RowMetrics;
        inference_ms_per_row: number;
      };
    }
  >;
};
export type Quality = {
  total_rows: number;
  years: {
    year: number;
    rows: number;
    channels: number;
    alarms: number;
    invalid_rows: number;
    unmapped_rows: number;
    quarantined: boolean;
    used_for_model: boolean;
    ambiguities: number | null;
  }[];
  catalog: {
    channel_count: number;
    object_count: number;
    sensor_types: Record<string, number>;
  };
  features: {
    rows: number;
    eligible_rows: number;
    episodes: Record<Kind, number>;
    limitations: string[];
  };
  rules: { title: string; detail: string }[];
  daily_2026: { day: string; rows: number; alarms: number; channels: number }[];
};
export type DecisionRow = Decision & {
  prediction_id: string;
  object_id: number;
  object_name: string;
  kind: Kind;
  forecast_at: string;
  action_label: string;
  created_at: string;
  updated_at: string;
  user_id: number;
};
