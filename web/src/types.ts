export type Kind = "fault" | "fire" | "flood" | "access";
export type Page =
  | "overview"
  | "map"
  | "journal"
  | "decisions"
  | "equipment"
  | "work"
  | "verification"
  | "thresholds"
  | "summary"
  | "evaluation"
  | "quality"
  | "users"
  | "audit"
  | "settings";
export type Role =
  "dispatcher" | "technician" | "analyst" | "manager" | "admin";
export type Permission =
  | "forecasts.view"
  | "decisions.view"
  | "decisions.write"
  | "notifications.all"
  | "notifications.critical"
  | "equipment.view"
  | "work.view"
  | "work.edit"
  | "data.import"
  | "labels.verify"
  | "retrain.run"
  | "model.view"
  | "summary.view"
  | "reports.export"
  | "thresholds.propose"
  | "thresholds.approve"
  | "users.manage"
  | "audit.view";
export type User = {
  id: number;
  name: string;
  role: Role;
  role_label: string;
  scope: string;
  permissions: Permission[];
  demo_mode: boolean;
};
export type RiskLevel = "critical" | "high" | "watch" | "low";
export type Decision = {
  id: number;
  action: string;
  reason: string;
  comment: string;
  work_status?: string | null;
  work_outcome?: string | null;
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
  risk: RiskLevel;
  risk_label: string;
  above_threshold: boolean;
  as_of: string;
  horizon_hours: number;
  valid_until: string;
  recommendation: string;
  model_version: string;
  split: string;
  source?: "archive" | "stream" | "import";
  source_label?: string;
  batch_id?: string;
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
  stream?: {
    job_id: string;
    as_of: string;
    label?: string;
    forecasts: Forecast[];
  } | null;
};
export type Topology = {
  nodes: {
    id: number;
    parent_id: number | null;
    name: string;
    level: number;
    kind: string;
    channels: number;
    risk: RiskLevel | "unknown";
    warnings?: number;
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
  recommendation_details?: {
    text: string;
    basis: string;
    source: "signals" | "general";
  }[];
  verification?: {
    alarm_messages: number;
    alarm_sensors: { sensor_type: string; channels: number }[];
    fault_channels: number;
    related_forecasts: {
      kind: Kind;
      kind_label: string;
      probability: number;
      risk: RiskLevel;
    }[];
    external_sources: { id: string; label: string; status: string }[];
    window_hours: number;
  };
  calendar?: {
    weekday: string;
    weekend: boolean;
    month: number;
    hour: number;
    weekday_share: number | null;
    weekday_ratio: number | null;
    history_from: string | null;
    kind_episodes: number;
    object_episodes_30d: number;
    object_last_episode: string | null;
  };
  trend?: { as_of: string; probability: number }[];
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
export type WorkOrder = {
  id: number;
  number: string;
  decision_id: number | null;
  prediction_id: string | null;
  object_id: number;
  object_name: string;
  kind: Kind;
  forecast_at: string | null;
  title: string;
  description: string;
  priority: "urgent" | "high" | "normal";
  priority_label: string;
  status: "draft" | "submitted" | "accepted" | "in_progress" | "done";
  status_label: string;
  outcome: "confirmed" | "not_confirmed" | "sensor_fault" | null;
  outcome_label: string | null;
  external_id: string | null;
  created_at: string;
  updated_at: string;
  submitted_at: string | null;
  synced_at: string | null;
};
export type DecisionRow = Decision & {
  prediction_id: string;
  object_id: number;
  object_name: string;
  kind: Kind;
  forecast_at: string;
  action_label: string;
  reason_label?: string;
  created_at: string;
  updated_at: string;
  user_id: number;
  user_name?: string | null;
  requires_work?: boolean;
  work_order?: WorkOrder | null;
  label_review?: {
    verdict: "accepted" | "rejected";
    label: number | null;
  } | null;
};
export type LabelRow = DecisionRow & {
  archive_occurred: boolean | null;
  suggested_label: 0 | 1 | null;
  label_basis: string;
};
export type EquipmentObject = {
  object_id: number;
  object_name: string;
  parent_id: number | null;
  forecast_id: string;
  probability: number;
  threshold: number;
  risk: RiskLevel;
  risk_label: string;
  above_threshold: boolean;
  faulty_now: number;
  channels_with_faults_24h: number;
  fault_messages_24h: number;
  open_orders: string[];
  channels: {
    channel_id: number;
    sensor_type: string;
    sensor_name: string;
    value: string;
    ts: string;
    current_fault: boolean;
    fault_messages: number;
  }[];
  recommendations: { text: string; basis: string; source: string }[];
};
export type Equipment = {
  as_of: string;
  objects: EquipmentObject[];
  totals: {
    objects: number;
    at_risk: number;
    faulty_now: number;
    objects_with_faults: number;
  };
  note: string;
};
export type PolicyMetrics = {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  alerts: number;
  true_alerts: number;
  false_alerts: number;
  eligible_episodes: number;
  warnings_now: number;
  warnings_per_day_7d: number;
  median_lead_hours?: number | null;
};
export type ThresholdPreview = {
  kind: Kind;
  period_label: string;
  current: PolicyMetrics;
  proposed: PolicyMetrics;
  model_threshold: number;
  curve: {
    threshold: number;
    precision: number;
    recall: number;
    f1: number;
    alerts: number;
  }[];
};
export type Proposal = {
  id: number;
  kind: Kind;
  kind_label: string;
  current_value: number;
  proposed_value: number;
  rationale: string;
  preview: {
    current: PolicyMetrics;
    proposed: PolicyMetrics;
    period_label: string;
  };
  status: "pending" | "approved" | "rejected";
  proposed_by: string | null;
  created_at: string;
  decided_by: string | null;
  decided_at: string | null;
  decision_note: string;
};
export type Summary = {
  as_of: string;
  totals: {
    forecasts: number;
    warnings: number;
    critical: number;
    objects_at_risk: number;
  };
  kinds: {
    id: Kind;
    label: string;
    warnings: number;
    critical: number;
    threshold: number;
  }[];
  nodes: {
    id: number;
    name: string;
    objects: number;
    warnings: number;
    critical: number;
    levels: Record<RiskLevel, number>;
    by_kind: Record<Kind, number>;
    max_probability: number | null;
  }[];
  trend: ({ as_of: string } & Partial<Record<Kind, number>>)[];
  seasonality: {
    year: number;
    month: number;
    alarms: number;
    quarantined: boolean;
  }[];
  quality: Record<
    Kind,
    {
      precision: number;
      recall: number;
      f1: number;
      baseline_f1: number;
      eligible_episodes: number;
      enabled: boolean;
    }
  >;
  workload: {
    decisions: number;
    actions: { id: string; label: string; count: number }[];
    false_alarm_share: number | null;
    orders_open: number;
    orders_done: number;
    outcomes: Record<"confirmed" | "not_confirmed" | "sensor_fault", number>;
    pending_proposals: boolean;
  };
};
export type AdminUser = {
  id: number;
  username: string;
  name: string;
  role: Role;
  role_label: string;
  scope: string;
  active: boolean;
};
export type AuditRow = {
  id: number;
  user_id: number | null;
  user_name: string | null;
  role: Role | null;
  action: string;
  detail: Record<string, unknown> | unknown[];
  created_at: string;
};
