import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import {
  ArrowDown,
  ArrowRight,
  ArrowUp,
  CalendarClock,
  Camera,
  Check,
  ChevronRight,
  CircleAlert,
  ClipboardList,
  Clock3,
  Droplets,
  Flame,
  Gauge,
  LoaderCircle,
  LockKeyhole,
  Maximize2,
  Radio,
  ShieldCheck,
  Wrench,
  X,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, clock, date, kindNames, num, pct } from "./api";
import { can } from "./roles";
import type {
  Detail,
  Forecast,
  Kind,
  RiskLevel,
  Topology,
  User,
  WorkOrder,
} from "./types";

export const riskNames: Record<RiskLevel, string> = {
  critical: "Критический",
  high: "Высокий",
  watch: "Повышенный",
  low: "Низкий",
};
export const workStatusNames: Record<string, string> = {
  draft: "Черновик заявки",
  submitted: "Заявка передана",
  accepted: "Заявка принята",
  in_progress: "Работы идут",
  done: "Работы выполнены",
};
export const outcomeNames: Record<string, string> = {
  confirmed: "Событие подтвердилось",
  not_confirmed: "Не подтвердилось",
  sensor_fault: "Сбой датчика или оборудования",
};
export function RiskBadge({ level }: { level: RiskLevel }) {
  return (
    <span className={`risk-badge ${level}`}>
      <i aria-hidden="true" />
      {riskNames[level]}
    </span>
  );
}
export function WorkBadge({ order }: { order: WorkOrder | null | undefined }) {
  if (!order) return null;
  return (
    <span className={`work-badge ${order.status}`}>
      {order.outcome_label || order.status_label}
    </span>
  );
}

export function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className="brand">
      <svg viewBox="0 0 36 36" aria-hidden="true">
        <path
          d="M6 29V17a12 12 0 0 1 24 0v12M12 29V17a6 6 0 0 1 12 0v12"
          fill="none"
          stroke="currentColor"
          strokeWidth="3.2"
          strokeLinecap="round"
        />
        <circle cx="18" cy="29" r="2" fill="currentColor" />
      </svg>
      {!compact && (
        <div>
          <strong>
            контур<span className="brand-edition">/ 01</span>
          </strong>
          <small>МОСКОЛЛЕКТОР</small>
        </div>
      )}
    </div>
  );
}
export function KindIcon({ kind, size = 18 }: { kind: Kind; size?: number }) {
  const Icon = {
    fault: Wrench,
    fire: Flame,
    flood: Droplets,
    access: LockKeyhole,
  }[kind];
  return <Icon size={size} strokeWidth={1.7} />;
}
export function KindTag({ kind }: { kind: Kind }) {
  return (
    <span className={`kind-tag ${kind}`}>
      <KindIcon kind={kind} size={14} />
      {kindNames[kind]}
    </span>
  );
}
export function Empty({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="empty-state">
      <ShieldCheck size={32} />
      <strong>{title}</strong>
      {children && <p>{children}</p>}
    </div>
  );
}
export function Loading({ text = "Загружаем данные…" }: { text?: string }) {
  return (
    <div className="loading">
      <LoaderCircle className="spin" size={22} />
      {text}
    </div>
  );
}
export function ErrorNotice({
  message,
  retry,
}: {
  message: string;
  retry?: () => void;
}) {
  return (
    <div className="error-notice">
      <CircleAlert size={19} />
      <span>{message}</span>
      {retry && <button onClick={retry}>Повторить</button>}
    </div>
  );
}
export function Risk({ forecast }: { forecast: Forecast }) {
  return (
    <div
      className={`probability ${forecast.above_threshold ? "high" : ""} level-${forecast.risk}`}
    >
      <strong>{pct(forecast.probability)}</strong>
      <span className="probability-track">
        <i style={{ width: `${Math.min(100, forecast.probability * 100)}%` }} />
        <b style={{ left: `${Math.min(100, forecast.threshold * 100)}%` }} />
      </span>
    </div>
  );
}
export function ForecastTable({
  forecasts,
  onSelect,
  compact = false,
}: {
  forecasts: Forecast[];
  onSelect: (f: Forecast) => void;
  compact?: boolean;
}) {
  return (
    <div className="table-scroll">
      <table className="forecast-table">
        <thead>
          <tr>
            <th>Объект / тип риска</th>
            <th>Вероятность · 24 ч</th>
            {!compact && <th className="recommendation-head">Рекомендация</th>}
            <th>Статус</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {forecasts.map((f) => (
            <tr
              key={f.id}
              onClick={() => onSelect(f)}
              tabIndex={0}
              onKeyDown={(e) => e.key === "Enter" && onSelect(f)}
              aria-label={`Открыть прогноз: ${f.object_name}, ${kindNames[f.kind]}`}
            >
              <td>
                <div className="object-title">
                  <span
                    className={`object-symbol ${f.above_threshold ? "alert" : ""}`}
                  >
                    <KindIcon kind={f.kind} />
                  </span>
                  <div>
                    <strong>{f.object_name}</strong>
                    <small>
                      {kindNames[f.kind]}{" "}
                      <span className="muted">· #{f.object_id}</span>
                    </small>
                    {f.source === "stream" && (
                      <span className="source-chip">
                        Поток СМВУ · {date(f.as_of, true)}
                      </span>
                    )}
                  </div>
                </div>
              </td>
              <td>
                <Risk forecast={f} />
                <RiskBadge level={f.risk} />
              </td>
              {!compact && (
                <td className="recommendation-cell">{f.recommendation}</td>
              )}
              <td>
                <span
                  className={`status ${f.decision ? "reviewed" : f.above_threshold ? "attention" : "normal"}`}
                >
                  {f.decision ? (
                    <>
                      <Check size={12} />
                      {f.decision.work_outcome
                        ? outcomeNames[f.decision.work_outcome]
                        : f.decision.work_status
                          ? workStatusNames[f.decision.work_status]
                          : "Решение принято"}
                    </>
                  ) : f.above_threshold ? (
                    "Требует проверки"
                  ) : f.threshold > 1 ? (
                    "Недостаточно данных"
                  ) : (
                    "Наблюдение"
                  )}
                </span>
              </td>
              <td>
                <ChevronRight size={17} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {!forecasts.length && (
        <Empty title="Прогнозов по выбранным условиям нет">
          Измените фильтр или дату архива.
        </Empty>
      )}
    </div>
  );
}

export function NetworkMap({
  topology,
  forecasts,
  onSelect,
  expanded = false,
}: {
  topology: Topology;
  forecasts: Forecast[];
  onSelect: (f: Forecast) => void;
  expanded?: boolean;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const [group, setGroup] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1);
  const roots = topology.nodes.filter((n) => n.level === 2);
  const leaves = topology.nodes.filter(
    (n) => n.level === 3 && (!group || n.parent_id === group),
  );
  const lanes = group ? roots.filter((n) => n.id === group) : roots;
  const width = 1100;
  const groupsPerRow = expanded ? 4 : 4;
  /* Row height follows the largest node in the row, so a node with more than
     six objects never overlaps the next row's labels. */
  const childRows = (id: number) =>
    Math.max(1, Math.ceil(leaves.filter((n) => n.parent_id === id).length / 6));
  const rowCount = Math.ceil(lanes.length / groupsPerRow);
  const tallest = (row: number) =>
    Math.max(
      1,
      ...lanes
        .slice(row * groupsPerRow, (row + 1) * groupsPerRow)
        .map((r) => childRows(r.id)),
    );
  const rowTops: number[] = [];
  for (let row = 0; row < rowCount; row++)
    rowTops.push(
      row === 0
        ? 35
        : rowTops[row - 1] +
            (expanded ? 120 : 95) +
            35 * (tallest(row - 1) - 1),
    );
  const cells = lanes.map((r, i) => ({
    ...r,
    x: 50 + (i % groupsPerRow) * 268,
    y: rowTops[Math.floor(i / groupsPerRow)],
  }));
  const contentHeight =
    (rowTops[rowCount - 1] ?? 35) + 60 + 35 * (tallest(rowCount - 1) - 1);
  const target = (id: number) => forecasts.find((f) => f.object_id === id);
  const active = topology.nodes.find((n) => n.id === hover);
  return (
    <div className={`network-container ${expanded ? "expanded" : ""}`}>
      <div className="map-tools">
        <span>
          <Radio size={13} /> Схема инфраструктуры
        </span>
        <select
          aria-label="Эксплуатационный узел"
          value={group || ""}
          onChange={(e) => setGroup(Number(e.target.value) || null)}
        >
          <option value="">Все узлы</option>
          {roots.map((r) => (
            <option key={r.id} value={r.id}>
              {r.name}
            </option>
          ))}
        </select>
        <button
          aria-label="Изменить масштаб схемы"
          onClick={() => setZoom((z) => (z === 1 ? 1.35 : 1))}
        >
          <Maximize2 size={15} />
        </button>
      </div>
      <div className="map-canvas">
        <svg
          viewBox={`0 0 ${width} ${Math.max(expanded ? 610 : 410, contentHeight)}`}
          style={{ width: `${zoom * 100}%`, minWidth: expanded ? 850 : 650 }}
          role="img"
          aria-label="Схематичная карта объектов по иерархии справочника"
        >
          <defs>
            <pattern
              id="map-grid"
              width="22"
              height="22"
              patternUnits="userSpaceOnUse"
            >
              <circle cx="1" cy="1" r=".7" fill="#cfcfcf" />
            </pattern>
            <filter id="node-shadow">
              <feDropShadow dx="0" dy="2" stdDeviation="2" floodOpacity=".08" />
            </filter>
          </defs>
          <rect width="1500" height="1400" fill="url(#map-grid)" />
          {cells.map((r) => {
            const children = leaves.filter((n) => n.parent_id === r.id);
            return (
              <g key={r.id}>
                <text x={r.x} y={r.y} className="map-group-name">
                  {r.name.replace("объект ", "").slice(0, 25)}
                </text>
                <path
                  d={`M${r.x + 6},${r.y + 25} H${r.x + 232}`}
                  stroke="#d6d6d6"
                  strokeWidth="5"
                  strokeLinecap="round"
                />
                {children.map((n, i) => {
                  const x = r.x + 12 + (i % 6) * 40,
                    y = r.y + 25 + Math.floor(i / 6) * 35;
                  const high = n.risk === "high" || n.risk === "critical",
                    f = target(n.id);
                  return (
                    <g
                      key={n.id}
                      className={`map-node ${f ? "clickable" : ""}`}
                      onMouseEnter={() => setHover(n.id)}
                      onMouseLeave={() => setHover(null)}
                      onFocus={() => setHover(n.id)}
                      onBlur={() => setHover(null)}
                      onClick={() => f && onSelect(f)}
                      tabIndex={f ? 0 : -1}
                      role="button"
                      aria-label={`${n.name}, ${n.risk === "unknown" ? "нет прогноза" : riskNames[n.risk].toLowerCase() + " риск"}`}
                      onKeyDown={(e) => e.key === "Enter" && f && onSelect(f)}
                    >
                      <title>
                        {n.name} · {n.channels} каналов
                        {n.probability !== null
                          ? ` · ${pct(n.probability)}`
                          : ""}
                      </title>
                      {i >= 6 && (
                        <path
                          d={`M${x},${r.y + 25} V${y}`}
                          stroke="#d6d6d6"
                          strokeWidth="2"
                        />
                      )}
                      {high && (
                        <circle
                          cx={x}
                          cy={y}
                          r="14"
                          className={`map-halo ${n.risk}`}
                          strokeDasharray="2 2"
                        />
                      )}
                      <circle
                        cx={x}
                        cy={y}
                        r={high ? 7 : n.risk === "watch" ? 6 : 5}
                        className={`map-dot ${n.risk}`}
                        strokeWidth={high ? 2 : 1.5}
                        filter="url(#node-shadow)"
                      />
                      <text
                        x={x}
                        y={y + 20}
                        textAnchor="middle"
                        className="map-node-label"
                      >
                        {n.id}
                      </text>
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>
      <div className="map-footer">
        <div className="map-legend">
          {(["critical", "high", "watch", "low"] as RiskLevel[]).map((l) => (
            <span key={l}>
              <i className={`legend-dot ${l}`} />
              {riskNames[l]}
            </span>
          ))}
          <span>
            <i className="legend-dot unknown" />
            Нет прогноза
          </span>
        </div>
        <span>
          {active ? `${active.name} · ${active.channels} каналов` : ""}
        </span>
      </div>
    </div>
  );
}

export function ForecastDrawer({
  forecast,
  user,
  onClose,
  onSaved,
}: {
  forecast: Forecast;
  user: User;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [detail, setDetail] = useState<Detail | null>(null),
    [error, setError] = useState("");
  const [tab, setTab] = useState<"analysis" | "events" | "decision">(
    "analysis",
  );
  const [action, setAction] = useState(forecast.decision?.action || "monitor");
  const [options, setOptions] = useState<{
    reasons: { id: string; label: string }[];
    actions: Record<string, string>;
  } | null>(null);
  const [order, setOrder] = useState<WorkOrder | null>(null),
    [orderBusy, setOrderBusy] = useState(false);
  const canDecide = can(user, "decisions.write"),
    canDraft = can(user, "work.edit");
  const [reason, setReason] = useState(
    forecast.decision?.reason || "sensor_pattern",
  );
  const [comment, setComment] = useState(forecast.decision?.comment || "");
  const [saving, setSaving] = useState(false),
    [saved, setSaved] = useState(false);
  const [retrospective, setRetrospective] = useState<{
    occurred: boolean;
    episodes: {
      start_ts: string;
      channel_count: number;
      duration_seconds: number;
    }[];
  } | null>(null);
  useEffect(() => {
    let cancelled = false;
    api<Detail>(
      forecast.batch_id
        ? `/imports/${forecast.batch_id}/forecast/${forecast.object_id}/${forecast.kind}`
        : `/forecast/${forecast.object_id}/${forecast.kind}?as_of=${encodeURIComponent(forecast.as_of)}`,
    )
      .then((d) => !cancelled && setDetail(d))
      .catch((e) => !cancelled && setError(e.message));
    api<{
      reasons: { id: string; label: string }[];
      actions: Record<string, string>;
    }>("/reasons")
      .then((r) => !cancelled && setOptions(r))
      .catch(() => {});
    const listener = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", listener);
    const before = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      cancelled = true;
      document.removeEventListener("keydown", listener);
      document.body.style.overflow = before;
    };
  }, [forecast, onClose]);
  async function save() {
    setSaving(true);
    setError("");
    try {
      await api("/decisions", {
        method: "POST",
        body: JSON.stringify({
          prediction_id: forecast.id,
          object_id: forecast.object_id,
          kind: forecast.kind,
          as_of: forecast.as_of,
          action,
          reason,
          comment,
          ...(forecast.batch_id ? { batch_id: forecast.batch_id } : {}),
        }),
      });
      setSaved(true);
      onSaved();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }
  async function draftOrder() {
    setOrderBusy(true);
    setError("");
    try {
      const body = forecast.decision?.id
        ? { decision_id: forecast.decision.id }
        : {
            object_id: forecast.object_id,
            kind: forecast.kind,
            as_of: forecast.as_of,
          };
      setOrder(
        await api<WorkOrder>("/work-orders", {
          method: "POST",
          body: JSON.stringify(body),
        }),
      );
      onSaved();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setOrderBusy(false);
    }
  }
  async function reveal() {
    try {
      setRetrospective(
        await api(
          `/retrospective/${forecast.object_id}/${forecast.kind}?as_of=${encodeURIComponent(forecast.as_of)}`,
        ),
      );
    } catch (e) {
      setError((e as Error).message);
    }
  }
  return (
    <div className="drawer-layer">
      <div className="drawer-backdrop" onClick={onClose} />
      <aside
        className="forecast-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="drawer-title"
      >
        <header className="drawer-header">
          <span className="eyebrow">
            {forecast.batch_id
              ? `ПОТОК СМВУ · ${forecast.batch_id.slice(0, 8).toUpperCase()}`
              : `ПРОГНОЗ ${forecast.id.slice(3, 11)}`}
          </span>
          <button
            className="icon-button"
            onClick={onClose}
            aria-label="Закрыть карточку"
          >
            <X size={22} />
          </button>
        </header>
        <div className="drawer-intro">
          <KindTag kind={forecast.kind} />
          <h2 id="drawer-title">{forecast.object_name}</h2>
          <p>
            Объект #{forecast.object_id} <span>·</span>{" "}
            {date(forecast.as_of, true)} МСК
          </p>
          <div className="risk-summary">
            <div>
              <span>Вероятность нового эпизода</span>
              <strong>{pct(forecast.probability)}</strong>
              <small>на следующие 24 часа</small>
            </div>
            <div
              className={`risk-gauge ${forecast.above_threshold ? "attention" : ""}`}
            >
              <Gauge size={31} />
              <RiskBadge level={forecast.risk} />
              <span>
                {forecast.threshold > 1
                  ? "Без предупреждений"
                  : forecast.above_threshold
                    ? "Требует проверки"
                    : "Наблюдение"}
              </span>
              <small>
                {forecast.threshold > 1
                  ? "Предупреждения отключены"
                  : `Порог ${pct(forecast.threshold)}`}
              </small>
            </div>
          </div>
        </div>
        <div className="drawer-tabs" role="tablist">
          {[
            ["analysis", "Обоснование"],
            ["events", "Исходные события"],
            ["decision", canDraft && !canDecide ? "Работы" : "Решение"],
          ].map(([id, label]) => (
            <button
              key={id}
              role="tab"
              aria-selected={tab === id}
              className={tab === id ? "active" : ""}
              onClick={() => setTab(id as typeof tab)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="drawer-content">
          {error && <ErrorNotice message={error} />}{" "}
          {!detail ? (
            <Loading text="Собираем факторы и исходные события…" />
          ) : (
            <>
              {tab === "analysis" && (
                <>
                  {detail.trend?.length ? (
                    <>
                      <div className="section-label">
                        <h3>Как менялся риск</h3>
                        <span>Последние 72 часа</span>
                      </div>
                      <div className="drawer-chart">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={detail.trend}>
                            <defs>
                              <linearGradient
                                id="risk-fill"
                                x1="0"
                                y1="0"
                                x2="0"
                                y2="1"
                              >
                                <stop
                                  offset="0%"
                                  stopColor="#171717"
                                  stopOpacity={0.2}
                                />
                                <stop
                                  offset="100%"
                                  stopColor="#171717"
                                  stopOpacity={0}
                                />
                              </linearGradient>
                            </defs>
                            <CartesianGrid vertical={false} stroke="#ebebeb" />
                            <XAxis
                              dataKey="as_of"
                              tickFormatter={(v) => clock(v)}
                              minTickGap={38}
                              tick={{ fontSize: 11 }}
                              axisLine={false}
                              tickLine={false}
                            />
                            <YAxis
                              tickFormatter={(v) => pct(v, 0)}
                              tick={{ fontSize: 10 }}
                              width={42}
                              axisLine={false}
                              tickLine={false}
                            />
                            <Tooltip
                              formatter={(v) => pct(Number(v))}
                              labelFormatter={(v) => date(String(v), true)}
                            />
                            <ReferenceLine
                              y={forecast.threshold}
                              stroke="#737373"
                              strokeDasharray="4 4"
                            />
                            <Area
                              type="stepAfter"
                              dataKey="probability"
                              name="Вероятность"
                              stroke="#171717"
                              strokeWidth={2}
                              fill="url(#risk-fill)"
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </>
                  ) : null}
                  <div className="section-label">
                    <h3>Факторы прогноза</h3>
                    <span>Вклад модели</span>
                  </div>
                  <div className="factors">
                    {detail.explanation.slice(0, 5).map((f) => (
                      <div className="factor" key={f.feature}>
                        <span
                          className={
                            f.contribution > 0 ? "factor-up" : "factor-down"
                          }
                        >
                          {f.contribution > 0 ? (
                            <ArrowUp size={14} />
                          ) : (
                            <ArrowDown size={14} />
                          )}
                        </span>
                        <div>
                          <strong>{f.label}</strong>
                          <small>
                            {typeof f.value === "number"
                              ? num(Math.round(f.value * 100) / 100)
                              : f.value || "Нет наблюдения"}
                          </small>
                        </div>
                        <div className="factor-bar">
                          <i
                            className={f.contribution > 0 ? "up" : "down"}
                            style={{
                              width: `${Math.min(100, (Math.abs(f.contribution) / (Math.abs(detail.explanation[0]?.contribution) || 1)) * 100)}%`,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="section-label">
                    <h3>Рекомендуемые действия</h3>
                    <Wrench size={15} />
                  </div>
                  <ol className="recommendation-list">
                    {(
                      detail.recommendation_details ||
                      detail.recommendations.map((text) => ({
                        text,
                        basis: "",
                        source: "general" as const,
                      }))
                    ).map((r, i) => (
                      <li key={r.text} className={`rec-${r.source}`}>
                        <span>{i + 1}</span>
                        <div>
                          {r.text}
                          {r.basis && (
                            <small>
                              {r.source === "signals" ? "По журналу: " : ""}
                              {r.basis}
                            </small>
                          )}
                        </div>
                      </li>
                    ))}
                  </ol>
                  {detail.verification && (
                    <>
                      <div className="section-label">
                        <h3>Проверка ситуации</h3>
                        <Camera size={15} />
                      </div>
                      <div className="verification-grid">
                        <div>
                          <span>Тревожных сообщений за сутки</span>
                          <strong>{detail.verification.alarm_messages}</strong>
                          <small>
                            {detail.verification.alarm_sensors.length
                              ? detail.verification.alarm_sensors
                                  .map(
                                    (s) =>
                                      `${s.sensor_type} · ${s.channels} кан.`,
                                  )
                                  .join("; ")
                              : "Сработавших датчиков нет"}
                          </small>
                        </div>
                        <div>
                          <span>Каналов в состоянии неисправности</span>
                          <strong>{detail.verification.fault_channels}</strong>
                          <small>
                            Сбой датчика не равен событию на объекте
                          </small>
                        </div>
                        <div>
                          <span>Другие риски объекта</span>
                          <strong>
                            {
                              detail.verification.related_forecasts.filter(
                                (r) =>
                                  r.risk === "high" || r.risk === "critical",
                              ).length
                            }
                          </strong>
                          <small>
                            {detail.verification.related_forecasts
                              .map(
                                (r) =>
                                  `${kindNames[r.kind]} ${pct(r.probability)}`,
                              )
                              .join(" · ")}
                          </small>
                        </div>
                      </div>
                      <ul className="source-status">
                        {detail.verification.external_sources.map((s) => (
                          <li key={s.id}>
                            <span>{s.label}</span>
                            <b>
                              {s.status === "not_connected"
                                ? "не подключено"
                                : s.status}
                            </b>
                          </li>
                        ))}
                      </ul>
                    </>
                  )}
                  {detail.calendar && (
                    <>
                      <div className="section-label">
                        <h3>Календарный контекст</h3>
                        <CalendarClock size={15} />
                      </div>
                      <div className="calendar-note">
                        <p>
                          {detail.calendar.weekday[0].toUpperCase() +
                            detail.calendar.weekday.slice(1)}
                          {detail.calendar.weekend ? ", выходной" : ""},{" "}
                          {detail.calendar.hour}:00.{" "}
                          {detail.calendar.weekday_ratio !== null &&
                            `В этот день недели эпизоды этого типа начинались в ${num(Math.round(detail.calendar.weekday_ratio * 100) / 100)} раза ${detail.calendar.weekday_ratio >= 1 ? "чаще" : "реже"} среднего. `}
                          На объекте за 30 дней:{" "}
                          {detail.calendar.object_episodes_30d} эпизодов
                          {detail.calendar.object_last_episode
                            ? `, последний ${date(detail.calendar.object_last_episode, true)}`
                            : ""}
                          .
                        </p>
                        <small>
                          По эпизодам, начавшимся до момента прогноза
                          {detail.calendar.history_from
                            ? ` (с ${date(detail.calendar.history_from)})`
                            : ""}
                          .
                        </small>
                      </div>
                    </>
                  )}
                  {can(user, "model.view") && !forecast.batch_id && (
                    <div className="retrospective">
                      <button onClick={reveal}>
                        <Clock3 size={15} />
                        Что произошло за следующие 24 часа
                        <ArrowRight size={15} />
                      </button>
                      {retrospective && (
                        <div className="retrospective-result">
                          <span className="eyebrow">ФАКТ ПО ЖУРНАЛУ</span>
                          <strong>
                            {retrospective.occurred
                              ? "Зарегистрирован эпизод"
                              : "Эпизод не зарегистрирован"}
                          </strong>
                          {retrospective.episodes.slice(0, 4).map((e, i) => (
                            <p key={i}>
                              {date(e.start_ts, true)} · {e.channel_count}{" "}
                              каналов
                            </p>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </>
              )}
              {tab === "events" && (
                <>
                  <div className="section-label">
                    <h3>Наблюдения до прогноза</h3>
                    <span>За 24 часа</span>
                  </div>
                  <div className="source-events">
                    {detail.source_events.map((e, i) => (
                      <div key={`${e.event_id}-${i}`} className="source-event">
                        <span
                          className={`event-dot ${e.alarm ? "alarm" : ""}`}
                        />
                        <div>
                          <strong>{e.value}</strong>
                          <p>
                            {e.sensor_name} <span>· #{e.channel_id}</span>
                          </p>
                          <small>
                            {date(e.ts, true)} · запись {e.event_id}
                          </small>
                        </div>
                      </div>
                    ))}
                  </div>
                  {!detail.source_events.length && (
                    <Empty title="За последние сутки записей нет">
                      Событийный журнал обновляется при изменении состояния.
                    </Empty>
                  )}
                </>
              )}
              {tab === "decision" && (
                <>
                  {canDecide && <h3>Зафиксировать решение</h3>}
                  {forecast.decision && (
                    <div className="current-decision">
                      <span className="eyebrow">ТЕКУЩЕЕ РЕШЕНИЕ</span>
                      <strong>
                        {options?.actions[forecast.decision.action] ||
                          forecast.decision.action}
                      </strong>
                      {forecast.decision.comment && (
                        <p>{forecast.decision.comment}</p>
                      )}
                      {forecast.decision.work_status && (
                        <small>
                          {forecast.decision.work_outcome
                            ? outcomeNames[forecast.decision.work_outcome]
                            : workStatusNames[forecast.decision.work_status]}
                        </small>
                      )}
                    </div>
                  )}
                  {canDraft && (
                    <div className="work-order-box">
                      <ClipboardList size={20} />
                      <div>
                        <strong>Заявка на работы</strong>

                        {order ? (
                          <p className="work-order-created">
                            Черновик {order.number} создан
                          </p>
                        ) : (
                          <button
                            className="primary-button"
                            disabled={orderBusy}
                            onClick={draftOrder}
                          >
                            {orderBusy ? (
                              <LoaderCircle className="spin" size={16} />
                            ) : (
                              <ClipboardList size={16} />
                            )}
                            Сформировать черновик заявки
                          </button>
                        )}
                      </div>
                    </div>
                  )}
                  {!canDecide ? (
                    !forecast.decision &&
                    !canDraft && <p className="muted">Решение ещё не принято</p>
                  ) : (
                    <form
                      onSubmit={(e) => {
                        e.preventDefault();
                        save();
                      }}
                      className="decision-form"
                    >
                      <label>
                        Действие
                        <select
                          value={action}
                          onChange={(e) => {
                            setAction(e.target.value);
                            setSaved(false);
                          }}
                        >
                          {Object.entries(
                            options?.actions || {
                              dispatch: "Выезд бригады",
                              inspect: "Направить бригаду на проверку",
                              monitor: "Мониторинг ситуации",
                              false_alarm: "Ложное срабатывание",
                              maintenance: "Запланировать ТО",
                            },
                          ).map(([id, label]) => (
                            <option key={id} value={id}>
                              {label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Основание
                        <select
                          value={reason}
                          onChange={(e) => setReason(e.target.value)}
                        >
                          {(options?.reasons || []).map((r) => (
                            <option key={r.id} value={r.id}>
                              {r.label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Комментарий
                        <textarea
                          value={comment}
                          onChange={(e) => setComment(e.target.value)}
                          placeholder="Что проверили и какие действия нужны"
                          maxLength={2000}
                          rows={5}
                          required={reason === "other"}
                        />
                      </label>
                      <button
                        className="primary-button"
                        disabled={saving}
                        type="submit"
                      >
                        {saving ? (
                          <LoaderCircle className="spin" size={17} />
                        ) : saved ? (
                          <Check size={17} />
                        ) : (
                          <ShieldCheck size={17} />
                        )}{" "}
                        {saved ? "Решение сохранено" : "Сохранить решение"}
                      </button>
                    </form>
                  )}
                </>
              )}
            </>
          )}
        </div>
        <footer className="drawer-bottom">
          <span>
            <Clock3 size={14} />
            Горизонт до {date(forecast.valid_until, true)}
          </span>
          {tab !== "decision" && (canDecide || canDraft) && (
            <button
              className="primary-button"
              onClick={() => setTab("decision")}
            >
              {canDecide ? "Принять решение" : "Заявка на работы"}
              <ArrowRight size={15} />
            </button>
          )}
        </footer>
      </aside>
    </div>
  );
}
