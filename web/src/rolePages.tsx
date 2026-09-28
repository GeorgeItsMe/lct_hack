import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  Ban,
  Check,
  ClipboardList,
  Download,
  FileSpreadsheet,
  FileText,
  Gauge,
  LoaderCircle,
  RefreshCw,
  Send,
  ShieldCheck,
  SlidersHorizontal,
  UserPlus,
  X,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, date, kindNames, num, pct } from "./api";
import {
  Empty,
  ErrorNotice,
  KindIcon,
  KindTag,
  Loading,
  RiskBadge,
  WorkBadge,
} from "./components";
import { can } from "./roles";
import type {
  AdminUser,
  AuditRow,
  DecisionRow,
  Equipment,
  Forecast,
  Kind,
  LabelRow,
  Proposal,
  Role,
  Summary,
  ThresholdPreview,
  User,
  WorkOrder,
} from "./types";

/* Categorical order validated with the dataviz palette checker (adjacent CVD ΔE ≥ 9). */
export const KIND_COLORS: Record<Kind, string> = {
  fault: "#2a78d6",
  fire: "#eb6834",
  flood: "#1baf7a",
  access: "#eda100",
};
const KINDS: Kind[] = ["fault", "fire", "flood", "access"];
const MONTHS = [
  "янв",
  "фев",
  "мар",
  "апр",
  "май",
  "июн",
  "июл",
  "авг",
  "сен",
  "окт",
  "ноя",
  "дек",
];

function useLoad<T>(path: string | null, refresh = 0) {
  const [data, setData] = useState<T | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    if (!path) return;
    let live = true;
    setError("");
    api<T>(path)
      .then((r) => live && setData(r))
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [path, refresh]);
  return { data, error, setData };
}

function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string | number;
  hint?: string;
}) {
  return (
    <div className="data-stat">
      <span>{label}</span>
      <strong>{value}</strong>
      {hint && <small>{hint}</small>}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Technical staff                                                     */
/* ------------------------------------------------------------------ */

export function EquipmentPage({
  asOf,
  forecasts,
  onSelect,
  user,
}: {
  asOf: string;
  forecasts: Forecast[];
  onSelect: (f: Forecast) => void;
  user: User;
}) {
  const [refresh, setRefresh] = useState(0);
  const { data, error } = useLoad<Equipment>(
    asOf ? `/equipment?as_of=${encodeURIComponent(asOf)}` : null,
    refresh,
  );
  const [onlyIssues, setOnlyIssues] = useState(true),
    [open, setOpen] = useState<number | null>(null),
    [message, setMessage] = useState("");
  if (error) return <ErrorNotice message={error} />;
  if (!data) return <Loading text="Собираем состояние каналов…" />;
  const rows = data.objects.filter(
    (o) =>
      !onlyIssues ||
      o.above_threshold ||
      o.risk === "watch" ||
      o.channels_with_faults_24h > 0,
  );
  const forecastFor = (id: string) => forecasts.find((f) => f.id === id);
  async function draft(objectId: number) {
    setMessage("");
    try {
      const order = await api<WorkOrder>("/work-orders", {
        method: "POST",
        body: JSON.stringify({
          object_id: objectId,
          kind: "fault",
          as_of: asOf,
        }),
      });
      setMessage(`Черновик ${order.number} создан`);
      setRefresh((r) => r + 1);
    } catch (e) {
      setMessage((e as Error).message);
    }
  }
  return (
    <div className="page-stack">
      <div className="four-column">
        <Stat
          label="Объектов под контролем"
          value={data.totals.objects}
          hint="Прогноз отказа на 24 часа"
        />
        <Stat label="С риском отказа выше порога" value={data.totals.at_risk} />
        <Stat
          label="Каналов неисправно сейчас"
          value={data.totals.faulty_now}
          hint="Последнее состояние канала за сутки"
        />
        <Stat
          label="Объектов с неисправностями"
          value={data.totals.objects_with_faults}
          hint="Хотя бы одно сообщение за сутки"
        />
      </div>
      {message && <div className="settings-message">{message}</div>}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Оборудование по объектам</h2>
            <span>{data.note}</span>
          </div>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={onlyIssues}
              onChange={(e) => setOnlyIssues(e.target.checked)}
            />
            Только с риском или неисправностями
          </label>
        </div>
        {!rows.length ? (
          <Empty title="Неисправностей и повышенного риска нет">
            Снимите фильтр, чтобы увидеть все объекты.
          </Empty>
        ) : (
          <div className="equipment-list">
            {rows.map((o) => {
              const forecast = forecastFor(o.forecast_id);
              const expanded = open === o.object_id;
              return (
                <article key={o.object_id} className="equipment-row">
                  <button
                    className="equipment-head"
                    onClick={() => setOpen(expanded ? null : o.object_id)}
                    aria-expanded={expanded}
                  >
                    <span className="object-symbol">
                      <Gauge size={18} />
                    </span>
                    <div>
                      <strong>{o.object_name}</strong>
                      <small>
                        Отказ: {pct(o.probability)} · порог{" "}
                        {pct(Math.min(o.threshold, 1))}
                      </small>
                    </div>
                    <RiskBadge level={o.risk} />
                    <span className="equipment-count">
                      <b>{o.faulty_now}</b> неисправно сейчас
                    </span>
                    <span className="equipment-count">
                      <b>{o.fault_messages_24h}</b> сообщений за сутки
                    </span>
                    <span className="equipment-orders">
                      {o.open_orders.length
                        ? `Заявки: ${o.open_orders.join(", ")}`
                        : "Заявок нет"}
                    </span>
                  </button>
                  {expanded && (
                    <div className="equipment-body">
                      {o.recommendations.length > 0 && (
                        <ol className="recommendation-list">
                          {o.recommendations.map((r, i) => (
                            <li key={r.text} className="rec-signals">
                              <span>{i + 1}</span>
                              <div>
                                {r.text}
                                <small>По журналу: {r.basis}</small>
                              </div>
                            </li>
                          ))}
                        </ol>
                      )}
                      {o.channels.length > 0 ? (
                        <div className="table-scroll">
                          <table>
                            <thead>
                              <tr>
                                <th>Канал</th>
                                <th>Тип датчика</th>
                                <th>Последнее состояние</th>
                                <th>Время</th>
                                <th>Сообщений о неисправности</th>
                              </tr>
                            </thead>
                            <tbody>
                              {o.channels.map((c) => (
                                <tr key={c.channel_id}>
                                  <td>
                                    {c.sensor_name}{" "}
                                    <span className="muted">
                                      #{c.channel_id}
                                    </span>
                                  </td>
                                  <td>{c.sensor_type}</td>
                                  <td>
                                    <span
                                      className={`status ${c.current_fault ? "attention" : "normal"}`}
                                    >
                                      {c.value}
                                    </span>
                                  </td>
                                  <td>{date(c.ts, true)}</td>
                                  <td>{c.fault_messages}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <p className="micro-note">
                          Неисправностей за сутки нет.
                        </p>
                      )}
                      <div className="row-actions">
                        {forecast && (
                          <button
                            className="secondary-button"
                            onClick={() => onSelect(forecast)}
                          >
                            Открыть прогноз отказа
                            <ArrowRight size={15} />
                          </button>
                        )}
                        {can(user, "work.edit") && (
                          <button
                            className="primary-button"
                            onClick={() => draft(o.object_id)}
                          >
                            <ClipboardList size={16} />
                            Черновик заявки на ТО
                          </button>
                        )}
                      </div>
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}

export function WorkPage({
  user,
  onChanged,
}: {
  user: User;
  onChanged: () => void;
}) {
  const [refresh, setRefresh] = useState(0);
  const tasks = useLoad<DecisionRow[]>("/tasks", refresh);
  const orders = useLoad<WorkOrder[]>("/work-orders", refresh);
  const [message, setMessage] = useState(""),
    [syncing, setSyncing] = useState(false);
  const editable = can(user, "work.edit");
  const sync = useCallback(
    async (quiet = false) => {
      setSyncing(true);
      try {
        const result = await api<{ changed: { number: string }[] }>(
          "/work-orders/sync",
          { method: "POST" },
        );
        if (!quiet || result.changed.length) {
          setMessage(
            result.changed.length
              ? `Обновлено из системы заявок: ${result.changed.map((c) => c.number).join(", ")}`
              : "Статусы актуальны",
          );
        }
        if (result.changed.length) {
          setRefresh((r) => r + 1);
          onChanged();
        }
      } catch (e) {
        setMessage((e as Error).message);
      } finally {
        setSyncing(false);
      }
    },
    [onChanged],
  );
  useEffect(() => {
    sync(true);
    const timer = setInterval(() => sync(true), 15000);
    return () => clearInterval(timer);
  }, [sync]);
  async function create(decisionId: number) {
    try {
      await api<WorkOrder>("/work-orders", {
        method: "POST",
        body: JSON.stringify({ decision_id: decisionId }),
      });
      setRefresh((r) => r + 1);
    } catch (e) {
      setMessage((e as Error).message);
    }
  }
  if (tasks.error || orders.error)
    return <ErrorNotice message={tasks.error || orders.error} />;
  if (!tasks.data || !orders.data) return <Loading />;
  const waiting = tasks.data.filter((t) => !t.work_order);
  return (
    <div className="page-stack">
      {message && <div className="settings-message">{message}</div>}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>
              Задачи от диспетчера{" "}
              <span className="count-badge">{waiting.length}</span>
            </h2>
            <span>Решения диспетчера без заявки</span>
          </div>
        </div>
        {!waiting.length ? (
          <Empty title="Новых задач нет" />
        ) : (
          <div className="decision-cards">
            {waiting.map((t) => (
              <div key={t.id} className="decision-card static">
                <span className="decision-icon">
                  <KindIcon kind={t.kind} size={19} />
                </span>
                <div>
                  <strong>{t.object_name}</strong>
                  <small>
                    {kindNames[t.kind]} · прогноз {date(t.forecast_at, true)} ·{" "}
                    {t.user_name || "диспетчер"}
                  </small>
                  <p>{t.comment || "Комментарий не указан"}</p>
                </div>
                <div>
                  <span className="status attention">{t.action_label}</span>
                </div>
                {editable && (
                  <button
                    className="primary-button"
                    onClick={() => create(t.id)}
                  >
                    <ClipboardList size={15} />
                    Черновик заявки
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Заявки</h2>
          </div>
          <button
            className="secondary-button"
            onClick={() => sync(false)}
            disabled={syncing}
          >
            <RefreshCw size={15} className={syncing ? "spin" : ""} />
            Обновить статусы
          </button>
        </div>
        {!orders.data.length ? (
          <Empty title="Заявок пока нет" />
        ) : (
          <div className="order-list">
            {orders.data.map((o) => (
              <OrderCard
                key={o.id}
                order={o}
                editable={editable}
                onChange={() => {
                  setRefresh((r) => r + 1);
                  onChanged();
                }}
              />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function OrderCard({
  order,
  editable,
  onChange,
}: {
  order: WorkOrder;
  editable: boolean;
  onChange: () => void;
}) {
  const [title, setTitle] = useState(order.title),
    [description, setDescription] = useState(order.description),
    [priority, setPriority] = useState(order.priority),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [expanded, setExpanded] = useState(order.status === "draft");
  const draft = order.status === "draft" && editable;
  async function save(submit: boolean) {
    setBusy(true);
    setError("");
    try {
      await api(`/work-orders/${order.id}`, {
        method: "PATCH",
        body: JSON.stringify({ title, description, priority }),
      });
      if (submit)
        await api(`/work-orders/${order.id}/submit`, { method: "POST" });
      onChange();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <article className={`order-card ${order.status}`}>
      <button className="order-head" onClick={() => setExpanded(!expanded)}>
        <span className="order-number">{order.number}</span>
        <div>
          <strong>{order.title}</strong>
          <small>
            {order.object_name} · {kindNames[order.kind]} · создана{" "}
            {date(order.created_at, true)}
            {order.external_id
              ? ` · во внешней системе ${order.external_id}`
              : ""}
          </small>
        </div>
        <span className={`priority ${order.priority}`}>
          {order.priority_label}
        </span>
        <WorkBadge order={order} />
      </button>
      {expanded && (
        <div className="order-body">
          {error && <ErrorNotice message={error} />}
          {draft ? (
            <div className="decision-form">
              <label>
                Заголовок
                <input
                  value={title}
                  maxLength={200}
                  onChange={(e) => setTitle(e.target.value)}
                />
              </label>
              <label>
                Приоритет
                <select
                  value={priority}
                  onChange={(e) =>
                    setPriority(e.target.value as WorkOrder["priority"])
                  }
                >
                  <option value="urgent">Срочно</option>
                  <option value="high">Высокий</option>
                  <option value="normal">Обычный</option>
                </select>
              </label>
              <label>
                Описание работ
                <textarea
                  rows={9}
                  value={description}
                  maxLength={5000}
                  onChange={(e) => setDescription(e.target.value)}
                />
              </label>
              <div className="row-actions">
                <button
                  className="secondary-button"
                  disabled={busy}
                  onClick={() => save(false)}
                >
                  Сохранить черновик
                </button>
                <button
                  className="primary-button"
                  disabled={busy}
                  onClick={() => save(true)}
                >
                  {busy ? (
                    <LoaderCircle className="spin" size={16} />
                  ) : (
                    <Send size={16} />
                  )}
                  Передать в систему заявок
                </button>
              </div>
            </div>
          ) : (
            <>
              <pre className="order-text">{order.description}</pre>
              <ol className="order-timeline">
                {(
                  ["submitted", "accepted", "in_progress", "done"] as const
                ).map((step) => {
                  const order_steps = [
                    "draft",
                    "submitted",
                    "accepted",
                    "in_progress",
                    "done",
                  ];
                  const reached =
                    order_steps.indexOf(order.status) >=
                    order_steps.indexOf(step);
                  return (
                    <li key={step} className={reached ? "reached" : ""}>
                      {
                        {
                          submitted: "Передана",
                          accepted: "Принята",
                          in_progress: "В работе",
                          done: "Выполнена",
                        }[step]
                      }
                    </li>
                  );
                })}
              </ol>
              {order.outcome_label && (
                <p className="order-outcome">
                  Итог: <b>{order.outcome_label}</b>
                </p>
              )}
            </>
          )}
        </div>
      )}
    </article>
  );
}

/* ------------------------------------------------------------------ */
/* Analyst                                                             */
/* ------------------------------------------------------------------ */

type LabelsData = {
  rows: LabelRow[];
  accepted: number;
  rejected: number;
  retraining: {
    id: number;
    created_at: string;
    labels: number;
    positives: number;
    file_name: string;
    status: string;
    note: string;
  }[];
};

export function VerificationPage({ user }: { user: User }) {
  const [refresh, setRefresh] = useState(0);
  const { data, error } = useLoad<LabelsData>("/labels", refresh);
  const [filter, setFilter] = useState<"pending" | "all">("pending"),
    [message, setMessage] = useState("");
  if (error) return <ErrorNotice message={error} />;
  if (!data) return <Loading text="Сопоставляем решения, заявки и архив…" />;
  const rows = data.rows.filter((r) => filter === "all" || !r.label_review);
  async function retrain() {
    setMessage("");
    try {
      const result = await api<{ labels: number; note: string }>(
        "/retraining",
        { method: "POST" },
      );
      setMessage(`Набор из ${result.labels} меток сохранён`);
      setRefresh((r) => r + 1);
    } catch (e) {
      setMessage((e as Error).message);
    }
  }
  return (
    <div className="page-stack">
      <div className="four-column">
        <Stat label="Решений в журнале" value={data.rows.length} />
        <Stat
          label="Ожидают проверки"
          value={data.rows.filter((r) => !r.label_review).length}
        />
        <Stat label="Приняты как метки" value={data.accepted} />
        <Stat label="Исключены" value={data.rejected} />
      </div>
      {message && <div className="settings-message">{message}</div>}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Пары «прогноз → итог»</h2>
            <span>Решение диспетчера, итог работ и факт по журналу</span>
          </div>
          <div className="heading-controls">
            <select
              aria-label="Фильтр проверки"
              value={filter}
              onChange={(e) => setFilter(e.target.value as "pending" | "all")}
            >
              <option value="pending">Ожидают проверки</option>
              <option value="all">Все решения</option>
            </select>
            <a className="secondary-button" href="/api/labels/export.csv">
              <Download size={15} />
              Принятые метки CSV
            </a>
          </div>
        </div>
        {!rows.length ? (
          <Empty title="Нет решений для проверки" />
        ) : (
          <div className="label-list">
            {rows.map((r) => (
              <LabelCard
                key={r.id}
                row={r}
                onSaved={() => setRefresh((x) => x + 1)}
              />
            ))}
          </div>
        )}
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Наборы меток</h2>
            <span>Проверенные пары «прогноз → итог» для будущего обучения</span>
          </div>
          {can(user, "retrain.run") && (
            <button className="primary-button" onClick={retrain}>
              <ShieldCheck size={16} />
              Сохранить набор меток
            </button>
          )}
        </div>
        {!data.retraining.length ? (
          <p className="micro-note">Наборов ещё нет.</p>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Создан</th>
                  <th>Меток</th>
                  <th>Из них «событие было»</th>
                  <th>Файл</th>
                  <th>Статус</th>
                </tr>
              </thead>
              <tbody>
                {data.retraining.map((r) => (
                  <tr key={r.id} title={r.note}>
                    <td>{date(r.created_at, true)}</td>
                    <td>{r.labels}</td>
                    <td>{r.positives}</td>
                    <td>
                      <code>{r.file_name}</code>
                    </td>
                    <td>{r.status === "prepared" ? "Сохранён" : r.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function LabelCard({ row, onSaved }: { row: LabelRow; onSaved: () => void }) {
  const [label, setLabel] = useState<0 | 1>(
      (row.label_review?.label ?? row.suggested_label ?? 0) as 0 | 1,
    ),
    [note, setNote] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  async function save(verdict: "accepted" | "rejected") {
    setBusy(true);
    setError("");
    try {
      await api(`/labels/${row.id}`, {
        method: "PUT",
        body: JSON.stringify({
          verdict,
          label: verdict === "accepted" ? label : null,
          note,
        }),
      });
      onSaved();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const archive =
    row.archive_occurred === null
      ? "нет данных"
      : row.archive_occurred
        ? "эпизод был"
        : "эпизода не было";
  return (
    <article className="label-card">
      <div className="label-main">
        <KindTag kind={row.kind} />
        <strong>{row.object_name}</strong>
        <small>
          Прогноз {date(row.forecast_at, true)} · {row.user_name || "диспетчер"}
        </small>
      </div>
      <dl className="label-evidence">
        <div>
          <dt>Решение</dt>
          <dd>
            {row.action_label}
            <small>{row.reason_label}</small>
          </dd>
        </div>
        <div>
          <dt>Итог работ</dt>
          <dd>
            {row.work_order?.outcome_label ||
              row.work_order?.status_label ||
              "работ не было"}
          </dd>
        </div>
        <div>
          <dt>Журнал, 24 ч</dt>
          <dd>{archive}</dd>
        </div>
        <div>
          <dt>Предложено</dt>
          <dd>
            {row.suggested_label === null
              ? "—"
              : row.suggested_label
                ? "1 · событие"
                : "0 · нет события"}
            <small>{row.label_basis}</small>
          </dd>
        </div>
      </dl>
      <div className="label-actions">
        {row.label_review ? (
          <span
            className={`status ${row.label_review.verdict === "accepted" ? "reviewed" : "attention"}`}
          >
            {row.label_review.verdict === "accepted"
              ? `Принята: ${row.label_review.label}`
              : "Исключена"}
          </span>
        ) : null}
        <select
          aria-label="Значение метки"
          value={label}
          onChange={(e) => setLabel(Number(e.target.value) as 0 | 1)}
        >
          <option value={1}>1 · событие</option>
          <option value={0}>0 · нет события</option>
        </select>
        <input
          placeholder="Комментарий (обязателен при исключении)"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          maxLength={1000}
        />
        <button
          className="primary-button"
          disabled={busy}
          onClick={() => save("accepted")}
        >
          <Check size={15} />
          Принять
        </button>
        <button
          className="secondary-button"
          disabled={busy}
          onClick={() => save("rejected")}
        >
          <Ban size={15} />
          Исключить
        </button>
      </div>
      {error && <ErrorNotice message={error} />}
    </article>
  );
}

/* ------------------------------------------------------------------ */
/* Thresholds: analyst proposes, unit head approves                    */
/* ------------------------------------------------------------------ */

export function ThresholdsPage({
  user,
  onApplied,
}: {
  user: User;
  onApplied: () => void;
}) {
  const [refresh, setRefresh] = useState(0);
  const settings = useLoad<{
    thresholds: Record<Kind, number>;
    defaults: Record<Kind, number>;
  }>("/settings", refresh);
  const proposals = useLoad<Proposal[]>("/thresholds/proposals", refresh);
  const [kind, setKind] = useState<Kind>("access"),
    [value, setValue] = useState<number | null>(null),
    [preview, setPreview] = useState<ThresholdPreview | null>(null),
    [rationale, setRationale] = useState(""),
    [message, setMessage] = useState(""),
    [busy, setBusy] = useState(false);
  const canPropose = can(user, "thresholds.propose"),
    canApprove = can(user, "thresholds.approve");
  const current = settings.data?.thresholds[kind];
  useEffect(() => {
    if (current !== undefined) setValue(current);
  }, [kind, current]);
  useEffect(() => {
    if (value === null) return;
    const handle = setTimeout(() => {
      api<ThresholdPreview>(
        `/thresholds/preview?kind=${kind}&threshold=${value}`,
      )
        .then(setPreview)
        .catch((e) => setMessage(e.message));
    }, 250);
    return () => clearTimeout(handle);
  }, [kind, value]);
  async function propose() {
    setMessage("");
    if (value === null || Math.abs(value - (current || 0)) < 1e-9) {
      setMessage("Сдвиньте порог: новое значение совпадает с действующим");
      return;
    }
    if (rationale.trim().length < 5) {
      setMessage("Укажите обоснование изменения");
      return;
    }
    setBusy(true);
    try {
      await api("/thresholds/proposals", {
        method: "POST",
        body: JSON.stringify({ kind, threshold: value, rationale }),
      });
      setMessage("Предложение отправлено руководителю подразделения");
      setRationale("");
      setRefresh((r) => r + 1);
    } catch (e) {
      setMessage((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function applyDirect() {
    setBusy(true);
    setMessage("");
    try {
      await api("/settings", {
        method: "PUT",
        body: JSON.stringify({ thresholds: { [kind]: value } }),
      });
      setMessage("Порог применён и записан в журнал аудита");
      setRefresh((r) => r + 1);
      onApplied();
    } catch (e) {
      setMessage((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  if (settings.error || proposals.error)
    return <ErrorNotice message={settings.error || proposals.error} />;
  if (!settings.data || !proposals.data || value === null) return <Loading />;
  const pending = proposals.data.filter((p) => p.status === "pending");
  const pendingForKind = pending.find((p) => p.kind === kind);
  const disabled = value > 1;
  return (
    <div className="page-stack">
      {canApprove && pending.length > 0 && (
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>
                На утверждении{" "}
                <span className="count-badge">{pending.length}</span>
              </h2>
            </div>
          </div>
          <div className="proposal-list">
            {pending.map((p) => (
              <ProposalCard
                key={p.id}
                proposal={p}
                onDecided={() => {
                  setRefresh((r) => r + 1);
                  onApplied();
                }}
              />
            ))}
          </div>
        </section>
      )}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>
              {canPropose
                ? "Подобрать порог"
                : canApprove
                  ? "Изменить порог"
                  : "Действующие пороги"}
            </h2>
          </div>
        </div>
        <div className="kind-tabs">
          {KINDS.map((k) => (
            <button
              key={k}
              className={kind === k ? "active" : ""}
              onClick={() => setKind(k)}
            >
              <KindIcon kind={k} size={17} />
              {kindNames[k]}
              <small>
                {settings.data!.thresholds[k] > 1
                  ? "выкл."
                  : pct(settings.data!.thresholds[k])}
              </small>
            </button>
          ))}
        </div>
        <div className="threshold-editor">
          <label>
            Порог: <b>{disabled ? "предупреждения выключены" : pct(value)}</b>
            <input
              type="range"
              min="1"
              max="101"
              step="0.5"
              value={Math.round(value * 1000) / 10}
              disabled={!canPropose && !canApprove}
              onChange={(e) => setValue(Number(e.target.value) / 100)}
            />
          </label>
          <small>
            Действует{" "}
            {current !== undefined && current > 1 ? "выкл." : pct(current || 0)}
            {" · "}по умолчанию{" "}
            {settings.data.defaults[kind] > 1
              ? "выкл."
              : pct(settings.data.defaults[kind])}
          </small>
        </div>
        {preview && preview.kind === kind && <PreviewBlock preview={preview} />}
        {message && <div className="settings-message">{message}</div>}
        {canPropose && pendingForKind && (
          <div className="settings-message">
            На утверждении: {pct(Math.min(pendingForKind.current_value, 1))} →{" "}
            {pendingForKind.proposed_value > 1
              ? "выкл."
              : pct(pendingForKind.proposed_value)}{" "}
            · {date(pendingForKind.created_at, true)}
          </div>
        )}
        {canPropose && (
          <div className="decision-form proposal-form">
            <label>
              Обоснование
              <textarea
                rows={3}
                value={rationale}
                maxLength={2000}
                placeholder="Причина изменения"
                onChange={(e) => setRationale(e.target.value)}
              />
            </label>
            <button
              className="primary-button"
              disabled={busy || !!pendingForKind}
              onClick={propose}
            >
              <Send size={16} />
              Отправить на утверждение
            </button>
          </div>
        )}
        {canApprove && (
          <div className="row-actions">
            <button
              className="primary-button"
              disabled={busy || Math.abs(value - (current || 0)) < 1e-9}
              onClick={applyDirect}
            >
              <Check size={16} />
              Утвердить и применить
            </button>
          </div>
        )}
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>История предложений</h2>
          </div>
        </div>
        {!proposals.data.length ? (
          <p className="micro-note">Предложений ещё не было.</p>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Создано</th>
                  <th>Тип</th>
                  <th>Было → предложено</th>
                  <th>Кто предложил</th>
                  <th>Статус</th>
                  <th>Решение</th>
                </tr>
              </thead>
              <tbody>
                {proposals.data.map((p) => (
                  <tr key={p.id}>
                    <td>{date(p.created_at, true)}</td>
                    <td>{p.kind_label}</td>
                    <td>
                      {pct(Math.min(p.current_value, 1))} →{" "}
                      {p.proposed_value > 1 ? "выкл." : pct(p.proposed_value)}
                    </td>
                    <td>{p.proposed_by}</td>
                    <td>
                      <span
                        className={`status ${p.status === "approved" ? "reviewed" : p.status === "pending" ? "attention" : "normal"}`}
                      >
                        {
                          {
                            pending: "На утверждении",
                            approved: "Утверждено",
                            rejected: "Отклонено",
                          }[p.status]
                        }
                      </span>
                    </td>
                    <td>
                      {p.decided_by
                        ? `${p.decided_by}${p.decision_note ? ` · ${p.decision_note}` : ""}`
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function MetricsTable({
  current,
  proposed,
}: {
  current: ThresholdPreview["current"];
  proposed: ThresholdPreview["proposed"];
}) {
  const rows: [string, (m: typeof current) => string][] = [
    ["Порог", (m) => (m.threshold > 1 ? "выкл." : pct(m.threshold))],
    ["Precision (доля верных)", (m) => pct(m.precision)],
    ["Recall (найдено эпизодов)", (m) => pct(m.recall)],
    ["Предупреждений за период", (m) => num(m.alerts)],
    ["Из них верных", (m) => num(m.true_alerts)],
    ["Ложных", (m) => num(m.false_alerts)],
    [
      "Нагрузка: предупреждений в сутки",
      (m) => num(Math.round(m.warnings_per_day_7d * 10) / 10),
    ],
    ["Выше порога в текущий момент", (m) => num(m.warnings_now)],
  ];
  return (
    <div className="table-scroll">
      <table className="compare-table">
        <thead>
          <tr>
            <th>Показатель</th>
            <th>Сейчас</th>
            <th>Предложено</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, f]) => (
            <tr key={label}>
              <td>{label}</td>
              <td>{f(current)}</td>
              <td>
                <b>{f(proposed)}</b>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PreviewBlock({ preview }: { preview: ThresholdPreview }) {
  const curve = preview.curve.filter((c) => c.threshold <= 1);
  return (
    <div className="preview-grid">
      <div>
        <h3>Эффект на периоде настройки</h3>
        <MetricsTable current={preview.current} proposed={preview.proposed} />
        <p className="micro-note">{preview.period_label}</p>
      </div>
      <div>
        <h3>Precision и Recall в зависимости от порога</h3>
        <div className="chart-box">
          <ResponsiveContainer width="100%" height={230}>
            <LineChart
              data={curve}
              margin={{ top: 22, right: 16, bottom: 4, left: 0 }}
            >
              <CartesianGrid vertical={false} stroke="#ebebeb" />
              <XAxis
                dataKey="threshold"
                type="number"
                domain={[0, 1]}
                tickFormatter={(v) => pct(v, 0)}
                tick={{ fontSize: 11 }}
              />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(v) => pct(v, 0)}
                tick={{ fontSize: 11 }}
                width={42}
              />
              <Tooltip
                formatter={(v, name) => [pct(Number(v)), name]}
                labelFormatter={(v) => `Порог ${pct(Number(v))}`}
              />
              <Legend />
              <ReferenceLine
                x={Math.min(preview.proposed.threshold, 1)}
                stroke="#171717"
                strokeDasharray="4 4"
                label={{ value: "предложено", fontSize: 11, position: "top" }}
              />
              <ReferenceLine
                x={Math.min(preview.current.threshold, 1)}
                stroke="#a3a3a3"
                strokeDasharray="2 3"
              />
              <Line
                type="monotone"
                dataKey="precision"
                name="Precision"
                stroke={KIND_COLORS.fault}
                strokeWidth={2}
                dot={{ r: 3 }}
              />
              <Line
                type="monotone"
                dataKey="recall"
                name="Recall"
                stroke={KIND_COLORS.fire}
                strokeWidth={2}
                dot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function ProposalCard({
  proposal,
  onDecided,
}: {
  proposal: Proposal;
  onDecided: () => void;
}) {
  const [note, setNote] = useState(""),
    [error, setError] = useState("");
  async function decide(approve: boolean) {
    setError("");
    try {
      await api(
        `/thresholds/proposals/${proposal.id}/${approve ? "approve" : "reject"}`,
        { method: "POST", body: JSON.stringify({ note }) },
      );
      onDecided();
    } catch (e) {
      setError((e as Error).message);
    }
  }
  return (
    <article className="proposal-card">
      <div className="proposal-head">
        <KindTag kind={proposal.kind} />
        <strong>
          {pct(Math.min(proposal.current_value, 1))} →{" "}
          {proposal.proposed_value > 1 ? "выкл." : pct(proposal.proposed_value)}
        </strong>
        <small>
          {proposal.proposed_by} · {date(proposal.created_at, true)}
        </small>
      </div>
      <p className="proposal-rationale">«{proposal.rationale}»</p>
      <MetricsTable
        current={proposal.preview.current}
        proposed={proposal.preview.proposed}
      />
      <div className="label-actions">
        <input
          placeholder="Комментарий (обязателен при отклонении)"
          value={note}
          onChange={(e) => setNote(e.target.value)}
        />
        <button className="primary-button" onClick={() => decide(true)}>
          <Check size={15} />
          Утвердить
        </button>
        <button className="secondary-button" onClick={() => decide(false)}>
          <X size={15} />
          Отклонить
        </button>
      </div>
      {error && <ErrorNotice message={error} />}
    </article>
  );
}

/* ------------------------------------------------------------------ */
/* Unit head summary                                                   */
/* ------------------------------------------------------------------ */

export function SummaryPage({
  asOf,
  user,
  onOpenThresholds,
}: {
  asOf: string;
  user: User;
  onOpenThresholds: () => void;
}) {
  const { data, error } = useLoad<Summary>(
    asOf ? `/summary?as_of=${encodeURIComponent(asOf)}` : null,
  );
  const seasonal = useMemo(() => {
    if (!data) return [];
    const years = [...new Set(data.seasonality.map((s) => s.year))].sort();
    return MONTHS.map((m, i) => {
      const row: Record<string, number | string> = { month: m };
      for (const y of years) {
        const found = data.seasonality.find(
          (s) => s.year === y && s.month === i + 1,
        );
        if (found) row[String(y)] = found.alarms;
      }
      return row;
    });
  }, [data]);
  if (error) return <ErrorNotice message={error} />;
  if (!data) return <Loading text="Собираем сводку подразделения…" />;
  const years = [...new Set(data.seasonality.map((s) => s.year))]
    .filter((y) => y >= 2022)
    .sort();
  const yearColors = ["#a3a3a3", "#737373", "#525252", "#2a78d6", "#171717"];
  const exportable = can(user, "reports.export");
  const workload = data.workload;
  return (
    <div className="page-stack">
      {workload.pending_proposals && can(user, "thresholds.approve") && (
        <div className="stream-notification" role="status">
          <SlidersHorizontal size={18} />
          <span>Аналитик предложил изменить порог предупреждений</span>
          <button className="text-link" onClick={onOpenThresholds}>
            Рассмотреть
          </button>
        </div>
      )}
      <div className="four-column">
        <Stat
          label="Предупреждений на 24 часа"
          value={data.totals.warnings}
          hint={`${data.totals.objects_at_risk} объектов`}
        />
        <Stat label="Критических" value={data.totals.critical} />
        <Stat
          label="Доля ложных срабатываний"
          value={
            workload.false_alarm_share === null
              ? "—"
              : pct(workload.false_alarm_share)
          }
          hint={`Среди ${workload.decisions} решений диспетчеров`}
        />
        <Stat
          label="Заявок в работе"
          value={workload.orders_open}
          hint={`Выполнено: ${workload.orders_done}`}
        />
      </div>
      {exportable && (
        <section className="panel report-panel">
          <div>
            <h2>Отчёт для руководства</h2>
            <span>На {date(data.as_of, true)}</span>
          </div>
          <div className="row-actions">
            <a
              className="secondary-button"
              href={`/api/reports/summary.xlsx?as_of=${encodeURIComponent(asOf)}`}
            >
              <FileSpreadsheet size={16} />
              XLSX
            </a>
            <a
              className="primary-button"
              href={`/api/reports/summary.pdf?as_of=${encodeURIComponent(asOf)}`}
            >
              <FileText size={16} />
              PDF
            </a>
          </div>
        </section>
      )}
      <div className="two-column">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>По типам риска</h2>
              <span>Предупреждения на выбранный момент</span>
            </div>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Тип</th>
                  <th>Предупреждений</th>
                  <th>Критических</th>
                  <th>Порог</th>
                </tr>
              </thead>
              <tbody>
                {data.kinds.map((k) => (
                  <tr key={k.id}>
                    <td>
                      <KindTag kind={k.id} />
                    </td>
                    <td>{k.warnings}</td>
                    <td>{k.critical}</td>
                    <td>{k.threshold > 1 ? "выключены" : pct(k.threshold)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Отработка предупреждений</h2>
              <span>Решения диспетчеров и итоги заявок</span>
            </div>
          </div>
          <div className="bar-list">
            {workload.actions.map((a) => (
              <div key={a.id} className="bar-row">
                <span>{a.label}</span>
                <i
                  style={{
                    width: `${workload.decisions ? (a.count / workload.decisions) * 100 : 0}%`,
                  }}
                />
                <b>{a.count}</b>
              </div>
            ))}
          </div>
          <p className="micro-note">
            Итоги выполненных заявок: подтвердилось{" "}
            {workload.outcomes.confirmed}, не подтвердилось{" "}
            {workload.outcomes.not_confirmed}, сбой датчика{" "}
            {workload.outcomes.sensor_fault}.
          </p>
        </section>
      </div>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Динамика предупреждений за 7 суток</h2>
            <span>Объекты выше порога</span>
          </div>
        </div>
        <div className="chart-box">
          <ResponsiveContainer width="100%" height={240}>
            <LineChart
              data={data.trend}
              margin={{ top: 8, right: 16, bottom: 4, left: 0 }}
            >
              <CartesianGrid vertical={false} stroke="#ebebeb" />
              <XAxis
                dataKey="as_of"
                tickFormatter={(v) => date(String(v))}
                minTickGap={40}
                tick={{ fontSize: 11 }}
              />
              <YAxis allowDecimals={false} width={32} tick={{ fontSize: 11 }} />
              <Tooltip labelFormatter={(v) => date(String(v), true)} />
              <Legend />
              {KINDS.map((k) => (
                <Line
                  key={k}
                  type="stepAfter"
                  dataKey={k}
                  name={kindNames[k]}
                  stroke={KIND_COLORS[k]}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Эксплуатационные узлы</h2>
          </div>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Узел</th>
                <th>Объектов</th>
                <th>Уровни риска</th>
                <th>Предупреждений</th>
                <th>Критических</th>
                {KINDS.map((k) => (
                  <th key={k}>{kindNames[k]}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.nodes.map((n) => {
                const total =
                  n.levels.critical +
                    n.levels.high +
                    n.levels.watch +
                    n.levels.low || 1;
                return (
                  <tr key={n.id}>
                    <td>{n.name}</td>
                    <td>{n.objects}</td>
                    <td>
                      <span
                        className="level-bar"
                        title={`Критический ${n.levels.critical} · высокий ${n.levels.high} · повышенный ${n.levels.watch} · низкий ${n.levels.low}`}
                      >
                        {(["critical", "high", "watch", "low"] as const).map(
                          (l) =>
                            n.levels[l] > 0 && (
                              <i
                                key={l}
                                className={l}
                                style={{
                                  width: `${(n.levels[l] / total) * 100}%`,
                                }}
                              />
                            ),
                        )}
                      </span>
                    </td>
                    <td>{n.warnings}</td>
                    <td>{n.critical}</td>
                    {KINDS.map((k) => (
                      <td key={k}>{n.by_kind[k] || "—"}</td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
      <div className="two-column">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Сезонность тревожных сообщений</h2>
              <span>Тревожные сообщения по месяцам</span>
            </div>
          </div>
          <div className="chart-box">
            <ResponsiveContainer width="100%" height={240}>
              <LineChart
                data={seasonal}
                margin={{ top: 8, right: 16, bottom: 4, left: 8 }}
              >
                <CartesianGrid vertical={false} stroke="#ebebeb" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis
                  width={52}
                  tick={{ fontSize: 11 }}
                  tickFormatter={(v) => num(Number(v))}
                />
                <Tooltip formatter={(v) => num(Number(v))} />
                <Legend />
                {years.map((y, i) => (
                  <Line
                    key={y}
                    type="monotone"
                    dataKey={String(y)}
                    name={String(y)}
                    stroke={yearColors[i % yearColors.length]}
                    strokeWidth={y === years[years.length - 1] ? 2.5 : 1.5}
                    dot={false}
                    connectNulls={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Качество прогноза</h2>
              <span>Отложенный тест, июнь 2026</span>
            </div>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Тип</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>F1 базы</th>
                </tr>
              </thead>
              <tbody>
                {KINDS.map((k) => {
                  const q = data.quality[k];
                  return (
                    <tr key={k}>
                      <td>{kindNames[k]}</td>
                      {q.enabled ? (
                        <>
                          <td>{pct(q.precision)}</td>
                          <td>{pct(q.recall)}</td>
                          <td>{q.f1.toFixed(3).replace(".", ",")}</td>
                          <td>{q.baseline_f1.toFixed(3).replace(".", ",")}</td>
                        </>
                      ) : (
                        <td colSpan={4}>Предупреждения выключены</td>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Administrator                                                       */
/* ------------------------------------------------------------------ */

type UsersData = {
  users: AdminUser[];
  roles: { id: Role; label: string; permissions: string[] }[];
  ldap_configured: boolean;
};

const PERMISSION_NAMES: Record<string, string> = {
  "forecasts.view": "Просмотр прогнозов",
  "decisions.view": "Журнал решений",
  "decisions.write": "Решения по прогнозам",
  "notifications.all": "Уведомления о предупреждениях",
  "notifications.critical": "Уведомления о критических",
  "equipment.view": "Состояние оборудования",
  "work.view": "Статусы работ",
  "work.edit": "Черновики заявок",
  "data.import": "Загрузка данных",
  "labels.verify": "Верификация данных",
  "retrain.run": "Наборы меток",
  "model.view": "Проверка модели",
  "summary.view": "Сводка",
  "reports.export": "Отчёты PDF/XLSX",
  "thresholds.propose": "Предложение порогов",
  "thresholds.approve": "Утверждение порогов",
  "users.manage": "Пользователи и роли",
  "audit.view": "Журнал аудита",
};

export function UsersPage({ user }: { user: User }) {
  const [refresh, setRefresh] = useState(0);
  const { data, error } = useLoad<UsersData>("/users", refresh);
  const topology = useLoad<{
    nodes: { id: number; name: string; level: number }[];
  }>("/topology");
  const [form, setForm] = useState({
      username: "",
      name: "",
      role: "dispatcher" as Role,
      scope: "district",
      password: "",
      ldap: false,
    }),
    [message, setMessage] = useState("");
  const nodes = (topology.data?.nodes || []).filter((n) => n.level === 2);
  async function update(id: number, patch: Partial<AdminUser>) {
    setMessage("");
    try {
      await api(`/users/${id}`, {
        method: "PATCH",
        body: JSON.stringify(patch),
      });
      setRefresh((r) => r + 1);
    } catch (e) {
      setMessage((e as Error).message);
    }
  }
  async function create(e: React.FormEvent) {
    e.preventDefault();
    setMessage("");
    try {
      await api("/users", {
        method: "POST",
        body: JSON.stringify({
          ...form,
          password: form.ldap ? null : form.password,
        }),
      });
      setForm({ ...form, username: "", name: "", password: "" });
      setMessage("Пользователь создан");
      setRefresh((r) => r + 1);
    } catch (err) {
      setMessage((err as Error).message);
    }
  }
  if (error) return <ErrorNotice message={error} />;
  if (!data) return <Loading />;
  const scopeLabel = (scope: string) =>
    scope === "district"
      ? "Весь район"
      : nodes.find((n) => `node:${n.id}` === scope)?.name || scope;
  return (
    <div className="page-stack">
      {message && <div className="settings-message">{message}</div>}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Учётные записи</h2>
            <span>
              {data.ldap_configured
                ? "Вход через LDAPS"
                : "Локальные учётные записи"}
            </span>
          </div>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Логин</th>
                <th>Имя</th>
                <th>Роль</th>
                <th>Зона ответственности</th>
                <th>Статус</th>
              </tr>
            </thead>
            <tbody>
              {data.users.map((u) => (
                <tr key={u.id} className={u.active ? "" : "inactive"}>
                  <td>
                    <code>{u.username}</code>
                  </td>
                  <td>{u.name}</td>
                  <td>
                    <select
                      aria-label={`Роль ${u.username}`}
                      value={u.role}
                      disabled={u.id === user.id}
                      onChange={(e) =>
                        update(u.id, { role: e.target.value as Role })
                      }
                    >
                      {data.roles.map((r) => (
                        <option key={r.id} value={r.id}>
                          {r.label}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td>
                    <select
                      aria-label={`Зона ${u.username}`}
                      value={u.scope}
                      onChange={(e) => update(u.id, { scope: e.target.value })}
                    >
                      <option value="district">Весь район</option>
                      {nodes.map((n) => (
                        <option key={n.id} value={`node:${n.id}`}>
                          {n.name}
                        </option>
                      ))}
                      {!nodes.length && u.scope !== "district" && (
                        <option value={u.scope}>{scopeLabel(u.scope)}</option>
                      )}
                    </select>
                  </td>
                  <td>
                    <button
                      className="text-link"
                      disabled={u.id === user.id}
                      onClick={() => update(u.id, { active: !u.active })}
                    >
                      {u.active
                        ? "Активна · заблокировать"
                        : "Заблокирована · включить"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <div className="two-column">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Новый пользователь</h2>
            </div>
            <UserPlus size={18} />
          </div>
          <form className="decision-form" onSubmit={create}>
            <label>
              Логин
              <input
                value={form.username}
                required
                pattern="[A-Za-z0-9_.@\-]{3,80}"
                onChange={(e) => setForm({ ...form, username: e.target.value })}
              />
            </label>
            <label>
              Имя
              <input
                value={form.name}
                required
                onChange={(e) => setForm({ ...form, name: e.target.value })}
              />
            </label>
            <label>
              Роль
              <select
                value={form.role}
                onChange={(e) =>
                  setForm({ ...form, role: e.target.value as Role })
                }
              >
                {data.roles.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Зона ответственности
              <select
                value={form.scope}
                onChange={(e) => setForm({ ...form, scope: e.target.value })}
              >
                <option value="district">Весь район</option>
                {nodes.map((n) => (
                  <option key={n.id} value={`node:${n.id}`}>
                    {n.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={form.ldap}
                onChange={(e) => setForm({ ...form, ldap: e.target.checked })}
              />
              Вход через LDAP/AD (без локального пароля)
            </label>
            {!form.ldap && (
              <label>
                Пароль (не короче 12 символов)
                <input
                  type="password"
                  value={form.password}
                  minLength={12}
                  required
                  onChange={(e) =>
                    setForm({ ...form, password: e.target.value })
                  }
                />
              </label>
            )}
            <button className="primary-button" type="submit">
              <UserPlus size={16} />
              Создать
            </button>
          </form>
        </section>
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Роли и права</h2>
            </div>
          </div>
          <div className="role-matrix">
            {data.roles.map((r) => (
              <div key={r.id}>
                <strong>{r.label}</strong>
                <p>
                  {r.permissions
                    .map((p) => PERMISSION_NAMES[p] || p)
                    .join(" · ")}
                </p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

const AUDIT_NAMES: Record<string, string> = {
  login_success: "Вход",
  login_failed: "Неудачный вход",
  decision_saved: "Решение по прогнозу",
  thresholds_updated: "Изменены пороги",
  threshold_proposed: "Предложен порог",
  threshold_approved: "Порог утверждён",
  threshold_rejected: "Порог отклонён",
  work_order_drafted: "Черновик заявки",
  work_order_edited: "Изменён черновик",
  work_order_submitted: "Заявка передана",
  work_orders_synced: "Статусы заявок обновлены",
  label_reviewed: "Верификация метки",
  retraining_requested: "Сохранён набор меток",
  report_exported: "Выгрузка отчёта",
  user_created: "Создан пользователь",
  user_updated: "Изменён пользователь",
  telemetry_import: "Загрузка телеметрии",
  stream_forecast_requested: "Пересчёт потока",
};

export function AuditPage() {
  const { data, error } = useLoad<AuditRow[]>("/audit");
  const [query, setQuery] = useState(""),
    [onlyActions, setOnlyActions] = useState(true);
  if (error) return <ErrorNotice message={error} />;
  if (!data) return <Loading />;
  const rows = data.filter(
    (r) =>
      (!onlyActions || !r.action.startsWith("GET ")) &&
      `${r.user_name || ""} ${r.action} ${AUDIT_NAMES[r.action] || ""}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  return (
    <section className="panel">
      <div className="journal-toolbar">
        <div className="search-field">
          <input
            aria-label="Поиск в журнале"
            placeholder="Пользователь или действие"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={onlyActions}
            onChange={(e) => setOnlyActions(e.target.checked)}
          />
          Скрыть просмотры страниц
        </label>
        <span className="result-count">
          {rows.length} записей из 300 последних
        </span>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Время, МСК</th>
              <th>Пользователь</th>
              <th>Действие</th>
              <th>Подробности</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td>{date(r.created_at, true)}</td>
                <td>{r.user_name || "—"}</td>
                <td>{AUDIT_NAMES[r.action] || <code>{r.action}</code>}</td>
                <td className="audit-detail">
                  <code>{JSON.stringify(r.detail).slice(0, 160)}</code>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

type Integrations = {
  sources: { id: string; label: string; status: string; access: string }[];
  imports_enabled: boolean;
  hosting_notice: string | null;
};

const SOURCE_STATUS: Record<string, string> = {
  available: "Работает",
  not_connected: "Не подключено",
  emulated: "Тестовый адаптер",
  configured: "Настроено",
  not_configured: "Не настроено",
  requires_worker_container: "Нужен worker-контейнер",
};

export function ParametersPage() {
  const integrations = useLoad<Integrations>("/integrations");
  const settings = useLoad<{
    thresholds: Record<Kind, number>;
    horizon_hours: number;
    cooldown_hours: number;
  }>("/settings");
  if (integrations.error || settings.error)
    return <ErrorNotice message={integrations.error || settings.error} />;
  if (!integrations.data || !settings.data) return <Loading />;
  return (
    <div className="two-column">
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Интеграции</h2>
          </div>
        </div>
        <ul className="source-status wide">
          {integrations.data.sources.map((s) => (
            <li key={s.id}>
              <span>{s.label}</span>
              <b>{SOURCE_STATUS[s.status] || s.status}</b>
            </li>
          ))}
        </ul>
        {integrations.data.hosting_notice && (
          <p className="micro-note">{integrations.data.hosting_notice}</p>
        )}
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Параметры прогноза</h2>
          </div>
        </div>
        <dl className="config-list">
          <div>
            <dt>Горизонт прогноза</dt>
            <dd>{settings.data.horizon_hours} часа</dd>
          </div>
          <div>
            <dt>Повтор предупреждения по объекту</dt>
            <dd>не чаще раза в {settings.data.cooldown_hours} часа</dd>
          </div>
          {KINDS.map((k) => (
            <div key={k}>
              <dt>Порог · {kindNames[k]}</dt>
              <dd>
                {settings.data!.thresholds[k] > 1
                  ? "выключен"
                  : pct(settings.data!.thresholds[k])}
              </dd>
            </div>
          ))}
          <div>
            <dt>Часовой пояс источника</dt>
            <dd>МСК · UTC+3</dd>
          </div>
          <div>
            <dt>Команды оборудованию</dt>
            <dd>не отправляются</dd>
          </div>
        </dl>
      </section>
    </div>
  );
}
