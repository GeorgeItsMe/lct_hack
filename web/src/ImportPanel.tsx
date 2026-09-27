import { useEffect, useRef, useState } from "react";
import { ArrowDownToLine, FileUp, RefreshCw } from "lucide-react";
import { api, date, kindNames, num, pct } from "./api";
import { ErrorNotice, Loading } from "./components";
import { can } from "./roles";
import type { Kind, User } from "./types";
type BatchForecastDetail = {
  id: string;
  batch_id: string;
  object_id: number;
  object_name: string;
  kind: Kind;
  kind_label: string;
  as_of: string;
  probability: number;
  explanation: {
    feature: string;
    label: string;
    value: number | string | null;
    contribution: number;
  }[];
  source_events: {
    channel_id: number;
    ts: string;
    value: string;
    alarm: boolean;
  }[];
  recommendations: string[];
};
type Batch = {
  id: string;
  status: string;
  as_of: string;
  accepted_rows: number;
  exact_duplicates: number;
  error?: string;
  mode?: string;
  result?: {
    model_version?: string;
    elapsed_seconds: number;
    history_rows: number;
    forecasts: {
      object_id: number;
      object_name: string;
      kind_label: string;
      kind: Kind;
      probability: number;
      above_threshold: boolean;
      expected_episodes?: number;
      notification_due?: boolean;
      notification_status?: string;
      recommendation: string;
    }[];
  };
};
type StreamStatus = {
  events: number;
  as_of: string | null;
  last_event: string | null;
  latest_job: Batch | null;
  forecast_status: string;
  forecast_error: string | null;
  receipts: { id: number; inserted_rows: number; duplicate_rows: number }[];
};
type IntegrationStatus = {
  imports_enabled: boolean;
  max_import_bytes: number;
  hosting_notice: string | null;
};
const statuses: Record<string, string> = {
  queued: "В очереди",
  running: "Расчёт",
  complete: "Готово",
  failed: "Не выполнен",
  waiting: "Ожидание событий",
  accepted_unprocessed: "Данные сохранены, прогноз не рассчитан",
};
export function ImportPanel({ user }: { user: User }) {
  const [file, setFile] = useState<File | null>(null),
    [asOf, setAsOf] = useState("2026-06-15T12:00"),
    [batches, setBatches] = useState<Batch[]>([]),
    [selected, setSelected] = useState<Batch | null>(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const [integration, setIntegration] = useState<IntegrationStatus | null>(
    null,
  );
  const [mode, setMode] = useState("preview"),
    [stream, setStream] = useState<StreamStatus | null>(null),
    [receipt, setReceipt] = useState("");
  const [detail, setDetail] = useState<BatchForecastDetail | null>(null);
  const [batchKind, setBatchKind] = useState<Kind | "all">("all");
  const [visibleCount, setVisibleCount] = useState(12);
  const filteredForecasts =
    selected?.result?.forecasts.filter(
      (f) => batchKind === "all" || f.kind === batchKind,
    ) ?? [];
  useEffect(() => {
    setDetail(null);
    setVisibleCount(12);
  }, [selected?.id]);
  async function openForecast(objectId: number, kind: Kind) {
    try {
      setDetail(
        await api<BatchForecastDetail>(
          `/imports/${selected!.id}/forecast/${objectId}/${kind}`,
        ),
      );
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function update() {
    try {
      const [latestBatches, currentStream, currentIntegration] =
        await Promise.all([
          api<Batch[]>("/imports"),
          api<StreamStatus>("/stream"),
          api<IntegrationStatus>("/integrations"),
        ]);
      setBatches(latestBatches);
      setStream(currentStream);
      setIntegration(currentIntegration);
      if (!selected && currentStream.latest_job?.status === "complete") {
        setSelected(
          await api<Batch>(`/imports/${currentStream.latest_job.id}`),
        );
      }
      if (selected) setSelected(await api<Batch>(`/imports/${selected.id}`));
    } catch (e) {
      setError((e as Error).message);
    }
  }
  useEffect(() => {
    update();
    const timer = setInterval(update, 5000);
    return () => clearInterval(timer);
  }, [selected?.id]);
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError("");
    setReceipt("");
    try {
      const maxBytes = integration?.max_import_bytes ?? 20 * 1024 * 1024;
      if (file.size > maxBytes)
        throw Error(
          `Размер файла превышает ${Math.floor(maxBytes / 1024 / 1024)} МБ`,
        );
      const extension = file.name.split(".").pop()?.toLowerCase();
      const response = await fetch(
        `/api/${mode === "stream" ? "stream/events" : "imports"}?format=${extension}&as_of=${encodeURIComponent(asOf)}`,
        {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/octet-stream" },
          body: file,
        },
      );
      const result = await response.json();
      if (!response.ok)
        throw Error(
          typeof result.detail === "string"
            ? result.detail
            : "Проверьте формат файла",
        );
      if (mode === "stream") {
        setSelected(result.job);
        setReceipt(
          `Пакет №${result.receipt.id}: добавлено ${result.receipt.inserted_rows}, повторов ${result.receipt.duplicate_rows}. Данные сохранены.`,
        );
        if (result.forecast_error) setError(result.forecast_error);
        setStream(await api<StreamStatus>("/stream"));
      } else setSelected(result);
      setBatches(await api<Batch[]>("/imports"));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function retryStream() {
    if (!stream?.as_of) return;
    setBusy(true);
    setError("");
    try {
      const job = await api<Batch>(
        `/stream/forecast?as_of=${encodeURIComponent(stream.as_of)}`,
        { method: "POST" },
      );
      setSelected(job);
      setStream(await api<StreamStatus>("/stream"));
      setBatches(await api<Batch[]>("/imports"));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="panel import-panel">
      <div className="panel-heading">
        <div>
          <h2>Приём телеметрии и расчёт</h2>
          <span>
            CSV, XLSX, JSON или XML · до{" "}
            {Math.floor(
              (integration?.max_import_bytes ?? 20 * 1024 * 1024) / 1024 / 1024,
            )}{" "}
            МБ и 100 000 записей
          </span>
        </div>
        <FileUp size={21} />
      </div>
      <div className="import-content">
        <p>
          Добавьте показания к доступной предыстории и получите прогноз на 24
          часа. В режиме потока пакеты накапливаются в собственной БД, повторы
          исключаются. Каждый расчёт создаёт отдельный снимок. Оперативная
          система мониторинга пока не подключена.
        </p>
        {integration && !integration.imports_enabled && (
          <ErrorNotice
            message={
              integration.hosting_notice || "Импорт на этом стенде отключён."
            }
          />
        )}
        <details>
          <summary>Формат и требования к данным</summary>
          <p>
            Поля: channel_id, ts, value, alarm. Время без часового пояса
            трактуется как МСК. Все события должны предшествовать моменту
            прогноза. Нужна неделя непрерывной истории общего потока. Новые
            каналы сначала добавляют в справочник.
          </p>
          <pre>
            {"channel_id,ts,value,alarm\n5122,2026-06-15T11:59:00,Норма,false"}
          </pre>
          <a href="/api/integrations" target="_blank" rel="noreferrer">
            Статус источников и контракт API
          </a>
        </details>
        {error && <ErrorNotice message={error} />}
        {stream && (
          <div className="stream-summary">
            <div>
              <span>В накопленном потоке</span>
              <strong>
                {stream.events.toLocaleString("ru-RU")} <small>событий</small>
              </strong>
            </div>
            <div>
              <span>Последний момент · МСК</span>
              <strong>
                {stream.as_of
                  ? date(stream.as_of, true)
                  : "Пакеты не поступали"}
              </strong>
            </div>
            <div>
              <span>Прогноз потока</span>
              <strong>
                {statuses[stream.forecast_status] || stream.forecast_status}
              </strong>
              {stream.latest_job && (
                <>
                  {" "}
                  <button
                    className="text-link"
                    onClick={() =>
                      api<Batch>(`/imports/${stream.latest_job!.id}`)
                        .then(setSelected)
                        .catch((e) => setError(e.message))
                    }
                  >
                    Последний расчёт потока
                  </button>
                </>
              )}
            </div>
          </div>
        )}
        {receipt && <p role="status">{receipt}</p>}
        {stream && stream.forecast_status !== "waiting" && (
          <div role="status">
            {stream.forecast_error && (
              <ErrorNotice message={stream.forecast_error} />
            )}
            {["failed", "accepted_unprocessed"].includes(
              stream.forecast_status,
            ) &&
              can(user, "data.import") && (
                <button
                  className="secondary-button"
                  disabled={busy}
                  onClick={retryStream}
                >
                  Повторить расчёт сохранённого потока
                </button>
              )}
          </div>
        )}
        {can(user, "data.import") && (
          <form className="import-form" onSubmit={submit}>
            <label>
              Способ обработки
              <select value={mode} onChange={(e) => setMode(e.target.value)}>
                <option value="preview">Разовый расчёт</option>
                <option value="stream">Добавить в накопленный поток</option>
              </select>
            </label>
            <label>
              Пакет телеметрии
              <span className="file-picker">
                <FileUp size={16} />
                <span title={file?.name}>{file?.name || "Выбрать файл"}</span>
                <input
                  type="file"
                  aria-label="Пакет телеметрии"
                  accept=".csv,.xlsx,.json,.xml"
                  required
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                  disabled={
                    !can(user, "data.import") ||
                    integration?.imports_enabled === false
                  }
                />
              </span>
            </label>
            <label>
              Момент прогноза · МСК
              <input
                type="datetime-local"
                value={asOf}
                onChange={(e) => setAsOf(e.target.value)}
                step={300}
                required
              />
            </label>
            <button
              className="primary-button"
              disabled={
                busy ||
                !can(user, "data.import") ||
                integration?.imports_enabled === false
              }
            >
              <FileUp size={17} />
              {busy ? "Проверяем пакет…" : "Рассчитать прогноз"}
            </button>
          </form>
        )}
        {!can(user, "data.import") && (
          <p className="micro-note">
            Загрузка данных доступна аналитику. Здесь видны результаты расчётов
            по поступившим пакетам.
          </p>
        )}
        {batches.length > 0 && (
          <div className="import-jobs">
            <div className="section-label">
              <h3>Последние расчёты</h3>
              <button className="text-link" onClick={update}>
                <RefreshCw size={15} />
                Обновить
              </button>
            </div>
            {batches.slice(0, 5).map((b) => (
              <button
                className={selected?.id === b.id ? "selected" : ""}
                key={b.id}
                onClick={() =>
                  api<Batch>(`/imports/${b.id}`)
                    .then(setSelected)
                    .catch((e) => setError(e.message))
                }
              >
                <strong>{date(b.as_of, true)}</strong>
                <span>
                  {b.mode === "accumulated_stream" ? "Поток · " : ""}
                  {b.accepted_rows} записей
                </span>
                <span>{statuses[b.status] || b.status}</span>
              </button>
            ))}
          </div>
        )}
        {selected && (
          <div className="batch-result">
            {["running", "queued"].includes(selected.status) && (
              <Loading text="Обрабатываем историю и вычисляем признаки…" />
            )}
            {selected.error && <ErrorNotice message={selected.error} />}{" "}
            {selected.result && (
              <>
                <div className="section-label">
                  <h3>
                    {selected.mode === "accumulated_stream"
                      ? "Прогноз по накопленному потоку"
                      : "Прогноз по пакету"}
                  </h3>
                  <a
                    className="text-link"
                    href={`/api/imports/${selected.id}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <ArrowDownToLine size={15} />
                    Результат JSON
                  </a>
                </div>
                <p>
                  {
                    selected.result.forecasts.filter((f) => f.above_threshold)
                      .length
                  }{" "}
                  прогнозов выше порога · расчёт{" "}
                  {selected.result.elapsed_seconds} с.
                </p>
                <p className="batch-model-version">
                  Версия расчёта:{" "}
                  {selected.result.model_version &&
                  selected.result.model_version !== "legacy"
                    ? selected.result.model_version
                    : "исходная архивная модель"}
                  . Сохранённые прогнозы не меняются при обновлении модели.
                </p>
                <div className="batch-result-tools">
                  <select
                    aria-label="Тип прогноза в пакете"
                    value={batchKind}
                    onChange={(e) => {
                      setBatchKind(e.target.value as Kind | "all");
                      setVisibleCount(12);
                      setDetail(null);
                    }}
                  >
                    <option value="all">Все типы рисков</option>
                    {(Object.keys(kindNames) as Kind[]).map((k) => (
                      <option key={k} value={k}>
                        {kindNames[k]}
                      </option>
                    ))}
                  </select>
                  <span>
                    Показано {Math.min(visibleCount, filteredForecasts.length)}{" "}
                    из {filteredForecasts.length}
                  </span>
                </div>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Объект</th>
                        <th>Тип</th>
                        <th>Вероятность · 24 ч</th>
                        <th>Рекомендация</th>
                        <th>Проверка</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredForecasts.slice(0, visibleCount).map((f) => (
                        <tr key={`${f.object_id}-${f.kind_label}`}>
                          <td>{f.object_name}</td>
                          <td>{f.kind_label}</td>
                          <td>
                            <strong>{pct(f.probability)}</strong>
                            {f.expected_episodes !== undefined && (
                              <div className="micro-note">
                                Ожидается эпизодов:{" "}
                                {num(Number(f.expected_episodes.toFixed(1)))}
                                {f.notification_due && (
                                  <div>Новое предупреждение</div>
                                )}
                                {f.notification_status ===
                                  "between_hourly_checks" && (
                                  <div>
                                    Следующая проверка повторов — в начале часа
                                  </div>
                                )}
                              </div>
                            )}
                          </td>
                          <td>{f.recommendation}</td>
                          <td>
                            <button
                              className="text-link"
                              onClick={() => openForecast(f.object_id, f.kind)}
                            >
                              Проверить прогноз
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {visibleCount < filteredForecasts.length && (
                  <button
                    className="secondary-button batch-load-more"
                    onClick={() => setVisibleCount((count) => count + 12)}
                  >
                    Показать ещё
                  </button>
                )}
                {detail && (
                  <BatchDecision key={detail.id} detail={detail} user={user} />
                )}
                <p className="micro-note">
                  Расчёт по новым данным использует зафиксированные модели и
                  пороги. Вероятности относятся к сигналам датчиков. Импорт не
                  меняет отложенный тест. Решение сохраняется с привязкой к
                  снимку расчёта.
                </p>
              </>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

function BatchDecision({
  detail,
  user,
}: {
  detail: BatchForecastDetail;
  user: User;
}) {
  const sectionRef = useRef<HTMLElement>(null);
  useEffect(() => {
    sectionRef.current?.focus({ preventScroll: true });
    sectionRef.current?.scrollIntoView({ block: "start", behavior: "auto" });
  }, []);
  const [options, setOptions] = useState<{
    actions: Record<string, string>;
    reasons: { id: string; label: string }[];
  } | null>(null);
  const [action, setAction] = useState("monitor"),
    [reason, setReason] = useState("insufficient_evidence"),
    [comment, setComment] = useState("");
  const [saved, setSaved] = useState(false),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  useEffect(() => {
    api<typeof options>("/reasons")
      .then(setOptions)
      .catch((e) => setError(e.message));
  }, []);
  async function save(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    setSaved(false);
    try {
      await api("/decisions", {
        method: "POST",
        body: JSON.stringify({
          prediction_id: detail.id,
          batch_id: detail.batch_id,
          object_id: detail.object_id,
          kind: detail.kind,
          as_of: detail.as_of,
          action,
          reason,
          comment,
        }),
      });
      setSaved(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <section
      ref={sectionRef}
      tabIndex={-1}
      className="batch-detail"
      aria-label="Проверка прогноза потока"
    >
      <h3>
        {detail.object_name} · {detail.kind_label} · {pct(detail.probability)}
      </h3>
      <p>
        Снимок на {date(detail.as_of, true)} · горизонт 24 часа. Факторы
        показывают вклад в логарифм шансов, а не физическую причину инцидента.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Фактор</th>
              <th>Значение</th>
              <th>Вклад</th>
            </tr>
          </thead>
          <tbody>
            {detail.explanation.map((f) => (
              <tr key={f.feature}>
                <td>{f.label}</td>
                <td>
                  {f.value === null
                    ? "Нет данных"
                    : typeof f.value === "number"
                      ? num(Math.round(f.value * 100) / 100)
                      : f.value}
                </td>
                <td>
                  {f.contribution > 0 ? "+" : ""}
                  {f.contribution.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <details>
        <summary>
          Исходные сообщения перед прогнозом ({detail.source_events.length})
        </summary>
        {detail.source_events.length === 0 && (
          <p>
            В последние 24 часа перед прогнозом сообщений этого объекта нет.
            Давность сигналов учтена в модели.
          </p>
        )}
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Время · МСК</th>
                <th>Канал</th>
                <th>Значение</th>
              </tr>
            </thead>
            <tbody>
              {detail.source_events.map((e, i) => (
                <tr key={i}>
                  <td>{date(e.ts, true)}</td>
                  <td>{e.channel_id}</td>
                  <td>
                    {e.value}
                    {e.alarm ? " · тревога" : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <ul>
        {detail.recommendations.map((r) => (
          <li key={r}>{r}</li>
        ))}
      </ul>
      {error && <ErrorNotice message={error} />}
      {options && can(user, "decisions.write") && (
        <form className="import-form" onSubmit={save}>
          <label>
            Решение по новому прогнозу
            <select
              value={action}
              onChange={(e) => {
                setAction(e.target.value);
                setSaved(false);
              }}
            >
              {Object.entries(options.actions).map(([id, label]) => (
                <option key={id} value={id}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Основание решения
            <select
              value={reason}
              onChange={(e) => {
                setReason(e.target.value);
                setSaved(false);
              }}
            >
              {options.reasons.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Комментарий к проверке
            <input
              value={comment}
              maxLength={2000}
              required={reason === "other"}
              onChange={(e) => {
                setComment(e.target.value);
                setSaved(false);
              }}
            />
          </label>
          <button className="primary-button" disabled={busy}>
            {busy ? "Сохраняем…" : "Сохранить решение"}
          </button>
        </form>
      )}
      {saved && (
        <p role="status">
          Решение сохранено в журнале «Решения и ТО» и связано с этим снимком.
        </p>
      )}
    </section>
  );
}
