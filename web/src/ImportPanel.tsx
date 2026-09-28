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
  input_rows?: number;
  file_name?: string;
  accepted_rows: number;
  exact_duplicates: number;
  invalid_rows?: number;
  unknown_channel_rows?: number;
  unknown_channel_count?: number;
  late_rows?: number;
  other_year_rows?: number;
  alarm_inferred?: boolean;
  inferred_alarm_rows?: number;
  as_of_auto?: boolean;
  first_record?: string;
  last_record?: string;
  error?: string;
  mode?: string;
  result?: {
    model_version?: string;
    elapsed_seconds: number;
    history_rows: number;
    history?: {
      covered_hours: number;
      required_hours: number;
      complete: boolean;
    };
    forecasts: {
      object_id: number;
      object_name: string;
      kind_label: string;
      kind: Kind;
      probability: number;
      threshold: number;
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
  max_import_rows: number;
  hosting_notice: string | null;
};
// How far the probability is from the warning threshold: 1,0× is the threshold itself.
const ratio = (f: { probability: number; threshold: number }) =>
  `${(f.probability / f.threshold).toFixed(1).replace(".", ",")}×`;
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
    [asOf, setAsOf] = useState(""),
    [batches, setBatches] = useState<Batch[]>([]),
    [selected, setSelected] = useState<Batch | null>(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const [integration, setIntegration] = useState<IntegrationStatus | null>(
    null,
  );
  const [stream, setStream] = useState<StreamStatus | null>(null);
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
    try {
      const maxBytes = integration?.max_import_bytes ?? 300 * 1024 * 1024;
      if (file.size > maxBytes)
        throw Error(
          `Размер файла превышает ${Math.floor(maxBytes / 1024 / 1024)} МБ`,
        );
      const extension = file.name.split(".").pop()?.toLowerCase() || "";
      if (!["csv", "xlsx", "json", "xml", "zip"].includes(extension))
        throw Error("Выберите файл CSV, XLSX, JSON, XML или ZIP");
      const moment = asOf ? `&as_of=${encodeURIComponent(asOf)}` : "";
      const response = await fetch(
        `/api/imports?format=${extension}&name=${encodeURIComponent(file.name)}${moment}`,
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
      setSelected(result);
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
          <h2>Импорт исторических данных</h2>
          <span>
            CSV, XLSX, JSON, XML или ZIP · до{" "}
            {Math.floor(
              (integration?.max_import_bytes ?? 300 * 1024 * 1024) /
                1024 /
                1024,
            )}{" "}
            МБ и {num(integration?.max_import_rows ?? 2000000)} записей
          </span>
        </div>
        <FileUp size={21} />
      </div>
      <div className="import-content">
        {integration && !integration.imports_enabled && (
          <ErrorNotice
            message={
              integration.hosting_notice || "Импорт на этом стенде отключён."
            }
          />
        )}
        <details>
          <summary>Формат файла</summary>
          <p>
            Журнал СМВУ в формате выгрузки: ид_канала_данных, дата, время,
            значение_датчика, тревожное. Также принимаются «ИД канала данных,
            Текущее значение, Дата записи» и channel_id, ts, value, alarm.
            Разделитель — запятая, точка с запятой или табуляция; кодировка
            UTF-8 или Windows-1251. Время — МСК.
          </p>
          <p>
            Прогноз строится на момент сразу после последней записи. Для полного
            расчёта нужна неделя данных до этого момента, минимум — сутки.
          </p>
          <pre>
            {
              "ид_события,ид_канала_данных,дата,время,значение_датчика,тревожное\n3404741043,77836,2026-06-15,11:27:04,Не замкнут,t"
            }
          </pre>
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
              Пакет телеметрии
              <span className="file-picker">
                <FileUp size={16} />
                <span title={file?.name}>{file?.name || "Выбрать файл"}</span>
                <input
                  type="file"
                  aria-label="Пакет телеметрии"
                  accept=".csv,.xlsx,.json,.xml,.zip"
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
              />
              <small className="field-hint">
                {asOf ? (
                  <button
                    type="button"
                    className="text-link"
                    onClick={() => setAsOf("")}
                  >
                    Взять по последней записи
                  </button>
                ) : (
                  "Пусто — сразу после последней записи"
                )}
              </small>
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
              {busy ? "Загружаем файл…" : "Загрузить и рассчитать"}
            </button>
          </form>
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
                  {b.mode === "accumulated_stream"
                    ? "Поток СМВУ"
                    : b.file_name || "Файл"}{" "}
                  · {num(b.accepted_rows)} записей
                </span>
                <span>{statuses[b.status] || b.status}</span>
              </button>
            ))}
          </div>
        )}
        {selected && (
          <div className="batch-result">
            {["running", "queued"].includes(selected.status) && (
              <Loading text="Считаем признаки и прогноз, обычно до минуты…" />
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
                <dl className="import-facts">
                  <div>
                    <dt>Прогноз на</dt>
                    <dd>
                      {date(selected.as_of, true)}
                      {selected.as_of_auto && (
                        <small>по последней записи</small>
                      )}
                    </dd>
                  </div>
                  <div>
                    <dt>Принято записей</dt>
                    <dd>
                      {num(selected.accepted_rows)}
                      {selected.input_rows !== undefined && (
                        <small>из {num(selected.input_rows)}</small>
                      )}
                    </dd>
                  </div>
                  <div>
                    <dt>Выше порога</dt>
                    <dd>
                      {
                        selected.result.forecasts.filter(
                          (f) => f.above_threshold,
                        ).length
                      }
                      <small>из {selected.result.forecasts.length}</small>
                    </dd>
                  </div>
                  {selected.result.history && (
                    <div>
                      <dt>Предыстория</dt>
                      <dd>
                        {selected.result.history.covered_hours} ч
                        <small>
                          из {selected.result.history.required_hours}
                        </small>
                      </dd>
                    </div>
                  )}
                </dl>
                {selected.result.history &&
                  !selected.result.history.complete && (
                    <div className="import-warning">
                      Данных меньше недели: недельные показатели неполные,
                      вероятности могут быть занижены. Для точного прогноза
                      загрузите журнал за 7 суток до момента прогноза.
                      <strong>Наибольший относительный риск</strong>
                      <ol>
                        {[...selected.result.forecasts]
                          .filter((f) => f.threshold <= 1)
                          .sort(
                            (a, b) =>
                              b.probability / b.threshold -
                              a.probability / a.threshold,
                          )
                          .slice(0, 5)
                          .map((f) => (
                            <li key={`${f.object_id}-${f.kind}`}>
                              {f.object_name} · {f.kind_label} ·{" "}
                              {pct(f.probability)} ({ratio(f)} порога)
                            </li>
                          ))}
                      </ol>
                    </div>
                  )}
                {(() => {
                  const notes = [
                    selected.unknown_channel_rows
                      ? `${num(selected.unknown_channel_rows)} записей каналов вне справочника (${selected.unknown_channel_count} кан.)`
                      : "",
                    selected.invalid_rows
                      ? `${num(selected.invalid_rows)} нераспознанных строк`
                      : "",
                    selected.late_rows
                      ? `${num(selected.late_rows)} записей позже момента прогноза`
                      : "",
                    selected.other_year_rows
                      ? `${num(selected.other_year_rows)} записей другого года`
                      : "",
                    selected.exact_duplicates
                      ? `${num(selected.exact_duplicates)} точных повторов`
                      : "",
                  ].filter(Boolean);
                  return (
                    <>
                      {notes.length > 0 && (
                        <p className="micro-note">
                          Пропущено: {notes.join("; ")}.
                        </p>
                      )}
                      {selected.alarm_inferred && (
                        <p className="micro-note">
                          В файле нет флага тревоги: он восстановлен по
                          значениям датчиков (
                          {num(selected.inferred_alarm_rows ?? 0)} тревожных
                          записей).
                        </p>
                      )}
                    </>
                  );
                })()}
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
                        <th>От порога</th>
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
                          <td>{f.threshold > 1 ? "выкл." : ratio(f)}</td>
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
      <p>Прогноз на {date(detail.as_of, true)} · горизонт 24 часа</p>
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
