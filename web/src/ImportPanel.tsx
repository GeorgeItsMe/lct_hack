import { useEffect, useState } from "react";
import { ArrowDownToLine, FileUp, RefreshCw } from "lucide-react";
import { api, date, pct } from "./api";
import { ErrorNotice, Loading } from "./components";
import type { User } from "./types";
type Batch = {
  id: string;
  status: string;
  as_of: string;
  accepted_rows: number;
  exact_duplicates: number;
  error?: string;
  result?: {
    elapsed_seconds: number;
    history_rows: number;
    forecasts: {
      object_id: number;
      object_name: string;
      kind_label: string;
      probability: number;
      above_threshold: boolean;
      recommendation: string;
    }[];
  };
};
const statuses: Record<string, string> = {
  queued: "В очереди",
  running: "Расчёт",
  complete: "Готово",
  failed: "Не выполнен",
};
export function ImportPanel({ user }: { user: User }) {
  const [file, setFile] = useState<File | null>(null),
    [asOf, setAsOf] = useState("2026-06-15T12:00"),
    [batches, setBatches] = useState<Batch[]>([]),
    [selected, setSelected] = useState<Batch | null>(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  async function update() {
    try {
      setBatches(await api<Batch[]>("/imports"));
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
      if (file.size > 20 * 1024 * 1024)
        throw Error("Размер файла превышает 20 МБ");
      const extension = file.name.split(".").pop()?.toLowerCase();
      const response = await fetch(
        `/api/imports?format=${extension}&as_of=${encodeURIComponent(asOf)}`,
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
  return (
    <section className="panel import-panel">
      <div className="panel-heading">
        <div>
          <h2>Расчёт по новому пакету</h2>
          <span>CSV, XLSX, JSON или XML · до 20 МБ и 100 000 записей</span>
        </div>
        <FileUp size={21} />
      </div>
      <div className="import-content">
        <p>
          Добавьте показания к доступной предыстории и получите прогноз на 24
          часа. Каждый расчёт создаёт отдельный снимок. Оперативная система
          мониторинга пока не подключена.
        </p>
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
        <form className="import-form" onSubmit={submit}>
          <label>
            Пакет телеметрии
            <input
              type="file"
              accept=".csv,.xlsx,.json,.xml"
              required
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              disabled={user.role === "analyst"}
            />
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
            disabled={busy || user.role === "analyst"}
          >
            <FileUp size={17} />
            {busy ? "Проверяем пакет…" : "Рассчитать прогноз"}
          </button>
        </form>
        {user.role === "analyst" && (
          <p className="micro-note">
            Импорт доступен диспетчеру и администратору.
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
                <span>{b.accepted_rows} записей</span>
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
                  <h3>Прогноз по пакету</h3>
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
                  {selected.result.elapsed_seconds} с. Показаны первые 12.
                  Полный результат доступен в JSON.
                </p>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Объект</th>
                        <th>Тип</th>
                        <th>Вероятность · 24 ч</th>
                        <th>Рекомендация</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selected.result.forecasts.slice(0, 12).map((f) => (
                        <tr key={`${f.object_id}-${f.kind_label}`}>
                          <td>{f.object_name}</td>
                          <td>{f.kind_label}</td>
                          <td>
                            <strong>{pct(f.probability)}</strong>
                          </td>
                          <td>{f.recommendation}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="micro-note">
                  Расчёт по новым данным использует зафиксированные модели и
                  пороги. Вероятности относятся к сигналам датчиков. Импорт не
                  меняет отложенный тест и архивный журнал решений.
                </p>
              </>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
