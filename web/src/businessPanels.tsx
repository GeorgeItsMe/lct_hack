import { Fragment, useEffect, useState } from "react";
import { ClipboardCheck, Fan, FlaskConical, X } from "lucide-react";
import { api, date, kindNames, num, pct } from "./api";
import { Empty, ErrorNotice, Loading } from "./components";
import type {
  Kind,
  PolygonQueueRow,
  PolygonQueue,
  PolygonSituation,
  PolygonTag,
  Shift,
  ShiftCounts,
  UnitState,
  Units,
} from "./types";

function useData<T>(path: string | null) {
  const [data, setData] = useState<T | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    if (!path) return;
    let live = true;
    setData(null);
    setError("");
    api<T>(path)
      .then((r) => live && setData(r))
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [path]);
  return { data, error };
}

export function RegistryTags({ tags }: { tags: PolygonTag[] }) {
  if (!tags.length) return null;
  return (
    <span className="registry-tags">
      {tags.map((t) => (
        <span key={t.label} className={`registry-tag ${t.kind}`}>
          {t.label}
        </span>
      ))}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Dispatcher: the polygon world in the queue                          */
/* ------------------------------------------------------------------ */

export function PolygonQueuePanel({ asOf }: { asOf: string }) {
  const [wholeMonth, setWholeMonth] = useState(false),
    [selected, setSelected] = useState<PolygonQueueRow | null>(null);
  const path = asOf
    ? wholeMonth
      ? "/polygon/queue?as_of=2026-07-01T00:00:00&hours=744"
      : `/polygon/queue?as_of=${encodeURIComponent(asOf)}&hours=24`
    : null;
  const { data, error } = useData<PolygonQueue>(path);
  if (error) return null; // The polygon is optional: no results, no panel.
  return (
    <section className="panel polygon-panel">
      <div className="panel-heading">
        <div>
          <h2>Полигон, мир {data?.world ?? "…"}</h2>
          <span>
            Тревоги и прогнозы, для которых известно, что было на самом деле.
            {data && ` ${data.note}`}
          </span>
        </div>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={wholeMonth}
            onChange={(e) => setWholeMonth(e.target.checked)}
          />
          Весь июнь
        </label>
      </div>
      {!data ? (
        <Loading text="Загружаем полигон…" />
      ) : !data.rows.length ? (
        <Empty title="За эти сутки на полигоне тревог нет">
          Отметьте «Весь июнь», чтобы увидеть все {data.feed_total} событий
          мира.
        </Empty>
      ) : (
        <div className="table-scroll capped">
          <table className="polygon-queue">
            <thead>
              <tr>
                <th>Время</th>
                <th>Объект</th>
                <th>Решение системы</th>
                <th>Реестры</th>
                <th>Правда полигона</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((r) => (
                <tr
                  key={`${r.time}-${r.object_id}-${r.scenario}`}
                  className="clickable"
                  onClick={() => setSelected(r)}
                >
                  <td>{date(r.time, true)}</td>
                  <td>
                    <strong>{r.object_name}</strong>
                    <small className="muted"> #{r.object_id}</small>
                  </td>
                  <td>
                    {r.scenario_label}
                    <small className="muted">{r.system}</small>
                  </td>
                  <td>
                    <RegistryTags tags={r.tags} />
                  </td>
                  <td>
                    <span
                      className={`status ${r.outcome === "верно" || r.outcome === "поймано" ? "normal" : "attention"}`}
                    >
                      {r.outcome}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {selected && (
        <PolygonCard row={selected} onClose={() => setSelected(null)} />
      )}
    </section>
  );
}

function PolygonCard({
  row,
  onClose,
}: {
  row: PolygonQueueRow;
  onClose: () => void;
}) {
  const { data, error } = useData<PolygonSituation>(
    `/polygon/situation?object_id=${row.object_id}&at=${encodeURIComponent(row.time)}`,
  );
  return (
    <div className="drawer-layer">
      <div className="drawer-backdrop" onClick={onClose} />
      <aside
        className="forecast-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="polygon-title"
      >
        <header className="drawer-header">
          <span className="eyebrow">Полигон, мир {data?.world ?? ""}</span>
          <button
            className="icon-button"
            onClick={onClose}
            aria-label="Закрыть карточку"
          >
            <X size={22} />
          </button>
        </header>
        <div className="drawer-intro">
          <h2 id="polygon-title">{row.object_name}</h2>
          <p>
            Объект #{row.object_id} <span>·</span> {date(row.time, true)} МСК
          </p>
          <p>
            <strong>{row.scenario_label}.</strong> {row.system}
          </p>
          <RegistryTags tags={row.tags} />
        </div>
        <div className="drawer-content">
          <div className="section-label">
            <h3>Проверка ситуации</h3>
            <ClipboardCheck size={15} />
          </div>
          {error && <ErrorNotice message={error} />}
          {!data && !error && <Loading text="Смотрим реестры…" />}
          {data && (
            <>
              <h4>Наряды-допуски на объекте (±12 ч)</h4>
              {data.permits.length ? (
                <ul className="source-status">
                  {data.permits.map((p) => (
                    <li key={p.permit_id}>
                      <span>
                        {p.permit_id} · {p.work_type} · {p.brigade}
                        <small className="muted">
                          {" "}
                          {date(p.start, true)} — {date(p.end, true)}
                        </small>
                      </span>
                      <b>{p.active ? "действует" : "вне момента"}</b>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="micro-note">
                  Нарядов на объекте нет: доступ и огневые работы не оформлены.
                </p>
              )}
              <h4>Огневые работы</h4>
              <p className="micro-note">
                {data.hot_works.length
                  ? data.hot_works
                      .map(
                        (p) =>
                          `${p.permit_id}: ${date(p.start, true)} — ${date(p.end, true)}, ${p.brigade}`,
                      )
                      .join("; ")
                  : "По графику огневых работ на объекте нет."}
              </p>
              <h4>Ремонты (±48 ч)</h4>
              <p className="micro-note">
                {data.repairs.length
                  ? data.repairs
                      .map(
                        (r) =>
                          `${r.work} ${date(r.start, true)} — ${date(r.end, true)}`,
                      )
                      .join("; ")
                  : "Аварийных ремонтов нет."}
              </p>
              <p className="micro-note">
                Агрегатов в паспортах объекта: насосов {data.units.pumps},
                вентиляторов {data.units.fans}.
              </p>
            </>
          )}
          <div className="section-label">
            <h3>Правда полигона</h3>
            <FlaskConical size={15} />
          </div>
          <p>
            {row.truth}. Итог для системы: <strong>{row.outcome}</strong>
            {row.lead_hours !== null &&
              `, заранее за ${num(Math.round(row.lead_hours * 10) / 10)} ч`}
            .
          </p>
          <p className="micro-note">
            Реестры и инциденты полигона синтетические, фоном служит реальный
            журнал СМВУ.
          </p>
        </div>
      </aside>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Technical staff: pumps and fans                                     */
/* ------------------------------------------------------------------ */

const UNIT_STATES: [UnitState, string][] = [
  ["running", "Включены"],
  ["stopped", "Выключены"],
  ["normal", "Норма"],
  ["problem", "Неисправны, обесточены, затоплены"],
  ["unknown", "Неопределено"],
  ["silent", "Нет сообщений за неделю"],
];

export function UnitsPanel({ asOf }: { asOf: string }) {
  const { data, error } = useData<Units>(
    asOf ? `/equipment/units?as_of=${encodeURIComponent(asOf)}` : null,
  );
  const [open, setOpen] = useState<number | null>(null);
  if (error) return <ErrorNotice message={error} />;
  if (!data)
    return <Loading text="Собираем состояние насосов и вентиляторов…" />;
  const real = data.real,
    polygon = data.polygon;
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h2>Насосы и вентиляторы</h2>
          <span>{real.note}</span>
        </div>
        <Fan size={20} />
      </div>
      <div className="four-column">
        <div className="data-stat">
          <span>Агрегатов</span>
          <strong>{num(real.pumps + real.fans)}</strong>
          <small>
            насосов {real.pumps}, вентиляторов {real.fans}
          </small>
        </div>
        <div className="data-stat">
          <span>Включены сейчас</span>
          <strong>{num(real.states.running)}</strong>
          <small>выключены {num(real.states.stopped)}</small>
        </div>
        <div className="data-stat">
          <span>С неисправностью</span>
          <strong>{num(real.states.problem)}</strong>
          <small>неисправен, обесточен, затоплен, отключён</small>
        </div>
        <div className="data-stat">
          <span>Без сообщений за неделю</span>
          <strong>{num(real.states.silent)}</strong>
          <small>проверить связь с агрегатом</small>
        </div>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Объект</th>
              {UNIT_STATES.map(([k, label]) => (
                <th key={k}>{label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {real.objects.map((o) => (
              <Fragment key={o.object_id}>
                <tr
                  className={o.problems.length ? "clickable" : undefined}
                  onClick={() =>
                    o.problems.length &&
                    setOpen(open === o.object_id ? null : o.object_id)
                  }
                >
                  <td>
                    <strong>{o.object_name}</strong>
                    <small className="muted">
                      {" "}
                      насосов {o.pumps}, вентиляторов {o.fans}
                    </small>
                  </td>
                  {UNIT_STATES.map(([k]) => (
                    <td key={k}>
                      {k === "problem" && o.states[k] > 0 ? (
                        <span className="status attention">{o.states[k]}</span>
                      ) : (
                        o.states[k]
                      )}
                    </td>
                  ))}
                </tr>
                {open === o.object_id &&
                  o.problems.map((p) => (
                    <tr key={`p-${p.channel_id}`} className="sub-row">
                      <td colSpan={UNIT_STATES.length + 1}>
                        {p.unit} {p.sensor_name}{" "}
                        <span className="muted">#{p.channel_id}</span> —{" "}
                        <b>{p.value}</b> с {date(p.ts, true)}
                      </td>
                    </tr>
                  ))}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
      {polygon && (
        <>
          <div className="section-label">
            <h3>Прогноз отказа по износу на полигоне, мир {polygon.world}</h3>
            <FlaskConical size={15} />
          </div>
          <p className="micro-note">
            На полигоне у агрегатов есть паспорта (ввод, последнее ТО, ресурс) и
            сценарии износа с известной правдой. Предупреждений за июнь:{" "}
            {polygon.warnings.length}, действует сейчас: {polygon.active}.
          </p>
          <div className="table-scroll capped">
            <table>
              <thead>
                <tr>
                  <th>Время</th>
                  <th>Объект</th>
                  <th>Агрегат</th>
                  <th>Паспорт</th>
                  <th>Правда полигона</th>
                </tr>
              </thead>
              <tbody>
                {polygon.warnings.map((w) => (
                  <tr key={`${w.time}-${w.object_id}-${w.unit}`}>
                    <td>
                      {date(w.time, true)}
                      {w.active && (
                        <span className="source-chip">действует</span>
                      )}
                    </td>
                    <td>{w.object_name}</td>
                    <td>
                      {w.unit}
                      {w.units_warned > 1 && ` ×${w.units_warned}`}{" "}
                      <span className="muted">
                        (всего на объекте {w.units_on_object})
                      </span>
                    </td>
                    <td>
                      <small>
                        старейший с {w.oldest_installed ?? "—"}, ТО{" "}
                        {w.last_maintenance ?? "—"}
                      </small>
                    </td>
                    <td>
                      <span
                        className={`status ${w.outcome === "поймано" ? "normal" : "attention"}`}
                      >
                        {w.outcome}
                      </span>
                      {w.lead_hours !== null && (
                        <small className="muted">
                          {" "}
                          за {num(Math.round(w.lead_hours * 10) / 10)} ч
                        </small>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Unit head: results of a dispatcher shift                            */
/* ------------------------------------------------------------------ */

const SHIFT_KINDS: Kind[] = ["access", "fire", "fault"];

function ShiftRow({ label, c }: { label: string; c: ShiftCounts }) {
  const round = (v: number) => num(Math.round(v * 10) / 10);
  return (
    <tr>
      <td>{label}</td>
      <td>{round(c.warnings)}</td>
      <td>{round(c.correct)}</td>
      <td>{round(c.false)}</td>
      <td>{round(c.pending)}</td>
      <td>
        {round(c.caught)} из {round(c.incident_days)}
      </td>
    </tr>
  );
}

export function ShiftPanel({ shift }: { shift: Shift }) {
  const resolved = shift.resolved.totals,
    average = shift.average.per_shift;
  const precision =
    resolved.correct + resolved.false
      ? resolved.correct / (resolved.correct + resolved.false)
      : null;
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h2>За смену</h2>
          <span>
            Смены по 12 часов с 08:00 и 20:00. {shift.rule} Итог предупреждения
            известен через сутки, поэтому результаты — за последнюю смену с
            известным итогом и в среднем за {shift.average.shifts} смен.
          </span>
        </div>
      </div>
      <div className="four-column">
        <div className="data-stat">
          <span>Текущая смена {shift.current.label}</span>
          <strong>{num(shift.current.totals.warnings)}</strong>
          <small>
            предупреждений выдано, ждут итога {shift.current.totals.pending}
          </small>
        </div>
        <div className="data-stat">
          <span>Смена {shift.resolved.label}</span>
          <strong>
            {num(resolved.correct)} из {num(resolved.warnings)}
          </strong>
          <small>
            верных предупреждений
            {precision !== null && ` (${pct(precision, 0)})`}, ложных{" "}
            {resolved.false}
          </small>
        </div>
        <div className="data-stat">
          <span>Поймано заранее</span>
          <strong>
            {num(resolved.caught)} из {num(resolved.incident_days)}
          </strong>
          <small>неспокойных суток, начавшихся в эту смену</small>
        </div>
        <div className="data-stat">
          <span>В среднем за смену</span>
          <strong>{num(Math.round(average.warnings * 10) / 10)}</strong>
          <small>
            предупреждений, из них ложных{" "}
            {num(Math.round(average.false * 10) / 10)}
          </small>
        </div>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Смена {shift.resolved.label}</th>
              <th>Предупреждений</th>
              <th>Верных</th>
              <th>Ложных</th>
              <th>Ждут итога</th>
              <th>Неспокойные сутки: поймано заранее</th>
            </tr>
          </thead>
          <tbody>
            {SHIFT_KINDS.filter((k) => shift.resolved.by_kind[k]).map((k) => (
              <ShiftRow
                key={k}
                label={kindNames[k]}
                c={shift.resolved.by_kind[k]}
              />
            ))}
            <ShiftRow label="В среднем за смену, все типы" c={average} />
          </tbody>
        </table>
      </div>
    </section>
  );
}
