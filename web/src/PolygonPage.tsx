import { useEffect, useMemo, useState } from "react";
import { Download, FileText } from "lucide-react";
import { api, date, num, pct } from "./api";
import { Empty, ErrorNotice, Loading } from "./components";
import type {
  PolygonFeedRow,
  PolygonForecast,
  PolygonReport,
  PolygonResults,
} from "./types";

function useData<T>(path: string) {
  const [data, setData] = useState<T | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    api<T>(path)
      .then((r) => live && setData(r))
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [path]);
  return { data, error };
}

const day = (n: number) =>
  n >= 10 ? num(Math.round(n)) : n.toFixed(n >= 1 ? 1 : 2).replace(".", ",");
const hours = (n: number | null | undefined) =>
  n == null ? "—" : `${n.toFixed(1).replace(".", ",")} ч`;

const RATE_LABELS: Record<string, string> = {
  intrusions_per_week: "Проникновений в неделю на весь парк",
  intrusion_probe_share: "Доля проникновений с пробой люка накануне",
  intrusion_off_hours_share: "Доля проникновений ночью или в выходные",
  real_fires_per_week: "Возгораний в неделю",
  fire_during_hot_work_share: "Доля возгораний во время огневых работ",
  smoulder_share: "Доля тлеющих возгораний",
  hot_works_per_week: "Огневых работ в неделю",
  heavy_hot_work_share: "Доля долгой резки (дым 10–30 минут)",
  hot_work_unregistered_share: "Доля огневых работ без наряда",
  permit_missing_share: "Доля визитов бригад без оформленного наряда",
  pump_failures_per_half_year: "Отказов насосов за полгода",
  fan_failures_per_half_year: "Отказов вентиляторов за полгода",
  cascades_per_week: "Каскадов сбоев в неделю",
};

function rate(key: string, value: number) {
  return key.endsWith("share") ? pct(value, 0) : num(value);
}

function forecastText(f: PolygonForecast, noun: string) {
  return `${f.caught_incident_days} из ${f.incident_days} ${noun} заранее, в среднем за ${hours(f.median_lead_hours)}; точность ${pct(f.precision, 0)}`;
}

type Line = {
  id: string;
  scenario: string;
  how: string;
  result: (r: PolygonResults) => string;
};

const LINES: Line[] = [
  {
    id: "fire_triage",
    scenario: "Пожарная тревога: возгорание или нет",
    how: "Решение через 10 минут после первого дыма",
    result: (r) =>
      `${r.fire_triage.caught} из ${r.fire_triage.incidents} возгораний поднято; ложных подъёмов ${r.fire_triage.false_escalations}; тревог от сварки поднято ${r.fire_triage.hot_work_escalated ?? 0}`,
  },
  {
    id: "fire_forecast",
    scenario: "Прогноз возгорания на сутки",
    how: "По нагреву на датчиках температуры",
    result: (r) => {
      const f = r.fire_forecast;
      const heated = f.with_visible_heating;
      return `${forecastText(f, "возгораний")}${heated ? `; где рядом есть датчик температуры — ${heated.caught_incident_days} из ${heated.incident_days}` : ""}`;
    },
  },
  {
    id: "access_triage",
    scenario: "Охранная тревога: проникновение или работа",
    how: "Решение через 5 минут: наряд, режим охраны, время, движение",
    result: (r) =>
      `${r.access_triage.caught} из ${r.access_triage.incidents} проникновений; точность ${pct(r.access_triage.precision, 0)}; выездов в сутки ${day(r.access_triage.escalated_per_day)}`,
  },
  {
    id: "pump_forecast",
    scenario: "Отказ насоса",
    how: "Частота и длительность включений относительно нормы агрегата, сбои, возраст",
    result: (r) => forecastText(r.pump_forecast, "отказов"),
  },
  {
    id: "fan_forecast",
    scenario: "Отказ вентилятора",
    how: "Короткие пуски и сбои относительно нормы агрегата, возраст, ТО",
    result: (r) => forecastText(r.fan_forecast, "отказов"),
  },
  {
    id: "gas_forecast",
    scenario: "Газ при отказе вентиляции",
    how: "Рост показаний газа и состояние вентиляторов объекта",
    result: (r) => {
      const g = r.gas_forecast.fan_caused;
      return g
        ? `${g.caught_incident_days} из ${g.incident_days} газовых тревог заранее, в среднем за ${hours(g.median_lead_hours)}; точность ${pct(r.gas_forecast.precision, 0)}`
        : forecastText(r.gas_forecast, "газовых тревог");
    },
  },
  {
    id: "cascades",
    scenario: "Каскад сбоев",
    how: "Пять и более каналов объекта в неисправности за 10 минут",
    result: (r) =>
      `${r.cascades.caught} из ${r.cascades.incidents} распознаны как сбой оборудования`,
  },
];

const OUTCOME_CLASS: Record<string, string> = {
  поймано: "reviewed",
  верно: "reviewed",
  пропущено: "attention",
  "ложный подъём": "normal",
  "ложное предупреждение": "normal",
};

function Feed({ rows }: { rows: PolygonFeedRow[] }) {
  const [scenario, setScenario] = useState("all"),
    [outcome, setOutcome] = useState("all");
  const scenarios = useMemo(
    () => [...new Set(rows.map((r) => r.scenario_label))].sort(),
    [rows],
  );
  const outcomes = useMemo(
    () => [...new Set(rows.map((r) => r.outcome))].sort(),
    [rows],
  );
  const shown = rows.filter(
    (r) =>
      (scenario === "all" || r.scenario_label === scenario) &&
      (outcome === "all" || r.outcome === outcome),
  );
  return (
    <>
      <div className="row-actions polygon-filters">
        <select
          aria-label="Сценарий"
          value={scenario}
          onChange={(e) => setScenario(e.target.value)}
        >
          <option value="all">Все сценарии</option>
          {scenarios.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
        <select
          aria-label="Итог"
          value={outcome}
          onChange={(e) => setOutcome(e.target.value)}
        >
          <option value="all">Все итоги</option>
          {outcomes.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
        <span className="micro-note">Строк: {num(shown.length)}</span>
      </div>
      {shown.length ? (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Время</th>
                <th>Объект</th>
                <th>Что произошло</th>
                <th>Что сказала система</th>
                <th>Итог</th>
                <th>Заранее</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((r, i) => (
                <tr key={`${r.time}-${r.object_id}-${i}`}>
                  <td>{date(r.time, true)}</td>
                  <td>{r.object_name}</td>
                  <td>
                    <b>{r.scenario_label}</b>
                    <br />
                    <small>{r.truth}</small>
                  </td>
                  <td>{r.system}</td>
                  <td>
                    <span
                      className={`status ${OUTCOME_CLASS[r.outcome] || "normal"}`}
                    >
                      {r.outcome}
                    </span>
                  </td>
                  <td>{hours(r.lead_hours)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty title="Нет строк для этих фильтров" />
      )}
    </>
  );
}

export function PolygonPage() {
  const report = useData<PolygonReport>("/polygon");
  const feed = useData<PolygonFeedRow[]>("/polygon/feed");
  const [journalDay, setJournalDay] = useState("2026-06-15");
  if (report.error) return <ErrorNotice message={report.error} />;
  if (!report.data) return <Loading text="Загружаем результаты полигона…" />;
  const r = report.data;
  const base = r.scenarios;
  const effect = r.effect;
  const forecasts = Object.values(effect.forecasts_per_day).reduce(
    (a, b) => a + b,
    0,
  );
  return (
    <div className="page-stack">
      <section className="panel">
        <p className="polygon-lead">
          Реальный журнал СМВУ 2026 года остаётся фоном. Поверх него записаны
          инциденты с известной правдой и реестры, которых нет в выгрузке:
          наряды-допуски, огневые работы, паспорта насосов и вентиляторов.
          Модели учились на январе–апреле {r.design.train_worlds.length} миров,
          пороги выбраны на мае, проверка — июнь {r.design.test_worlds.length}{" "}
          других миров. Цифры ниже показывают, как система отрабатывает
          сценарии; точность на реальных авариях они не измеряют.
        </p>
      </section>
      <div className="four-column">
        <div className="data-stat">
          <span>Пожарных тревог в сутки</span>
          <strong>
            {day(effect.fire.alarms_per_day)} →{" "}
            {day(effect.fire.raised_per_day)}
          </strong>
          <small>
            поднимается как возгорание; ещё {day(effect.fire.hot_work_per_day)}{" "}
            — огневые работы по наряду
          </small>
        </div>
        <div className="data-stat">
          <span>Охранных тревог в сутки</span>
          <strong>
            {day(effect.access.alarms_per_day)} →{" "}
            {day(effect.access.trips_per_day)}
          </strong>
          <small>
            выезд; {day(effect.access.permitted_per_day)} закрывает наряд,{" "}
            {day(effect.access.calls_per_day)} — звонок
          </small>
        </div>
        <div className="data-stat">
          <span>Проникновений распознано</span>
          <strong>
            {base.access_triage.caught} из {base.access_triage.incidents}
          </strong>
          <small>в июне {r.design.test_worlds.length} миров</small>
        </div>
        <div className="data-stat">
          <span>Прогнозов в сутки</span>
          <strong>{day(forecasts)}</strong>
          <small>пожар, насосы, вентиляторы, газ — на весь парк</small>
        </div>
      </div>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Сценарии и результат</h2>
            <span>
              Июнь, {r.design.test_worlds.length} миров; «если хитрее» —
              инциденты труднее отличить от работы, модели и пороги те же
            </span>
          </div>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Сценарий</th>
                <th>Как распознаётся</th>
                <th>Результат</th>
                <th>Если хитрее</th>
              </tr>
            </thead>
            <tbody>
              {LINES.map((line) => (
                <tr key={line.id}>
                  <td>
                    <b>{line.scenario}</b>
                  </td>
                  <td>{line.how}</td>
                  <td>{line.result(base)}</td>
                  <td>{line.result(r.hard)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Лента июня, мир {r.design.feed_world}</h2>
            <span>
              Что произошло, что сказала система и за сколько часов, включая
              пропуски и ложные срабатывания
            </span>
          </div>
        </div>
        {feed.error ? (
          <ErrorNotice message={feed.error} />
        ) : feed.data ? (
          <Feed rows={feed.data} />
        ) : (
          <Loading text="Загружаем ленту…" />
        )}
      </section>
      <div className="two-column">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Допущения генератора</h2>
              <span>Обычные миры / хитрые миры</span>
            </div>
          </div>
          <dl className="config-list">
            {Object.entries(r.design.rates).map(([key, value]) => (
              <div key={key}>
                <dt>{RATE_LABELS[key] || key}</dt>
                <dd>
                  {rate(key, value)}
                  {key in r.design.hard_rates &&
                    ` / ${rate(key, r.design.hard_rates[key])}`}
                </dd>
              </div>
            ))}
          </dl>
        </section>
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Синтетический датасет</h2>
              <span>Мир {r.design.feed_world}</span>
            </div>
          </div>
          <div className="page-stack">
            <a className="secondary-button" href="/api/polygon/dataset.zip">
              <Download size={16} />
              Реестры, правда и вставленные записи (ZIP)
            </a>
            <div className="row-actions">
              <input
                type="date"
                aria-label="День журнала"
                min="2026-06-01"
                max="2026-06-30"
                value={journalDay}
                onChange={(e) => setJournalDay(e.target.value)}
              />
              <a
                className="secondary-button"
                href={`/api/polygon/journal.csv?day=${journalDay}`}
              >
                <FileText size={16} />
                Журнал полигона за сутки (CSV)
              </a>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
