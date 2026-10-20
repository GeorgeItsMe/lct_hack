import { useEffect, useState } from "react";
import {
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  Clock3,
  Database,
  FileCheck2,
  Filter,
  Wrench,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
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
  WorkBadge,
} from "./components";
import type {
  DailyRuleMatch,
  DecisionRow,
  Evaluation,
  Forecast,
  Kind,
  Quality,
} from "./types";

function useResource<T>(path: string, refresh = 0) {
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
  }, [path, refresh]);
  return { data, error };
}

const MODEL_NAMES: Record<string, string> = {
  archive_classifier: "Классификатор «эпизод за 24 часа»",
  count_head: "Модель числа эпизодов",
};
const modelName = (name: string) => {
  const blend = /^blend_count_(\d\.\d+)$/.exec(name);
  if (!blend) return MODEL_NAMES[name] || name;
  const w = Number(blend[1]);
  return `Смесь: ${num(w)} числа эпизодов + ${num(1 - w)} классификатора`;
};
const SELECTION_RULES: Record<string, string> = {
  "max_recall_at_precision_0.7": "наибольшая полнота при точности не ниже 0,7",
  "best_f1_precision_0.7_unreachable":
    "точность 0,7 недостижима, выбран лучший F1",
};
const selectionRule = (rule: string) => SELECTION_RULES[rule] || rule;
const MATCHES_PAGE = 40;

export function EvaluationPage() {
  const { data, error } = useResource<Evaluation>("/evaluation"),
    [kind, setKind] = useState<Kind>("fault"),
    [shown, setShown] = useState(MATCHES_PAGE),
    [diagnostics, setDiagnostics] = useState(false);
  const [matches, setMatches] = useState<DailyRuleMatch[]>([]);
  useEffect(() => {
    setShown(MATCHES_PAGE);
    api<DailyRuleMatch[]>(`/evaluation/matches/${kind}`)
      .then(setMatches)
      .catch(() => setMatches([]));
  }, [kind]);
  if (error) return <ErrorNotice message={error} />;
  if (!data) return <Loading text="Загружаем результаты отложенного теста…" />;
  const m = data.models[kind],
    metric = m.test.alerts,
    base = m.test.baseline_alerts;
  const rule = data.daily_rule?.kinds[kind],
    june = rule?.june;
  const sensitivity = data.uncertainty?.models[kind]?.by_parent;
  const rangeText = (range: { low: number | null; high: number | null }) =>
    range.low === null || range.high === null
      ? "Недостаточно данных"
      : `${range.low.toFixed(3)} … ${range.high.toFixed(3)}`;
  const comparison = [
    {
      name: "Точность предупреждений",
      model: metric.precision,
      baseline: base.precision,
    },
    { name: "Полнота эпизодов", model: metric.recall, baseline: base.recall },
    { name: "F1", model: metric.f1, baseline: base.f1 },
  ];
  const oneDecimal = (v: number) => num(Math.round(v * 10) / 10);
  return (
    <div className="page-stack">
      <div className="kind-tabs">
        {(Object.keys(kindNames) as Kind[]).map((k) => (
          <button
            key={k}
            className={kind === k ? "active" : ""}
            onClick={() => setKind(k)}
          >
            <KindIcon kind={k} size={17} />
            {kindNames[k]}
          </button>
        ))}
      </div>
      {!rule || !june ? (
        <div className="evidence-note">
          <CircleAlert size={20} />
          <div>
            <strong>Автоматические предупреждения отключены</strong>
            <p>
              Недостаточно событий для выбора порога, поэтому для этого типа нет
              предупреждений и метрик отложенного теста.
            </p>
          </div>
        </div>
      ) : (
        <>
          <div className="metrics-grid evaluation-metrics">
            <div className="metric-card">
              <div className="metric-top">
                <span>Точность</span>
                <CheckCircle2 size={17} />
              </div>
              <strong>{pct(june.precision)}</strong>
              <p>
                {june.true_alerts} верных из {june.alerts} предупреждений
              </p>
            </div>
            <div className="metric-card">
              <div className="metric-top">
                <span>Полнота</span>
                <Filter size={17} />
              </div>
              <strong>{pct(june.recall)}</strong>
              <p>
                {june.caught_incident_days} из {june.incident_days} неспокойных
                суток пойманы заранее
              </p>
            </div>
            <div className="metric-card">
              <div className="metric-top">
                <span>Медианное упреждение</span>
                <Clock3 size={17} />
              </div>
              <strong>
                {june.median_lead_hours === null
                  ? "—"
                  : oneDecimal(june.median_lead_hours)}
                <small> ч</small>
              </strong>
              <p>До начала неспокойных суток</p>
            </div>
            <div className="metric-card">
              <div className="metric-top">
                <span>Предупреждений в сутки</span>
                <CircleAlert size={17} />
              </div>
              <strong>{oneDecimal(june.alerts_per_day)}</strong>
              <p>
                На весь парк, из них ложных{" "}
                {oneDecimal(june.false_alerts_per_day)}
              </p>
            </div>
          </div>
          <p className="evaluation-rule">
            Июнь 2026, расчёт каждый час, не больше одного предупреждения на
            объект и тип в сутки. Модель: {modelName(rule.model).toLowerCase()},
            порог {pct(rule.threshold)}. Полнота по отдельным эпизодам, один к
            одному: {pct(june.recall_strict)}.
          </p>
          <section className="panel">
            <div className="panel-heading">
              <div>
                <h2>Выбор модели и порога на 15–31 мая</h2>
                <span>
                  Правило: {selectionRule(rule.selection_rule)}. Июнь открыт
                  один раз, для выбранного варианта.
                </span>
              </div>
            </div>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Вариант</th>
                    <th>Порог</th>
                    <th>Точность</th>
                    <th>Полнота</th>
                    <th>F1</th>
                    <th>Предупреждений</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(rule.selection_candidates).map(
                    ([name, c]) => (
                      <tr
                        key={name}
                        className={name === rule.model ? "chosen-row" : ""}
                      >
                        <td>
                          {modelName(name)}
                          {name === rule.model && (
                            <span className="status reviewed">выбран</span>
                          )}
                        </td>
                        <td>{pct(c.threshold)}</td>
                        <td>{pct(c.precision)}</td>
                        <td>{pct(c.recall)}</td>
                        <td>{c.f1.toFixed(3).replace(".", ",")}</td>
                        <td>{num(c.alerts)}</td>
                      </tr>
                    ),
                  )}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Предупреждение и факт</h2>
            <span>
              Июнь 2026: каждое предупреждение и то, что записано в журнале за
              следующие 24 часа
            </span>
          </div>
          <span className="count-badge">{matches.length}</span>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Момент прогноза</th>
                <th>Объект</th>
                <th>Вероятность</th>
                <th>Итог</th>
                <th>Упреждение</th>
              </tr>
            </thead>
            <tbody>
              {matches.slice(0, shown).map((r) => (
                <tr key={`${r.object_id}-${r.as_of}`}>
                  <td>{date(r.as_of, true)}</td>
                  <td>#{r.object_id}</td>
                  <td>{pct(r.probability)}</td>
                  <td>
                    <span
                      className={`status ${r.hit ? "reviewed" : "attention"}`}
                    >
                      {r.hit
                        ? "Начались неспокойные сутки"
                        : "Ложное предупреждение"}
                    </span>
                  </td>
                  <td>
                    {r.lead_hours === null
                      ? "—"
                      : `${oneDecimal(r.lead_hours)} ч`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!matches.length && (
            <Empty title="Предупреждений на тесте нет">
              Для этого типа автоматические предупреждения отключены.
            </Empty>
          )}
        </div>
        {matches.length > shown && (
          <div className="table-more">
            <button
              className="secondary-button"
              onClick={() => setShown(matches.length)}
            >
              Показать все {num(matches.length)}
            </button>
            <span>
              {num(shown)} из {num(matches.length)}
            </span>
          </div>
        )}
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Протокол проверки</h2>
            <span>Разделение периодов и ограничения оценки</span>
          </div>
          <FileCheck2 size={20} />
        </div>
        <div className="split-timeline">
          {[
            ["train", "Обучение", "до марта 2026"],
            ["validation", "Ранняя остановка обучения", "март — апрель"],
            ["calibration", "Калибровка", "1–15 мая"],
            ["policy", "Выбор модели и порога", "15–31 мая"],
            ["test", "Отложенный тест", "июнь 2026"],
          ].map(([key, title, period], i) => (
            <div key={key} className={key === "test" ? "test" : ""}>
              <span>0{i + 1}</span>
              <strong>{title}</strong>
              <small>{period}</small>
            </div>
          ))}
        </div>
        <div className="protocol-details">
          <div>
            <span>Наблюдений в обучении</span>
            <strong>{num(m.training_rows)}</strong>
          </div>
          <div>
            <span>Расчёт на тесте</span>
            <strong>каждый час</strong>
          </div>
          <div>
            <span>Предупреждений в июне</span>
            <strong>{june ? num(june.alerts) : "—"}</strong>
          </div>
          <div>
            <span>Разрыв между выборками</span>
            <strong>{data.purge_hours} часов</strong>
          </div>
        </div>
        <div className="panel-footnote">
          Точность и полнота считаются в одной единице, неспокойных сутках:
          эпизоды объекта и типа в пределах 24 часов от первого составляют один
          случай. Предупреждение верно, если в следующие 24 часа начались новые
          неспокойные сутки.
        </div>
      </section>
      <details
        className="panel research-panel"
        onToggle={(e) => setDiagnostics(e.currentTarget.open)}
      >
        <summary>
          Диагностика исходного классификатора{" "}
          <span>окна каждые 3 часа, без правила 24 часов</span>
        </summary>
        {diagnostics && (
          <div className="research-body">
            <p>
              Здесь проверяется сам классификатор «эпизод за 24 часа» на
              отложенном июне: насколько точны его вероятности и чем он лучше
              частоты эпизодов объекта. Предупреждения сопоставлены с отдельными
              эпизодами, поэтому числа не совпадают с метриками правила 24 часов
              выше.
              {rule && rule.model !== "archive_classifier"
                ? " Для этого типа в работе другая модель, она указана выше."
                : ""}
            </p>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Точность</th>
                    <th>Полнота эпизодов</th>
                    <th>F1</th>
                    <th>PR-AUC на окнах</th>
                    <th>Окон в тесте</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>{pct(metric.precision)}</td>
                    <td>{pct(metric.recall)}</td>
                    <td>{metric.f1.toFixed(3).replace(".", ",")}</td>
                    <td>{m.test.rows.pr_auc.toFixed(3).replace(".", ",")}</td>
                    <td>{num(m.test.rows.rows)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            {sensitivity && (
              <p>
                Устойчивость: 95% диапазон разницы F1 «модель − база» при
                повторной выборке целых узлов{" "}
                {rangeText(sensitivity.percentile_95.f1_difference)} (
                {num(sensitivity.replicates)} выборок, {sensitivity.clusters}{" "}
                узлов).{" "}
                {(sensitivity.percentile_95.f1_difference.low ?? 0) <= 0
                  ? "Устойчивое превосходство над базой пока не подтверждено."
                  : "Прирост сохраняется в этой проверке."}
              </p>
            )}
            <div className="two-column">
              <div>
                <h3>Сравнение с базовым прогнозом</h3>
                <div className="large-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={comparison} barGap={7}>
                      <CartesianGrid vertical={false} stroke="#ebebeb" />
                      <XAxis
                        dataKey="name"
                        tick={{ fontSize: 10 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis
                        tickFormatter={(v) => pct(v, 0)}
                        domain={[0, 1]}
                        tick={{ fontSize: 11 }}
                        width={40}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip formatter={(v) => pct(Number(v))} />
                      <Legend
                        iconType="circle"
                        iconSize={7}
                        wrapperStyle={{ fontSize: 12, paddingTop: 18 }}
                      />
                      <Bar
                        dataKey="model"
                        name="Классификатор"
                        fill="#171717"
                        radius={[4, 4, 0, 0]}
                        maxBarSize={42}
                      />
                      <Bar
                        dataKey="baseline"
                        name="Частота эпизодов объекта"
                        fill="#8a8a8a"
                        radius={[4, 4, 0, 0]}
                        maxBarSize={42}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div>
                <h3>Насколько точны вероятности</h3>
                <div className="large-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={m.test.rows.calibration_curve.map((r) => ({
                        ...r,
                        ideal: r.predicted,
                      }))}
                    >
                      <CartesianGrid stroke="#ebebeb" vertical={false} />
                      <XAxis
                        dataKey="predicted"
                        type="number"
                        domain={[0, 1]}
                        tickFormatter={(v) => pct(v, 0)}
                        tick={{ fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis
                        domain={[0, 1]}
                        tickFormatter={(v) => pct(v, 0)}
                        tick={{ fontSize: 11 }}
                        width={40}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        formatter={(v) => pct(Number(v))}
                        labelFormatter={(v) => `Прогноз ${pct(Number(v))}`}
                      />
                      <Legend
                        iconType="circle"
                        iconSize={7}
                        wrapperStyle={{ fontSize: 12, paddingTop: 18 }}
                      />
                      <Line
                        dataKey="observed"
                        name="Наблюдаемая частота"
                        stroke="#171717"
                        strokeWidth={2}
                        dot={{ r: 4 }}
                      />
                      <Line
                        dataKey="ideal"
                        name="Идеальная калибровка"
                        stroke="#737373"
                        strokeDasharray="5 4"
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
            {data.research?.models[kind] && (
              <>
                <h3>Варианты классификатора на трёх прошлых периодах, F1</h3>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Период</th>
                        <th>Исходный подход</th>
                        <th>Контекст узла</th>
                        <th>Простые деревья</th>
                        <th>Ансамбль 50/50</th>
                        <th>Эпизодов</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.research.models[kind]!.rows.map((row) => (
                        <tr key={row.period}>
                          <td>{row.period}</td>
                          <td>{row.reference.toFixed(3)}</td>
                          <td>
                            {row.context === null
                              ? "—"
                              : row.context.toFixed(3)}
                          </td>
                          <td>{row.regularized.toFixed(3)}</td>
                          <td>{row.blend.toFixed(3)}</td>
                          <td>{num(row.eligible_episodes)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        )}
      </details>
    </div>
  );
}

export function QualityPage() {
  const { data, error } = useResource<Quality>("/quality");
  if (error) return <ErrorNotice message={error} />;
  if (!data) return <Loading />;
  return (
    <div className="page-stack">
      <div className="data-hero">
        <div>
          <span className="eyebrow">Полный исходный массив</span>
          <h2>
            {num(data.total_rows)}
            <small>записей</small>
          </h2>
          <p>8 годовых журналов за 2019–2026, исходные значения сохранены</p>
        </div>
        <Database size={58} strokeWidth={1} />
      </div>
      <div className="three-column">
        <div className="data-stat">
          <span>Каналы датчиков</span>
          <strong>{num(data.catalog.channel_count)}</strong>
          <small>Связаны с объектами справочника</small>
        </div>
        <div className="data-stat">
          <span>Объекты инфраструктуры</span>
          <strong>{data.catalog.object_count}</strong>
          <small>Три уровня иерархии</small>
        </div>
        <div className="data-stat">
          <span>Окна для моделирования</span>
          <strong>{num(data.features.eligible_rows)}</strong>
          <small>Полная предыстория и наблюдаемое будущее</small>
        </div>
      </div>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Поступление событий в 2026 году</h2>
            <span>Число исходных строк за сутки</span>
          </div>
        </div>
        <div className="wide-chart">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data.daily_2026}>
              <CartesianGrid stroke="#ebebeb" vertical={false} />
              <XAxis
                dataKey="day"
                tickFormatter={(v) => date(v)}
                tick={{ fontSize: 10 }}
                minTickGap={45}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tickFormatter={(v) => `${Math.round(v / 1000)} тыс.`}
                tick={{ fontSize: 10 }}
                width={60}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                formatter={(v) => num(Number(v))}
                labelFormatter={(v) => date(String(v))}
              />
              <Area
                dataKey="rows"
                name="Записей"
                stroke="#171717"
                fill="#f1f1f1"
                strokeWidth={1.5}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="panel-footnote">0 — нет записей в выгрузке.</div>
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Покрытие и качество по годам</h2>
            <span>Показатели рассчитаны на всём исходном массиве</span>
          </div>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Год</th>
                <th>Записей</th>
                <th>Каналов</th>
                <th>Тревожных строк</th>
                <th>Без справочника</th>
                <th>Использование</th>
              </tr>
            </thead>
            <tbody>
              {data.years.map((y) => (
                <tr key={y.year}>
                  <td>
                    <strong>{y.year}</strong>
                  </td>
                  <td>{num(y.rows)}</td>
                  <td>{num(y.channels)}</td>
                  <td>{num(y.alarms)}</td>
                  <td>{num(y.unmapped_rows)}</td>
                  <td>
                    <span
                      className={`status ${y.quarantined ? "attention" : y.used_for_model ? "reviewed" : "normal"}`}
                    >
                      {y.quarantined
                        ? "Карантин"
                        : y.used_for_model
                          ? "В моделировании"
                          : "Архив для сравнения"}
                    </span>
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
            <h2>Правила обработки</h2>
          </div>
          <div className="quality-rules">
            {data.rules.map((r, i) => (
              <div key={r.title}>
                <span>0{i + 1}</span>
                <div>
                  <h3>{r.title}</h3>
                  <p>{r.detail}</p>
                </div>
              </div>
            ))}
          </div>
        </section>
        <section className="panel">
          <div className="panel-heading">
            <h2>Независимые эпизоды</h2>
            <span>После обработки 2022–2026</span>
          </div>
          <div className="episode-counts">
            {Object.entries(data.features.episodes).map(([kind, n]) => (
              <div key={kind}>
                <KindTag kind={kind as Kind} />
                <strong>{num(n)}</strong>
              </div>
            ))}
          </div>
          <div className="inset-note">
            <CircleAlert size={18} />
            <p>
              Количество сообщений датчиков не равно количеству инцидентов.
              Каскады объединены, конфликтующие состояния отмечены отдельно.
            </p>
          </div>
          <div className="inset-note">
            <MapIcon />
            <p>
              Координаты не предоставлены. Схема в интерфейсе показывает связи
              справочника и не является географической картой.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
function MapIcon() {
  return <Database size={18} />;
}

export function DecisionsPage({
  refresh,
  onSelect,
}: {
  refresh: number;
  onSelect: (f: Forecast) => void;
}) {
  const { data, error } = useResource<DecisionRow[]>("/decisions", refresh),
    [filter, setFilter] = useState("all"),
    [localError, setLocalError] = useState("");
  const rows =
    data?.filter((d) => filter === "all" || d.action === filter) || [];
  async function open(row: DecisionRow) {
    try {
      const f = await api<Forecast>(
        `/forecast/${row.object_id}/${row.kind}?as_of=${encodeURIComponent(row.forecast_at)}`,
      );
      onSelect({ ...f, decision: row });
    } catch (e) {
      setLocalError((e as Error).message);
    }
  }
  if (error) return <ErrorNotice message={error} />;
  if (!data) return <Loading />;
  return (
    <div className="page-stack">
      {localError && <ErrorNotice message={localError} />}
      <div className="four-column">
        <div className="data-stat">
          <span>Решений в журнале</span>
          <strong>{data.length}</strong>
          <small>С сохранением автора и времени</small>
        </div>
        <div className="data-stat">
          <span>Требуют работ</span>
          <strong>{data.filter((d) => d.requires_work).length}</strong>
          <small>Выезд, проверка или ТО</small>
        </div>
        <div className="data-stat">
          <span>Работы выполнены</span>
          <strong>
            {data.filter((d) => d.work_order?.status === "done").length}
          </strong>
          <small>Статус из системы учёта заявок</small>
        </div>
        <div className="data-stat">
          <span>Ложные срабатывания</span>
          <strong>
            {data.filter((d) => d.action === "false_alarm").length}
          </strong>
          <small>По решению диспетчера</small>
        </div>
      </div>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Решения и результаты отработки</h2>
            <span>Открывайте запись, чтобы вернуться к исходному прогнозу</span>
          </div>
          <select
            aria-label="Фильтр решений"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          >
            <option value="all">Все действия</option>
            <option value="dispatch">Выезд бригады</option>
            <option value="inspect">Направить бригаду на проверку</option>
            <option value="monitor">Мониторинг ситуации</option>
            <option value="maintenance">Запланировать ТО</option>
            <option value="false_alarm">Ложное срабатывание</option>
          </select>
        </div>
        {!rows.length ? (
          <Empty title="Решений пока нет">
            Откройте карточку прогноза, проверьте основания и сохраните действие
            с причиной.
          </Empty>
        ) : (
          <div className="decision-cards">
            {rows.map((r) => (
              <button
                key={r.id}
                onClick={() => open(r)}
                className="decision-card"
              >
                <span className="decision-icon">
                  <Wrench size={20} />
                </span>
                <div>
                  <strong>{r.object_name}</strong>
                  <small>
                    {kindNames[r.kind]} · прогноз {date(r.forecast_at, true)}
                    {r.user_name ? ` · ${r.user_name}` : ""}
                  </small>
                  <p>
                    {r.reason_label ? `${r.reason_label}. ` : ""}
                    {r.comment || "Комментарий не указан"}
                  </p>
                </div>
                <div>
                  <span className="status reviewed">{r.action_label}</span>
                  {r.work_order ? (
                    <WorkBadge order={r.work_order} />
                  ) : r.requires_work ? (
                    <small>Заявка ещё не сформирована</small>
                  ) : null}
                  <small>{date(r.updated_at, true)}</small>
                </div>
                <ChevronRight size={18} />
              </button>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
