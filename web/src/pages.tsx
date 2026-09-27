import { useEffect, useState } from "react";
import {
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  Clock3,
  Database,
  FileCheck2,
  Filter,
  FlaskConical,
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
import type { DecisionRow, Evaluation, Forecast, Kind, Quality } from "./types";

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

export function EvaluationPage() {
  const { data, error } = useResource<Evaluation>("/evaluation"),
    [kind, setKind] = useState<Kind>("fault");
  const [matches, setMatches] = useState<
    {
      object_id: number;
      as_of: string;
      probability: number;
      hit: boolean;
      episode_id: string | null;
      lead_hours: number | null;
    }[]
  >([]);
  useEffect(() => {
    api<typeof matches>(`/evaluation/matches/${kind}`)
      .then(setMatches)
      .catch(() => setMatches([]));
  }, [kind]);
  if (error) return <ErrorNotice message={error} />;
  if (!data) return <Loading text="Загружаем результаты отложенного теста…" />;
  const m = data.models[kind],
    metric = m.test.alerts,
    base = m.test.baseline_alerts;
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
  return (
    <div className="page-stack">
      <div className="method-banner">
        <span className="method-icon">
          <FlaskConical size={25} />
        </span>
        <div>
          <strong>Будущее отделено от обучения</strong>
          <p>
            Тест исходной архивной версии: июнь 2026. Разрывы выгрузки и окна
            без полного будущего исключены. Одна тревога сопоставляется с одним
            эпизодом.
          </p>
        </div>
        <span className="soft-chip">Горизонт 24 часа</span>
      </div>
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
      {data.operational_quality?.kind === kind && (
        <section className="panel research-panel">
          <div className="research-body">
            <h3>Новая модель охранных эпизодов</h3>
            <p>
              Прогноз числа эпизодов и учёт незавершённых предупреждений. Одна
              тревога сопоставляется с одним эпизодом.
            </p>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Период</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>Эпизодов</th>
                  </tr>
                </thead>
                <tbody>
                  {data.operational_quality.rows.map((row) => (
                    <tr key={row.period}>
                      <td>{row.period}</td>
                      <td>{pct(row.precision)}</td>
                      <td>{pct(row.recall)}</td>
                      <td>{num(Number(row.f1.toFixed(3)))}</td>
                      <td>{num(row.eligible_episodes)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p>
              {data.operational_quality.target_summary ||
                "Цель 75% / 50% выполнена суммарно на трёх исходных периодах."}{" "}
              {data.operational_quality.below_target}. Оценены сигналы датчиков,
              а не подтверждённые физические инциденты.
            </p>
            <p>
              Условный диапазон прироста F1 по узлам в прежнем сравнении на
              ноябре, феврале и мае:{" "}
              {rangeText(data.operational_quality.f1_gain_interval)}. Прирост
              относится к модели вместе с правилом повторных предупреждений.
            </p>
            {data.operational_quality.additional_periods && (
              <p>
                Декабрь и март добавлены после выбора семейства модели. Для
                каждого периода модель обучалась на более ранней истории;
                конфигурация по результатам этих месяцев не менялась.
              </p>
            )}
            <small>
              Повторно использованные исторические периоды; это не новый слепой
              тест. Окончательные переобученные веса ещё не проверены на новом
              периоде. Версия новых расчётов:{" "}
              {data.operational_quality.model_version}. Июньская оценка исходной
              версии ниже сохранена.
            </small>
          </div>
        </section>
      )}
      {data.research?.models[kind] && (
        <details className="panel research-panel">
          <summary>
            Предыдущий цикл экспериментов <span>3 временных периода</span>
          </summary>
          <div className="research-body">
            <p>
              Сравнение семейств моделей на прошлых периодах. Каждая модель
              обучена только на более ранних данных. Указан событийный F1;
              больше — лучше.
            </p>
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
                        {row.context === null ? "—" : row.context.toFixed(3)}
                      </td>
                      <td>{row.regularized.toFixed(3)}</td>
                      <td>{row.blend.toFixed(3)}</td>
                      <td>{num(row.eligible_episodes)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p>
              {data.research.models[kind]!.selected === "reference"
                ? "В этом цикле новые варианты не прошли заранее заданные условия улучшения. Его результаты не описывают последующие обновления модели."
                : data.research.deployed_changed
                  ? "Новый вариант используется в новых пакетных расчётах. Он прошёл ретроспективное сравнение; для независимого подтверждения нужен новый период. Архивные прогнозы сохранены."
                  : "Кандидат прошёл ретроспективное сравнение. Для независимого подтверждения нужен новый период."}
            </p>
            {data.research.models[kind]!.uncertainty && (
              <p>
                Приближённый 95% диапазон разницы F1 по эксплуатационным узлам:{" "}
                {rangeText(
                  data.research.models[kind]!.uncertainty!.by_parent
                    .percentile_95.f1_difference,
                )}
                . Диапазон условен для уже выбранной модели и не учитывает сам
                отбор вариантов.
              </p>
            )}
            <small>
              Эти даты уже рассматривались в исследовании. Это дополнительная
              ретроспективная проверка; июньский тест исходной версии ниже не
              пересчитывался. Прочерк — вариант не выводился на подтверждение.
            </small>
          </div>
        </details>
      )}
      {m.threshold > 1 && (
        <div className="evidence-note">
          <CircleAlert size={20} />
          <div>
            <strong>Автоматические предупреждения отключены</strong>
            <p>
              В периоде настройки недостаточно независимых событий для выбора
              устойчивого порога. Вероятности доступны для исследовательского
              просмотра.
            </p>
          </div>
        </div>
      )}
      <div className="metrics-grid evaluation-metrics">
        <div className="metric-card">
          <div className="metric-top">
            <span>Точность предупреждений</span>
            <CheckCircle2 size={17} />
          </div>
          <strong>{pct(metric.precision)}</strong>
          <p>
            {metric.true_alerts} подтверждений из {metric.alerts} предупреждений
          </p>
        </div>
        <div className="metric-card">
          <div className="metric-top">
            <span>Полнота эпизодов</span>
            <Filter size={17} />
          </div>
          <strong>{pct(metric.recall)}</strong>
          <p>
            {metric.true_alerts} из {metric.eligible_episodes} доступных
            эпизодов
          </p>
        </div>
        <div className="metric-card">
          <div className="metric-top">
            <span>Медианное упреждение</span>
            <Clock3 size={17} />
          </div>
          <strong>
            {metric.median_lead_hours === null
              ? "—"
              : num(Math.round(metric.median_lead_hours * 10) / 10)}
            <small> ч</small>
          </strong>
          <p>До начала сопоставленного эпизода</p>
        </div>
        <div className="metric-card">
          <div className="metric-top">
            <span>Ложных на объект в сутки</span>
            <CircleAlert size={17} />
          </div>
          <strong>
            {num(Math.round(metric.false_alerts_per_object_day * 1000) / 1000)}
          </strong>
          <p>{metric.false_alerts} ложных предупреждений на тесте</p>
        </div>
      </div>
      {sensitivity && (
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Насколько устойчив результат</h2>
              <span>
                Приближённые 95% диапазоны при повторной выборке целых
                эксплуатационных узлов
              </span>
            </div>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>F1 модели</th>
                  <th>F1 базового прогноза</th>
                  <th>Разница F1: модель − база</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>{rangeText(sensitivity.percentile_95.f1)}</td>
                  <td>{rangeText(sensitivity.percentile_95.baseline_f1)}</td>
                  <td>{rangeText(sensitivity.percentile_95.f1_difference)}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="evidence-note">
            <CircleAlert size={20} />
            <div>
              <strong>
                {(sensitivity.percentile_95.f1_difference.low ?? 0) <= 0
                  ? "Устойчивое превосходство над базой пока не подтверждено"
                  : "Прирост сохраняется в этой проверке устойчивости"}
              </strong>
              <p>
                {num(sensitivity.replicates)} повторных выборок,{" "}
                {sensitivity.clusters} узлов. Верно предсказанные эпизоды
                приходятся на {sensitivity.clusters_with_true_alerts} из них.
                Истории внутри узла сохраняются вместе. Общие сбои между узлами,
                сезонность и качество на новом месяце эта проверка не оценивает.
              </p>
            </div>
          </div>
        </section>
      )}
      <div className="two-column">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Сравнение с базовым прогнозом</h2>
              <span>База — частота эпизодов объекта на обучении</span>
            </div>
          </div>
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
                  name="Модель"
                  fill="#171717"
                  radius={[4, 4, 0, 0]}
                  maxBarSize={42}
                />
                <Bar
                  dataKey="baseline"
                  name="Базовый прогноз"
                  fill="#8a8a8a"
                  radius={[4, 4, 0, 0]}
                  maxBarSize={42}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Насколько точны вероятности</h2>
              <span>Частота события в группах прогнозов</span>
            </div>
          </div>
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
        </section>
      </div>
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
            ["policy", "Выбор модели и порога", "16–31 мая"],
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
            <span>Прогнозных окон в тесте</span>
            <strong>{num(m.test.rows.rows)}</strong>
          </div>
          <div>
            <span>PR-AUC на тестовых окнах</span>
            <strong>{num(Math.round(m.test.rows.pr_auc * 1000) / 1000)}</strong>
          </div>
          <div>
            <span>Разрыв между выборками</span>
            <strong>{data.purge_hours} часов</strong>
          </div>
        </div>
        <div className="panel-footnote">
          Оцениваются прокси-эпизоды по журналам датчиков. Истинные пожары,
          проникновения и подтверждённые отказы отдельно не размечены.
          Вероятность относится к зарегистрированному событию.
        </div>
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Предупреждение и факт</h2>
            <span>
              Первые 40 предупреждений отложенного теста, по объектам и времени
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
                <th>Результат</th>
                <th>Упреждение</th>
              </tr>
            </thead>
            <tbody>
              {matches.slice(0, 40).map((r, i) => (
                <tr key={i}>
                  <td>{date(r.as_of, true)}</td>
                  <td>#{r.object_id}</td>
                  <td>{pct(r.probability)}</td>
                  <td>
                    <span
                      className={`status ${r.hit ? "reviewed" : "attention"}`}
                    >
                      {r.hit
                        ? "Эпизод зарегистрирован"
                        : "Нет сопоставленного эпизода"}
                    </span>
                  </td>
                  <td>
                    {r.lead_hours === null
                      ? "—"
                      : `${num(Math.round(r.lead_hours * 10) / 10)} ч`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!matches.length && (
            <Empty title="Предупреждения на тесте не сформированы">
              Смотрите статус политики и доступность событий для этого сценария.
            </Empty>
          )}
        </div>
      </section>
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
          <span className="eyebrow">ПОЛНЫЙ ИСХОДНЫЙ МАССИВ</span>
          <h2>
            {num(data.total_rows)}
            <small>записей</small>
          </h2>
          <p>8 годовых журналов · 2019–2026 · исходные значения сохранены</p>
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
        <div className="panel-footnote">
          Ноль означает отсутствие записей в выгрузке, а не исправность
          оборудования. Окна рядом с разрывами исключены из оценки.
        </div>
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
