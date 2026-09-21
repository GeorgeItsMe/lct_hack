import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  Bell,
  CalendarDays,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Database,
  FlaskConical,
  LayoutDashboard,
  ListChecks,
  LogOut,
  Map,
  Pause,
  Play,
  Radio,
  RefreshCw,
  Search,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Wrench,
} from "lucide-react";
import { Area, AreaChart, ResponsiveContainer } from "recharts";
import { api, date, kindNames, num } from "./api";
import {
  Brand,
  ErrorNotice,
  ForecastDrawer,
  ForecastTable,
  KindIcon,
  Loading,
  NetworkMap,
} from "./components";
import {
  DecisionsPage,
  EvaluationPage,
  QualityPage,
  SettingsPage,
} from "./pages";
import { ImportPanel } from "./ImportPanel";
import type { Forecast, Kind, Overview, Page, Topology, User } from "./types";

const nav = [
  { id: "overview", label: "Обзор", icon: LayoutDashboard },
  { id: "map", label: "Схема объектов", icon: Map },
  { id: "journal", label: "Журнал прогнозов", icon: ListChecks },
  { id: "decisions", label: "Решения и ТО", icon: Wrench },
  { id: "evaluation", label: "Проверка модели", icon: FlaskConical },
  { id: "quality", label: "Данные", icon: Database },
] as const;
const titles: Record<Page, [string, string]> = {
  overview: [
    "Состояние инфраструктуры",
    "Риски на ближайшие 24 часа и приоритеты для проверки",
  ],
  map: [
    "Схема объектов",
    "Иерархия инфраструктуры и распределение предупреждений",
  ],
  journal: [
    "Журнал прогнозов",
    "Вероятности, основания и результаты работы диспетчера",
  ],
  decisions: [
    "Решения и обслуживание",
    "Зафиксированные действия по прогнозам и планирование ТО",
  ],
  evaluation: [
    "Проверка модели",
    "Прогнозы сопоставлены с событиями отложенного периода",
  ],
  quality: [
    "Данные и их качество",
    "От исходных журналов к воспроизводимым эпизодам",
  ],
  settings: [
    "Параметры предупреждений",
    "Пороги риска и конфигурация рабочего места",
  ],
};

function Login({ onLogin }: { onLogin: (user: User) => void }) {
  const [username, setUsername] = useState("dispatcher"),
    [password, setPassword] = useState("contour-demo");
  const [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [demo, setDemo] = useState(false);
  useEffect(() => {
    api<{ demo_mode: boolean }>("/health")
      .then((r) => {
        setDemo(r.demo_mode);
        if (!r.demo_mode) {
          setUsername("");
          setPassword("");
        }
      })
      .catch(() => setError("Сервис недоступен. Проверьте запуск API."));
  }, []);
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const result = await api<{ user: User; demo_mode: boolean }>(
        "/auth/login",
        { method: "POST", body: JSON.stringify({ username, password }) },
      );
      onLogin({ ...result.user, demo_mode: result.demo_mode });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="login-page">
      <div className="login-story">
        <Brand />
        <div>
          <span className="eyebrow">ИНЖЕНЕРНАЯ ИНФРАСТРУКТУРА МОСКВЫ</span>
          <h1>
            Знать о риске.
            <br />
            Успеть проверить.
          </h1>
          <p>
            Рабочее место диспетчера с прогнозом состояния оборудования,
            объяснением предупреждений и историей решений.
          </p>
          <div className="login-lines" aria-hidden="true">
            <i />
            <i />
            <i />
            <b />
            <b />
            <b />
          </div>
        </div>
        <footer>
          Предиктивная аналитика <span>01 / МОСКОЛЛЕКТОР</span>
        </footer>
      </div>
      <div className="login-panel">
        <div className="login-form">
          <span className="label-with-dot">
            <i className="dot green" />
            КОНТУР · РАБОЧЕЕ МЕСТО
          </span>
          <h2>Вход в систему</h2>
          <p>Выберите роль для работы с инфраструктурой.</p>
          {demo && (
            <div className="demo-login-label">
              Демонстрационный стенд · реальные архивные данные
            </div>
          )}
          {error && <ErrorNotice message={error} />}
          <form onSubmit={submit}>
            {demo ? (
              <label>
                Роль
                <select
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                >
                  <option value="dispatcher">Диспетчер ОДС</option>
                  <option value="analyst">Аналитик</option>
                  <option value="admin">Администратор</option>
                </select>
              </label>
            ) : (
              <>
                <label>
                  Логин
                  <input
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    autoComplete="username"
                    required
                  />
                </label>
                <label>
                  Пароль
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    autoComplete="current-password"
                    required
                  />
                </label>
              </>
            )}
            <button className="primary-button" disabled={busy}>
              {busy ? "Входим…" : "Открыть рабочее место"}
              <ArrowRight size={18} />
            </button>
          </form>
          <div className="login-note">
            <ShieldCheck size={18} />
            <span>
              Прогноз помогает принять решение.
              <br />
              Управление оборудованием остаётся у диспетчера.
            </span>
          </div>
        </div>
        <small>Задача №8 · Лидеры цифровой трансформации · 2026</small>
      </div>
    </div>
  );
}

export default function App() {
  const [user, setUser] = useState<User | null>(null),
    [authReady, setAuthReady] = useState(false);
  const [page, setPage] = useState<Page>("overview"),
    [overview, setOverview] = useState<Overview | null>(null),
    [topology, setTopology] = useState<Topology | null>(null);
  const [asOf, setAsOf] = useState(""),
    [times, setTimes] = useState<string[]>([]),
    [timeIndex, setTimeIndex] = useState(0),
    [playing, setPlaying] = useState(false);
  const [kind, setKind] = useState<Kind | "all">("all"),
    [query, setQuery] = useState(""),
    [onlyWarnings, setOnlyWarnings] = useState(false);
  const [selected, setSelected] = useState<Forecast | null>(null),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(false);
  const [refresh, setRefresh] = useState(0),
    [toast, setToast] = useState(""),
    [profileOpen, setProfileOpen] = useState(false);
  const closeDrawer = useCallback(() => setSelected(null), []);
  const [streamAlert, setStreamAlert] = useState<{
    asOf: string | null;
    message: string;
  } | null>(null);
  useEffect(() => {
    if (!user) {
      setStreamAlert(null);
      return;
    }
    let active = true;
    async function pollStream() {
      try {
        const state = await api<{
          as_of: string | null;
          forecast_status: string;
          forecast_error: string | null;
          latest_job: { id: string; status: string; as_of: string } | null;
        }>("/stream");
        let message = "";
        if (state.forecast_status === "waiting") {
          if (active) setStreamAlert(null);
          return;
        }
        if (state.forecast_status === "complete" && state.latest_job) {
          const result = await api<{
            result: { forecasts: { above_threshold: boolean }[] };
          }>(`/imports/${state.latest_job.id}`);
          message = `Прогноз готов · выше порога: ${result.result.forecasts.filter((f) => f.above_threshold).length}`;
        } else if (state.forecast_status === "failed")
          message = `Расчёт не выполнен: ${state.forecast_error || "проверьте журнал"}`;
        else if (state.forecast_status === "accepted_unprocessed")
          message = "Данные сохранены, актуальный прогноз ещё не рассчитан";
        else
          message =
            state.forecast_status === "running"
              ? "Расчёт нового прогноза выполняется"
              : "Новый прогноз ожидает расчёта";
        if (active) setStreamAlert({ asOf: state.as_of, message });
      } catch {
        if (active)
          setStreamAlert({
            asOf: null,
            message:
              "Не удалось обновить состояние потока. Проверьте соединение.",
          });
      }
    }
    pollStream();
    const timer = setInterval(pollStream, 15000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [user]);
  useEffect(() => {
    api<User>("/auth/me")
      .then(setUser)
      .catch(() => {})
      .finally(() => setAuthReady(true));
  }, []);
  useEffect(() => {
    if (!user || times.length) return;
    api<{ times: string[]; default: string }>("/replay")
      .then((r) => {
        setTimes(r.times);
        setAsOf(r.default);
        setTimeIndex(Math.max(0, r.times.indexOf(r.default)));
      })
      .catch((e) => setError(e.message));
  }, [user, refresh]);
  useEffect(() => {
    if (!user || !asOf) return;
    let live = true;
    setLoading(true);
    setError("");
    Promise.all([
      api<Overview>(`/overview?as_of=${encodeURIComponent(asOf)}`),
      api<Topology>(`/topology?as_of=${encodeURIComponent(asOf)}`),
    ])
      .then(([o, t]) => {
        if (live) {
          setOverview(o);
          setTopology(t);
        }
      })
      .catch((e) => live && setError(e.message))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [user, asOf, refresh]);
  useEffect(() => {
    if (!playing || !times.length) return;
    const t = setInterval(
      () =>
        setTimeIndex((i) => {
          const next = Math.min(times.length - 1, i + 1);
          setAsOf(times[next]);
          if (next === times.length - 1) setPlaying(false);
          return next;
        }),
      2400,
    );
    return () => clearInterval(t);
  }, [playing, times]);
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(""), 4500);
    return () => clearTimeout(t);
  }, [toast]);
  const filtered = useMemo(
    () =>
      overview?.forecasts.filter(
        (f) =>
          (kind === "all" || f.kind === kind) &&
          (!onlyWarnings || f.above_threshold) &&
          `${f.object_name} ${f.object_id} ${f.kind_label}`
            .toLowerCase()
            .includes(query.toLowerCase()),
      ) || [],
    [overview, kind, query, onlyWarnings],
  );
  const warnings =
    overview?.forecasts.filter((f) => f.above_threshold && !f.decision) || [];
  const jump = (i: number) => {
    const n = Math.max(0, Math.min(times.length - 1, i));
    setTimeIndex(n);
    setAsOf(times[n]);
    setPlaying(false);
  };
  const saved = () => {
    setToast("Решение сохранено в журнале");
    setRefresh((r) => r + 1);
  };
  async function logout() {
    await api("/auth/logout", { method: "POST" });
    setUser(null);
    setProfileOpen(false);
    setPlaying(false);
  }
  if (!authReady)
    return (
      <div className="startup">
        <Brand />
        <Loading />
      </div>
    );
  if (!user) return <Login onLogin={setUser} />;
  const title = titles[page];
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Brand />
        <div className="workspace-switch">
          <span className="workspace-icon">
            <Map size={17} />
          </span>
          <div>
            <strong>Эксплуатационный район</strong>
            <small>Объединённая диспетчерская</small>
          </div>
        </div>
        <span className="nav-label">РАБОЧЕЕ ПРОСТРАНСТВО</span>
        <nav>
          {nav.slice(0, 4).map((n) => (
            <button
              key={n.id}
              className={page === n.id ? "active" : ""}
              onClick={() => setPage(n.id)}
            >
              <n.icon size={18} strokeWidth={1.7} />
              <span>{n.label}</span>
              {n.id === "journal" && warnings.length > 0 && (
                <b>{warnings.length}</b>
              )}
            </button>
          ))}
          <span className="nav-label second">АНАЛИТИКА</span>
          {nav.slice(4).map((n) => (
            <button
              key={n.id}
              className={page === n.id ? "active" : ""}
              onClick={() => setPage(n.id)}
            >
              <n.icon size={18} strokeWidth={1.7} />
              <span>{n.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="model-status">
            <span>
              <i className="dot mint" />
              Модель подключена
            </span>
            <small>
              {overview
                ? `Версия ${overview.model_version}`
                : "Загрузка артефактов"}{" "}
              · горизонт 24 ч
            </small>
          </div>
          <button
            className={page === "settings" ? "active" : ""}
            onClick={() => setPage("settings")}
          >
            <Settings2 size={18} />
            <span>Параметры</span>
          </button>
          <button
            onClick={() => {
              setPage("evaluation");
              setToast(
                "Методика и ограничения доступны в разделе проверки модели",
              );
            }}
          >
            <CircleHelp size={18} />
            <span>Методика работы</span>
          </button>
          <div className="sidebar-credit">
            МОСКОЛЛЕКТОР <span>2026</span>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            Рабочее место <ChevronRight size={13} />
            <strong>
              {nav.find((n) => n.id === page)?.label || "Параметры"}
            </strong>
          </div>
          <div className="topbar-actions">
            <span className="environment-label">
              <i className="dot green" />
              Архивный поток
            </span>
            <button
              className="notification-button"
              aria-label={`Открыть ${warnings.length} предупреждений`}
              onClick={() => {
                setPage("journal");
                setOnlyWarnings(true);
              }}
            >
              <Bell size={18} />
              {warnings.length > 0 && <i />}
            </button>
            <div className="profile">
              <button
                onClick={() => setProfileOpen(!profileOpen)}
                aria-label="Меню пользователя"
              >
                <span className="avatar">
                  {user.role === "admin"
                    ? "АД"
                    : user.role === "analyst"
                      ? "АН"
                      : "ОД"}
                </span>
                <div>
                  <strong>{user.name}</strong>
                  <small>
                    {user.role === "admin"
                      ? "Администратор"
                      : user.role === "analyst"
                        ? "Аналитик"
                        : "Диспетчер"}
                  </small>
                </div>
              </button>
              {profileOpen && (
                <div className="profile-menu">
                  <button onClick={logout}>
                    <LogOut size={15} />
                    Выйти из системы
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>
        <main>
          {streamAlert && (
            <div className="stream-notification" role="status">
              <Bell size={18} />
              <span>
                Поступивший поток
                {streamAlert.asOf ? ` · ${date(streamAlert.asOf, true)}` : ""} ·{" "}
                {streamAlert.message}
              </span>
              <button className="text-link" onClick={() => setPage("quality")}>
                Открыть проверку
              </button>
            </div>
          )}
          <div className="page-heading">
            <div>
              <span className="eyebrow">ОПЕРАТИВНЫЙ КОНТУР</span>
              <h1>{title[0]}</h1>
              <p>{title[1]}</p>
            </div>
            <div className="heading-actions">
              <button
                className="secondary-button"
                onClick={() => setRefresh((r) => r + 1)}
                aria-label="Обновить данные"
              >
                <RefreshCw size={16} className={loading ? "spin" : ""} />
              </button>
              {overview && (
                <a
                  className="secondary-button"
                  href={`/api/export/forecasts.csv?as_of=${encodeURIComponent(asOf)}`}
                >
                  <ArrowDownToLine size={16} />
                  Экспорт
                </a>
              )}
            </div>
          </div>
          {["overview", "map", "journal"].includes(page) && (
            <div className="replay-bar">
              <div className="replay-title">
                <span className="replay-icon">
                  <Radio size={17} />
                </span>
                <div>
                  <strong>Воспроизведение истории</strong>
                  <small>Данные до выбранного момента · МСК</small>
                </div>
              </div>
              <div className="replay-controls">
                <button
                  onClick={() => jump(timeIndex - 1)}
                  disabled={!times.length || timeIndex === 0}
                  aria-label="Предыдущий момент"
                >
                  <ChevronLeft size={16} />
                </button>
                <button
                  className={playing ? "playing" : ""}
                  onClick={() => setPlaying(!playing)}
                  disabled={!times.length}
                  aria-label={
                    playing ? "Приостановить" : "Воспроизводить историю"
                  }
                >
                  {playing ? <Pause size={15} /> : <Play size={15} />}
                </button>
                <button
                  onClick={() => jump(timeIndex + 1)}
                  disabled={!times.length || timeIndex === times.length - 1}
                  aria-label="Следующий момент"
                >
                  <ChevronRight size={16} />
                </button>
              </div>
              <input
                type="range"
                aria-label="Момент архивного прогноза"
                min={0}
                max={Math.max(0, times.length - 1)}
                value={timeIndex}
                onChange={(e) => setTimeIndex(Number(e.target.value))}
                onPointerUp={() => jump(timeIndex)}
                onKeyUp={() => jump(timeIndex)}
              />
              <div className="replay-date">
                <CalendarDays size={16} />
                {asOf ? date(asOf, true) : "Подготовка…"}
              </div>
              <span className="horizon-chip">+24 часа</span>
            </div>
          )}
          {error && (
            <ErrorNotice
              message={error}
              retry={() => setRefresh((r) => r + 1)}
            />
          )}
          {!overview && ["overview", "map", "journal"].includes(page) ? (
            <Loading text="Подключаем прогнозы и справочник объектов…" />
          ) : (
            <>
              {page === "overview" && overview && (
                <>
                  <div className="metrics-grid">
                    <div className="metric-card">
                      <div className="metric-top">
                        <span>Объекты в прогнозе</span>
                        <Map size={17} />
                      </div>
                      <strong>
                        {num(overview.stats.objects)}
                        <small> / {overview.stats.catalog_objects}</small>
                      </strong>
                      <p>
                        <i className="dot green" />
                        {num(overview.stats.channels)} каналов в справочнике
                      </p>
                    </div>
                    <div className="metric-card warning">
                      <div className="metric-top">
                        <span>Требуют проверки</span>
                        <Bell size={17} />
                      </div>
                      <strong>
                        {num(overview.stats.objects_at_risk)}
                        <small> объектов</small>
                      </strong>
                      <p>
                        {num(overview.stats.warnings)} предупреждений на 24 часа
                      </p>
                    </div>
                    <div className="metric-card">
                      <div className="metric-top">
                        <span>Изменений за сутки</span>
                        <Activity size={17} />
                      </div>
                      <strong>{num(overview.stats.events_24h)}</strong>
                      <div className="mini-chart">
                        <ResponsiveContainer width="100%" height={32}>
                          <AreaChart data={overview.activity}>
                            <Area
                              type="monotone"
                              dataKey="events"
                              stroke="#55846d"
                              strokeWidth={1.5}
                              fill="#eaf2ec"
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-top">
                        <span>Ожидают решения</span>
                        <ListChecks size={17} />
                      </div>
                      <strong>{num(warnings.length)}</strong>
                      <button
                        className="text-link"
                        onClick={() => {
                          setPage("journal");
                          setOnlyWarnings(true);
                        }}
                      >
                        Перейти к проверке
                        <ArrowRight size={14} />
                      </button>
                    </div>
                  </div>
                  <div className="overview-middle">
                    <section className="panel map-panel">
                      <div className="panel-heading">
                        <div>
                          <h2>Инфраструктура района</h2>
                          <span>
                            {topology?.nodes.filter((n) => n.level === 2)
                              .length || 0}{" "}
                            эксплуатационных узлов
                          </span>
                        </div>
                        <button
                          className="text-link"
                          onClick={() => setPage("map")}
                        >
                          Открыть схему
                          <ArrowRight size={15} />
                        </button>
                      </div>
                      {topology && (
                        <NetworkMap
                          topology={topology}
                          forecasts={overview.forecasts}
                          onSelect={setSelected}
                        />
                      )}
                    </section>
                    <section className="panel priority-panel">
                      <div className="panel-heading">
                        <div>
                          <h2>В фокусе диспетчера</h2>
                          <span>Сценарии для проверки</span>
                        </div>
                        <SlidersHorizontal size={17} />
                      </div>
                      <div className="scenario-list">
                        {overview.kinds.map((k) => (
                          <button
                            key={k.id}
                            onClick={() => {
                              setKind(k.id);
                              setPage("journal");
                              setOnlyWarnings(true);
                            }}
                          >
                            <span className={`scenario-symbol ${k.id}`}>
                              <KindIcon kind={k.id} size={21} />
                            </span>
                            <div>
                              <strong>{kindNames[k.id]}</strong>
                              <small>
                                {k.threshold > 1
                                  ? "Предупреждения отключены"
                                  : k.count
                                    ? "Есть предупреждения"
                                    : "Выше порога нет"}
                              </small>
                            </div>
                            <b className={k.count ? "has-risk" : ""}>
                              {k.count}
                            </b>
                            <ChevronRight size={14} />
                          </button>
                        ))}
                      </div>
                      <div className="focus-note">
                        <ShieldCheck size={22} />
                        <strong>
                          У каждого прогноза
                          <br />
                          есть проверяемое основание
                        </strong>
                        <p>
                          Факторы модели, исходные события и результат проверки
                          доступны в карточке.
                        </p>
                        <button
                          className="text-link"
                          onClick={() => setPage("evaluation")}
                        >
                          Как проверяли качество
                          <ArrowRight size={14} />
                        </button>
                      </div>
                    </section>
                  </div>
                  <section className="panel">
                    <div className="panel-heading">
                      <div>
                        <h2>
                          Приоритетные прогнозы{" "}
                          <span className="count-badge">
                            {
                              overview.forecasts.filter(
                                (f) => f.above_threshold,
                              ).length
                            }
                          </span>
                        </h2>
                        <span>
                          Сначала предупреждения, затем ближайшие к порогу риски
                        </span>
                      </div>
                      <button
                        className="text-link"
                        onClick={() => setPage("journal")}
                      >
                        Весь журнал
                        <ArrowRight size={15} />
                      </button>
                    </div>
                    <ForecastTable
                      forecasts={overview.forecasts.slice(0, 6)}
                      onSelect={setSelected}
                    />
                  </section>
                </>
              )}
              {page === "map" && overview && topology && (
                <section className="panel">
                  <div className="panel-heading">
                    <div>
                      <h2>Объекты и эксплуатационные узлы</h2>
                      <span>
                        Выберите объект, чтобы открыть приоритетный прогноз
                      </span>
                    </div>
                    <span className="soft-chip">Схематичное представление</span>
                  </div>
                  <NetworkMap
                    topology={topology}
                    forecasts={overview.forecasts}
                    onSelect={setSelected}
                    expanded
                  />
                  <div className="panel-footnote">
                    Расположение условное. Связи построены по иерархии
                    справочника заказчика.
                  </div>
                </section>
              )}
              {page === "journal" && overview && (
                <section className="panel">
                  <div className="journal-toolbar">
                    <div className="search-field">
                      <Search size={17} />
                      <input
                        aria-label="Поиск объекта"
                        placeholder="Объект, номер или тип риска"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                      />
                    </div>
                    <select
                      aria-label="Тип риска"
                      value={kind}
                      onChange={(e) => setKind(e.target.value as Kind | "all")}
                    >
                      <option value="all">Все типы рисков</option>
                      {Object.entries(kindNames).map(([k, v]) => (
                        <option value={k} key={k}>
                          {v}
                        </option>
                      ))}
                    </select>
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={onlyWarnings}
                        onChange={(e) => setOnlyWarnings(e.target.checked)}
                      />
                      Только выше порога
                    </label>
                    <span className="result-count">
                      {filtered.length} прогнозов
                    </span>
                  </div>
                  <ForecastTable forecasts={filtered} onSelect={setSelected} />
                </section>
              )}
              {page === "decisions" && (
                <DecisionsPage
                  refresh={refresh}
                  onSelect={(f) => {
                    setAsOf(f.as_of);
                    setSelected(f);
                  }}
                />
              )}
              {page === "evaluation" && <EvaluationPage />}
              {page === "quality" && (
                <div className="page-stack">
                  <ImportPanel user={user} />
                  <QualityPage />
                </div>
              )}
              {page === "settings" && (
                <SettingsPage
                  user={user}
                  onSaved={() => {
                    setRefresh((r) => r + 1);
                    setToast("Пороги предупреждений сохранены");
                  }}
                />
              )}
            </>
          )}
          <footer className="main-footer">
            <span>
              <i className="dot green" />
              Контур · поддержка решений диспетчера
            </span>
            <span>
              Внешние источники — только чтение <ShieldCheck size={13} />
            </span>
          </footer>
        </main>
      </div>
      {selected && (
        <ForecastDrawer
          forecast={selected}
          user={user}
          onClose={closeDrawer}
          onSaved={saved}
        />
      )}{" "}
      {toast && (
        <div className="toast">
          <ShieldCheck size={18} />
          {toast}
        </div>
      )}
    </div>
  );
}
