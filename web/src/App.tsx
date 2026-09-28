import {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  Bell,
  BellRing,
  CalendarDays,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  ListChecks,
  LogOut,
  Map,
  Pause,
  Play,
  Radio,
  RefreshCw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
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
  riskNames,
} from "./components";
import { DecisionsPage, EvaluationPage, QualityPage } from "./pages";
import {
  AuditPage,
  EquipmentPage,
  ParametersPage,
  SummaryPage,
  ThresholdsPage,
  UsersPage,
  VerificationPage,
  WorkPage,
} from "./rolePages";
import { ImportPanel } from "./ImportPanel";
import {
  DEMO_ROLES,
  HOME,
  NAVIGATION,
  REPLAY_PAGES,
  ROLE_SHORT,
  TITLES,
  allowedPages,
  can,
} from "./roles";
import type { Forecast, Kind, Overview, Page, Topology, User } from "./types";

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
  async function login(name: string, secret: string) {
    setBusy(true);
    setError("");
    try {
      const result = await api<{ user: User; demo_mode: boolean }>(
        "/auth/login",
        {
          method: "POST",
          body: JSON.stringify({ username: name, password: secret }),
        },
      );
      onLogin({ ...result.user, demo_mode: result.demo_mode });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  function submit(e: React.FormEvent) {
    e.preventDefault();
    login(username, password);
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
            Прогноз отказов и инцидентов в инженерных коллекторах на 24 часа.
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
          <p>{demo ? "Выберите роль" : "Корпоративная учётная запись"}</p>
          {error && <ErrorNotice message={error} />}
          {demo ? (
            <div className="role-picker">
              {DEMO_ROLES.map((r) => (
                <button
                  key={r.username}
                  disabled={busy}
                  onClick={() => login(r.username, "contour-demo")}
                >
                  <span className="avatar">
                    {ROLE_SHORT[r.username as keyof typeof ROLE_SHORT]}
                  </span>
                  <div>
                    <strong>{r.label}</strong>
                  </div>
                  <ArrowRight size={17} />
                </button>
              ))}
              <button
                className="service-login"
                disabled={busy}
                onClick={() => login("admin", "contour-demo")}
              >
                Служебный вход: администратор
              </button>
            </div>
          ) : (
            <form onSubmit={submit}>
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
              <button className="primary-button" disabled={busy}>
                {busy ? "Входим…" : "Открыть рабочее место"}
                <ArrowRight size={18} />
              </button>
            </form>
          )}
        </div>
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
  const [source, setSource] = useState<"all" | "archive" | "stream">("all");
  const [kind, setKind] = useState<Kind | "all">("all"),
    [query, setQuery] = useState(""),
    [onlyWarnings, setOnlyWarnings] = useState(false);
  const [selected, setSelected] = useState<Forecast | null>(null),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(false);
  const [refresh, setRefresh] = useState(0),
    [toast, setToast] = useState(""),
    [profileOpen, setProfileOpen] = useState(false);
  const [alertToast, setAlertToast] = useState<{
    title: string;
    items: Forecast[];
  } | null>(null);
  const seenAlerts = useRef<Set<string> | null>(null);
  const closeDrawer = useCallback(() => setSelected(null), []);
  const searchInput = useRef<HTMLInputElement>(null);
  const [focusSearch, setFocusSearch] = useState(false);
  const pages = useMemo(() => (user ? allowedPages(user.role) : []), [user]);
  const go = useCallback(
    (target: Page) => {
      if (!user) return;
      setPage(pages.includes(target) ? target : HOME[user.role]);
    },
    [user, pages],
  );
  const openSearch = useCallback(() => {
    if (!pages.includes("journal")) return;
    setPage("journal");
    setSelected(null);
    setPlaying(false);
    setKind("all");
    setOnlyWarnings(false);
    setFocusSearch(true);
  }, [pages]);
  useEffect(() => {
    if (focusSearch && page === "journal" && overview) {
      searchInput.current?.focus();
      searchInput.current?.select();
      setFocusSearch(false);
    }
  }, [focusSearch, page, overview]);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (
        user &&
        (event.metaKey || event.ctrlKey) &&
        event.key.toLowerCase() === "k"
      ) {
        event.preventDefault();
        openSearch();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [user, openSearch]);
  const [streamAlert, setStreamAlert] = useState<{
    asOf: string | null;
    message: string;
  } | null>(null);
  const followsStream =
    can(user, "data.import") || can(user, "decisions.write");
  useEffect(() => {
    if (!user || !followsStream) {
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
            result: {
              forecasts: {
                above_threshold: boolean;
                notification_due?: boolean;
              }[];
            };
          }>(`/imports/${state.latest_job.id}`);
          message = `Новые данные мониторинга · выше порога: ${result.result.forecasts.filter((f) => f.above_threshold).length}`;
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
  }, [user, followsStream]);
  useEffect(() => {
    api<User>("/auth/me")
      .then((u) => {
        setUser(u);
        setPage(HOME[u.role]);
      })
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
  }, [user, refresh, times.length]);
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
  useEffect(() => {
    if (!alertToast) return;
    const t = setTimeout(() => setAlertToast(null), 9000);
    return () => clearTimeout(t);
  }, [alertToast]);
  /* Notifications (section 10): the dispatcher is told about every new warning,
     the unit head only about critical ones. A warning is "new" when its object
     and risk type were not above the threshold in the previous snapshot. */
  const allForecasts: Forecast[] = useMemo(() => {
    const rows = [
      ...(overview?.stream?.forecasts || []),
      ...(overview?.forecasts || []),
    ];
    // Warnings first; among them fresh monitoring-stream data before the archive moment.
    return rows.sort(
      (a, b) =>
        Number(b.above_threshold) - Number(a.above_threshold) ||
        Number(b.source === "stream") - Number(a.source === "stream") ||
        b.probability / Math.max(b.threshold, 0.001) -
          a.probability / Math.max(a.threshold, 0.001),
    );
  }, [overview]);
  const queueStats = useMemo(() => {
    const above = allForecasts.filter((f) => f.above_threshold);
    return {
      warnings: above.length,
      objects: new Set(above.map((f) => f.object_id)).size,
      critical: allForecasts.filter((f) => f.risk === "critical").length,
      byKind: Object.fromEntries(
        (["fault", "fire", "flood", "access"] as Kind[]).map((k) => [
          k,
          above.filter((f) => f.kind === k).length,
        ]),
      ) as Record<Kind, number>,
    };
  }, [allForecasts]);
  const notified: Forecast[] = useMemo(() => {
    if (!overview || !user) return [];
    if (can(user, "notifications.all"))
      return allForecasts.filter((f) => f.above_threshold && !f.decision);
    if (can(user, "notifications.critical"))
      return allForecasts.filter((f) => f.risk === "critical");
    return [];
  }, [overview, user, allForecasts]);
  useEffect(() => {
    if (!overview || !user) return;
    const keys = new Set(notified.map((f) => `${f.object_id}:${f.kind}`));
    const previous = seenAlerts.current;
    seenAlerts.current = keys;
    if (!previous) return;
    const fresh = notified.filter(
      (f) => !previous.has(`${f.object_id}:${f.kind}`),
    );
    if (fresh.length)
      setAlertToast({
        title: can(user, "notifications.all")
          ? `Новых предупреждений: ${fresh.length}`
          : `Новых критических рисков: ${fresh.length}`,
        items: fresh.slice(0, 3),
      });
  }, [notified, overview, user]);
  const filtered = useMemo(
    () =>
      allForecasts.filter(
        (f) =>
          (source === "all" || (f.source || "archive") === source) &&
          (kind === "all" || f.kind === kind) &&
          (!onlyWarnings || f.above_threshold) &&
          `${f.object_name} ${f.object_id} ${f.kind_label}`
            .toLowerCase()
            .includes(query.toLowerCase()),
      ) || [],
    [allForecasts, source, kind, query, onlyWarnings],
  );
  const warnings = allForecasts.filter((f) => f.above_threshold && !f.decision);
  const jump = (i: number) => {
    const n = Math.max(0, Math.min(times.length - 1, i));
    setTimeIndex(n);
    setAsOf(times[n]);
    setPlaying(false);
  };
  const saved = () => {
    setToast("Сохранено в журнале");
    setRefresh((r) => r + 1);
  };
  const changed = useCallback(() => setRefresh((r) => r + 1), []);
  async function logout() {
    await api("/auth/logout", { method: "POST" });
    setUser(null);
    setProfileOpen(false);
    setPlaying(false);
    setOverview(null);
    setTimes([]);
    seenAlerts.current = null;
  }
  if (!authReady)
    return (
      <div className="startup">
        <Brand />
        <Loading />
      </div>
    );
  if (!user)
    return (
      <Login
        onLogin={(u) => {
          setUser(u);
          setPage(HOME[u.role]);
        }}
      />
    );
  const title = TITLES[page];
  const sections = NAVIGATION[user.role];
  const navIndex = pages.indexOf(page);
  const replay = REPLAY_PAGES.includes(page);
  const bellTarget: Page = can(user, "notifications.all")
    ? "journal"
    : "summary";
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Brand />
        {pages.includes("journal") && (
          <button
            className="workspace-search"
            onClick={openSearch}
            aria-label="Найти объект"
          >
            <Search size={16} />
            <span>Найти объект</span>
            <kbd>⌘ K</kbd>
          </button>
        )}
        <div className="workspace-switch">
          <span className="workspace-icon">
            <Map size={17} />
          </span>
          <div>
            <strong>{user.role_label}</strong>
            <small>
              {user.scope === "district"
                ? "Весь район · инженерные коллекторы"
                : `Зона: ${user.scope}`}
            </small>
          </div>
        </div>
        <nav>
          {sections.map((section, s) => (
            <Fragment key={section.label}>
              <span className={`nav-label ${s ? "second" : ""}`}>
                {section.label}
              </span>
              {section.items.map((n) => (
                <button
                  key={n.id}
                  aria-label={n.label}
                  title={n.label}
                  aria-current={page === n.id ? "page" : undefined}
                  className={page === n.id ? "active" : ""}
                  onClick={() => go(n.id)}
                >
                  <n.icon size={18} strokeWidth={1.7} />
                  <span>{n.label}</span>
                  {n.id === bellTarget &&
                    (n.id === "journal" || n.id === "summary") &&
                    notified.length > 0 && <b>{notified.length}</b>}
                </button>
              ))}
            </Fragment>
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
          {pages.includes("evaluation") && (
            <button
              aria-label="Методика работы"
              title="Методика работы"
              onClick={() => {
                go("evaluation");
              }}
            >
              <CircleHelp size={18} />
              <span>Методика работы</span>
            </button>
          )}
          <div className="sidebar-credit">
            МОСКОЛЛЕКТОР <span>2026</span>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            {user.role_label} <ChevronRight size={13} />
            <strong>{title[0]}</strong>
          </div>
          <div className="topbar-actions">
            <span className="environment-label">
              <i className="dot green" />
              Архив
            </span>
            {(can(user, "notifications.all") ||
              can(user, "notifications.critical")) && (
              <button
                className="notification-button"
                aria-label={`Открыть уведомления: ${notified.length}`}
                title={
                  can(user, "notifications.all")
                    ? "Предупреждения без решения"
                    : "Критические риски"
                }
                onClick={() => {
                  if (can(user, "notifications.all")) {
                    setOnlyWarnings(true);
                    go("journal");
                  } else go("summary");
                }}
              >
                <Bell size={18} />
                {notified.length > 0 && <i />}
              </button>
            )}
            <div className="profile">
              <button
                onClick={() => setProfileOpen(!profileOpen)}
                aria-label="Меню пользователя"
              >
                <span className="avatar">{ROLE_SHORT[user.role]}</span>
                <div>
                  <strong>{user.name}</strong>
                  <small>{user.role_label}</small>
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
                Поток СМВУ
                {streamAlert.asOf
                  ? ` · ${date(streamAlert.asOf, true)}`
                  : ""} · {streamAlert.message}
              </span>
              <button
                className="text-link"
                onClick={() =>
                  go(can(user, "data.import") ? "quality" : "overview")
                }
              >
                Открыть
              </button>
            </div>
          )}
          <div className="page-heading">
            <div>
              <span className="eyebrow">
                {String(navIndex + 1 || 1).padStart(2, "0")} /{" "}
                {user.role_label.toUpperCase()}
              </span>
              <h1>{title[0]}</h1>
              <p>{title[1]}</p>
            </div>
            <div className="heading-actions">
              {page === "overview" &&
                can(user, "decisions.write") &&
                warnings.length > 0 && (
                  <button
                    className="primary-button review-next"
                    onClick={() => setSelected(warnings[0])}
                  >
                    Начать проверку <ArrowRight size={16} />
                  </button>
                )}
              {replay && (
                <button
                  className="secondary-button"
                  onClick={() => setRefresh((r) => r + 1)}
                  aria-label="Обновить данные"
                >
                  <RefreshCw size={16} className={loading ? "spin" : ""} />
                </button>
              )}
              {overview && ["overview", "journal"].includes(page) && (
                <a
                  className="secondary-button"
                  href={`/api/export/forecasts.csv?as_of=${encodeURIComponent(asOf)}`}
                >
                  <ArrowDownToLine size={16} />
                  Экспорт CSV
                </a>
              )}
            </div>
          </div>
          {replay && (
            <div className="replay-bar">
              <div className="replay-title">
                <span className="replay-icon">
                  <Radio size={17} />
                </span>
                <div>
                  <strong>Момент прогноза</strong>
                  <small>МСК</small>
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
                <div className="dashboard-content">
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
                        {num(queueStats.objects)}
                        <small> объектов</small>
                      </strong>
                      <p>
                        {num(queueStats.warnings)} предупреждений на 24 часа ·
                        критических {queueStats.critical}
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
                              stroke="#787878"
                              strokeWidth={1.5}
                              fill="#f0f0f0"
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
                          setOnlyWarnings(true);
                          go("journal");
                        }}
                      >
                        Перейти к проверке
                        <ArrowRight size={14} />
                      </button>
                    </div>
                  </div>
                  <section className="panel overview-queue">
                    <div className="panel-heading">
                      <div>
                        <h2>
                          Очередь проверки{" "}
                          <span className="count-badge">
                            {
                              allForecasts.filter((f) => f.above_threshold)
                                .length
                            }
                          </span>
                        </h2>
                        <span>
                          Приоритетные предупреждения и ближайшие к порогу риски
                        </span>
                      </div>
                      <button
                        className="text-link"
                        onClick={() => go("journal")}
                      >
                        Весь журнал
                        <ArrowRight size={15} />
                      </button>
                    </div>
                    <ForecastTable
                      forecasts={allForecasts.slice(0, 6)}
                      onSelect={setSelected}
                    />
                  </section>
                  <div className="overview-middle">
                    <section className="panel priority-panel">
                      <div className="panel-heading">
                        <div>
                          <h2>По направлениям</h2>
                          <span>Предупреждения на 24 часа</span>
                        </div>
                        <SlidersHorizontal size={17} />
                      </div>
                      <div className="scenario-list">
                        {overview.kinds.map((k) => (
                          <button
                            key={k.id}
                            onClick={() => {
                              setKind(k.id);
                              setOnlyWarnings(true);
                              go("journal");
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
                                  : queueStats.byKind[k.id]
                                    ? "Есть предупреждения"
                                    : "Выше порога нет"}
                              </small>
                            </div>
                            <b
                              className={
                                queueStats.byKind[k.id] ? "has-risk" : ""
                              }
                            >
                              {queueStats.byKind[k.id]}
                            </b>
                            <ChevronRight size={14} />
                          </button>
                        ))}
                      </div>
                    </section>
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
                        {pages.includes("map") && (
                          <button
                            className="text-link"
                            onClick={() => go("map")}
                          >
                            Открыть схему
                            <ArrowRight size={15} />
                          </button>
                        )}
                      </div>
                      {topology && (
                        <NetworkMap
                          topology={topology}
                          forecasts={overview.forecasts}
                          onSelect={setSelected}
                        />
                      )}
                    </section>
                  </div>
                </div>
              )}
              {page === "map" && overview && topology && (
                <section className="panel">
                  <div className="panel-heading">
                    <div>
                      <h2>Объекты и эксплуатационные узлы</h2>
                      <span>Наибольший уровень риска по объекту</span>
                    </div>
                  </div>
                  <NetworkMap
                    topology={topology}
                    forecasts={overview.forecasts}
                    onSelect={setSelected}
                    expanded
                  />
                </section>
              )}
              {page === "journal" && overview && (
                <section className="panel">
                  <div className="journal-toolbar">
                    <div className="search-field">
                      <Search size={17} />
                      <input
                        ref={searchInput}
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
                    {overview.stream && (
                      <select
                        aria-label="Источник данных"
                        value={source}
                        onChange={(e) =>
                          setSource(e.target.value as typeof source)
                        }
                      >
                        <option value="all">Все источники</option>
                        <option value="stream">Поток СМВУ</option>
                        <option value="archive">Архив</option>
                      </select>
                    )}
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
              {page === "equipment" && (
                <EquipmentPage
                  asOf={asOf}
                  forecasts={overview?.forecasts || []}
                  onSelect={setSelected}
                  user={user}
                />
              )}
              {page === "work" && <WorkPage user={user} onChanged={changed} />}
              {page === "verification" && <VerificationPage user={user} />}
              {page === "thresholds" && (
                <ThresholdsPage user={user} onApplied={changed} />
              )}
              {page === "summary" && (
                <SummaryPage
                  asOf={asOf}
                  user={user}
                  onOpenThresholds={() => go("thresholds")}
                />
              )}
              {page === "evaluation" && <EvaluationPage />}
              {page === "quality" && (
                <div className="page-stack">
                  {can(user, "data.import") && <ImportPanel user={user} />}
                  <QualityPage />
                </div>
              )}
              {page === "users" && <UsersPage user={user} />}
              {page === "audit" && <AuditPage />}
              {page === "settings" && <ParametersPage />}
            </>
          )}
        </main>
      </div>
      {selected && (
        <ForecastDrawer
          forecast={selected}
          user={user}
          onClose={closeDrawer}
          onSaved={saved}
        />
      )}
      {alertToast && (
        <div className="alert-toast" role="alert">
          <BellRing size={19} />
          <div>
            <strong>{alertToast.title}</strong>
            {alertToast.items.map((f) => (
              <button
                key={f.id}
                onClick={() => {
                  setSelected(f);
                  setAlertToast(null);
                }}
              >
                {f.object_name} · {kindNames[f.kind]} ·{" "}
                {riskNames[f.risk].toLowerCase()}
              </button>
            ))}
          </div>
          <button
            className="icon-button"
            aria-label="Закрыть уведомление"
            onClick={() => setAlertToast(null)}
          >
            ×
          </button>
        </div>
      )}
      {toast && (
        <div className="toast">
          <ShieldCheck size={18} />
          {toast}
        </div>
      )}
    </div>
  );
}
