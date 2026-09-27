import {
  ClipboardCheck,
  Database,
  Inbox,
  FlaskConical,
  Gauge,
  History,
  LayoutDashboard,
  ListChecks,
  Map,
  PieChart,
  ScrollText,
  SlidersHorizontal,
  Users,
  Wrench,
} from "lucide-react";
import type { Page, Permission, Role, User } from "./types";

export const can = (user: User | null, permission: Permission) =>
  !!user?.permissions.includes(permission);

export type NavItem = {
  id: Page;
  label: string;
  icon: typeof Map;
};
type NavSection = { label: string; items: NavItem[] };

const item = (id: Page, label: string, icon: typeof Map): NavItem => ({
  id,
  label,
  icon,
});

/* Section 16 of the task: each role sees only what its work needs. */
export const NAVIGATION: Record<Role, NavSection[]> = {
  dispatcher: [
    {
      label: "РАБОТА С ПРЕДУПРЕЖДЕНИЯМИ",
      items: [
        item("overview", "Очередь проверки", LayoutDashboard),
        item("map", "Схема объектов", Map),
        item("journal", "Журнал прогнозов", ListChecks),
        item("decisions", "Журнал отработки", History),
        item("incoming", "Поступившие данные", Inbox),
      ],
    },
  ],
  technician: [
    {
      label: "ОБОРУДОВАНИЕ И РАБОТЫ",
      items: [
        item("equipment", "Состояние оборудования", Gauge),
        item("work", "Задачи и заявки", Wrench),
        item("map", "Схема объектов", Map),
        item("journal", "Журнал прогнозов", ListChecks),
      ],
    },
  ],
  analyst: [
    {
      label: "ДОСТОВЕРНОСТЬ ДАННЫХ",
      items: [
        item("verification", "Верификация данных", ClipboardCheck),
        item("quality", "Данные и загрузка", Database),
        item("evaluation", "Проверка модели", FlaskConical),
        item("thresholds", "Пороги предупреждений", SlidersHorizontal),
      ],
    },
    {
      label: "ОБЗОР",
      items: [
        item("summary", "Сводка подразделения", PieChart),
        item("journal", "Журнал прогнозов", ListChecks),
        item("decisions", "Журнал отработки", History),
      ],
    },
  ],
  manager: [
    {
      label: "УПРАВЛЕНИЕ ПОДРАЗДЕЛЕНИЕМ",
      items: [
        item("summary", "Сводка и отчёты", PieChart),
        item("thresholds", "Пороги предупреждений", SlidersHorizontal),
        item("decisions", "Журнал отработки", History),
        item("map", "Схема объектов", Map),
      ],
    },
    {
      label: "АНАЛИТИКА",
      items: [
        item("evaluation", "Проверка модели", FlaskConical),
        item("quality", "Качество данных", Database),
        item("journal", "Журнал прогнозов", ListChecks),
      ],
    },
  ],
  admin: [
    {
      label: "АДМИНИСТРИРОВАНИЕ",
      items: [
        item("users", "Пользователи и роли", Users),
        item("audit", "Журнал аудита", ScrollText),
        item("settings", "Интеграции и параметры", SlidersHorizontal),
      ],
    },
    {
      label: "ПРОСМОТР",
      items: [
        item("summary", "Сводка подразделения", PieChart),
        item("overview", "Прогнозы", LayoutDashboard),
        item("journal", "Журнал прогнозов", ListChecks),
      ],
    },
  ],
};

export const HOME: Record<Role, Page> = {
  dispatcher: "overview",
  technician: "equipment",
  analyst: "verification",
  manager: "summary",
  admin: "users",
};

export function allowedPages(role: Role): Page[] {
  return NAVIGATION[role].flatMap((s) => s.items.map((i) => i.id));
}

export const TITLES: Record<Page, [string, string]> = {
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
    "Журнал отработки",
    "Решения по прогнозам, статусы работ и их результаты",
  ],
  incoming: [
    "Поступившие данные",
    "Прогнозы по новым пакетам телеметрии и решения по ним",
  ],
  equipment: [
    "Состояние оборудования",
    "Прогноз отказов и неисправности, зафиксированные в журнале датчиков",
  ],
  work: [
    "Задачи и заявки",
    "Работы по решениям диспетчера и заявки во внешней системе учёта",
  ],
  verification: [
    "Верификация данных",
    "Какие решения и итоги работ можно использовать для дообучения",
  ],
  thresholds: [
    "Пороги предупреждений",
    "Аналитик предлагает порог, руководитель подразделения утверждает",
  ],
  summary: [
    "Сводка подразделения",
    "Риски по узлам, отработка предупреждений, сезонность и отчёты",
  ],
  evaluation: [
    "Проверка модели",
    "Прогнозы сопоставлены с событиями отложенного периода",
  ],
  quality: [
    "Данные и их качество",
    "От исходных журналов к воспроизводимым эпизодам",
  ],
  users: ["Пользователи и роли", "Учётные записи, роли и зоны ответственности"],
  audit: ["Журнал аудита", "Все действия пользователей с автором и временем"],
  settings: [
    "Интеграции и параметры",
    "Источники данных, режим работы и параметры прогноза",
  ],
};

/* Pages that follow the archive replay moment. */
export const REPLAY_PAGES: Page[] = [
  "overview",
  "map",
  "journal",
  "equipment",
  "summary",
];

export const ROLE_SHORT: Record<Role, string> = {
  dispatcher: "ДП",
  technician: "ТП",
  analyst: "АН",
  manager: "РП",
  admin: "АД",
};

export const DEMO_ROLES: { username: string; label: string; hint: string }[] = [
  {
    username: "dispatcher",
    label: "Диспетчер ОДС",
    hint: "Разбирает предупреждения и принимает решения",
  },
  {
    username: "technician",
    label: "Технический персонал",
    hint: "Состояние оборудования, задачи и заявки",
  },
  {
    username: "analyst",
    label: "Аналитик",
    hint: "Верификация данных, загрузка, модель и пороги",
  },
  {
    username: "manager",
    label: "Руководитель подразделения",
    hint: "Сводка, отчёты и утверждение порогов",
  },
];
