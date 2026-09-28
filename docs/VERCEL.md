# Развёртывание на Vercel

Репозиторий содержит FastAPI-функцию `api/index.py`, React-сборку и уменьшенный
набор аналитических файлов в `vercel_runtime/`. React собирается в статику `web/dist` и
раздаётся CDN, запросы `/api/*` переписываются на Python Function (`vercel.json`).
Зависимости ставятся из `uv.lock`.

## Подготовка репозитория

После обновления моделей или архивных данных пересоберите runtime и закоммитьте
получившиеся файлы:

```bash
.venv/bin/python scripts/build_vercel_runtime.py
npm --prefix web run build
```

Runtime предназначен для архивного демо с 1 мая 2026 года. В нём остаются
исходные сообщения начиная с 30 апреля, поэтому карточки прогноза показывают
строго предшествующие 24 часа. Полный датасет и исследовательские артефакты в
deployment не входят.

## Настройки проекта

1. Импортируйте GitHub-репозиторий в Vercel, оставьте Root Directory равным
   корню репозитория. Framework, Build и Output вручную не переопределяйте —
   они заданы в `vercel.json`.
2. Подключите PostgreSQL через Marketplace (Neon, Supabase или другой
   совместимый сервис) и добавьте `DATABASE_URL`. Поддерживаются URL вида
   `postgres://`, `postgresql://` и `postgresql+psycopg://`.
3. Добавьте переменные окружения:

```text
CONTOUR_DEMO=true
VERCEL_SUPPORT_LARGE_FUNCTIONS=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
```

`CONTOUR_DEMO=true` создаёт пять демонстрационных учётных записей (по одной на роль) с паролем
`contour-demo`. Для закрытого стенда задайте `CONTOUR_DEMO=false` и
`CONTOUR_ADMIN_PASSWORD` длиной не менее 12 символов в новой пустой базе.

`VERCEL_SUPPORT_LARGE_FUNCTIONS=1` обязателен. Стандартный лимит Python Function —
500 МБ в распакованном виде. Linux-колёса зависимостей занимают около 870 МБ: из
них 264 МБ приходится на `_catboost.so` и 151 МБ на PyArrow. Ещё 77 МБ занимает
`vercel_runtime/`. Large Functions (бета, до 5 ГБ) требует Fluid compute; в
`vercel.json` он включён. Итоговая функция около 960 МБ. Ей нужно примерно 470 МБ памяти, поэтому лимита тарифа Hobby (2 ГБ) хватает.

## Проверка

После preview deployment проверьте:

```bash
curl -fsS https://<preview-domain>/api/health
curl -fsS https://<preview-domain>/api/ready
```

`/api/health` должен показать `model_ready: true`, `serverless_mode: true` и
`persistent_database: true`. `/api/ready` должен вернуть активную модель
`op-d87946a7b7fd` и 95 объектов.

## Ограничения Vercel-стенда

Архивный прогноз, объяснения, ретроспектива, экспорт и решения диспетчера
работают полностью. Импорт телеметрии и накопительный поток отключены: их
worker использует отдельный процесс и локальную файловую очередь. Эти функции
остаются в Docker-развёртывании. Без `DATABASE_URL` приложение использует
временную SQLite в `/tmp`; это годится только для проверки запуска, поскольку
сеансы и решения могут исчезнуть при новом экземпляре Function.
