"""Role model (RBAC). One table drives server checks and what the interface shows.

Users come from section 3 of the task (dispatchers, technical staff, unit heads) and
section 12 (analyst responsible for data verification). The administrator is a service
role required by section 11 (RBAC, directory integration, audit of all actions).
"""

from __future__ import annotations

ROLES = {
    "dispatcher": "Диспетчер",
    "technician": "Технический персонал",
    "analyst": "Аналитик",
    "manager": "Руководитель подразделения",
    "admin": "Администратор",
}

# Permission -> roles. Admin sees forecasts read-only and never takes operational decisions.
PERMISSIONS = {
    "forecasts.view": {"dispatcher", "technician", "analyst", "manager", "admin"},
    "decisions.view": {"dispatcher", "technician", "analyst", "manager", "admin"},
    "decisions.write": {"dispatcher"},
    "notifications.all": {"dispatcher"},
    "notifications.critical": {"manager"},
    "equipment.view": {"dispatcher", "technician", "analyst", "manager", "admin"},
    "work.view": {"dispatcher", "technician", "analyst", "manager"},
    "work.edit": {"technician"},
    "data.import": {"analyst"},
    "labels.verify": {"analyst"},
    "retrain.run": {"analyst"},
    "model.view": {"analyst", "manager"},
    "summary.view": {"analyst", "manager", "admin"},
    "reports.export": {"analyst", "manager"},
    "thresholds.propose": {"analyst"},
    "thresholds.approve": {"manager"},
    "users.manage": {"admin"},
    "audit.view": {"admin"},
}

DENIED = {
    "decisions.write": "Решение по прогнозу фиксирует диспетчер",
    "work.edit": "Заявки формирует технический персонал",
    "data.import": "Загрузка данных доступна аналитику",
    "labels.verify": "Верификация данных доступна аналитику",
    "retrain.run": "Дообучение запускает аналитик",
    "model.view": "Проверка модели доступна аналитику и руководителю",
    "summary.view": "Сводка доступна руководителю, аналитику и администратору",
    "reports.export": "Отчёты выгружают аналитик и руководитель",
    "thresholds.propose": "Предлагать пороги может аналитик",
    "thresholds.approve": "Утверждать пороги может руководитель подразделения",
    "users.manage": "Нужна роль администратора",
    "audit.view": "Нужна роль администратора",
    "work.view": "Статусы работ недоступны для этой роли",
}

# Demo accounts are local-only; production refuses to start while they exist.
DEMO_ACCOUNTS = [
    ("dispatcher", "Диспетчер ОДС", "dispatcher"),
    ("technician", "Технический персонал", "technician"),
    ("analyst", "Аналитик", "analyst"),
    ("manager", "Руководитель подразделения", "manager"),
    ("admin", "Администратор", "admin"),
]


def allowed(role: str, permission: str) -> bool:
    return role in PERMISSIONS[permission]


def permissions_for(role: str) -> list[str]:
    return sorted(p for p, roles in PERMISSIONS.items() if role in roles)
