"""Transparent operational recommendations; they never issue equipment commands."""

KIND_LABELS = {
    "fault": "Неисправность оборудования",
    "fire": "Пожарный сигнал",
    "flood": "Сигнал подтопления",
    "access": "Охранный сигнал",
}
KIND_SHORT = {"fault": "Оборудование", "fire": "Пожарный риск", "flood": "Подтопление", "access": "Доступ"}
RECOMMENDATIONS = {
    "fault": [
        "Проверить питание и связь группы каналов на объекте.",
        "Сопоставить повторные неисправности с плановыми работами.",
        "Подготовить проверку оборудования; решение о выезде принимает диспетчер.",
    ],
    "fire": [
        "Проверить показания дыма и температуры на соседних каналах.",
        "Уточнить наличие огневых или регламентных работ.",
        "При подтверждении опасности действовать по действующему регламенту ОДС.",
    ],
    "flood": [
        "Проверить состояние насосов и сигнализаторов затопления.",
        "Уточнить сведения об осадках и работах на сетях.",
        "При подтверждении признаков воды подготовить осмотр объекта.",
    ],
    "access": [
        "Проверить режим охраны и разрешённые работы.",
        "Сопоставить контактные и объёмные датчики, при наличии — видеонаблюдение.",
        "Зафиксировать результат проверки и необходимость выезда.",
    ],
}
REASONS = [
    {"id": "sensor_pattern", "label": "Повторные отклонения датчиков"},
    {"id": "confirmed_remotely", "label": "Подтверждено дополнительной проверкой"},
    {"id": "planned_work", "label": "Плановые работы"},
    {"id": "maintenance_test", "label": "Проверка оборудования"},
    {"id": "insufficient_evidence", "label": "Недостаточно данных"},
    {"id": "false_signal", "label": "Ложное срабатывание"},
    {"id": "other", "label": "Другая причина"},
]
ACTIONS = {
    "dispatch": "Направить бригаду",
    "monitor": "Продолжить наблюдение",
    "false_alarm": "Ложное срабатывание",
    "maintenance": "Запланировать ТО",
}
FEATURE_LABELS = {
    "events": "Изменения показаний",
    "alarms": "Тревожные сообщения",
    "fault_reports": "Сообщения о неисправности",
    "power_reports": "Отклонения питания",
    "unknown_reports": "Неопределённые состояния",
    "smoke_reports": "Пожарные сигналы",
    "flood_reports": "Сигналы воды",
    "access_reports": "Охранные сигналы",
    "pump_switches": "Переключения насосов",
    "ambiguous": "Конфликтующие состояния",
    "technical_codes": "Служебные коды",
    "reporting_channels_24h": "Каналы с обновлениями за сутки",
    "temperature": "Последняя средняя температура",
    "temperature_max": "Последний максимум температуры",
    "gas": "Последнее среднее показание газовых датчиков",
    "hour": "Час суток",
    "day_of_week": "День недели",
    "month": "Месяц",
    "weekend": "Выходной день",
    "channel_count": "Каналы в справочнике",
    "object_id": "Исторический профиль объекта",
    "parent_id": "Эксплуатационный узел",
    "object_kind": "Тип объекта",
    "temperature_channels": "Температурные каналы",
    "smoke_channels": "Дымовые каналы",
    "pump_channels": "Каналы насосов",
    "water_channels": "Каналы затопления",
}


def feature_label(name):
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("past_"):
        parts = name.split("_")
        kind = KIND_LABELS.get(parts[1], parts[1]).lower()
        if "recency" in name:
            return f"Часов с подтверждения эпизода: {kind}"
        return f"Прошлые эпизоды ({kind}) за {parts[-1][:-1]} ч"
    if name.endswith("_recency_h"):
        return f"Давность сообщений, ч: {FEATURE_LABELS.get(name[:-10], name)}"
    if name.endswith("_burst"):
        return f"Рост частоты за 6 ч: {FEATURE_LABELS.get(name[:-6], name)}"
    for suffix in ("_1h", "_6h", "_24h", "_168h"):
        if name.endswith(suffix):
            return f"{FEATURE_LABELS.get(name[: -len(suffix)], name)} за {suffix[1:-1]} ч"
    if name.endswith("_age_h"):
        return f"Давность: {FEATURE_LABELS.get(name[:-6], name)}"
    if name.endswith("_change_24h"):
        return f"Изменение за сутки: {FEATURE_LABELS.get(name[:-11], name)}"
    return name
