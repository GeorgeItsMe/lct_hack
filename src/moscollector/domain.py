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
    {"id": "sensor_fault", "label": "Сбой датчика"},
    {"id": "insufficient_evidence", "label": "Недостаточно данных"},
    {"id": "false_signal", "label": "Ложное срабатывание"},
    {"id": "other", "label": "Другая причина"},
]
# Wording follows the dispatcher scenarios in section 12 of the task.
ACTIONS = {
    "dispatch": "Выезд бригады",
    "inspect": "Направить бригаду на проверку",
    "monitor": "Мониторинг ситуации",
    "false_alarm": "Ложное срабатывание",
    "maintenance": "Запланировать ТО",
}
# Decisions that create work for technical staff.
WORK_ACTIONS = ("dispatch", "inspect", "maintenance")
WORK_STATUSES = {
    "draft": "Черновик",
    "submitted": "Передана в систему заявок",
    "accepted": "Принята",
    "in_progress": "В работе",
    "done": "Выполнена",
}
WORK_OUTCOMES = {
    "confirmed": "Событие подтвердилось",
    "not_confirmed": "Не подтвердилось",
    "sensor_fault": "Сбой датчика или оборудования",
}
# Levels are defined relative to the kind's warning threshold, so they stay meaningful
# when an approved threshold changes.
RISK_LEVELS = {
    "critical": "Критический",
    "high": "Высокий",
    "watch": "Повышенный",
    "low": "Низкий",
}


def risk_level(probability: float, threshold: float) -> str:
    if threshold > 1:
        return "low" if probability < 0.5 else "watch"
    if probability >= threshold:
        return "critical" if probability >= min(0.95, 2 * threshold) else "high"
    return "watch" if probability >= 0.5 * threshold else "low"


FAULT_STATES = {"Неисправен", "Обесточен", "Отключено устройство", "Питание от батарей", "Батарея разряжена"}
SENTINEL_PREFIX = "01.01.1970"


def is_fault_state(value) -> bool:
    text = str(value or "")
    return text in FAULT_STATES or text.startswith(SENTINEL_PREFIX)


def data_recommendations(kind: str, events: list[dict]) -> list[dict]:
    """Recommendations derived from the object's recorded signals before the forecast.

    Rules only restate what the journal shows and which check follows from it; generic
    guidance for the risk type is appended so a card is never empty.
    """
    rules = []
    seen = set()

    def add(key, text, basis):
        if key not in seen:
            seen.add(key)
            rules.append({"text": text, "basis": basis, "source": "signals"})

    fault_channels = {e["channel_id"] for e in events if is_fault_state(e.get("value"))}
    if len(fault_channels) >= 5:
        add(
            "cascade",
            f"Сбой одновременно на {len(fault_channels)} каналах: в первую очередь проверить шлейф "
            "и контроллер участка.",
            "Неисправность сразу многих каналов обычно указывает на общую линию, а не на датчики.",
        )
    for e in events:
        value = str(e.get("value") or "")
        sensor = e.get("sensor_type") or e.get("sensor_name") or "датчик"
        if value in ("Питание от батарей", "Батарея разряжена"):
            add(
                "battery",
                "ИБП работает от батарей: проверить внешнее питание и заряд АКБ, при разряде заменить батарею.",
                f"Сообщение «{value}» ({sensor}).",
            )
        elif value == "Обесточен":
            add(
                f"power:{sensor}",
                f"Канал «{sensor}» обесточен: проверить питание и автоматы на участке.",
                f"Сообщение «{value}».",
            )
        elif value == "Неисправен":
            add(
                f"fault:{sensor}",
                f"«{sensor}» сообщает о неисправности: проверить датчик и линию связи, при повторе заменить.",
                f"Сообщение «{value}».",
            )
        elif value == "Отключено устройство":
            add(
                f"off:{sensor}",
                f"«{sensor}»: устройство отключено. Уточнить, идут ли работы; если нет, восстановить подключение.",
                f"Сообщение «{value}».",
            )
        elif value.startswith(SENTINEL_PREFIX):
            add(
                "sentinel",
                "В показаниях служебные коды даты 1970 года: проверить часы и прошивку контроллера.",
                f"Значение «{value}» ({sensor}) — технический код, не измерение.",
            )
        elif value == "Обнаружен дым":
            add(
                "smoke",
                "Зафиксирован дым: сверить с тепловыми датчиками и газоанализатором; при подтверждении "
                "действовать по регламенту ОДС.",
                f"Сообщение «{value}» ({sensor}).",
            )
        elif value == "Обнаружен газ":
            add(
                "gas",
                "Сработал газоанализатор: проверить концентрацию и вентиляцию; допуск в коллектор только "
                "после замера.",
                f"Сообщение «{value}» ({sensor}).",
            )
        elif value in ("Затоплен", "Работают все насосы в АНС"):
            add(
                "water",
                "Признаки воды: проверить насосную станцию, приямки и работу всех насосов.",
                f"Сообщение «{value}» ({sensor}).",
            )
        elif value in ("Не замкнут", "Обнаружено движение", "Рычаг сдернут") and e.get("alarm"):
            add(
                f"access:{sensor}",
                f"Сработал «{sensor}»: сверить с допуском и разрешёнными работами; без допуска нужна "
                "проверка на месте.",
                f"Тревожное сообщение «{value}».",
            )
    generic = [
        {"text": text, "basis": "Общий порядок проверки для этого типа риска.", "source": "general"}
        for text in RECOMMENDATIONS[kind]
    ]
    return rules[:4] + generic


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
