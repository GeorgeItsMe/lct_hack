"""Draft repair requests and the emulated external work-management system.

Section 6 of the task: work orders live in the customer's own system; this service only
submits drafts and reads statuses back. The organizers confirmed that a mock REST
integration is sufficient, provided it behaves plausibly. The emulator therefore
advances a submitted order by elapsed time and takes the outcome from the archive
(whether the forecast episode really started), never from random numbers.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from moscollector.domain import ACTIONS, KIND_LABELS, WORK_OUTCOMES, WORK_STATUSES

PRIORITIES = {"urgent": "Срочно", "high": "Высокий", "normal": "Обычный"}
EMULATOR_STEP_SECONDS = float(os.getenv("CONTOUR_WORKORDER_STEP_SECONDS", "20"))
EMULATOR_FLOW = ("submitted", "accepted", "in_progress", "done")


def now():
    return datetime.now(UTC).replace(tzinfo=None)


def order_number(order_id: int, created: datetime) -> str:
    return f"З-{created.year}-{order_id:05d}"


def priority_for(risk: str | None) -> str:
    return {"critical": "urgent", "high": "high"}.get(risk or "", "normal")


def draft_text(forecast: dict, action: str | None, recommendations: list[str], factors: list[dict]):
    """Title and body of a draft built only from the forecast card the technician sees."""
    work = ACTIONS.get(action or "", "Работы по состоянию оборудования")
    title = f"{work}: {forecast['object_name']} — {KIND_LABELS[forecast['kind']].lower()}"
    lines = [
        f"Объект: {forecast['object_name']} (#{forecast['object_id']})",
        f"Тип риска: {KIND_LABELS[forecast['kind']]}",
        f"Прогноз на {forecast['as_of'][:16].replace('T', ' ')} МСК: "
        f"вероятность {forecast['probability'] * 100:.1f}% на 24 часа "
        f"(порог {min(forecast['threshold'], 1) * 100:.1f}%)",
    ]
    if action:
        lines.append(f"Решение диспетчера: {ACTIONS[action]}")
    if factors:
        lines.append("Основные факторы прогноза:")
        lines += [f"  • {f['label']}" for f in factors[:3]]
    if recommendations:
        lines.append("Рекомендуемые работы:")
        lines += [f"  {i}. {text}" for i, text in enumerate(recommendations[:5], 1)]
    lines.append("Черновик сформирован сервисом прогнозирования; решение о работах принимает исполнитель.")
    return title[:200], "\n".join(lines)


def emulated_status(submitted_at: datetime | None, at: datetime | None = None) -> str:
    """Status the external system reports for an order submitted at the given time."""
    if submitted_at is None:
        return "draft"
    elapsed = ((at or now()) - submitted_at).total_seconds()
    step = min(len(EMULATOR_FLOW) - 1, int(max(0.0, elapsed) // max(EMULATOR_STEP_SECONDS, 0.001)))
    return EMULATOR_FLOW[step]


def archive_outcome(kind: str, occurred: bool | None, fault_signals: bool) -> str | None:
    """Outcome recorded when an order closes: follows what the journal shows afterwards."""
    if occurred is None:
        return None
    if occurred:
        return "confirmed"
    return "sensor_fault" if fault_signals else "not_confirmed"


def serialize(order, object_name: str | None = None) -> dict:
    return {
        "id": order.id,
        "number": order.number,
        "decision_id": order.decision_id,
        "prediction_id": order.prediction_id,
        "object_id": order.object_id,
        "object_name": object_name or str(order.object_id),
        "kind": order.kind,
        "forecast_at": order.forecast_at,
        "title": order.title,
        "description": order.description,
        "priority": order.priority,
        "priority_label": PRIORITIES.get(order.priority, order.priority),
        "status": order.status,
        "status_label": WORK_STATUSES.get(order.status, order.status),
        "outcome": order.outcome,
        "outcome_label": WORK_OUTCOMES.get(order.outcome) if order.outcome else None,
        "external_id": order.external_id,
        "created_by": order.created_by,
        "created_at": order.created_at.isoformat(),
        "updated_at": order.updated_at.isoformat(),
        "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
        "synced_at": order.synced_at.isoformat() if order.synced_at else None,
    }


def suggested_label(kind: str, action: str, outcome: str | None, archive_occurred: bool | None):
    """Label proposed to the analyst, with the evidence it rests on.

    A closed work order is the strongest evidence; a dispatcher's "false alarm" comes next;
    without either, the archive record of the following 24 hours is used.
    """
    if outcome == "confirmed":
        return 1, "Заявка закрыта: событие подтвердилось"
    if outcome == "sensor_fault":
        return (
            (1, "Заявка закрыта: подтверждён отказ оборудования")
            if kind == "fault"
            else (
                0,
                "Заявка закрыта: сбой датчика, а не событие",
            )
        )
    if outcome == "not_confirmed":
        return 0, "Заявка закрыта: событие не подтвердилось"
    if action == "false_alarm":
        return 0, "Диспетчер отметил ложное срабатывание"
    if archive_occurred is not None:
        return (
            (1, "По журналу эпизод начался в течение 24 часов")
            if archive_occurred
            else (
                0,
                "По журналу эпизода в течение 24 часов не было",
            )
        )
    return None, "Итог неизвестен"
