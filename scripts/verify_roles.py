"""End-to-end check of the five-role workflow against a running demo server.

dispatcher decides → technician drafts and submits a work order → the emulated
work-order system closes it → analyst verifies the label and freezes it for
retraining, proposes a threshold → unit head approves it and exports reports →
administrator sees every step in the audit log.

Usage: python scripts/verify_roles.py --url http://127.0.0.1:8000
The server must run in demo mode. Work-order steps default to 20 s each.
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

PASSWORD = "contour-demo"


def session(url, username):
    client = httpx.Client(base_url=url, timeout=120)
    response = client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
    response.raise_for_status()
    return client, response.json()["user"]


def check(condition, message):
    print(("OK   " if condition else "FAIL ") + message)
    if not condition:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--wait", type=float, default=75, help="Seconds to wait for the emulated order to close")
    args = parser.parse_args()

    clients = {}
    for username in ("dispatcher", "technician", "analyst", "manager", "admin"):
        clients[username], user = session(args.url, username)
        check(user["role"] == username, f"{username}: вход, роль {user['role_label']}")

    dispatcher = clients["dispatcher"]
    moment = dispatcher.get("/api/replay").json()["default"]
    overview = dispatcher.get("/api/overview", params={"as_of": moment}).json()
    target = next(
        (f for f in overview["forecasts"] if f["above_threshold"] and not f["decision"]),
        None,
    ) or next(f for f in overview["forecasts"] if not f["decision"])
    check(target["risk"] in ("critical", "high", "watch", "low"), f"уровень риска: {target['risk_label']}")
    detail = dispatcher.get(f"/api/forecast/{target['object_id']}/{target['kind']}", params={"as_of": moment}).json()
    check(len(detail["recommendation_details"]) > 0, "рекомендации с основаниями в карточке")
    check(
        {s["id"] for s in detail["verification"]["external_sources"]} >= {"cameras"},
        "блок проверки ситуации в карточке",
    )
    body = {
        "prediction_id": target["id"],
        "object_id": target["object_id"],
        "kind": target["kind"],
        "as_of": target["as_of"],
        "action": "inspect",
        "reason": "sensor_pattern",
        "comment": "Проверка роли: направить бригаду на проверку",
    }
    check(dispatcher.post("/api/decisions", json=body).status_code == 200, "диспетчер сохранил решение")
    check(
        clients["technician"].post("/api/decisions", json=body).status_code == 403,
        "техперсонал не может принять решение",
    )

    technician = clients["technician"]
    task = next(t for t in technician.get("/api/tasks").json() if t["prediction_id"] == target["id"])
    order = technician.post("/api/work-orders", json={"decision_id": task["id"]})
    check(order.status_code == 201, "техперсонал сформировал черновик заявки")
    order = order.json()
    submitted = technician.post(f"/api/work-orders/{order['id']}/submit").json()
    check(submitted["status"] == "submitted", f"заявка {submitted['number']} передана во внешнюю систему")
    equipment = technician.get("/api/equipment", params={"as_of": moment}).json()
    check(equipment["totals"]["objects"] > 0, f"состояние оборудования: {equipment['totals']}")

    deadline = time.monotonic() + args.wait
    status = submitted["status"]
    while time.monotonic() < deadline and status != "done":
        time.sleep(5)
        technician.post("/api/work-orders/sync")
        status = next(o for o in technician.get("/api/work-orders").json() if o["id"] == order["id"])["status"]
    closed = next(o for o in technician.get("/api/work-orders").json() if o["id"] == order["id"])
    check(closed["status"] == "done", f"эмулятор закрыл заявку, итог: {closed['outcome_label']}")

    analyst = clients["analyst"]
    labels = analyst.get("/api/labels").json()["rows"]
    row = next(r for r in labels if r["prediction_id"] == target["id"])
    check(row["suggested_label"] is not None, f"предложена метка {row['suggested_label']}: {row['label_basis']}")
    check(
        analyst.put(
            f"/api/labels/{row['id']}",
            json={"verdict": "accepted", "label": row["suggested_label"], "note": "Проверка роли"},
        ).status_code
        == 200,
        "аналитик принял метку",
    )
    retrain = analyst.post("/api/retraining")
    check(retrain.status_code == 201, f"набор для дообучения: {retrain.json().get('file_name')}")
    settings = analyst.get("/api/settings").json()["thresholds"]
    proposed = round(min(0.95, settings["fire"] + 0.05), 4)
    preview = analyst.get("/api/thresholds/preview", params={"kind": "fire", "threshold": proposed}).json()
    check(
        preview["proposed"]["alerts"] <= preview["current"]["alerts"],
        f"предпросмотр порога: предупреждений {preview['current']['alerts']} → {preview['proposed']['alerts']}",
    )
    proposal = analyst.post(
        "/api/thresholds/proposals",
        json={"kind": "fire", "threshold": proposed, "rationale": "Проверка роли: снизить нагрузку на смену"},
    )
    check(proposal.status_code == 201, "аналитик предложил порог")
    proposal = proposal.json()
    check(
        analyst.post(f"/api/thresholds/proposals/{proposal['id']}/approve", json={}).status_code == 403,
        "аналитик не может утвердить свой порог",
    )

    manager = clients["manager"]
    approved = manager.post(f"/api/thresholds/proposals/{proposal['id']}/approve", json={"note": "Согласовано"})
    check(approved.status_code == 200, "руководитель утвердил порог")
    check(manager.get("/api/settings").json()["thresholds"]["fire"] == proposed, "порог применён")
    summary = manager.get("/api/summary", params={"as_of": moment}).json()
    check(summary["workload"]["decisions"] >= 1, f"сводка: {summary['totals']}")
    xlsx = manager.get("/api/reports/summary.xlsx", params={"as_of": moment})
    pdf = manager.get("/api/reports/summary.pdf", params={"as_of": moment})
    check(xlsx.status_code == 200 and xlsx.content[:2] == b"PK", f"отчёт XLSX, {len(xlsx.content)} байт")
    check(pdf.status_code == 200 and pdf.content[:4] == b"%PDF", f"отчёт PDF, {len(pdf.content)} байт")
    check(
        manager.put("/api/settings", json={"thresholds": {"fire": settings["fire"]}}).status_code == 200,
        "руководитель вернул исходный порог",
    )

    admin = clients["admin"]
    check(admin.get("/api/users").status_code == 200, "администратор видит пользователей")
    check(admin.put("/api/settings", json={"thresholds": {"fire": 0.5}}).status_code == 403, "администратор не меняет пороги")
    actions = {r["action"] for r in admin.get("/api/audit").json()}
    expected = {
        "decision_saved",
        "work_order_drafted",
        "work_order_submitted",
        "label_reviewed",
        "retraining_requested",
        "threshold_proposed",
        "threshold_approved",
        "report_exported",
    }
    check(expected <= actions, "все шаги сценария записаны в журнал аудита")
    print("Сценарий пяти ролей пройден.")


if __name__ == "__main__":
    main()
