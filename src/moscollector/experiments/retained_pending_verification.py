"""Accept only a complete, immutable replay of the delayed-owner experiment."""

from pathlib import Path

from moscollector.experiments.flood_research import validate_hashes
from moscollector.experiments.goal90_research import read
from moscollector.experiments.retained_pending_research import ARMS, PARENT, PLAN, ROOT, compare
from moscollector.prepare import sha256


def verified_evidence(root=ROOT):
    proof = read(root / "policy-replay.json")
    validate_hashes(proof)
    plan, report = read(root / "plan.json"), read(root / "report.json")
    if (
        proof["status"] != "exact_policy_replay_passed"
        or proof["new_model_fits"] != 0
        or proof["report"] != report
        or proof["code_hashes"] != plan["code_hashes"]
        or set(report) != set(PLAN["kinds"])
        or any(proof["source_hashes"].get(p) != h for p, h in plan["source_hashes"].items())
    ):
        raise ValueError("Incomplete or mismatched retained-warning proof")
    required = {str(p) for p in root.rglob("*") if p.is_file() and p.name != "policy-replay.json"}
    if not required.issubset(proof["source_hashes"]):
        raise ValueError("Retained proof omits result files")
    for kind, outcome in report.items():
        rows = outcome["periods"]
        selection = compare(rows)
        expected_status = (
            "requires_separate_confirmation" if selection["passed_screen"] else "screen_rejected"
        )
        if (
            [r["fold"] for r in rows] != PLAN["folds"][kind]
            or outcome["selection"] != selection
            or outcome["status"] != expected_status
            or outcome["serving_changed"] is not False
            or outcome["goal_achieved"] is not False
        ):
            raise ValueError("Changed retained selection, coverage or release claim")
        for row in rows:
            directory = root / kind / row["fold"]
            parent_path = PARENT / kind / row["fold"] / "result.json"
            parent = read(parent_path)
            references = dict(parent["references"])
            if kind == "flood":
                references["deadline_uniform_old"] = parent["arms"]["uniform_old"]["scores"]
            outputs = {
                str(directory / f"{arm}-{part}.parquet") for arm in ARMS for part in ("policy", "test")
            }
            outputs.add(str(directory / "retained-frontier.json"))
            if (
                row != read(directory / "result.json")
                or row["kind"] != kind
                or row["plan_sha256"] != sha256(root / "plan.json")
                or row["parent_sha256"] != sha256(parent_path)
                or row["references"] != references
                or row["arms"]["legacy_control"]["scores"] != parent["arms"]["count_control"]["scores"]
                or set(row["outputs"]) != outputs
                or len(read(directory / "retained-frontier.json")) != (120 if kind == "flood" else 456)
            ):
                raise ValueError("Changed retained output signature")
            for path, digest in row["outputs"].items():
                if sha256(Path(path)) != digest:
                    raise ValueError("Changed retained forecast/frontier")
    if report["flood"]["selection"]["scores"]["retained_selected"]["eligible_episodes"] != 40:
        raise ValueError("Changed retained flood cohort")
    return proof


if __name__ == "__main__":
    print(verified_evidence()["status"])
