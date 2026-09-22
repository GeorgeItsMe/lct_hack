"""Read only a complete, hash-checked temporal-count weight-replay proof."""

from pathlib import Path

from moscollector.flood_research import FIVE, validate_hashes
from moscollector.goal90_research import pooled, read
from moscollector.prepare import sha256
from moscollector.temporal_count_research import KINDS, PLAN, ROOT, screen


def verified_evidence(root=ROOT):
    proof = read(root / "weight-replay.json")
    validate_hashes(proof)
    plan = read(root / "plan.json")
    report = read(root / "report.json")
    if (
        proof["status"] != "exact_replay_passed"
        or set(report) != set(KINDS)
        or proof["report"] != report
        or proof["code_hashes"] != plan["code_hashes"]
        or any(proof["source_hashes"].get(p) != h for p, h in plan["source_hashes"].items())
    ):
        raise ValueError("Incomplete or mismatched temporal study proof")
    required = {str(p) for p in root.rglob("*") if p.is_file() and p.name != "weight-replay.json"}
    if not required.issubset(proof["source_hashes"]):
        raise ValueError("Temporal proof omits local result files")
    expected = []
    for kind in KINDS:
        row = report[kind]
        first = row["periods"] if kind == "flood" else row["periods"][:2]
        selection = screen(first)
        if row["selection"] != selection or row["serving_changed"] or row["goal_achieved"]:
            raise ValueError("Changed temporal selection or unsupported promotion")
        folds = list(FIVE) if kind == "flood" or selection["passed_screen"] else ["screen_1", "screen_2"]
        if [p["fold"] for p in row["periods"]] != folds:
            raise ValueError("Missing temporal evaluation")
        for result in row["periods"]:
            fold = result["fold"]
            directory = root / kind / fold
            fit = read(directory / "fit.json")
            expected_outputs = {
                str(directory / f"{arm}-{part}.parquet")
                for arm in PLAN["arms"]
                for part in ("calibration", "policy", "test")
            } | {str(directory / f"{arm}-frontier.json") for arm in PLAN["arms"]}
            if (
                result != read(directory / "result.json")
                or fit["signature"] != result["signature"]
                or result["signature"]["plan_sha256"] != sha256(root / "plan.json")
                or result["model_sha256"] != sha256(directory / "model.cbm")
                or fit["model_sha256"] != result["model_sha256"]
                or set(result["outputs"]) != expected_outputs
            ):
                raise ValueError("Temporal model or result signature differs")
            for path, digest in result["outputs"].items():
                if sha256(Path(path)) != digest:
                    raise ValueError("Temporal forecast/frontier hash differs")
            expected.append([kind, fold])
    if proof["periods"] != expected:
        raise ValueError("Weight replay omitted a required fold")
    flood = pooled([r["arms"]["profile"]["scores"] for r in report["flood"]["periods"]])
    if flood["eligible_episodes"] != 40:
        raise ValueError("Rare flood cohort changed")
    return proof


if __name__ == "__main__":
    print(verified_evidence()["status"])
