"""Require all fixed-source projections, controls and periods in a policy replay."""

from pathlib import Path

from moscollector.count_bound_research import CANDIDATES, PARENT, PLAN, ROOT, selection
from moscollector.flood_research import validate_hashes
from moscollector.goal90_research import read
from moscollector.prepare import sha256


def verified_evidence(root=ROOT):
    proof = read(root / "policy-replay.json")
    validate_hashes(proof)
    plan, report = read(root / "plan.json"), read(root / "report.json")
    if (
        proof["status"] != "exact_policy_replay_passed"
        or proof["new_model_fits"] != 0
        or proof["report"] != report
        or set(report) != set(PLAN["kinds"])
        or proof["code_hashes"] != plan["code_hashes"]
        or any(proof["source_hashes"].get(p) != h for p, h in plan["source_hashes"].items())
    ):
        raise ValueError("Incomplete or mismatched count-bound proof")
    required = {str(p) for p in root.rglob("*") if p.is_file() and p.name != "policy-replay.json"}
    if not required.issubset(proof["source_hashes"]):
        raise ValueError("Count-bound proof omits result files")
    for kind, outcome in report.items():
        rows = outcome["periods"]
        chosen = selection(rows)
        status = "requires_separate_confirmation" if chosen["selected"] else "screen_rejected"
        if (
            [r["fold"] for r in rows] != PLAN["folds"][kind]
            or chosen != outcome["selection"]
            or outcome["status"] != status
            or outcome["serving_changed"] is not False
            or outcome["goal_achieved"] is not False
        ):
            raise ValueError("Changed count-bound coverage, selection or release claim")
        for row in rows:
            directory = root / kind / row["fold"]
            parent_path = PARENT / kind / row["fold"] / "result.json"
            parent = read(parent_path)
            outputs = {
                str(directory / f"{name}-{part}.parquet")
                for name in CANDIDATES
                for part in ("calibration", "policy", "test")
            } | {str(directory / f"{name}-frontier.json") for name in CANDIDATES}
            if (
                row != read(directory / "result.json")
                or row["kind"] != kind
                or row["plan_sha256"] != sha256(root / "plan.json")
                or row["parent_sha256"] != sha256(parent_path)
                or row["references"] != parent["references"]
                or set(row["outputs"]) != outputs
            ):
                raise ValueError("Changed count-bound result signature")
            for name, old in (
                ("poisson_control", "poisson_corrected"),
                ("tweedie_control", "tweedie"),
                ("legacy_control", "poisson_legacy"),
            ):
                for key in ("scores", "policy"):
                    if row["arms"][name][key] != parent["arms"][old][key]:
                        raise ValueError("Changed count-bound comparator")
            for name in CANDIDATES:
                if len(read(directory / f"{name}-frontier.json")) != (120 if kind == "flood" else 456):
                    raise ValueError("Incomplete projected-policy frontier")
                for part, stats in row["arms"][name]["projection"].items():
                    source = parent["signature"]["parts"][part]["rows"]
                    # Exact query rows are already replayed; calibrated fine grids
                    # can exceed source hours and must never be smaller.
                    if (
                        stats["rows"] < source
                        or not 0 <= stats["raised_rows"] <= stats["rows"]
                        or stats["projected_count_sum"] < stats["source_count_sum"]
                    ):
                        raise ValueError("Invalid projection row/mass diagnostic")
            for path, digest in row["outputs"].items():
                if sha256(Path(path)) != digest:
                    raise ValueError("Changed projected forecasts/frontier")
    if report["flood"]["selection"]["scores"][CANDIDATES[0]]["eligible_episodes"] != 40:
        raise ValueError("Changed count-bound flood40 cohort")
    return proof


if __name__ == "__main__":
    print(verified_evidence()["status"])
