"""Read a complete weight replay of the fixed Tweedie comparison."""

from pathlib import Path

from catboost import CatBoostRegressor

from moscollector.experiments.flood_research import validate_hashes
from moscollector.experiments.goal90_research import read
from moscollector.experiments.tweedie_count_research import (
    PARENT,
    PLAN,
    ROOT,
    base_model,
    check_native_parameters,
    comparison,
)
from moscollector.prepare import sha256


def verified_evidence(root=ROOT):
    proof = read(root / "weight-replay.json")
    validate_hashes(proof)
    plan, report = read(root / "plan.json"), read(root / "report.json")
    if (
        proof["status"] != "exact_replay_passed"
        or proof["report"] != report
        or set(report) != set(PLAN["kinds"])
        or proof["code_hashes"] != plan["code_hashes"]
        or any(proof["source_hashes"].get(p) != h for p, h in plan["source_hashes"].items())
    ):
        raise ValueError("Incomplete or mismatched Tweedie proof")
    required = {str(p) for p in root.rglob("*") if p.is_file() and p.name != "weight-replay.json"}
    if not required.issubset(proof["source_hashes"]):
        raise ValueError("Tweedie proof omits model/result files")
    for kind, outcome in report.items():
        rows = outcome["periods"]
        selection = comparison(rows)
        status = "requires_separate_confirmation" if selection["passed_screen"] else "screen_rejected"
        if (
            [r["fold"] for r in rows] != PLAN["folds"][kind]
            or outcome["selection"] != selection
            or outcome["status"] != status
            or outcome["serving_changed"] is not False
            or outcome["goal_achieved"] is not False
        ):
            raise ValueError("Changed Tweedie coverage, selection or unsupported release")
        for row in rows:
            directory = root / kind / row["fold"]
            parent = read(PARENT / kind / row["fold"] / "result.json")
            fit = read(directory / "fit.json")
            outputs = {
                str(directory / f"tweedie-{part}.parquet") for part in ("calibration", "policy", "test")
            } | {str(directory / "tweedie-frontier.json")}
            if (
                row != read(directory / "result.json")
                or row["kind"] != kind
                or row["signature"] != fit["signature"]
                or row["signature"]["plan_sha256"] != sha256(root / "plan.json")
                or row["model_sha256"] != fit["model_sha256"]
                or fit["model_sha256"] != sha256(directory / "model.cbm")
                or row["references"] != parent["references"]
                or row["arms"]["poisson_corrected"]["scores"] != parent["arms"]["retained_selected"]["scores"]
                or row["arms"]["poisson_legacy"]["scores"] != parent["arms"]["legacy_control"]["scores"]
                or set(row["outputs"]) != outputs
                or len(read(directory / "tweedie-frontier.json")) != (120 if kind == "flood" else 456)
            ):
                raise ValueError("Changed Tweedie model, control or result signature")
            for path, digest in row["outputs"].items():
                if sha256(Path(path)) != digest:
                    raise ValueError("Changed Tweedie forecast/frontier")
            model, old = CatBoostRegressor(), CatBoostRegressor()
            model.load_model(str(directory / "model.cbm"))
            old.load_model(str(base_model(kind, row["fold"])[0]))
            if check_native_parameters(model, old) != fit["native_parameter_differences"]:
                raise ValueError("Unmatched Tweedie/Poisson training parameters")
    if report["flood"]["selection"]["scores"]["tweedie"]["eligible_episodes"] != 40:
        raise ValueError("Changed original flood40 cohort")
    return proof


if __name__ == "__main__":
    print(verified_evidence()["status"])
