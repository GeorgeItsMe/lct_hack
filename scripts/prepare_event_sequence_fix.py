"""Preserve the failed schema check and reuse its two unchanged completed fits."""

import ast
import copy
import shutil
from pathlib import Path

from moscollector.event_sequence_research import lock_plan, saved_json
from moscollector.goal90_research import read
from moscollector.prepare import sha256

old = Path("artifacts/research-v37")
root = Path("artifacts/research-v37-fixed")
archive = old / "frozen-event_sequence_research.py"
current = Path("src/moscollector/event_sequence_research.py").resolve()
old_plan = read(old / "plan.json")
assert sha256(archive) == old_plan["code_hashes"][str(current)]
for category in ("source_hashes", "code_hashes"):
    for path, digest in old_plan[category].items():
        source = archive if Path(path).resolve() == current else Path(path)
        assert sha256(source) == digest, path


def definitions(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef):
            result[node.name] = ast.dump(node, include_attributes=False)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                    "PLAN",
                    "VARIANTS",
                    "COUNT_CONTROLS",
                    "REFERENCES",
                ):
                    result[target.id] = ast.dump(node, include_attributes=False)
    return result


before, after = definitions(archive), definitions(current)
assert before.keys() == after.keys()
for name in before:
    if name not in ("evaluate", "lock_plan"):
        assert before[name] == after[name], name


def copy_exact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha256(source) == sha256(target), target
    else:
        shutil.copyfile(source, target)


sources = {old / "plan.json", archive}
for name in ("compute-preflight.json", "input-audit.json", "control-preflight.json"):
    copy_exact(old / name, root / name)
    sources.add(old / name)
directory = old / "access/screen_1"
sources.update(directory.glob("*.json"))
sources.update(directory.glob("*.parquet"))
for variant in ("current_mlp", "event_gru"):
    sources.update({directory / variant / "fit.json", directory / variant / "model.pt"})
fix = {
    "reason": "V33 policy predictions have no persisted alert column. The first v37 evaluation attached alerts before exact archive comparison and failed only on this extra column. Recompute the absent archived policy alerts with its unchanged selected policy, then compare every column exactly.",
    "observed_before_fix": "Only access screen_1 two neural policies were computed. Fresh control policy/metrics/frontier already matched before the table-shape assertion. No selection or additional month occurred.",
    "changes": "Only evaluation archive-schema compatibility, optional fix-provenance registration and default output directory. AST equality confirms PLAN, candidates, fit_models, screening, confirmation and run unchanged. All other originally frozen source/code hashes match.",
    "reuse": "Copy two completed best weight files exactly. Derived fit signatures refer to this new plan, with original fit/plan/weight hashes retained in reused_from. Full input signatures are recomputed by fit_network before accepting the cache; no optimizer checkpoint or incomplete training is reused.",
    "original_plan_sha256": sha256(old / "plan.json"),
    "original_code_sha256": sha256(archive),
    "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
    "code_hashes": {str(Path(__file__)): sha256(Path(__file__))},
}
saved_json(root / "schema-fix.json", fix)
lock_plan(root, "mps")
new_directory = root / "access/screen_1"
for name in ("current-codec.json", "event-codec.json"):
    copy_exact(directory / name, new_directory / name)
data = copy.deepcopy(read(directory / "data.json"))
data["provenance"]["plan_sha256"] = sha256(root / "plan.json")
saved_json(new_directory / "data.json", data)
for variant in ("current_mlp", "event_gru"):
    original_fit = directory / variant / "fit.json"
    original_weights = directory / variant / "model.pt"
    fit = copy.deepcopy(read(original_fit))
    assert fit["signature"]["plan_sha256"] == sha256(old / "plan.json")
    assert fit["model_sha256"] == sha256(original_weights)
    target = new_directory / variant
    copy_exact(original_weights, target / "model.pt")
    fit["signature"]["plan_sha256"] = sha256(root / "plan.json")
    fit["weights_file"] = str(target / "model.pt")
    fit["reused_from"] = {
        "fit_file": str(original_fit),
        "fit_sha256": sha256(original_fit),
        "weights_file": str(original_weights),
        "model_sha256": sha256(original_weights),
        "original_plan_sha256": sha256(old / "plan.json"),
    }
    saved_json(target / "fit.json", fit)
print("Preserved original run; froze corrected plan; reused exactly two unchanged best-weight files")
