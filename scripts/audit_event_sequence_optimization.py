"""Describe optimization exposure; do not attribute model errors to it causally."""

import ast
import math
from pathlib import Path

from moscollector.goal90_research import read
from moscollector.prepare import sha256, write_json

root = Path("artifacts/research-v37-fixed")
legacy = Path("artifacts/research-v28")
training_source = Path("src/moscollector/neural_count_research.py")
training = next(
    ast.literal_eval(node.value)
    for node in ast.parse(training_source.read_text()).body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "TRAINING" for target in node.targets)
)
sources, rows = {root / "plan.json", legacy / "plan.json"}, []
for kind in ("access", "fire", "fault"):
    for fold in ("screen_1", "screen_2"):
        for current, previous in (("current_mlp", "mlp"), ("event_gru", "gru")):
            new_path = root / kind / fold / current / "fit.json"
            old_path = legacy / kind / fold / "models" / previous / "fit.json"
            new, old = read(new_path), read(old_path)
            new_sizes, old_sizes = new["signature"]["sizes"], old["sizes"]
            assert new["signature"]["settings"] == training
            for part in ("train", "validation"):
                assert new_sizes[part]["original_rows"] == old_sizes[part]["rows"]
                assert new_sizes[part]["eligible_episodes"] == old_sizes[part]["eligible_episodes"]
                assert abs(new_sizes[part]["weight_sum"] - old_sizes[part]["rows"]) < 1e-7
            now_steps = math.ceil(new_sizes["train"]["rows"] / training["batch_size"])
            old_steps = math.ceil(old_sizes["train"]["rows"] / training["batch_size"])
            rows.append(
                {
                    "kind": kind,
                    "fold": fold,
                    "variant": current,
                    "original_training_rows": old_sizes["train"]["rows"],
                    "augmented_training_rows": new_sizes["train"]["rows"],
                    "optimizer_steps_per_epoch": now_steps,
                    "old_optimizer_steps_per_epoch": old_steps,
                    "step_ratio": now_steps / old_steps,
                    "best_epoch": new["best_epoch"],
                    "completed_epochs": new["completed_epochs"],
                    "initial_validation_loss": new["history"][0]["validation_loss"],
                    "first_epoch_validation_loss": new["history"][1]["validation_loss"],
                    "best_validation_loss": new["best_validation_loss"],
                    "first_epoch_improves_initial": new["history"][1]["validation_loss"]
                    < new["history"][0]["validation_loss"] - training["min_delta"],
                }
            )
            sources.update({new_path, old_path})
result = {
    "scope": "Post-result descriptive audit of the fixed v37 optimization schedule; no new fitting, checkpoint selection, forecast evaluation or June reads.",
    "rows": rows,
    "best_epoch_at_most_one": sum(r["best_epoch"] <= 1 for r in rows),
    "best_epoch_zero": sum(r["best_epoch"] == 0 for r in rows),
    "step_ratio_range": [min(r["step_ratio"] for r in rows), max(r["step_ratio"] for r in rows)],
    "interpretation": "Snapshot mass and original event support match v28, but one augmented epoch has about3times its optimizer steps and only end-of-epoch validation. Both v37 candidates share this schedule. No intermediate trained checkpoints were evaluated, so useful earlier states and the cause of poor event precision/recall remain unknown. A different monitoring/step budget requires a separately frozen experiment; current outcomes and best-weight selection stay unchanged. Loss values across v28/v37 are not compared because validation queries and weights differ.",
    "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
    "code_hashes": {str(p): sha256(p) for p in (Path(__file__), training_source)},
}
write_json(Path("artifacts/event_sequence_optimization_audit.json"), result)
print({k: result[k] for k in ("best_epoch_at_most_one", "best_epoch_zero", "step_ratio_range")})
