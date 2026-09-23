from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from checkpoint_planner_runtime.config import dump_json, ensure_dir, load_yaml
from checkpoint_planner_runtime.runtime import CheckpointPlannerRuntime


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "checkpoint_planner_runtime_smoke_test.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run token-first checkpoint-planner smoke inference.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config path.")
    parser.add_argument("--num-steps", type=int, default=None, help="Override synthetic.num_steps.")
    return parser.parse_args()


def _shape_map(features: dict[str, Any]) -> dict[str, Any]:
    shapes: dict[str, Any] = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            shapes[key] = list(value.shape)
        else:
            shapes[key] = type(value).__name__
    return shapes


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)

    paths_cfg = config["paths"]
    runtime_cfg = config.get("runtime", {})
    synthetic_cfg = config.get("synthetic", {})
    output_cfg = config.get("output", {})

    output_dir = ensure_dir(paths_cfg["output_dir"])
    runtime = CheckpointPlannerRuntime(
        planner_train_config_path=paths_cfg["planner_train_config"],
        planner_checkpoint_path=paths_cfg["planner_checkpoint"],
        manifest_root_override=paths_cfg.get("planner_manifest_root"),
        device=str(runtime_cfg.get("device", "cuda")),
        precision=str(runtime_cfg.get("precision", "fp16")),
        patch_count=int(runtime_cfg.get("patch_count", synthetic_cfg.get("patch_count", 196))),
        value_variant=str(runtime_cfg.get("value_variant", "final")),
        uncertainty_source=str(runtime_cfg.get("uncertainty_source", "raw")),
        timestamp_source=str(runtime_cfg.get("timestamp_source", "step_index")),
    )

    seed = int(synthetic_cfg.get("seed", 1234))
    rng = np.random.default_rng(seed)
    num_steps = int(args.num_steps if args.num_steps is not None else synthetic_cfg.get("num_steps", 35))
    patch_count = int(synthetic_cfg.get("patch_count", runtime.patch_count))
    patch_dim = int(synthetic_cfg.get("patch_dim", runtime.patch_dim))
    state_dim = int(synthetic_cfg.get("state_dim", runtime.state_dim))
    tactile_dim = int(synthetic_cfg.get("tactile_dim", runtime.tactile_dim))
    ooi_present_period = max(1, int(synthetic_cfg.get("ooi_present_period", 2)))
    token_scale = float(synthetic_cfg.get("token_scale", 0.02))
    state_scale = float(synthetic_cfg.get("state_scale", 0.1))
    tactile_scale = float(synthetic_cfg.get("tactile_scale", 0.1))

    last_summary: dict[str, Any] | None = None
    last_feature_shapes: dict[str, Any] | None = None
    branch_counts = {"posterior": 0, "prior": 0}
    runtime.reset()

    progress = tqdm(range(num_steps), desc="Checkpoint planner token smoke", unit="step")
    for step in progress:
        ooi_target_present = (step % ooi_present_period) == 0
        head_left_tokens = rng.normal(scale=token_scale, size=(patch_count, patch_dim)).astype(np.float32)
        wrist_left_tokens = rng.normal(scale=token_scale, size=(patch_count, patch_dim)).astype(np.float32)
        wrist_right_tokens = rng.normal(scale=token_scale, size=(patch_count, patch_dim)).astype(np.float32)
        ooi_tokens = rng.normal(scale=token_scale, size=(patch_count, patch_dim)).astype(np.float32)
        state_65d = rng.normal(scale=state_scale, size=(state_dim,)).astype(np.float32)
        tactile_60d = rng.normal(scale=tactile_scale, size=(tactile_dim,)).astype(np.float32)

        result = runtime.infer_from_tokens(
            head_left_tokens=head_left_tokens,
            wrist_left_tokens=wrist_left_tokens,
            wrist_right_tokens=wrist_right_tokens,
            ooi_tokens=ooi_tokens,
            state_65d=state_65d,
            tactile_60d=tactile_60d,
            ooi_target_present=ooi_target_present,
        )
        branch_counts[result.selected_branch] += 1
        last_summary = result.to_summary_dict()
        last_feature_shapes = _shape_map(result.to_openpi_features())
        progress.set_postfix(
            branch=result.selected_branch,
            hist=result.history_length,
            cref=result.c_ref_after,
            ckpt=result.selected.final_checkpoint_index,
        )

    summary = {
        "config": str(args.config),
        "planner_train_config": str(paths_cfg["planner_train_config"]),
        "planner_checkpoint": str(paths_cfg["planner_checkpoint"]),
        "output_dir": str(output_dir),
        "num_steps": num_steps,
        "sequence_length": runtime.sequence_length,
        "patch_count": runtime.patch_count,
        "patch_dim": runtime.patch_dim,
        "state_dim": runtime.state_dim,
        "tactile_dim": runtime.tactile_dim,
        "branch_counts": branch_counts,
        "last_result": last_summary,
        "openpi_feature_shapes": last_feature_shapes,
    }

    if bool(output_cfg.get("save_summary", True)):
        dump_json(output_dir / str(output_cfg.get("summary_name", "checkpoint_planner_smoke_summary.json")), summary)

    print("Checkpoint planner token smoke")
    print(f"  planner_ckpt   : {paths_cfg['planner_checkpoint']}")
    print(f"  trainer_config : {paths_cfg['planner_train_config']}")
    print(f"  sequence_length: {runtime.sequence_length}")
    print(f"  steps          : {num_steps}")
    print(f"  branch_counts  : {branch_counts}")
    print(f"  feature_shapes : {last_feature_shapes}")
    print(f"  output_dir     : {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
