from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from openpi_runtime.config import dump_json, load_openpi_runtime_config, load_yaml
from openpi_runtime.runtime import OpenPICompActionChunkRuntime
from openpi_runtime.tactile import (
    TACTILE_DEFORM_GRID_SHAPE,
    TACTILE_RAW_GRID_SHAPE,
    prepare_tactile_from_observation,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "openpi_comp_action_chunk_smoke_test.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenPI comp-action-chunk inference smoke tests.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--input-only",
        action="store_true",
        help="Validate observation/tactile preprocessing without loading the OpenPI checkpoint.",
    )
    return parser.parse_args()


def _resolve_path(raw_path: str | Path, *, config_dir: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return path


def make_planner_features(rng: np.random.Generator) -> dict[str, Any]:
    belief = rng.random(29, dtype=np.float32)
    belief /= np.maximum(np.sum(belief), np.float32(1e-6))
    return {
        "planner_available": True,
        "planner_state_belief": belief,
        "planner_progress_transition": rng.random(2, dtype=np.float32),
        "planner_uncertainty": rng.random(3, dtype=np.float32),
        "planner_history_latent": rng.normal(size=512).astype(np.float32),
    }


def make_observation(rng: np.random.Generator, *, raw_mode: str) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "observation/image/head_left": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image/head_right": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image/wrist_left": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image/wrist_right": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": rng.normal(size=65).astype(np.float32),
        "observation/state/joint_torque": np.zeros((65,), dtype=np.float32),
        "observation/tactile": rng.normal(size=60).astype(np.float32),
        "observation/image/tactile_deform": rng.integers(0, 256, size=TACTILE_DEFORM_GRID_SHAPE, dtype=np.uint8),
        "prompt": "fold paper into airplane",
    }
    if raw_mode == "present":
        observation["observation/image/tactile_raw"] = rng.integers(0, 256, size=TACTILE_RAW_GRID_SHAPE, dtype=np.uint8)
    elif raw_mode == "zero":
        observation["observation/image/tactile_raw"] = np.zeros(TACTILE_RAW_GRID_SHAPE, dtype=np.uint8)
    elif raw_mode == "invalid":
        observation["observation/image/tactile_raw"] = np.zeros((1, 1, 3), dtype=np.uint8)
    elif raw_mode != "missing":
        raise ValueError(f"Unsupported raw_mode={raw_mode!r}.")
    return observation


def main() -> int:
    args = parse_args()
    smoke_cfg = load_yaml(args.config)
    config_dir = args.config.resolve().parent
    paths_cfg = dict(smoke_cfg.get("paths", {}))
    output_cfg = dict(smoke_cfg.get("output", {}))
    synthetic_cfg = dict(smoke_cfg.get("synthetic", {}))

    runtime_config_path = _resolve_path(
        paths_cfg.get("runtime_config", "./configs/openpi_comp_action_chunk_runtime.yaml"),
        config_dir=config_dir,
    )
    runtime_cfg = load_openpi_runtime_config(runtime_config_path)
    rng = np.random.default_rng(int(synthetic_cfg.get("seed", 1234)))

    raw_modes = []
    if bool(synthetic_cfg.get("test_raw_present", True)):
        raw_modes.append("present")
    if bool(synthetic_cfg.get("test_raw_missing", True)):
        raw_modes.append("missing")
    if bool(synthetic_cfg.get("test_raw_invalid", True)):
        raw_modes.append("invalid")
    raw_modes.append("zero")

    summary: dict[str, Any] = {
        "runtime_config": str(runtime_config_path),
        "input_only": bool(args.input_only),
        "cases": [],
    }

    runtime = None if args.input_only else OpenPICompActionChunkRuntime(runtime_cfg)
    planner_features = make_planner_features(rng)
    for raw_mode in raw_modes:
        observation = make_observation(rng, raw_mode=raw_mode)
        tactile = prepare_tactile_from_observation(
            observation,
            output_size=runtime_cfg.tactile_image_size,
            tolerate_invalid_optional_raw=runtime_cfg.tolerate_invalid_optional_raw,
            raw_zero_is_unavailable=runtime_cfg.raw_zero_is_unavailable,
        )
        case: dict[str, Any] = {
            "raw_mode": raw_mode,
            "tactile": tactile.to_debug_dict(),
        }
        if runtime is not None:
            result = runtime.infer_from_observation(observation, planner_features=planner_features)
            case["result"] = result.to_summary_dict()
        summary["cases"].append(case)

    if output_cfg.get("output_dir"):
        output_dir = _resolve_path(output_cfg["output_dir"], config_dir=config_dir)
        dump_json(output_dir / str(output_cfg.get("summary_name", "openpi_comp_action_chunk_smoke_summary.json")), summary)

    print("OpenPI comp-action-chunk smoke")
    print(f"  runtime_config: {runtime_config_path}")
    print(f"  input_only    : {bool(args.input_only)}")
    for case in summary["cases"]:
        print(
            "  "
            f"{case['raw_mode']}: raw_available={case['tactile']['raw_available']} "
            f"raw_status={case['tactile']['raw_status']}"
        )
        if "result" in case:
            print(f"    actions_shape={case['result']['actions_shape']} finite={case['result']['actions_finite']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
