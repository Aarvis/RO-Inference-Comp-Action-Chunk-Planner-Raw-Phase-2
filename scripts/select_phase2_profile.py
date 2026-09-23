from __future__ import annotations

"""Select the Phase-2 H25/S1 or H15/S2 inference/replay profile.

The public inference contract contains only ``action_horizon``.  The stride is
kept exclusively in the OpenPI training snapshot and dataset-replay target
comparison, where it determines ground-truth positions t + j * stride.
"""

import argparse
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

PROFILES: dict[str, dict[str, Any]] = {
    "h15_s2": {
        "action_horizon": 15,
        "action_chunk_stride": 2,
        "vendored_source_root": "../../vendor/openpi/src",
        "source_config_file": "vendor/openpi/src/openpi/training/config.py",
    },
    "h25_s1": {
        "action_horizon": 25,
        "action_chunk_stride": 1,
        "vendored_source_root": "../../vendor/openpi_phase2_h25_s1/src",
        "source_config_file": "vendor/openpi_phase2_h25_s1/src/openpi/training/config.py",
    },
}


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _update_openpi_runtime(path: Path, profile: dict[str, Any]) -> None:
    payload = _read_yaml(path)
    openpi = payload.setdefault("openpi", {})
    model = openpi.setdefault("model", {})
    data = openpi.setdefault("data", {})
    inputs = payload.setdefault("input", {})
    openpi["config_name"] = "pi05_origami_comp_action_chunk_phase2"
    openpi["vendored_source_root"] = profile["vendored_source_root"]
    openpi["source_config_file"] = profile["source_config_file"]
    model["action_horizon"] = profile["action_horizon"]
    data["action_chunk_stride"] = profile["action_chunk_stride"]
    inputs["action_horizon"] = profile["action_horizon"]
    _write_yaml(path, payload)


def _update_runtime_config(path: Path, profile: dict[str, Any]) -> None:
    payload = _read_yaml(path)
    server = payload.setdefault("server", {})
    replay = payload.setdefault("dataset_replay", {})
    server["action_horizon"] = profile["action_horizon"]
    # action_chunk_stride is intentionally not a server/public-metadata field.
    server.pop("action_chunk_stride", None)
    replay["action_horizon"] = profile["action_horizon"]
    replay["action_chunk_stride"] = profile["action_chunk_stride"]
    _write_yaml(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Select an immutable Phase-2 inference profile.")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=ROOT / "configs" / "dataset_replay.yaml",
        help="Standalone runtime config to update.",
    )
    parser.add_argument(
        "--openpi-runtime-config",
        type=Path,
        default=ROOT / "OpenPI_Module" / "configs" / "openpi_comp_action_chunk_runtime.yaml",
        help="OpenPI runtime snapshot to update.",
    )
    parser.add_argument(
        "--planner-off-runtime-config",
        type=Path,
        default=ROOT / "configs" / "dataset_replay_planner_off.yaml",
        help="Planner-off runtime config to keep aligned with the selected profile.",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=None,
        help="Optional populated model_bundle to update too.",
    )
    args = parser.parse_args()
    profile = PROFILES[args.profile]

    _update_runtime_config(args.runtime_config.resolve(), profile)
    planner_off_config = args.planner_off_runtime_config.resolve()
    if planner_off_config != args.runtime_config.resolve():
        _update_runtime_config(planner_off_config, profile)
    _update_openpi_runtime(args.openpi_runtime_config.resolve(), profile)
    if args.bundle_root is not None:
        bundle_root = args.bundle_root.resolve()
        _update_runtime_config(bundle_root / "bundle.yaml", profile)
        _update_openpi_runtime(bundle_root / "configs" / "openpi_comp_action_chunk_runtime.yaml", profile)

    print(
        "Selected Phase-2 profile "
        f"{args.profile}: action_horizon={profile['action_horizon']}, "
        f"replay_action_chunk_stride={profile['action_chunk_stride']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
