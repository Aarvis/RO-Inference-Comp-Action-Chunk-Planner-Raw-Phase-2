from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class JaxRuntimeConfig:
    preallocate: bool | None = False
    mem_fraction: float | None = 0.60
    allocator: str | None = "platform"
    platforms: str | None = None


@dataclasses.dataclass(frozen=True)
class OpenPICompActionChunkRuntimeConfig:
    config_name: str
    checkpoint_dir: Path
    vendored_source_root: Path | None
    openpi_source_root: Path | None
    tokenizer_model_path: Path | None
    asset_id: str | None
    default_prompt: str
    policy_sample_steps: int
    openpi_param_dtype: str | None
    random_seed: int
    image_size: int
    state_dim: int
    action_dim: int
    action_horizon: int
    tactile_dim: int
    tactile_image_size: int
    tolerate_invalid_optional_raw: bool
    raw_zero_is_unavailable: bool
    require_finite_actions: bool
    jax: JaxRuntimeConfig


def as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def ensure_dir(path: str | Path) -> Path:
    target = as_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = as_path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping in {config_path}, got {type(data)!r}.")
    return data


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = as_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_path(raw_path: str | Path | None, *, config_dir: Path) -> Path | None:
    if raw_path in (None, ""):
        return None
    path = as_path(raw_path)
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return path


def load_openpi_runtime_config(config_path: str | Path) -> OpenPICompActionChunkRuntimeConfig:
    path = as_path(config_path).resolve()
    payload = load_yaml(path)
    config_dir = path.parent

    paths_cfg = dict(payload.get("paths", {}))
    openpi_cfg = dict(payload.get("openpi", {}))
    runtime_cfg = dict(payload.get("runtime", {}))
    input_cfg = dict(payload.get("input", {}))
    output_cfg = dict(payload.get("output", {}))
    jax_cfg = dict(runtime_cfg.get("jax", {}))
    param_dtype = os.environ.get(
        "ORIGAMI_OPENPI_PARAM_DTYPE",
        runtime_cfg.get("openpi_param_dtype", runtime_cfg.get("restore_dtype", "bfloat16")),
    )

    checkpoint_dir = _resolve_path(paths_cfg.get("checkpoint_dir"), config_dir=config_dir)
    if checkpoint_dir is None:
        raise ValueError(f"{path}: paths.checkpoint_dir is required.")

    return OpenPICompActionChunkRuntimeConfig(
        config_name=str(openpi_cfg.get("config_name", "pi05_origami_comp_action_chunk")),
        checkpoint_dir=checkpoint_dir,
        vendored_source_root=_resolve_path(openpi_cfg.get("vendored_source_root"), config_dir=config_dir),
        openpi_source_root=_resolve_path(paths_cfg.get("openpi_source_root"), config_dir=config_dir),
        tokenizer_model_path=_resolve_path(paths_cfg.get("tokenizer_model_path"), config_dir=config_dir),
        asset_id=paths_cfg.get("asset_id"),
        default_prompt=str(payload.get("prompt", "fold paper into airplane")),
        policy_sample_steps=int(runtime_cfg.get("policy_sample_steps", 10)),
        openpi_param_dtype=None if param_dtype in (None, "") else str(param_dtype),
        random_seed=int(runtime_cfg.get("random_seed", 0)),
        image_size=int(input_cfg.get("image_size", 224)),
        state_dim=int(input_cfg.get("state_dim", 65)),
        action_dim=int(input_cfg.get("action_dim", 65)),
        action_horizon=int(input_cfg.get("action_horizon", 10)),
        tactile_dim=int(input_cfg.get("tactile_dim", 60)),
        tactile_image_size=int(input_cfg.get("tactile_image_size", 224)),
        tolerate_invalid_optional_raw=bool(input_cfg.get("tolerate_invalid_optional_raw", True)),
        raw_zero_is_unavailable=bool(input_cfg.get("raw_zero_is_unavailable", True)),
        require_finite_actions=bool(output_cfg.get("require_finite_actions", True)),
        jax=JaxRuntimeConfig(
            preallocate=(
                None
                if jax_cfg.get("preallocate") is None
                else bool(jax_cfg.get("preallocate"))
            ),
            mem_fraction=(
                None
                if jax_cfg.get("mem_fraction") is None
                else float(jax_cfg.get("mem_fraction"))
            ),
            allocator=(
                None
                if jax_cfg.get("allocator") in (None, "")
                else str(jax_cfg.get("allocator"))
            ),
            platforms=(
                None
                if jax_cfg.get("platforms") in (None, "")
                else str(jax_cfg.get("platforms"))
            ),
        ),
    )
