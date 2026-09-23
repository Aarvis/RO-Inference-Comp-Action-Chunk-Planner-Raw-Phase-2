from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import yaml


def as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = as_path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping in {config_path}, got {type(payload)!r}.")
    return payload


def dump_json(path: str | Path, payload: Any) -> None:
    target = as_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_path(raw_path: str | Path | None, *, config_dir: Path) -> Path | None:
    if raw_path in (None, ""):
        return None
    path = as_path(raw_path)
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return path


def _require_path(raw_path: str | Path | None, *, config_dir: Path, field_name: str) -> Path:
    path = _resolve_path(raw_path, config_dir=config_dir)
    if path is None:
        raise ValueError(f"Missing required path: {field_name}")
    return path


def _resolve_model_or_path(raw_value: str | Path | None, *, config_dir: Path) -> str | None:
    if raw_value in (None, ""):
        return None
    value = str(raw_value)
    if value.startswith(".") or value.startswith("/") or "\\" in value or ":" in value:
        resolved = _resolve_path(value, config_dir=config_dir)
        return None if resolved is None else str(resolved)
    return value


def _as_tuple_str(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _as_tuple_int(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value in (None, ""):
        return default
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(item) for item in value)


@dataclasses.dataclass(frozen=True)
class RuntimePathsConfig:
    ooi_config: Path | None
    ooi_checkpoint: Path | None
    checkpoint_planner_config: Path | None
    checkpoint_planner_checkpoint: Path | None
    checkpoint_planner_manifest_root: Path | None
    openpi_runtime_config: Path
    dino_model_name_or_path: str | None
    dino_model_cache_dir: Path | None
    dataset_root: Path
    output_dir: Path


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cuda:0"
    precision: str = "fp16"
    camera_image_size: int = 224
    ooi_input_mode: str = "input_224"
    dino_local_files_only: bool | None = None
    dino_trust_remote_code: bool | None = None
    # When false, DINO, OOI, and the checkpoint planner are not constructed.
    # OpenPI instead receives the all-zero, masked planner prefix used by
    # planner-dropout training rows.
    planner_enabled: bool = True
    # This mode deliberately distinguishes the planner's prior *branch* from
    # the gamma continuity post-processing prior. See README_CONFIGURATION_ALIGNMENT.md.
    planner_feature_mode: str = "selected_branch_raw_no_continuity"
    planner_value_variant: str = "raw"
    planner_uncertainty_source: str = "raw"
    planner_transition_completion_threshold: float | None = None
    planner_timestamp_source: str = "step_index"


@dataclasses.dataclass(frozen=True)
class DatasetReplayConfig:
    warmup_inferences: int | None = None
    warmup_training_loss: bool = True
    episode_uids: tuple[str, ...] = ()
    max_episodes: int | None = None
    action_horizon: int = 10
    action_chunk_stride: int = 1
    step_stride: int = 10
    phase_offsets: tuple[int, ...] = ()
    prompt: str = "fold paper into airplane"
    tactile_modes: tuple[str, ...] = ("deform_only", "deform_plus_raw", "mixed_50")
    mixed_raw_probability: float = 0.5
    mixed_seed: int = 1234
    require_raw_for_deform_plus_raw: bool = True
    compute_action_error: bool = True
    compute_training_loss: bool = True
    loss_rng_seed: int = 1234
    loss_noise_samples: int = 1
    loss_train_mode: bool = False
    loss_every_n_samples: int = 1
    max_loss_samples: int | None = None
    max_samples_per_episode: int | None = None
    max_total_samples: int | None = None
    save_per_sample: bool = True
    per_sample_filename: str = "per_sample_metrics.jsonl"
    summary_filename: str = "replay_summary.json"
    all_modes_summary_filename: str = "replay_summary_all_modes.json"
    videos_subdir: str = "videos"
    arrays_subdir: str = "arrays"
    head_left_video: str = "head_left.mp4"
    wrist_left_video: str = "wrist_left.mp4"
    wrist_right_video: str = "wrist_right.mp4"
    tactile_deform_video: str = "tactile_deform.mp4"
    tactile_raw_video: str = "tactile_raw.mp4"
    state_array: str = "state_65d.npy"
    action_array: str = "action_65d.npy"
    tactile_array: str = "tactile_60d.npy"
    timestamps_array: str = "timestamps.npy"
    frame_index_array: str = "frame_index.npy"
    include_dataset_timestamps: bool = False

    @property
    def resolved_phase_offsets(self) -> tuple[int, ...]:
        if self.phase_offsets:
            return tuple(int(offset) for offset in self.phase_offsets)
        return tuple(range(int(self.action_horizon)))


@dataclasses.dataclass(frozen=True)
class ServerConfig:
    policy_name: str = "origami_comp_action_chunk"
    action_horizon: int = 10
    action_dim: int = 65
    execution_mode: str = "async"
    warmup_inferences: int = 1
    require_prompt: bool = True
    allow_observation_timestamp: bool = False
    tolerate_invalid_optional_raw: bool = True
    log_inference_requests: bool = True
    log_inference_every_n: int = 1


@dataclasses.dataclass(frozen=True)
class OrigamiCompActionChunkRuntimeConfig:
    config_path: Path
    paths: RuntimePathsConfig
    runtime: RuntimeConfig
    dataset_replay: DatasetReplayConfig
    server: ServerConfig = dataclasses.field(default_factory=ServerConfig)
    bundle_root: Path | None = None


def load_origami_comp_action_chunk_config(config_path: str | Path) -> OrigamiCompActionChunkRuntimeConfig:
    path = as_path(config_path).resolve()
    payload = load_yaml(path)
    config_dir = path.parent
    return _parse_origami_comp_action_chunk_config(path=path, payload=payload, config_dir=config_dir, bundle_root=None)


def load_origami_comp_action_chunk_bundle(bundle_root_or_config: str | Path) -> OrigamiCompActionChunkRuntimeConfig:
    bundle_path = as_path(bundle_root_or_config).resolve()
    if bundle_path.is_dir():
        bundle_path = bundle_path / "bundle.yaml"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Bundle config not found: {bundle_path}")
    payload = load_yaml(bundle_path)
    bundle_root = bundle_path.parent
    return _parse_origami_comp_action_chunk_config(
        path=bundle_path,
        payload=payload,
        config_dir=bundle_root,
        bundle_root=bundle_root,
    )


def load_runtime_config(config_path: str | Path | None = None, *, bundle_root: str | Path | None = None) -> OrigamiCompActionChunkRuntimeConfig:
    if bundle_root not in (None, ""):
        return load_origami_comp_action_chunk_bundle(bundle_root)
    if config_path in (None, ""):
        raise ValueError("config_path is required when bundle_root is not provided.")
    return load_origami_comp_action_chunk_config(config_path)


def _parse_origami_comp_action_chunk_config(
    *,
    path: Path,
    payload: dict[str, Any],
    config_dir: Path,
    bundle_root: Path | None,
) -> OrigamiCompActionChunkRuntimeConfig:

    paths_cfg = dict(payload.get("paths", {}))
    runtime_cfg = dict(payload.get("runtime", {}))
    replay_cfg = dict(payload.get("dataset_replay", {}))
    server_cfg = dict(payload.get("server", {}))

    output_dir = _resolve_path(paths_cfg.get("output_dir"), config_dir=config_dir)
    if output_dir is None:
        output_dir = (config_dir.parent / "outputs" / "dataset_replay").resolve()
    dataset_root = _resolve_path(paths_cfg.get("dataset_root"), config_dir=config_dir)
    if dataset_root is None:
        dataset_root = (config_dir / "dataset_replay").resolve()

    runtime = RuntimeConfig(
        device=str(runtime_cfg.get("device", "cuda:0")),
        precision=str(runtime_cfg.get("precision", "fp16")),
        camera_image_size=int(runtime_cfg.get("camera_image_size", 224)),
        ooi_input_mode=str(runtime_cfg.get("ooi_input_mode", "input_224")),
        dino_local_files_only=(
            None if runtime_cfg.get("dino_local_files_only") is None else bool(runtime_cfg.get("dino_local_files_only"))
        ),
        dino_trust_remote_code=(
            None
            if runtime_cfg.get("dino_trust_remote_code") is None
            else bool(runtime_cfg.get("dino_trust_remote_code"))
        ),
        planner_enabled=bool(runtime_cfg.get("planner_enabled", True)),
        planner_feature_mode=str(
            runtime_cfg.get("planner_feature_mode", "selected_branch_raw_no_continuity")
        ),
        planner_value_variant=str(runtime_cfg.get("planner_value_variant", "raw")),
        planner_uncertainty_source=str(runtime_cfg.get("planner_uncertainty_source", "raw")),
        planner_transition_completion_threshold=(
            None
            if runtime_cfg.get("planner_transition_completion_threshold") in (None, "")
            else float(runtime_cfg.get("planner_transition_completion_threshold"))
        ),
        planner_timestamp_source=str(runtime_cfg.get("planner_timestamp_source", "step_index")),
    )
    if runtime.planner_feature_mode != "selected_branch_raw_no_continuity":
        raise ValueError(
            "runtime.planner_feature_mode must be "
            "'selected_branch_raw_no_continuity'."
        )
    if runtime.planner_value_variant != "raw":
        raise ValueError(
            "runtime.planner_value_variant must be 'raw' for "
            "selected_branch_raw_no_continuity."
        )

    paths = RuntimePathsConfig(
        ooi_config=(
            _require_path(paths_cfg.get("ooi_config"), config_dir=config_dir, field_name="paths.ooi_config")
            if runtime.planner_enabled
            else _resolve_path(paths_cfg.get("ooi_config"), config_dir=config_dir)
        ),
        ooi_checkpoint=_resolve_path(paths_cfg.get("ooi_checkpoint"), config_dir=config_dir),
        checkpoint_planner_config=(
            _require_path(
                paths_cfg.get("checkpoint_planner_config"),
                config_dir=config_dir,
                field_name="paths.checkpoint_planner_config",
            )
            if runtime.planner_enabled
            else _resolve_path(paths_cfg.get("checkpoint_planner_config"), config_dir=config_dir)
        ),
        checkpoint_planner_checkpoint=(
            _require_path(
                paths_cfg.get("checkpoint_planner_checkpoint"),
                config_dir=config_dir,
                field_name="paths.checkpoint_planner_checkpoint",
            )
            if runtime.planner_enabled
            else _resolve_path(paths_cfg.get("checkpoint_planner_checkpoint"), config_dir=config_dir)
        ),
        checkpoint_planner_manifest_root=_resolve_path(
            paths_cfg.get("checkpoint_planner_manifest_root"),
            config_dir=config_dir,
        ),
        openpi_runtime_config=_require_path(
            paths_cfg.get("openpi_runtime_config"),
            config_dir=config_dir,
            field_name="paths.openpi_runtime_config",
        ),
        dino_model_name_or_path=_resolve_model_or_path(paths_cfg.get("dino_model_name_or_path"), config_dir=config_dir),
        dino_model_cache_dir=_resolve_path(paths_cfg.get("dino_model_cache_dir"), config_dir=config_dir),
        dataset_root=dataset_root,
        output_dir=output_dir,
    )

    replay = DatasetReplayConfig(
        warmup_inferences=(
            None
            if replay_cfg.get("warmup_inferences") in (None, "")
            else int(replay_cfg.get("warmup_inferences"))
        ),
        warmup_training_loss=bool(replay_cfg.get("warmup_training_loss", True)),
        episode_uids=_as_tuple_str(replay_cfg.get("episode_uids"), ()),
        max_episodes=(
            None if replay_cfg.get("max_episodes") in (None, "") else int(replay_cfg.get("max_episodes"))
        ),
        action_horizon=int(replay_cfg.get("action_horizon", 10)),
        action_chunk_stride=int(replay_cfg.get("action_chunk_stride", 1)),
        step_stride=int(replay_cfg.get("step_stride", 10)),
        phase_offsets=_as_tuple_int(replay_cfg.get("phase_offsets"), ()),
        prompt=str(replay_cfg.get("prompt", "fold paper into airplane")),
        tactile_modes=_as_tuple_str(
            replay_cfg.get("tactile_modes"),
            ("deform_only", "deform_plus_raw", "mixed_50"),
        ),
        mixed_raw_probability=float(replay_cfg.get("mixed_raw_probability", 0.5)),
        mixed_seed=int(replay_cfg.get("mixed_seed", 1234)),
        require_raw_for_deform_plus_raw=bool(replay_cfg.get("require_raw_for_deform_plus_raw", True)),
        compute_action_error=bool(replay_cfg.get("compute_action_error", True)),
        compute_training_loss=bool(replay_cfg.get("compute_training_loss", True)),
        loss_rng_seed=int(replay_cfg.get("loss_rng_seed", 1234)),
        loss_noise_samples=int(replay_cfg.get("loss_noise_samples", 1)),
        loss_train_mode=bool(replay_cfg.get("loss_train_mode", False)),
        loss_every_n_samples=int(replay_cfg.get("loss_every_n_samples", 1)),
        max_loss_samples=(
            None if replay_cfg.get("max_loss_samples") in (None, "") else int(replay_cfg.get("max_loss_samples"))
        ),
        max_samples_per_episode=(
            None
            if replay_cfg.get("max_samples_per_episode") in (None, "")
            else int(replay_cfg.get("max_samples_per_episode"))
        ),
        max_total_samples=(
            None if replay_cfg.get("max_total_samples") in (None, "") else int(replay_cfg.get("max_total_samples"))
        ),
        save_per_sample=bool(replay_cfg.get("save_per_sample", True)),
        per_sample_filename=str(replay_cfg.get("per_sample_filename", "per_sample_metrics.jsonl")),
        summary_filename=str(replay_cfg.get("summary_filename", "replay_summary.json")),
        all_modes_summary_filename=str(
            replay_cfg.get("all_modes_summary_filename", "replay_summary_all_modes.json")
        ),
        videos_subdir=str(replay_cfg.get("videos_subdir", "videos")),
        arrays_subdir=str(replay_cfg.get("arrays_subdir", "arrays")),
        head_left_video=str(replay_cfg.get("head_left_video", "head_left.mp4")),
        wrist_left_video=str(replay_cfg.get("wrist_left_video", "wrist_left.mp4")),
        wrist_right_video=str(replay_cfg.get("wrist_right_video", "wrist_right.mp4")),
        tactile_deform_video=str(replay_cfg.get("tactile_deform_video", "tactile_deform.mp4")),
        tactile_raw_video=str(replay_cfg.get("tactile_raw_video", "tactile_raw.mp4")),
        state_array=str(replay_cfg.get("state_array", "state_65d.npy")),
        action_array=str(replay_cfg.get("action_array", "action_65d.npy")),
        tactile_array=str(replay_cfg.get("tactile_array", "tactile_60d.npy")),
        timestamps_array=str(replay_cfg.get("timestamps_array", "timestamps.npy")),
        frame_index_array=str(replay_cfg.get("frame_index_array", "frame_index.npy")),
        include_dataset_timestamps=bool(replay_cfg.get("include_dataset_timestamps", False)),
    )

    server = ServerConfig(
        policy_name=str(server_cfg.get("policy_name", "origami_comp_action_chunk")),
        action_horizon=int(server_cfg.get("action_horizon", replay.action_horizon)),
        action_dim=int(server_cfg.get("action_dim", 65)),
        execution_mode=str(server_cfg.get("execution_mode", "async")),
        warmup_inferences=int(server_cfg.get("warmup_inferences", 1)),
        require_prompt=bool(server_cfg.get("require_prompt", True)),
        allow_observation_timestamp=bool(server_cfg.get("allow_observation_timestamp", False)),
        tolerate_invalid_optional_raw=bool(server_cfg.get("tolerate_invalid_optional_raw", True)),
        log_inference_requests=bool(server_cfg.get("log_inference_requests", True)),
        log_inference_every_n=max(1, int(server_cfg.get("log_inference_every_n", 1))),
    )
    if replay.action_horizon < 1 or replay.action_chunk_stride < 1:
        raise ValueError("dataset_replay.action_horizon and action_chunk_stride must both be positive.")
    if server.action_horizon < 1:
        raise ValueError("server.action_horizon must be positive.")
    if server.action_horizon != replay.action_horizon:
        raise ValueError("server and dataset_replay must use the same action_horizon.")

    return OrigamiCompActionChunkRuntimeConfig(
        config_path=path,
        paths=paths,
        runtime=runtime,
        dataset_replay=replay,
        server=server,
        bundle_root=bundle_root,
    )
