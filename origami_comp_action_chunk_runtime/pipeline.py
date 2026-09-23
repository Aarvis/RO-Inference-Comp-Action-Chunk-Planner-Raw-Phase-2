from __future__ import annotations

import dataclasses
import logging
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .config import OrigamiCompActionChunkRuntimeConfig, _resolve_path, load_yaml


logger = logging.getLogger(__name__)

HEAD_LEFT_KEY = "observation/image/head_left"
WRIST_LEFT_KEY = "observation/image/wrist_left"
WRIST_RIGHT_KEY = "observation/image/wrist_right"
TACTILE_DEFORM_KEY = "observation/image/tactile_deform"
TACTILE_RAW_KEY = "observation/image/tactile_raw"
STATE_KEY = "observation/state"
TACTILE_VECTOR_KEY = "observation/tactile"

# These dimensions are part of the Origami VLA checkpoint-planner prefix
# contract. They remain available in planner-disabled deployments so OpenPI
# receives precisely the masked zero arrays used by planner-dropout training.
PLANNER_FEATURE_SHAPES = {
    "planner_state_belief": (29,),
    "planner_progress_transition": (2,),
    "planner_uncertainty": (3,),
    "planner_history_latent": (512,),
}


@dataclasses.dataclass(frozen=True)
class CompActionChunkInferenceResult:
    actions: np.ndarray
    ooi_summary: dict[str, Any]
    planner_summary: dict[str, Any]
    openpi_summary: dict[str, Any]
    openpi_loss_summary: dict[str, Any] | None
    tactile_raw_requested: bool | None
    timings_ms: dict[str, float]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "actions_shape": list(self.actions.shape),
            "actions_dtype": str(self.actions.dtype),
            "actions_finite": bool(np.isfinite(self.actions).all()),
            "ooi": self.ooi_summary,
            "planner": self.planner_summary,
            "openpi": self.openpi_summary,
            "openpi_loss": self.openpi_loss_summary,
            "tactile_raw_requested": (
                None if self.tactile_raw_requested is None else bool(self.tactile_raw_requested)
            ),
            "timings_ms": dict(self.timings_ms),
        }


class CompActionChunkPipeline:
    def __init__(self, config: OrigamiCompActionChunkRuntimeConfig) -> None:
        self.config = config
        self.module_root = Path(__file__).resolve().parents[1]
        self._install_module_paths()

        from OpenPI_Module.openpi_runtime import OpenPICompActionChunkRuntime

        self.planner_enabled = bool(config.runtime.planner_enabled)
        self.ooi_runtime: Any | None = None
        self.dino_runtime: Any | None = None
        self.planner_runtime: Any | None = None
        if self.planner_enabled:
            from Checkpoint_Planner_Module.checkpoint_planner_runtime import CheckpointPlannerRuntime
            from OOI_Module.ooi_runtime import OOIZoomRuntime, SharedDinoRuntime

            if (
                config.paths.ooi_config is None
                or config.paths.checkpoint_planner_config is None
                or config.paths.checkpoint_planner_checkpoint is None
            ):
                raise ValueError("Planner-enabled runtime requires OOI and checkpoint-planner paths.")
            ooi_payload = load_yaml(config.paths.ooi_config)
            ooi_paths_cfg = dict(ooi_payload.get("paths", {}))
            ooi_runtime_cfg = dict(ooi_payload.get("runtime", {}))
            ooi_checkpoint = config.paths.ooi_checkpoint
            if ooi_checkpoint is None:
                ooi_checkpoint = _resolve_path(ooi_paths_cfg.get("checkpoint"), config_dir=config.paths.ooi_config.parent)
            if ooi_checkpoint is None:
                raise ValueError("OOI checkpoint must be set in top-level paths.ooi_checkpoint or OOI config paths.checkpoint.")

            model_cache_dir = config.paths.dino_model_cache_dir
            if model_cache_dir is None:
                model_cache_dir = _resolve_path(ooi_runtime_cfg.get("model_cache_dir"), config_dir=config.paths.ooi_config.parent)

            dino_local_files_only = (
                bool(ooi_runtime_cfg.get("local_files_only", True))
                if config.runtime.dino_local_files_only is None
                else bool(config.runtime.dino_local_files_only)
            )
            dino_trust_remote_code = (
                bool(ooi_runtime_cfg.get("trust_remote_code", False))
                if config.runtime.dino_trust_remote_code is None
                else bool(config.runtime.dino_trust_remote_code)
            )

            self.ooi_runtime = OOIZoomRuntime(
                ooi_checkpoint,
                device=config.runtime.device,
                precision=config.runtime.precision,
                model_cache_dir=model_cache_dir,
                local_files_only=dino_local_files_only,
                trust_remote_code=dino_trust_remote_code,
                output_crop_size=int(ooi_runtime_cfg.get("output_crop_size", config.runtime.camera_image_size)),
                confidence_threshold=float(ooi_runtime_cfg.get("confidence_threshold", 0.1)),
                black_frame_below_threshold=bool(ooi_runtime_cfg.get("black_frame_below_threshold", True)),
                load_encoder=False,
            )
            dino_source = (
                config.paths.dino_model_name_or_path
                or ooi_runtime_cfg.get("dino_model_name_or_path")
                or self.ooi_runtime.model_config["dino_model_name"]
            )
            self.dino_runtime = SharedDinoRuntime(
                str(dino_source),
                device=config.runtime.device,
                precision=config.runtime.precision,
                cache_dir=model_cache_dir,
                local_files_only=dino_local_files_only,
                trust_remote_code=dino_trust_remote_code,
            )
            self.planner_runtime = CheckpointPlannerRuntime(
                planner_train_config_path=config.paths.checkpoint_planner_config,
                planner_checkpoint_path=config.paths.checkpoint_planner_checkpoint,
                manifest_root_override=config.paths.checkpoint_planner_manifest_root,
                device=config.runtime.device,
                precision=config.runtime.precision,
                value_variant=config.runtime.planner_value_variant,  # type: ignore[arg-type]
                uncertainty_source=config.runtime.planner_uncertainty_source,  # type: ignore[arg-type]
                transition_completion_threshold=config.runtime.planner_transition_completion_threshold,
                timestamp_source=config.runtime.planner_timestamp_source,
                apply_continuity_prior=False,
            )
        self.openpi_runtime = OpenPICompActionChunkRuntime(config.paths.openpi_runtime_config)

    def _install_module_paths(self) -> None:
        for subdir in ("OOI_Module", "Checkpoint_Planner_Module", "OpenPI_Module"):
            module_path = str(self.module_root / subdir)
            if module_path not in sys.path:
                sys.path.insert(0, module_path)

    def reset(self) -> None:
        if self.planner_runtime is not None:
            self.planner_runtime.reset()
        self.openpi_runtime.reset()

    def warmup(
        self,
        count: int = 1,
        *,
        prompt: str = "fold paper into airplane",
        compute_training_loss: bool = False,
        loss_noise_samples: int = 1,
        loss_train_mode: bool = False,
    ) -> None:
        for index in range(max(0, int(count))):
            observation = {
                HEAD_LEFT_KEY: np.zeros((self.config.runtime.camera_image_size, self.config.runtime.camera_image_size, 3), dtype=np.uint8),
                WRIST_LEFT_KEY: np.zeros((self.config.runtime.camera_image_size, self.config.runtime.camera_image_size, 3), dtype=np.uint8),
                WRIST_RIGHT_KEY: np.zeros((self.config.runtime.camera_image_size, self.config.runtime.camera_image_size, 3), dtype=np.uint8),
                STATE_KEY: np.zeros((65,), dtype=np.float32),
                TACTILE_VECTOR_KEY: np.zeros((60,), dtype=np.float32),
                TACTILE_DEFORM_KEY: np.zeros((480, 1200, 3), dtype=np.uint8),
                "prompt": prompt,
            }
            self.infer(
                observation,
                prompt=prompt,
                use_raw_tactile=False,
                target_actions_65d=np.zeros(
                    (
                        int(self.openpi_runtime.config.action_horizon),
                        int(self.openpi_runtime.config.action_dim),
                    ),
                    dtype=np.float32,
                ),
                compute_training_loss=bool(compute_training_loss),
                loss_rng_seed=index,
                loss_noise_samples=int(loss_noise_samples),
                loss_train_mode=bool(loss_train_mode),
            )
        self.reset()

    def close(self) -> None:
        for runtime in (
            getattr(self, "ooi_runtime", None),
            getattr(self, "dino_runtime", None),
            getattr(self, "planner_runtime", None),
            getattr(self, "openpi_runtime", None),
        ):
            close = getattr(runtime, "close", None)
            if callable(close):
                close()

    def infer(
        self,
        observation: Mapping[str, Any],
        *,
        prompt: str | None = None,
        use_raw_tactile: bool | None = None,
        require_raw_tactile: bool = False,
        planner_prefix_enabled: bool | None = None,
        target_actions_65d: np.ndarray | None = None,
        compute_training_loss: bool = False,
        loss_rng_seed: int = 0,
        loss_noise_samples: int = 1,
        loss_train_mode: bool = False,
    ) -> CompActionChunkInferenceResult:
        started = time.perf_counter()
        head_left_input = self._require_rgb(observation, HEAD_LEFT_KEY)
        wrist_left_input = self._require_rgb(observation, WRIST_LEFT_KEY)
        wrist_right_input = self._require_rgb(observation, WRIST_RIGHT_KEY)
        head_left = self._to_model_image(head_left_input, HEAD_LEFT_KEY)
        wrist_left = self._to_model_image(wrist_left_input, WRIST_LEFT_KEY)
        wrist_right = self._to_model_image(wrist_right_input, WRIST_RIGHT_KEY)

        state_65d = self._vector(observation[STATE_KEY], name=STATE_KEY, dim=65)
        tactile_60d = self._vector(observation[TACTILE_VECTOR_KEY], name=TACTILE_VECTOR_KEY, dim=60)
        tactile_deform_grid = self._require_rgb(observation, TACTILE_DEFORM_KEY)
        tactile_raw_grid = observation.get(TACTILE_RAW_KEY)
        if use_raw_tactile is False:
            raw_grid_for_openpi = None
            raw_available = False
        elif use_raw_tactile is True:
            if tactile_raw_grid is None and require_raw_tactile:
                raise ValueError("tactile_mode requested raw tactile, but observation/image/tactile_raw is missing.")
            raw_grid_for_openpi = tactile_raw_grid
            raw_available = True
        else:
            raw_grid_for_openpi = tactile_raw_grid
            raw_available = None

        prefix_enabled = self.planner_enabled if planner_prefix_enabled is None else bool(planner_prefix_enabled)
        use_planner = bool(self.planner_enabled and prefix_enabled)
        dino_camera_ms = 0.0
        ooi_ms = 0.0
        dino_ooi_ms = 0.0
        planner_ms = 0.0
        if use_planner:
            assert self.dino_runtime is not None
            assert self.ooi_runtime is not None
            assert self.planner_runtime is not None
            dino_started = time.perf_counter()
            camera_tokens = self.dino_runtime.encode_rgb_images([head_left, wrist_left, wrist_right])
            dino_camera_ms = (time.perf_counter() - dino_started) * 1000.0

            ooi_started = time.perf_counter()
            ooi_result = self.ooi_runtime.infer_from_tokens(
                head_left_tokens=camera_tokens[0],
                wrist_left_tokens=camera_tokens[1],
                wrist_right_tokens=camera_tokens[2],
                head_left_model_rgb=head_left,
                source_input_size=int(head_left.shape[0]),
                input_mode=self.config.runtime.ooi_input_mode,  # type: ignore[arg-type]
            )
            ooi_ms = (time.perf_counter() - ooi_started) * 1000.0

            ooi_tokens = None
            if ooi_result.target_present:
                ooi_dino_started = time.perf_counter()
                ooi_tokens = self.dino_runtime.encode_rgb_images([ooi_result.ooi_crop_rgb])[0]
                dino_ooi_ms = (time.perf_counter() - ooi_dino_started) * 1000.0

            planner_started = time.perf_counter()
            planner_result = self.planner_runtime.infer_from_tokens(
                head_left_tokens=camera_tokens[0],
                wrist_left_tokens=camera_tokens[1],
                wrist_right_tokens=camera_tokens[2],
                ooi_tokens=ooi_tokens,
                state_65d=state_65d,
                tactile_60d=tactile_60d,
                ooi_target_present=bool(ooi_result.target_present),
                timestamp=self._optional_float(observation.get("observation_timestamp")),
            )
            planner_ms = (time.perf_counter() - planner_started) * 1000.0
            planner_features = self._planner_features_for_policy(planner_result.to_openpi_features())
            planner_summary = planner_result.to_summary_dict()
            planner_summary.update(
                {
                    "planner_runtime_ran": True,
                    "planner_runtime_available": True,
                    "planner_prefix_enabled": True,
                    "planner_available": True,
                    "feature_mode": self.config.runtime.planner_feature_mode,
                }
            )
            ooi_summary = ooi_result.to_json_dict()
        else:
            planner_features = self._planner_disabled_features()
            planner_summary = {
                "planner_runtime_ran": False,
                "planner_runtime_available": False,
                "planner_prefix_enabled": False,
                "planner_available": False,
                "feature_mode": "planner_disabled_masked_zero_prefix",
            }
            ooi_summary = {
                "runtime_ran": False,
                "target_present": False,
                "reason": "planner_disabled",
            }

        openpi_started = time.perf_counter()
        openpi_result = self.openpi_runtime.infer(
            head_left_rgb=head_left,
            wrist_left_rgb=wrist_left,
            wrist_right_rgb=wrist_right,
            state_65d=state_65d,
            tactile_60d=tactile_60d,
            planner_features=planner_features,
            tactile_deform_grid=tactile_deform_grid,
            tactile_raw_grid=raw_grid_for_openpi,
            tactile_raw_available=raw_available,
            prompt=prompt or observation.get("prompt"),
        )
        openpi_ms = (time.perf_counter() - openpi_started) * 1000.0

        loss_summary = None
        loss_ms = 0.0
        if compute_training_loss:
            if target_actions_65d is None:
                raise ValueError("target_actions_65d is required when compute_training_loss=True.")
            loss_started = time.perf_counter()
            loss_result = self.openpi_runtime.compute_training_loss(
                head_left_rgb=head_left,
                wrist_left_rgb=wrist_left,
                wrist_right_rgb=wrist_right,
                state_65d=state_65d,
                tactile_60d=tactile_60d,
                planner_features=planner_features,
                target_actions_65d=target_actions_65d,
                tactile_deform_grid=tactile_deform_grid,
                tactile_raw_grid=raw_grid_for_openpi,
                tactile_raw_available=raw_available,
                prompt=prompt or observation.get("prompt"),
                rng_seed=int(loss_rng_seed),
                noise_samples=int(loss_noise_samples),
                sample_weight=1.0,
                train=bool(loss_train_mode),
            )
            loss_ms = (time.perf_counter() - loss_started) * 1000.0
            loss_summary = loss_result.to_summary_dict()

        total_ms = (time.perf_counter() - started) * 1000.0
        return CompActionChunkInferenceResult(
            actions=np.asarray(openpi_result.actions, dtype=np.float32),
            ooi_summary=ooi_summary,
            planner_summary=planner_summary,
            openpi_summary=openpi_result.to_summary_dict(),
            openpi_loss_summary=loss_summary,
            tactile_raw_requested=use_raw_tactile,
            timings_ms={
                "dino_camera": dino_camera_ms,
                "ooi": ooi_ms,
                "dino_ooi": dino_ooi_ms,
                "planner": planner_ms,
                "openpi": openpi_ms,
                "openpi_policy_only": float(openpi_result.policy_timing_ms),
                "training_loss": loss_ms,
                "total": total_ms,
            },
        )

    @staticmethod
    def _planner_features_for_policy(planner_features: Mapping[str, Any]) -> dict[str, np.ndarray | bool]:
        """Normalize a real raw planner result to the fixed OpenPI contract."""
        return {
            "planner_available": bool(planner_features["planner_available"]),
            **{
                name: np.asarray(planner_features[name], dtype=np.float32)
                for name in PLANNER_FEATURE_SHAPES
            },
        }

    @staticmethod
    def _planner_disabled_features() -> dict[str, np.ndarray | bool]:
        """Build the exact masked planner-dropout prefix without planner modules."""
        return {
            "planner_available": False,
            **{
                name: np.zeros(shape, dtype=np.float32)
                for name, shape in PLANNER_FEATURE_SHAPES.items()
            },
        }

    def _to_model_image(self, image_rgb: np.ndarray, name: str) -> np.ndarray:
        if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
            raise ValueError(f"{name} must have shape [H, W, 3], got {image_rgb.shape}.")
        if image_rgb.dtype != np.uint8:
            raise TypeError(f"{name} must have dtype uint8, got {image_rgb.dtype}.")
        size = int(self.config.runtime.camera_image_size)
        if image_rgb.shape[:2] == (size, size):
            return np.ascontiguousarray(image_rgb)
        return np.ascontiguousarray(cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA))

    @staticmethod
    def _require_rgb(observation: Mapping[str, Any], key: str) -> np.ndarray:
        if key not in observation:
            raise KeyError(f"Observation is missing required key {key!r}.")
        value = np.asarray(observation[key])
        if value.dtype != np.uint8:
            raise TypeError(f"{key} must have dtype uint8, got {value.dtype}.")
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"{key} must have shape [H, W, 3], got {value.shape}.")
        return np.ascontiguousarray(value)

    @staticmethod
    def _vector(value: Any, *, name: str, dim: int) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        if tuple(array.shape) != (dim,):
            raise ValueError(f"{name} must have shape {(dim,)}, got {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf.")
        return np.ascontiguousarray(array, dtype=np.float32)

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value in (None, ""):
            return None
        return float(np.asarray(value).reshape(()))
