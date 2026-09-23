from __future__ import annotations

import dataclasses
import inspect
import logging
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import OpenPICompActionChunkRuntimeConfig, load_openpi_runtime_config
from .tactile import (
    TACTILE_DEFORM_KEY,
    TACTILE_RAW_KEY,
    TACTILE_VECTOR_KEY,
    TactileImageInputs,
    prepare_tactile_from_observation,
    prepare_tactile_image_inputs,
    validate_rgb_image,
    validate_vector,
)
from .vendor_bootstrap import (
    configure_jax_environment,
    ensure_openpi_on_path,
    patch_paligemma_tokenizer_download,
)


logger = logging.getLogger(__name__)

HEAD_LEFT_KEY = "observation/image/head_left"
WRIST_LEFT_KEY = "observation/image/wrist_left"
WRIST_RIGHT_KEY = "observation/image/wrist_right"
STATE_KEY = "observation/state"


@dataclasses.dataclass(frozen=True)
class OpenPICompActionChunkInferenceResult:
    actions: np.ndarray
    prompt: str
    planner_available: bool
    tactile_raw_available: bool
    tactile_raw_status: str
    policy_timing_ms: float
    total_runtime_ms: float
    policy_debug: dict[str, Any]

    @property
    def first_action(self) -> np.ndarray:
        return self.actions[0]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "actions_shape": list(self.actions.shape),
            "actions_dtype": str(self.actions.dtype),
            "actions_finite": bool(np.isfinite(self.actions).all()),
            "prompt": self.prompt,
            "planner_available": bool(self.planner_available),
            "tactile_raw_available": bool(self.tactile_raw_available),
            "tactile_raw_status": self.tactile_raw_status,
            "policy_timing_ms": float(self.policy_timing_ms),
            "total_runtime_ms": float(self.total_runtime_ms),
            "policy_debug": self.policy_debug,
        }


@dataclasses.dataclass(frozen=True)
class OpenPICompActionChunkLossResult:
    metrics: dict[str, float]
    chunked_loss: np.ndarray
    transformed_actions: np.ndarray
    noise_samples: int
    rng_seed: int

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "chunked_loss_shape": list(self.chunked_loss.shape),
            "chunked_loss_mean": float(np.mean(self.chunked_loss)),
            "transformed_actions_shape": list(self.transformed_actions.shape),
            "noise_samples": int(self.noise_samples),
            "rng_seed": int(self.rng_seed),
        }


class OpenPICompActionChunkRuntime:
    def __init__(self, config: OpenPICompActionChunkRuntimeConfig | str | Path) -> None:
        if isinstance(config, (str, Path)):
            config = load_openpi_runtime_config(config)
        self.config = config

        configure_jax_environment(self.config.jax)
        ensure_openpi_on_path(self.config.vendored_source_root or self.config.openpi_source_root)

        from openpi.shared import download as openpi_download

        self.tokenizer_model_path = patch_paligemma_tokenizer_download(
            openpi_download,
            self.config.tokenizer_model_path,
        )

        from openpi.policies import policy_config as openpi_policy_config
        from openpi.models import model as openpi_model
        from openpi.shared import nnx_utils as openpi_nnx_utils
        from openpi.training import config as openpi_training_config

        self._openpi_policy_config = openpi_policy_config
        self._openpi_model = openpi_model
        self._openpi_nnx_utils = openpi_nnx_utils
        self._openpi_training_config = openpi_training_config

        self.checkpoint_dir = self.config.checkpoint_dir.resolve()
        self._validate_checkpoint_dir()

        self.train_config = self._openpi_training_config.get_config(self.config.config_name)
        self.model_config = self.train_config.model
        self._validate_loaded_config()

        restore_dtype = self._resolve_openpi_restore_dtype(self.config.openpi_param_dtype)
        normalized_restore_dtype = (
            None
            if self.config.openpi_param_dtype is None
            else str(self.config.openpi_param_dtype).strip().lower()
        )
        if normalized_restore_dtype in ("fp32", "float32"):
            logger.warning(
                "Loading OpenPI params in float32 for testing. This increases GPU memory versus bfloat16."
            )
        create_policy_kwargs: dict[str, Any] = {
            "sample_kwargs": {"num_steps": int(self.config.policy_sample_steps)},
            "default_prompt": self.config.default_prompt,
        }
        # OpenPI source revisions in active use differ here. Older/current
        # policy_config creates policies at checkpoint/native dtype and does
        # not accept restore_dtype; newer revisions may expose the override.
        create_policy_signature = inspect.signature(self._openpi_policy_config.create_trained_policy)
        if "restore_dtype" in create_policy_signature.parameters:
            create_policy_kwargs["restore_dtype"] = restore_dtype
        elif normalized_restore_dtype not in (None, "", "checkpoint", "native", "none"):
            logger.info(
                "Vendored OpenPI create_trained_policy has no restore_dtype option; using checkpoint-native parameter dtype."
            )
        self.policy = self._openpi_policy_config.create_trained_policy(
            self.train_config,
            self.checkpoint_dir,
            **create_policy_kwargs,
        )
        if getattr(self.policy, "_is_pytorch_model", False):
            self._compute_loss_and_metrics = None
        else:
            self._compute_loss_and_metrics = self._openpi_nnx_utils.module_jit(
                self.policy._model.compute_loss_and_metrics,  # type: ignore[attr-defined]
                static_argnames=("train",),
            )
        self.reset()
        logger.info(
            (
                "OpenPI comp-action-chunk runtime ready checkpoint_dir=%s config_name=%s "
                "action_horizon=%d action_dim=%d openpi_param_dtype=%s"
            ),
            self.checkpoint_dir,
            self.config.config_name,
            self.config.action_horizon,
            self.config.action_dim,
            self.config.openpi_param_dtype or "checkpoint",
        )

    @staticmethod
    def _resolve_openpi_restore_dtype(dtype_name: str | None):
        import jax.numpy as jnp

        if dtype_name in (None, ""):
            return None
        normalized = str(dtype_name).strip().lower()
        if normalized in ("checkpoint", "native", "none"):
            return None
        if normalized in ("bf16", "bfloat16"):
            return jnp.bfloat16
        if normalized in ("fp32", "float32"):
            return jnp.float32
        if normalized in ("fp16", "float16"):
            return jnp.float16
        raise ValueError(
            "OpenPI restore dtype must be one of bfloat16/bf16, float32/fp32, "
            f"float16/fp16, or checkpoint/native/none. Got {dtype_name!r}."
        )

    def _validate_checkpoint_dir(self) -> None:
        if not self.checkpoint_dir.is_dir():
            raise FileNotFoundError(f"OpenPI checkpoint dir not found: {self.checkpoint_dir}")
        params_dir = self.checkpoint_dir / "params"
        assets_dir = self.checkpoint_dir / "assets"
        if not params_dir.is_dir():
            raise FileNotFoundError(f"OpenPI checkpoint params dir not found: {params_dir}")
        if not assets_dir.is_dir():
            raise FileNotFoundError(f"OpenPI checkpoint assets dir not found: {assets_dir}")
        if self.config.asset_id not in (None, ""):
            norm_stats = assets_dir / str(self.config.asset_id) / "norm_stats.json"
            if not norm_stats.is_file():
                raise FileNotFoundError(f"OpenPI checkpoint norm stats not found: {norm_stats}")

    def _validate_loaded_config(self) -> None:
        model = self.model_config
        if int(model.action_horizon) != self.config.action_horizon:
            raise ValueError(f"Expected action_horizon={self.config.action_horizon}, got {model.action_horizon}.")
        if int(model.action_dim) != self.config.action_dim:
            raise ValueError(f"Expected action_dim={self.config.action_dim}, got {model.action_dim}.")
        if int(model.state_dim or model.action_dim) != self.config.state_dim:
            raise ValueError(f"Expected state_dim={self.config.state_dim}, got {model.state_dim}.")
        origami = model.origami_vla
        if not bool(origami.enabled):
            raise ValueError("Loaded OpenPI config does not have origami_vla.enabled=True.")
        if str(origami.action_mode) != "action_chunk":
            raise ValueError(f"Expected origami action_mode='action_chunk', got {origami.action_mode!r}.")
        if not bool(origami.ftp_tactile_enabled):
            raise ValueError("Loaded OpenPI config does not have ftp_tactile_enabled=True.")
        if int(origami.tactile_dim) != self.config.tactile_dim:
            raise ValueError(f"Expected tactile_dim={self.config.tactile_dim}, got {origami.tactile_dim}.")

    def reset(self) -> None:
        if hasattr(self.policy, "_rng"):
            import jax

            self.policy._rng = jax.random.key(int(self.config.random_seed))  # type: ignore[attr-defined]

    def build_policy_input(
        self,
        *,
        head_left_rgb: np.ndarray,
        wrist_left_rgb: np.ndarray,
        wrist_right_rgb: np.ndarray,
        state_65d: np.ndarray,
        tactile_60d: np.ndarray,
        planner_features: Mapping[str, Any],
        tactile_inputs: TactileImageInputs,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        prompt_value = prompt if prompt not in (None, "") else self.config.default_prompt
        state = validate_vector(state_65d, name="state_65d", expected_dim=self.config.state_dim)
        tactile = validate_vector(tactile_60d, name="tactile_60d", expected_dim=self.config.tactile_dim)
        planner_available = bool(planner_features.get("planner_available", True))

        return {
            "image": {
                "base_0_rgb": validate_rgb_image(head_left_rgb, name="head_left_rgb", output_size=self.config.image_size),
                "left_wrist_0_rgb": validate_rgb_image(wrist_left_rgb, name="wrist_left_rgb", output_size=self.config.image_size),
                "right_wrist_0_rgb": validate_rgb_image(wrist_right_rgb, name="wrist_right_rgb", output_size=self.config.image_size),
            },
            "image_mask": {
                "base_0_rgb": np.asarray(True),
                "left_wrist_0_rgb": np.asarray(True),
                "right_wrist_0_rgb": np.asarray(True),
            },
            "state": state,
            "prompt": str(prompt_value),
            "tactile_prompt": tactile,
            "tactile_prompt_mask": np.ones((self.config.tactile_dim,), dtype=bool),
            "state_mask": np.ones((self.config.state_dim,), dtype=bool),
            "planner_available": np.asarray(planner_available, dtype=bool),
            "planner_state_belief": self._planner_array(planner_features, "planner_state_belief", (29,)),
            "planner_progress_transition": self._planner_array(planner_features, "planner_progress_transition", (2,)),
            "planner_uncertainty": self._planner_array(planner_features, "planner_uncertainty", (3,)),
            "planner_history_latent": self._planner_array(planner_features, "planner_history_latent", (512,)),
            "tactile_deform_images": tactile_inputs.deform_images,
            "tactile_raw_images": tactile_inputs.raw_images,
            "tactile_raw_available": np.asarray(tactile_inputs.raw_available, dtype=bool),
        }

    @staticmethod
    def _planner_array(features: Mapping[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
        if key not in features:
            raise KeyError(f"Missing planner feature {key!r}.")
        array = np.asarray(features[key], dtype=np.float32)
        if array.ndim == len(shape) + 1 and array.shape[0] == 1:
            array = array[0]
        if tuple(array.shape) != shape:
            raise ValueError(f"{key} must have shape {shape}, got {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains NaN or Inf.")
        return np.ascontiguousarray(array, dtype=np.float32)

    def infer(
        self,
        *,
        head_left_rgb: np.ndarray,
        wrist_left_rgb: np.ndarray,
        wrist_right_rgb: np.ndarray,
        state_65d: np.ndarray,
        tactile_60d: np.ndarray,
        planner_features: Mapping[str, Any],
        tactile_deform_grid: Any | None = None,
        tactile_raw_grid: Any | None = None,
        tactile_deform_images: Any | None = None,
        tactile_raw_images: Any | None = None,
        tactile_raw_available: bool | None = None,
        prompt: str | None = None,
    ) -> OpenPICompActionChunkInferenceResult:
        tactile_inputs = prepare_tactile_image_inputs(
            tactile_deform_grid=tactile_deform_grid,
            tactile_raw_grid=tactile_raw_grid,
            tactile_deform_images=tactile_deform_images,
            tactile_raw_images=tactile_raw_images,
            tactile_raw_available=tactile_raw_available,
            output_size=self.config.tactile_image_size,
            tolerate_invalid_optional_raw=self.config.tolerate_invalid_optional_raw,
            raw_zero_is_unavailable=self.config.raw_zero_is_unavailable,
        )
        return self._infer_with_tactile_inputs(
            head_left_rgb=head_left_rgb,
            wrist_left_rgb=wrist_left_rgb,
            wrist_right_rgb=wrist_right_rgb,
            state_65d=state_65d,
            tactile_60d=tactile_60d,
            planner_features=planner_features,
            tactile_inputs=tactile_inputs,
            prompt=prompt,
        )

    def compute_training_loss(
        self,
        *,
        head_left_rgb: np.ndarray,
        wrist_left_rgb: np.ndarray,
        wrist_right_rgb: np.ndarray,
        state_65d: np.ndarray,
        tactile_60d: np.ndarray,
        planner_features: Mapping[str, Any],
        target_actions_65d: np.ndarray,
        tactile_deform_grid: Any | None = None,
        tactile_raw_grid: Any | None = None,
        tactile_deform_images: Any | None = None,
        tactile_raw_images: Any | None = None,
        tactile_raw_available: bool | None = None,
        prompt: str | None = None,
        rng_seed: int = 0,
        noise_samples: int = 1,
        sample_weight: float | None = 1.0,
        train: bool = False,
    ) -> OpenPICompActionChunkLossResult:
        tactile_inputs = prepare_tactile_image_inputs(
            tactile_deform_grid=tactile_deform_grid,
            tactile_raw_grid=tactile_raw_grid,
            tactile_deform_images=tactile_deform_images,
            tactile_raw_images=tactile_raw_images,
            tactile_raw_available=tactile_raw_available,
            output_size=self.config.tactile_image_size,
            tolerate_invalid_optional_raw=self.config.tolerate_invalid_optional_raw,
            raw_zero_is_unavailable=self.config.raw_zero_is_unavailable,
        )
        policy_input = self.build_policy_input(
            head_left_rgb=head_left_rgb,
            wrist_left_rgb=wrist_left_rgb,
            wrist_right_rgb=wrist_right_rgb,
            state_65d=state_65d,
            tactile_60d=tactile_60d,
            planner_features=planner_features,
            tactile_inputs=tactile_inputs,
            prompt=prompt,
        )
        return self.compute_training_loss_from_policy_input(
            policy_input,
            target_actions_65d=target_actions_65d,
            rng_seed=rng_seed,
            noise_samples=noise_samples,
            sample_weight=sample_weight,
            train=train,
        )

    def compute_training_loss_from_policy_input(
        self,
        policy_input: Mapping[str, Any],
        *,
        target_actions_65d: np.ndarray,
        rng_seed: int = 0,
        noise_samples: int = 1,
        sample_weight: float | None = 1.0,
        train: bool = False,
    ) -> OpenPICompActionChunkLossResult:
        if self._compute_loss_and_metrics is None:
            raise NotImplementedError("Training-style loss evaluation is only implemented for JAX OpenPI checkpoints.")

        expected_shape = (self.config.action_horizon, self.config.action_dim)
        target_actions = np.asarray(target_actions_65d, dtype=np.float32)
        if tuple(target_actions.shape) != expected_shape:
            raise ValueError(f"target_actions_65d must have shape {expected_shape}, got {target_actions.shape}.")
        if not np.isfinite(target_actions).all():
            raise ValueError("target_actions_65d contains NaN or Inf.")
        if noise_samples <= 0:
            raise ValueError(f"noise_samples must be positive, got {noise_samples}.")

        import jax
        import jax.numpy as jnp

        data = self._copy_tree(dict(policy_input))
        data["actions"] = np.array(target_actions, dtype=np.float32, copy=True)
        data["action_mask"] = np.ones(expected_shape, dtype=bool)
        if sample_weight is not None:
            data["sample_weight"] = np.asarray(float(sample_weight), dtype=np.float32)

        transformed = self.policy._input_transform(data)  # type: ignore[attr-defined]
        if "actions" not in transformed:
            raise RuntimeError("OpenPI input transforms did not preserve transformed actions for loss evaluation.")

        batched = jax.tree.map(lambda value: jnp.asarray(value)[np.newaxis, ...], transformed)
        observation = self._openpi_model.Observation.from_dict(batched)
        actions = batched["actions"]

        base_rng = jax.random.key(int(rng_seed))
        metric_values: dict[str, list[float]] = {}
        chunked_losses: list[np.ndarray] = []
        for noise_index in range(int(noise_samples)):
            rng = jax.random.fold_in(base_rng, int(noise_index))
            chunked_loss, metrics = self._compute_loss_and_metrics(rng, observation, actions, train=bool(train))
            chunked_np = np.asarray(jax.device_get(chunked_loss), dtype=np.float32)
            if chunked_np.shape[0] == 1:
                chunked_np = chunked_np[0]
            chunked_losses.append(chunked_np)
            metrics_np = jax.device_get(metrics)
            for key, value in metrics_np.items():
                metric_values.setdefault(str(key), []).append(float(np.asarray(value)))

        reduced_metrics = {
            key: float(np.mean(values, dtype=np.float64))
            for key, values in sorted(metric_values.items())
        }
        return OpenPICompActionChunkLossResult(
            metrics=reduced_metrics,
            chunked_loss=np.mean(np.stack(chunked_losses, axis=0), axis=0).astype(np.float32),
            transformed_actions=np.asarray(jax.device_get(actions[0]), dtype=np.float32),
            noise_samples=int(noise_samples),
            rng_seed=int(rng_seed),
        )

    @staticmethod
    def _copy_tree(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return np.array(value, copy=True)
        if isinstance(value, dict):
            return {key: OpenPICompActionChunkRuntime._copy_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [OpenPICompActionChunkRuntime._copy_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(OpenPICompActionChunkRuntime._copy_tree(item) for item in value)
        return value

    def infer_from_observation(
        self,
        observation: Mapping[str, Any],
        *,
        planner_features: Mapping[str, Any],
        prompt: str | None = None,
    ) -> OpenPICompActionChunkInferenceResult:
        tactile_inputs = prepare_tactile_from_observation(
            observation,
            output_size=self.config.tactile_image_size,
            tolerate_invalid_optional_raw=self.config.tolerate_invalid_optional_raw,
            raw_zero_is_unavailable=self.config.raw_zero_is_unavailable,
        )
        return self._infer_with_tactile_inputs(
            head_left_rgb=observation[HEAD_LEFT_KEY],
            wrist_left_rgb=observation[WRIST_LEFT_KEY],
            wrist_right_rgb=observation[WRIST_RIGHT_KEY],
            state_65d=observation[STATE_KEY],
            tactile_60d=observation[TACTILE_VECTOR_KEY],
            planner_features=planner_features,
            tactile_inputs=tactile_inputs,
            prompt=prompt or observation.get("prompt"),
        )

    def _infer_with_tactile_inputs(
        self,
        *,
        head_left_rgb: np.ndarray,
        wrist_left_rgb: np.ndarray,
        wrist_right_rgb: np.ndarray,
        state_65d: np.ndarray,
        tactile_60d: np.ndarray,
        planner_features: Mapping[str, Any],
        tactile_inputs: TactileImageInputs,
        prompt: str | None,
    ) -> OpenPICompActionChunkInferenceResult:
        started = time.perf_counter()
        policy_input = self.build_policy_input(
            head_left_rgb=head_left_rgb,
            wrist_left_rgb=wrist_left_rgb,
            wrist_right_rgb=wrist_right_rgb,
            state_65d=state_65d,
            tactile_60d=tactile_60d,
            planner_features=planner_features,
            tactile_inputs=tactile_inputs,
            prompt=prompt,
        )
        policy_started = time.perf_counter()
        policy_output = self.policy.infer(policy_input)
        policy_elapsed_ms = (time.perf_counter() - policy_started) * 1000.0

        actions = np.asarray(policy_output["actions"], dtype=np.float32)
        expected_shape = (self.config.action_horizon, self.config.action_dim)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(f"OpenPI policy returned actions shape {actions.shape}, expected {expected_shape}.")
        if self.config.require_finite_actions and not np.isfinite(actions).all():
            raise ValueError("OpenPI policy returned actions containing NaN or Inf.")

        total_runtime_ms = (time.perf_counter() - started) * 1000.0
        return OpenPICompActionChunkInferenceResult(
            actions=np.ascontiguousarray(actions, dtype=np.float32),
            prompt=str(policy_input["prompt"]),
            planner_available=bool(policy_input["planner_available"]),
            tactile_raw_available=bool(tactile_inputs.raw_available),
            tactile_raw_status=tactile_inputs.raw_status,
            policy_timing_ms=policy_elapsed_ms,
            total_runtime_ms=total_runtime_ms,
            policy_debug={
                "policy_timing": dict(policy_output.get("policy_timing", {})),
                "checkpoint_dir": str(self.checkpoint_dir),
                "config_name": self.config.config_name,
                "tokenizer_model_path": str(self.tokenizer_model_path),
                "policy_sample_steps": int(self.config.policy_sample_steps),
                "tactile": tactile_inputs.to_debug_dict(),
            },
        )
