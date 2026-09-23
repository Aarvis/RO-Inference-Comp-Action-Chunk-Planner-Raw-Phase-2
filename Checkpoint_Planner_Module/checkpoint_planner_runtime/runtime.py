from __future__ import annotations

import contextlib
import dataclasses
import logging
import math
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .config import as_path, load_json, load_yaml
from .history import PlannerHistoryBuffer
from .planner_model import CheckpointPlannerModel
from .prior_state import PlannerPriorState


Precision = Literal["fp32", "fp16", "bf16"]
BranchName = Literal["posterior", "prior"]
ValueVariant = Literal["raw", "final"]
TokenArray = torch.Tensor | np.ndarray

logger = logging.getLogger(__name__)


def compute_uncertainty_features(checkpoint_probs: np.ndarray, eps: float) -> np.ndarray:
    probs = np.asarray(checkpoint_probs, dtype=np.float32)
    if probs.ndim != 1:
        raise ValueError(f"Expected checkpoint probabilities with rank 1, got shape {probs.shape}.")
    entropy = -np.sum(probs * np.log(np.clip(probs, eps, None)))
    denom = math.log(probs.shape[0])
    entropy_norm = entropy / denom if denom > 0 else entropy
    sorted_probs = np.sort(probs, axis=0)
    top1 = float(sorted_probs[-1])
    top2 = float(sorted_probs[-2]) if probs.shape[0] > 1 else 0.0
    margin = top1 - top2
    return np.asarray([entropy_norm, top1, margin], dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class CheckpointPlannerBranchResult:
    branch: BranchName
    raw_checkpoint_index: int
    final_checkpoint_index: int
    raw_state_index: int
    final_state_index: int
    phase_pred_index: int
    raw_checkpoint_label: str
    final_checkpoint_label: str
    raw_state_label: str
    final_state_label: str
    phase_pred_label: str
    prog_pred: float
    trans_pred: float
    raw_checkpoint_probs: np.ndarray
    final_checkpoint_probs: np.ndarray
    phase_probs: np.ndarray
    raw_state_belief: np.ndarray
    final_state_belief: np.ndarray
    progress_transition: np.ndarray
    uncertainty_features: np.ndarray
    temporal_latent: np.ndarray

    def state_belief(self, variant: ValueVariant) -> np.ndarray:
        return self.final_state_belief if variant == "final" else self.raw_state_belief

    def checkpoint_probs(self, variant: ValueVariant) -> np.ndarray:
        return self.final_checkpoint_probs if variant == "final" else self.raw_checkpoint_probs

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "raw_checkpoint_index": self.raw_checkpoint_index,
            "final_checkpoint_index": self.final_checkpoint_index,
            "raw_state_index": self.raw_state_index,
            "final_state_index": self.final_state_index,
            "phase_pred_index": self.phase_pred_index,
            "raw_checkpoint_label": self.raw_checkpoint_label,
            "final_checkpoint_label": self.final_checkpoint_label,
            "raw_state_label": self.raw_state_label,
            "final_state_label": self.final_state_label,
            "phase_pred_label": self.phase_pred_label,
            "prog_pred": self.prog_pred,
            "trans_pred": self.trans_pred,
            "uncertainty_entropy_norm": float(self.uncertainty_features[0]),
            "uncertainty_pmax": float(self.uncertainty_features[1]),
            "uncertainty_margin": float(self.uncertainty_features[2]),
        }


@dataclasses.dataclass(frozen=True)
class CheckpointPlannerInferenceResult:
    sequence_length: int
    history_length: int
    timestamp: float
    step_index: int
    ooi_target_present: bool
    selected_branch: BranchName
    state_belief_variant: ValueVariant
    uncertainty_source: ValueVariant
    c_ref_before: int
    c_ref_after: int
    continuity_prior_applied: bool
    selected: CheckpointPlannerBranchResult
    posterior: CheckpointPlannerBranchResult
    prior: CheckpointPlannerBranchResult
    planner_available: bool = True

    @property
    def planner_state_belief(self) -> np.ndarray:
        return self.selected.state_belief(self.state_belief_variant)

    @property
    def planner_progress_transition(self) -> np.ndarray:
        return self.selected.progress_transition

    @property
    def planner_uncertainty(self) -> np.ndarray:
        return self.selected.uncertainty_features

    @property
    def planner_history_latent(self) -> np.ndarray:
        return self.selected.temporal_latent

    def to_openpi_features(self) -> dict[str, np.ndarray | bool]:
        return {
            "planner_available": bool(self.planner_available),
            "planner_state_belief": self.planner_state_belief.astype(np.float32, copy=False),
            "planner_progress_transition": self.planner_progress_transition.astype(np.float32, copy=False),
            "planner_uncertainty": self.planner_uncertainty.astype(np.float32, copy=False),
            "planner_history_latent": self.planner_history_latent.astype(np.float32, copy=False),
        }

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "history_length": self.history_length,
            "timestamp": self.timestamp,
            "step_index": self.step_index,
            "ooi_target_present": self.ooi_target_present,
            "selected_branch": self.selected_branch,
            "state_belief_variant": self.state_belief_variant,
            "uncertainty_source": self.uncertainty_source,
            "planner_available": self.planner_available,
            "c_ref_before": self.c_ref_before,
            "c_ref_after": self.c_ref_after,
            "continuity_prior_applied": self.continuity_prior_applied,
            "selected": self.selected.to_summary_dict(),
            "posterior": self.posterior.to_summary_dict(),
            "prior": self.prior.to_summary_dict(),
        }


class CheckpointPlannerRuntime:
    def __init__(
        self,
        *,
        planner_train_config_path: str | Path,
        planner_checkpoint_path: str | Path,
        manifest_root_override: str | Path | None = None,
        device: str = "cuda",
        precision: Precision = "fp16",
        patch_count: int | None = None,
        value_variant: ValueVariant = "final",
        uncertainty_source: ValueVariant = "raw",
        transition_completion_threshold: float | None = None,
        timestamp_source: str = "step_index",
        apply_continuity_prior: bool = True,
    ) -> None:
        self.device = self._resolve_device(device)
        self.precision = str(precision)
        self.value_variant = self._validate_value_variant(value_variant, name="value_variant")
        self.uncertainty_source = self._validate_value_variant(uncertainty_source, name="uncertainty_source")
        self.timestamp_source = str(timestamp_source)
        self.apply_continuity_prior = bool(apply_continuity_prior)
        if not self.apply_continuity_prior and self.value_variant != "raw":
            raise ValueError("value_variant must be 'raw' when apply_continuity_prior is false.")
        self.step_index = 0

        self.planner_train_config_path = as_path(planner_train_config_path)
        self.planner_checkpoint_path = as_path(planner_checkpoint_path)
        if not self.planner_train_config_path.is_file():
            raise FileNotFoundError(f"Planner train/runtime config not found: {self.planner_train_config_path}")
        if not self.planner_checkpoint_path.is_file():
            raise FileNotFoundError(f"Planner checkpoint not found: {self.planner_checkpoint_path}")

        self.planner_config = load_yaml(self.planner_train_config_path)
        manifest_root_or_file = self._resolve_manifest_root(manifest_root_override)
        if manifest_root_or_file.is_dir():
            manifest_path = manifest_root_or_file / "manifest.json"
            self.manifest_root = manifest_root_or_file
        else:
            manifest_path = manifest_root_or_file
            self.manifest_root = manifest_path.parent
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Planner manifest not found: {manifest_path}")
        self.manifest = load_json(manifest_path)

        self.checkpoint_names = [*self.manifest["checkpoint_names"], "DONE"]
        self.state_labels = list(self.manifest.get("state_labels") or self.manifest["labels"])
        self.done_index = len(self.checkpoint_names) - 1
        self.sequence_length = int(self.planner_config["data"].get("sequence_length", 30))
        model_cfg = self.planner_config["model"]
        self.patch_dim = int(model_cfg.get("patch_dim", 384))
        self.patch_count = int(patch_count or self.planner_config.get("runtime", {}).get("patch_count", 196))
        self.state_dim = int(model_cfg.get("state_dim", len(self.manifest["state_stats"]["mean"])))
        self.tactile_dim = int(model_cfg.get("tactile_dim", len(self.manifest.get("tactile_stats", {}).get("mean", []))))
        self.nominal_dt_seconds = float(model_cfg.get("nominal_dt_seconds", 0.5))
        self.prior_cfg = dict(self.planner_config.get("prior", {}))
        if transition_completion_threshold is not None:
            self.prior_cfg["transition_completion_threshold"] = float(transition_completion_threshold)
        self.transition_completion_threshold = float(self.prior_cfg.get("transition_completion_threshold", 0.65))

        logger.info(
            "Loading checkpoint planner checkpoint=%s manifest=%s device=%s sequence_length=%d patch_count=%d patch_dim=%d",
            self.planner_checkpoint_path,
            manifest_path,
            self.device,
            self.sequence_length,
            self.patch_count,
            self.patch_dim,
        )
        planner_checkpoint = self._torch_load_checkpoint(self.planner_checkpoint_path)
        model_state = self._extract_model_state(planner_checkpoint)
        tactile_stats = self.manifest.get("tactile_stats") or {}
        self.model = CheckpointPlannerModel(
            config=self.planner_config,
            num_checkpoints=int(self.manifest["num_checkpoints"]),
            state_mean=self.manifest["state_stats"]["mean"],
            state_std=self.manifest["state_stats"]["std"],
            tactile_mean=tactile_stats.get("mean"),
            tactile_std=tactile_stats.get("std"),
        )
        self.model.load_state_dict(model_state, strict=True)
        self.model.to(self.device).eval()

        self.history = PlannerHistoryBuffer(
            sequence_length=self.sequence_length,
            patch_count=self.patch_count,
            patch_dim=self.patch_dim,
            state_dim=self.state_dim,
            tactile_dim=self.tactile_dim,
            device=self.device,
        )
        self.prior_state = PlannerPriorState(
            transition_completion_threshold=self.transition_completion_threshold,
            done_index=self.done_index,
        )

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if str(device_name).startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested for checkpoint planner, but torch.cuda.is_available() is false. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_name)

    @staticmethod
    def _validate_value_variant(value: str, *, name: str) -> ValueVariant:
        if value not in {"raw", "final"}:
            raise ValueError(f"{name} must be 'raw' or 'final', got {value!r}.")
        return value  # type: ignore[return-value]

    @staticmethod
    def _autocast_dtype(precision: str) -> torch.dtype | None:
        if precision == "fp16":
            return torch.float16
        if precision == "bf16":
            return torch.bfloat16
        return None

    @staticmethod
    def _torch_load_checkpoint(path: Path) -> Any:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    @classmethod
    def _extract_model_state(cls, checkpoint: Any) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("model", "model_state_dict", "state_dict"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return cls._strip_module_prefix_if_needed(value)
            return cls._strip_module_prefix_if_needed(checkpoint)
        raise TypeError(f"Unsupported checkpoint object type: {type(checkpoint)!r}")

    @staticmethod
    def _strip_module_prefix_if_needed(state_dict: dict[str, Any]) -> dict[str, Any]:
        keys = list(state_dict.keys())
        if keys and all(isinstance(key, str) and key.startswith("module.") for key in keys):
            return {key[len("module.") :]: value for key, value in state_dict.items()}
        return state_dict

    def _resolve_manifest_root(self, manifest_root_override: str | Path | None) -> Path:
        manifest_value: str | Path | None = manifest_root_override
        if manifest_value in (None, ""):
            paths_cfg = self.planner_config.get("paths", {})
            if not isinstance(paths_cfg, dict):
                paths_cfg = {}
            manifest_value = (
                paths_cfg.get("manifest_root")
                or paths_cfg.get("manifest_path")
                or self.planner_config.get("manifest_root")
                or self.planner_config.get("manifest_path")
            )
        if manifest_value in (None, ""):
            raise KeyError("Planner runtime config must define paths.manifest_root or paths.manifest_path.")

        manifest_path = as_path(manifest_value)
        if manifest_path.is_absolute():
            return manifest_path
        return (self.planner_train_config_path.parent / manifest_path).resolve()

    def reset(self) -> None:
        self.history.reset()
        self.prior_state.reset()
        self.step_index = 0

    def infer_from_tokens(
        self,
        *,
        head_left_tokens: TokenArray,
        wrist_left_tokens: TokenArray,
        wrist_right_tokens: TokenArray,
        ooi_tokens: TokenArray | None,
        state_65d: np.ndarray | torch.Tensor,
        tactile_60d: np.ndarray | torch.Tensor | None,
        ooi_target_present: bool,
        timestamp: float | None = None,
    ) -> CheckpointPlannerInferenceResult:
        head_left = self._to_token_tensor(head_left_tokens, name="head_left_tokens")
        wrist_left = self._to_token_tensor(wrist_left_tokens, name="wrist_left_tokens")
        wrist_right = self._to_token_tensor(wrist_right_tokens, name="wrist_right_tokens")
        ooi = self._to_token_tensor(ooi_tokens, name="ooi_tokens") if ooi_tokens is not None else torch.zeros_like(head_left)
        state = self._to_vector_tensor(state_65d, expected_dim=self.state_dim, name="state_65d")
        tactile = self._to_vector_tensor(tactile_60d, expected_dim=self.tactile_dim, name="tactile_60d")
        resolved_timestamp = self._resolve_timestamp(timestamp)

        self.history.append(
            head_left_tokens=head_left,
            wrist_left_tokens=wrist_left,
            wrist_right_tokens=wrist_right,
            ooi_tokens=ooi,
            state=state,
            tactile_60d=tactile,
            timestamp=resolved_timestamp,
            ooi_valid=bool(ooi_target_present),
        )

        c_ref_before = int(self.prior_state.last_completed_checkpoint)
        batch = self.history.build_batch(last_completed_checkpoint=c_ref_before)
        autocast_dtype = self._autocast_dtype(self.precision)
        autocast_context = (
            torch.autocast(device_type=self.device.type, dtype=autocast_dtype)
            if self.device.type == "cuda" and autocast_dtype is not None
            else contextlib.nullcontext()
        )
        with torch.inference_mode():
            with autocast_context:
                outputs = self.model(
                    batch,
                    apply_prior=self.apply_continuity_prior,
                    prior_config=self.prior_cfg if self.apply_continuity_prior else None,
                )

        posterior = self._branch_result("posterior", outputs["posterior"])
        prior = self._branch_result("prior", outputs["prior"])
        selected_branch: BranchName = "posterior" if bool(ooi_target_present) else "prior"
        selected = posterior if selected_branch == "posterior" else prior

        if self.apply_continuity_prior:
            self.prior_state.update(
                chosen_checkpoint_index=selected.final_checkpoint_index,
                phase_pred_index=selected.phase_pred_index,
                trans_pred=selected.trans_pred,
            )
        c_ref_after = int(self.prior_state.last_completed_checkpoint)
        result = CheckpointPlannerInferenceResult(
            sequence_length=self.sequence_length,
            history_length=self.history.num_valid_steps,
            timestamp=float(resolved_timestamp),
            step_index=int(self.step_index),
            ooi_target_present=bool(ooi_target_present),
            selected_branch=selected_branch,
            state_belief_variant=self.value_variant,
            uncertainty_source=self.uncertainty_source,
            c_ref_before=c_ref_before,
            c_ref_after=c_ref_after,
            continuity_prior_applied=self.apply_continuity_prior,
            selected=selected,
            posterior=posterior,
            prior=prior,
        )
        self.step_index += 1
        return result

    def _branch_result(self, branch: BranchName, outputs: dict[str, torch.Tensor]) -> CheckpointPlannerBranchResult:
        raw_checkpoint_probs = outputs["raw_checkpoint_probs"][0].detach().float().cpu().numpy().astype(np.float32, copy=False)
        final_checkpoint_probs = outputs["final_checkpoint_probs"][0].detach().float().cpu().numpy().astype(np.float32, copy=False)
        phase_probs = torch.softmax(outputs["phase_logits"], dim=-1)[0].detach().float().cpu().numpy().astype(np.float32, copy=False)
        raw_state_belief = outputs["raw_state_belief"][0].detach().float().cpu().numpy().astype(np.float32, copy=False)
        final_state_belief = outputs["final_state_belief"][0].detach().float().cpu().numpy().astype(np.float32, copy=False)
        temporal_latent = outputs["temporal_latent"][0].detach().float().cpu().numpy().astype(np.float32, copy=False)
        prog_pred = float(outputs["prog_pred"][0].detach().float().cpu().item())
        trans_pred = float(outputs["trans_pred"][0].detach().float().cpu().item())

        raw_checkpoint_index = int(np.argmax(raw_checkpoint_probs))
        final_checkpoint_index = int(np.argmax(final_checkpoint_probs))
        raw_state_index = int(np.argmax(raw_state_belief))
        final_state_index = int(np.argmax(final_state_belief))
        if final_checkpoint_index < self.done_index:
            phase_pred_index = int(np.argmax(phase_probs[final_checkpoint_index]))
            phase_pred_label = "P" if phase_pred_index == 0 else "T"
        else:
            phase_pred_index = -1
            phase_pred_label = "DONE"

        uncertainty_probs = raw_checkpoint_probs if self.uncertainty_source == "raw" else final_checkpoint_probs
        return CheckpointPlannerBranchResult(
            branch=branch,
            raw_checkpoint_index=raw_checkpoint_index,
            final_checkpoint_index=final_checkpoint_index,
            raw_state_index=raw_state_index,
            final_state_index=final_state_index,
            phase_pred_index=phase_pred_index,
            raw_checkpoint_label=self.checkpoint_names[raw_checkpoint_index],
            final_checkpoint_label=self.checkpoint_names[final_checkpoint_index],
            raw_state_label=self.state_labels[raw_state_index],
            final_state_label=self.state_labels[final_state_index],
            phase_pred_label=phase_pred_label,
            prog_pred=prog_pred,
            trans_pred=trans_pred,
            raw_checkpoint_probs=raw_checkpoint_probs.copy(),
            final_checkpoint_probs=final_checkpoint_probs.copy(),
            phase_probs=phase_probs.copy(),
            raw_state_belief=raw_state_belief.copy(),
            final_state_belief=final_state_belief.copy(),
            progress_transition=np.asarray([prog_pred, trans_pred], dtype=np.float32),
            uncertainty_features=compute_uncertainty_features(
                uncertainty_probs,
                eps=float(self.prior_cfg.get("eps", 1e-6)),
            ),
            temporal_latent=temporal_latent.copy(),
        )

    def _to_token_tensor(self, tokens: TokenArray, *, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(tokens).to(device=self.device)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        expected_shape = (self.patch_count, self.patch_dim)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}.")
        return tensor.to(device=self.device, dtype=torch.float16)

    def _to_vector_tensor(
        self,
        values: np.ndarray | torch.Tensor | None,
        *,
        expected_dim: int,
        name: str,
    ) -> torch.Tensor:
        if values is None:
            return torch.zeros((expected_dim,), dtype=torch.float32, device=self.device)
        if isinstance(values, np.ndarray) and not values.flags.writeable:
            values = np.array(values, copy=True)
        tensor = torch.as_tensor(values, dtype=torch.float32).to(device=self.device)
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tuple(tensor.shape) != (expected_dim,):
            raise ValueError(f"{name} must have shape {(expected_dim,)}, got {tuple(tensor.shape)}.")
        return tensor

    def _resolve_timestamp(self, explicit_timestamp: float | None) -> float:
        if explicit_timestamp is not None:
            return float(explicit_timestamp)
        if self.timestamp_source == "step_index":
            return float(self.step_index) * self.nominal_dt_seconds
        if self.timestamp_source == "monotonic":
            return float(time.perf_counter())
        raise ValueError(f"Unsupported timestamp_source={self.timestamp_source!r}.")
