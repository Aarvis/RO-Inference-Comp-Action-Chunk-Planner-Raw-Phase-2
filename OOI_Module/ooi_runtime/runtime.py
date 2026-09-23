from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .losses import gather_predictions
from .model import OOIZoomModel, load_compatible_state_dict
from .utils import (
    cxcy_side_to_xyxy,
    preprocess_rgb_for_dino,
    resize_rgb_square,
    scale_xyxy,
    square_crop_rgb,
    validate_square_rgb_uint8,
)


InputMode = Literal["auto", "input_224", "input_480"]
Precision = Literal["fp32", "fp16", "bf16"]
TokenArray = torch.Tensor | np.ndarray


@dataclasses.dataclass(frozen=True)
class OOIBoxPrediction:
    cx_norm: float
    cy_norm: float
    side_norm: float
    confidence: float
    xyxy_model: tuple[float, float, float, float]
    xyxy_source: tuple[float, float, float, float]


@dataclasses.dataclass(frozen=True)
class OOIInferenceResult:
    source_input_size: int
    model_input_size: int
    output_crop_size: int
    resolved_input_mode: str
    confidence_threshold: float
    target_present: bool
    used_black_frame: bool
    prediction: OOIBoxPrediction
    head_left_model_rgb: np.ndarray
    wrist_left_model_rgb: np.ndarray | None
    wrist_right_model_rgb: np.ndarray | None
    ooi_crop_rgb: np.ndarray

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source_input_size": self.source_input_size,
            "model_input_size": self.model_input_size,
            "output_crop_size": self.output_crop_size,
            "resolved_input_mode": self.resolved_input_mode,
            "confidence_threshold": self.confidence_threshold,
            "target_present": self.target_present,
            "used_black_frame": self.used_black_frame,
            "prediction": {
                "cx_norm": self.prediction.cx_norm,
                "cy_norm": self.prediction.cy_norm,
                "side_norm": self.prediction.side_norm,
                "confidence": self.prediction.confidence,
                "xyxy_model": list(self.prediction.xyxy_model),
                "xyxy_source": list(self.prediction.xyxy_source),
            },
        }


class OOIZoomRuntime:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        precision: Precision = "fp16",
        model_cache_dir: str | Path | None = None,
        local_files_only: bool | None = True,
        trust_remote_code: bool | None = False,
        output_crop_size: int | None = None,
        confidence_threshold: float = 0.1,
        black_frame_below_threshold: bool = True,
        load_encoder: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"OOI checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if "config" not in checkpoint or "model" not in checkpoint:
            raise KeyError(f"Checkpoint is missing required keys 'config' and 'model': {self.checkpoint_path}")

        self.raw_checkpoint = checkpoint
        self.checkpoint_config = checkpoint["config"]
        self.model_config = dict(self.checkpoint_config["model"])
        if model_cache_dir is not None:
            self.model_config["model_cache_dir"] = str(model_cache_dir)
        if local_files_only is not None:
            self.model_config["local_files_only"] = bool(local_files_only)
        if trust_remote_code is not None:
            self.model_config["trust_remote_code"] = bool(trust_remote_code)

        self.model_input_size = int(self.checkpoint_config["data"].get("image_size", 224))
        self.output_crop_size = int(output_crop_size or self.model_input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.black_frame_below_threshold = bool(black_frame_below_threshold)
        self.device = self._resolve_device(device)
        self.precision = str(precision)
        self.load_encoder = bool(load_encoder)

        self.model = OOIZoomModel(self.model_config, load_encoder=self.load_encoder)
        incompatible = load_compatible_state_dict(self.model, checkpoint["model"], strict=self.load_encoder)
        if not self.load_encoder:
            unexpected = sorted(incompatible.unexpected_keys)
            missing = sorted(incompatible.missing_keys)
            non_encoder_unexpected = [key for key in unexpected if not key.startswith("encoder.")]
            if missing or non_encoder_unexpected:
                raise RuntimeError(
                    "Shared-token OOI load encountered incompatible keys. "
                    f"missing={missing} unexpected_non_encoder={non_encoder_unexpected}"
                )
        self.model.to(self.device).eval()

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if str(device_name).startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_name)

    @staticmethod
    def _autocast_dtype(precision: str) -> torch.dtype | None:
        if precision == "fp16":
            return torch.float16
        if precision == "bf16":
            return torch.bfloat16
        return None

    def infer(
        self,
        head_left_rgb: np.ndarray,
        wrist_left_rgb: np.ndarray | None,
        wrist_right_rgb: np.ndarray | None,
        *,
        input_mode: InputMode = "auto",
    ) -> OOIInferenceResult:
        if self.model.encoder is None:
            raise RuntimeError("OOIZoomRuntime.infer() requires load_encoder=True. Use infer_from_tokens() instead.")

        source_size = int(head_left_rgb.shape[0])
        resolved_input_mode = self._resolve_input_mode(source_size, input_mode)
        head_left_model_rgb, wrist_left_model_rgb, wrist_right_model_rgb = self.prepare_model_images(
            head_left_rgb,
            wrist_left_rgb,
            wrist_right_rgb,
        )

        top_tensor = self._to_tensor(head_left_model_rgb)
        wrist_left_tensor = self._to_tensor(wrist_left_model_rgb) if wrist_left_model_rgb is not None else None
        wrist_right_tensor = self._to_tensor(wrist_right_model_rgb) if wrist_right_model_rgb is not None else None

        if bool(self.model_config.get("use_wrist_inputs", False)) and (wrist_left_tensor is None or wrist_right_tensor is None):
            raise ValueError("This OOI checkpoint expects wrist inputs, but one or both wrist images are missing.")

        with torch.inference_mode():
            with self._maybe_autocast():
                outputs = self.model(top_tensor, wrist_left_tensor, wrist_right_tensor)
        return self._result_from_outputs(
            outputs,
            source_input_size=source_size,
            resolved_input_mode=resolved_input_mode,
            head_left_model_rgb=head_left_model_rgb,
            wrist_left_model_rgb=wrist_left_model_rgb,
            wrist_right_model_rgb=wrist_right_model_rgb,
        )

    def infer_from_tokens(
        self,
        *,
        head_left_tokens: TokenArray,
        wrist_left_tokens: TokenArray | None,
        wrist_right_tokens: TokenArray | None,
        head_left_model_rgb: np.ndarray,
        source_input_size: int | None = None,
        input_mode: InputMode = "auto",
    ) -> OOIInferenceResult:
        validate_square_rgb_uint8(head_left_model_rgb, name="head_left_model_rgb")
        source_size = int(source_input_size or head_left_model_rgb.shape[0])
        resolved_input_mode = self._resolve_input_mode(source_size, input_mode)
        head_left_model_rgb = self._to_model_resolution(head_left_model_rgb)

        top_tokens = self._to_batched_tokens(head_left_tokens)
        left_tokens = self._to_batched_tokens(wrist_left_tokens) if wrist_left_tokens is not None else None
        right_tokens = self._to_batched_tokens(wrist_right_tokens) if wrist_right_tokens is not None else None
        if bool(self.model_config.get("use_wrist_inputs", False)) and (left_tokens is None or right_tokens is None):
            raise ValueError("This OOI checkpoint expects wrist tokens, but one or both wrist token tensors are missing.")

        with torch.inference_mode():
            with self._maybe_autocast():
                outputs = self.model.forward_from_tokens(
                    top_tokens,
                    left_tokens,
                    right_tokens,
                    tokens_are_projected=False,
                )
        return self._result_from_outputs(
            outputs,
            source_input_size=source_size,
            resolved_input_mode=resolved_input_mode,
            head_left_model_rgb=head_left_model_rgb,
            wrist_left_model_rgb=None,
            wrist_right_model_rgb=None,
        )

    def prepare_model_images(
        self,
        head_left_rgb: np.ndarray,
        wrist_left_rgb: np.ndarray | None,
        wrist_right_rgb: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        validate_square_rgb_uint8(head_left_rgb, name="head_left_rgb")
        if wrist_left_rgb is not None:
            validate_square_rgb_uint8(wrist_left_rgb, name="wrist_left_rgb")
        if wrist_right_rgb is not None:
            validate_square_rgb_uint8(wrist_right_rgb, name="wrist_right_rgb")
        return (
            self._to_model_resolution(head_left_rgb),
            self._to_model_resolution(wrist_left_rgb) if wrist_left_rgb is not None else None,
            self._to_model_resolution(wrist_right_rgb) if wrist_right_rgb is not None else None,
        )

    def _result_from_outputs(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        source_input_size: int,
        resolved_input_mode: str,
        head_left_model_rgb: np.ndarray,
        wrist_left_model_rgb: np.ndarray | None,
        wrist_right_model_rgb: np.ndarray | None,
    ) -> OOIInferenceResult:
        prediction = gather_predictions(outputs)[0].detach().float().cpu().numpy()
        cx_norm, cy_norm, side_norm, confidence = [float(value) for value in prediction.tolist()]
        xyxy_model = cxcy_side_to_xyxy(
            cx_norm,
            cy_norm,
            side_norm,
            self.model_input_size,
            self.model_input_size,
        )
        scale = float(source_input_size) / float(self.model_input_size)
        xyxy_source = scale_xyxy(xyxy_model, scale, scale)

        target_present = bool(confidence >= self.confidence_threshold)
        used_black_frame = bool(not target_present and self.black_frame_below_threshold)
        if used_black_frame:
            ooi_crop_rgb = np.zeros((self.output_crop_size, self.output_crop_size, 3), dtype=np.uint8)
        else:
            ooi_crop_rgb = square_crop_rgb(
                head_left_model_rgb,
                *xyxy_model,
                output_size=self.output_crop_size,
            )
        return OOIInferenceResult(
            source_input_size=int(source_input_size),
            model_input_size=self.model_input_size,
            output_crop_size=self.output_crop_size,
            resolved_input_mode=resolved_input_mode,
            confidence_threshold=self.confidence_threshold,
            target_present=target_present,
            used_black_frame=used_black_frame,
            prediction=OOIBoxPrediction(
                cx_norm=cx_norm,
                cy_norm=cy_norm,
                side_norm=side_norm,
                confidence=confidence,
                xyxy_model=xyxy_model,
                xyxy_source=xyxy_source,
            ),
            head_left_model_rgb=head_left_model_rgb,
            wrist_left_model_rgb=wrist_left_model_rgb,
            wrist_right_model_rgb=wrist_right_model_rgb,
            ooi_crop_rgb=ooi_crop_rgb,
        )

    def _maybe_autocast(self) -> contextlib.AbstractContextManager[None]:
        autocast_dtype = self._autocast_dtype(self.precision)
        if self.device.type == "cuda" and autocast_dtype is not None:
            return torch.autocast(device_type=self.device.type, dtype=autocast_dtype)
        return contextlib.nullcontext()

    def _resolve_input_mode(self, source_size: int, input_mode: InputMode) -> str:
        if input_mode == "auto":
            if source_size == 224:
                return "input_224"
            if source_size == 480:
                return "input_480"
            return "input_custom"
        if input_mode == "input_224" and source_size != 224:
            raise ValueError(f"input_mode='input_224' expects 224x224 input, got {source_size}x{source_size}.")
        if input_mode == "input_480" and source_size != 480:
            raise ValueError(f"input_mode='input_480' expects 480x480 input, got {source_size}x{source_size}.")
        return input_mode

    def _to_model_resolution(self, image_rgb: np.ndarray | None) -> np.ndarray | None:
        if image_rgb is None:
            return None
        if image_rgb.shape[0] == self.model_input_size and image_rgb.shape[1] == self.model_input_size:
            return np.ascontiguousarray(image_rgb)
        return np.ascontiguousarray(resize_rgb_square(image_rgb, self.model_input_size))

    def _to_tensor(self, image_rgb: np.ndarray) -> torch.Tensor:
        chw = preprocess_rgb_for_dino(image_rgb)
        return torch.from_numpy(chw).float().unsqueeze(0).to(self.device)

    def _to_batched_tokens(self, tokens: TokenArray) -> torch.Tensor:
        tensor = torch.as_tensor(tokens, device=self.device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(f"Expected OOI token tensor with shape [N,D] or [B,N,D], got {tuple(tensor.shape)}.")
        return tensor
