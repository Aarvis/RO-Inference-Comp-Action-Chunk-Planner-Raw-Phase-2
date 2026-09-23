from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .model import resolve_model_source
from .utils import preprocess_rgb_for_dino, validate_square_rgb_uint8


Precision = Literal["fp32", "fp16", "bf16"]
logger = logging.getLogger(__name__)


class SharedDinoRuntime:
    """Inference-only DINO patch-token extractor shared by OOI and planner."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cuda",
        precision: Precision = "fp16",
        cache_dir: str | Path | None = None,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        output_dtype: torch.dtype = torch.float16,
    ) -> None:
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("Install transformers to use the shared DINO runtime.") from exc

        self.device = self._resolve_device(device)
        self.precision = str(precision)
        self.output_dtype = output_dtype
        model_source = resolve_model_source(
            model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        logger.info(
            "Loading shared DINO source=%s device=%s precision=%s local_only=%s",
            model_source,
            self.device,
            self.precision,
            bool(local_files_only),
        )
        self.encoder = AutoModel.from_pretrained(
            model_source,
            local_files_only=bool(local_files_only),
            trust_remote_code=bool(trust_remote_code),
        ).to(self.device)
        self.encoder.eval()

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

    @staticmethod
    def extract_patch_tokens(tokens: torch.Tensor) -> torch.Tensor:
        n_tokens = int(tokens.shape[1])
        grid = int(math.sqrt(n_tokens))
        if grid * grid == n_tokens:
            return tokens
        grid_without_cls = int(math.sqrt(n_tokens - 1))
        if grid_without_cls * grid_without_cls == n_tokens - 1:
            return tokens[:, 1:, :]
        patch_count = grid * grid
        if patch_count <= 0:
            raise RuntimeError(f"Could not infer patch-token grid from {n_tokens} tokens.")
        return tokens[:, -patch_count:, :]

    def encode_rgb_images(self, images_rgb: list[np.ndarray]) -> torch.Tensor:
        if not images_rgb:
            raise ValueError("encode_rgb_images() requires at least one image.")
        preprocessed = []
        for index, image_rgb in enumerate(images_rgb):
            validate_square_rgb_uint8(image_rgb, name=f"images_rgb[{index}]")
            preprocessed.append(preprocess_rgb_for_dino(image_rgb))
        batch = torch.from_numpy(np.stack(preprocessed, axis=0)).float().to(self.device)

        autocast_dtype = self._autocast_dtype(self.precision)
        context = (
            torch.autocast(device_type=self.device.type, dtype=autocast_dtype)
            if self.device.type == "cuda" and autocast_dtype is not None
            else torch.no_grad()
        )
        with torch.inference_mode():
            with context:
                outputs = self.encoder(pixel_values=batch)
        if hasattr(outputs, "last_hidden_state"):
            tokens = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            tokens = outputs[0]
        else:
            raise RuntimeError("DINO encoder output does not expose last_hidden_state.")
        tokens = self.extract_patch_tokens(tokens)
        return tokens.to(dtype=self.output_dtype)
