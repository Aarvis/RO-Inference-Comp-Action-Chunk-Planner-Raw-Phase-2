from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import cv2
import numpy as np


TACTILE_DEFORM_KEY = "observation/image/tactile_deform"
TACTILE_RAW_KEY = "observation/image/tactile_raw"
TACTILE_VECTOR_KEY = "observation/tactile"

TACTILE_DEFORM_GRID_SHAPE = (480, 1200, 3)
TACTILE_RAW_GRID_SHAPE = (480, 1600, 3)

FINGER_ORDER = (
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_little",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_little",
)


@dataclasses.dataclass(frozen=True)
class TactileImageInputs:
    deform_images: np.ndarray
    raw_images: np.ndarray
    raw_available: bool
    raw_status: str

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "deform_images_shape": list(self.deform_images.shape),
            "raw_images_shape": list(self.raw_images.shape),
            "raw_available": bool(self.raw_available),
            "raw_status": self.raw_status,
        }


def _is_uint8_image(value: Any, expected_shape: tuple[int, int, int]) -> bool:
    if not isinstance(value, np.ndarray):
        return False
    return value.dtype == np.dtype(np.uint8) and tuple(value.shape) == expected_shape


def _is_all_zero(value: np.ndarray) -> bool:
    return bool(value.size > 0 and not np.any(value))


def validate_vector(
    value: Any,
    *,
    name: str,
    expected_dim: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if tuple(array.shape) != (expected_dim,):
        raise ValueError(f"{name} must have shape {(expected_dim,)}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return np.ascontiguousarray(array, dtype=np.float32)


def validate_rgb_image(
    value: Any,
    *,
    name: str,
    output_size: int = 224,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array.")
    if value.dtype != np.dtype(np.uint8):
        raise TypeError(f"{name} must have dtype uint8, got {value.dtype}.")
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [H, W, 3], got {value.shape}.")
    if value.shape[:2] != (output_size, output_size):
        value = cv2.resize(value, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(value, dtype=np.uint8)


def split_tactile_grid(
    frame: np.ndarray,
    *,
    expected_shape: tuple[int, int, int],
    rows: int = 2,
    cols: int = 5,
    output_size: int = 224,
    name: str,
) -> np.ndarray:
    if not _is_uint8_image(frame, expected_shape):
        raise ValueError(f"{name} must be uint8{expected_shape}, got {getattr(frame, 'dtype', None)}{getattr(frame, 'shape', None)}.")
    height, width, channels = frame.shape
    if channels != 3 or height % rows or width % cols:
        raise ValueError(f"{name} has invalid tactile grid shape {frame.shape} for {rows}x{cols}.")
    cell_h = height // rows
    cell_w = width // cols
    cells: list[np.ndarray] = []
    for row in range(rows):
        for col in range(cols):
            cell = frame[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w]
            cells.append(cv2.resize(cell, (output_size, output_size), interpolation=cv2.INTER_LINEAR))
    return np.ascontiguousarray(np.stack(cells, axis=0).transpose(0, 3, 1, 2), dtype=np.uint8)


def coerce_tactile_crops(
    value: Any,
    *,
    name: str,
    output_size: int = 224,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.uint8):
        raise TypeError(f"{name} must have dtype uint8, got {array.dtype}.")
    if tuple(array.shape) == (10, 3, output_size, output_size):
        return np.ascontiguousarray(array, dtype=np.uint8)
    if tuple(array.shape) == (10, output_size, output_size, 3):
        return np.ascontiguousarray(array.transpose(0, 3, 1, 2), dtype=np.uint8)
    raise ValueError(
        f"{name} must have shape (10, 3, {output_size}, {output_size}) or "
        f"(10, {output_size}, {output_size}, 3), got {array.shape}."
    )


def prepare_tactile_image_inputs(
    *,
    tactile_deform_grid: Any | None = None,
    tactile_raw_grid: Any | None = None,
    tactile_deform_images: Any | None = None,
    tactile_raw_images: Any | None = None,
    tactile_raw_available: bool | None = None,
    output_size: int = 224,
    tolerate_invalid_optional_raw: bool = True,
    raw_zero_is_unavailable: bool = True,
) -> TactileImageInputs:
    if tactile_deform_images is not None:
        deform_images = coerce_tactile_crops(tactile_deform_images, name="tactile_deform_images", output_size=output_size)
    elif tactile_deform_grid is not None:
        deform_images = split_tactile_grid(
            tactile_deform_grid,
            expected_shape=TACTILE_DEFORM_GRID_SHAPE,
            output_size=output_size,
            name=TACTILE_DEFORM_KEY,
        )
    else:
        raise ValueError("tactile_deform_grid or tactile_deform_images is required.")

    raw_status = "missing"
    raw_available = False
    if tactile_raw_available is not None:
        raw_available = bool(tactile_raw_available)

    try:
        if tactile_raw_images is not None:
            raw_images = coerce_tactile_crops(tactile_raw_images, name="tactile_raw_images", output_size=output_size)
            if raw_zero_is_unavailable and _is_all_zero(raw_images):
                raw_status = "all_zero"
                raw_available = False
            else:
                raw_status = "provided_crops"
                raw_available = bool(tactile_raw_available) if tactile_raw_available is not None else True
        elif tactile_raw_grid is not None:
            if not _is_uint8_image(tactile_raw_grid, TACTILE_RAW_GRID_SHAPE):
                raise ValueError(
                    f"{TACTILE_RAW_KEY} must be uint8{TACTILE_RAW_GRID_SHAPE}, got "
                    f"{getattr(tactile_raw_grid, 'dtype', None)}{getattr(tactile_raw_grid, 'shape', None)}."
                )
            if raw_zero_is_unavailable and _is_all_zero(tactile_raw_grid):
                raw_images = np.zeros_like(deform_images)
                raw_status = "all_zero"
                raw_available = False
            else:
                raw_images = split_tactile_grid(
                    tactile_raw_grid,
                    expected_shape=TACTILE_RAW_GRID_SHAPE,
                    output_size=output_size,
                    name=TACTILE_RAW_KEY,
                )
                raw_status = "provided_grid"
                raw_available = bool(tactile_raw_available) if tactile_raw_available is not None else True
        else:
            raw_images = np.zeros_like(deform_images)
            raw_status = "missing"
            raw_available = False
            raw_available = False
    except Exception:
        if not tolerate_invalid_optional_raw:
            raise
        raw_images = np.zeros_like(deform_images)
        raw_status = "invalid"
        raw_available = False

    if not raw_available:
        raw_images = np.zeros_like(deform_images)
    return TactileImageInputs(
        deform_images=np.ascontiguousarray(deform_images, dtype=np.uint8),
        raw_images=np.ascontiguousarray(raw_images, dtype=np.uint8),
        raw_available=bool(raw_available),
        raw_status=raw_status,
    )


def prepare_tactile_from_observation(
    observation: Mapping[str, Any],
    *,
    output_size: int = 224,
    tolerate_invalid_optional_raw: bool = True,
    raw_zero_is_unavailable: bool = True,
) -> TactileImageInputs:
    return prepare_tactile_image_inputs(
        tactile_deform_grid=observation.get(TACTILE_DEFORM_KEY),
        tactile_raw_grid=observation.get(TACTILE_RAW_KEY),
        output_size=output_size,
        tolerate_invalid_optional_raw=tolerate_invalid_optional_raw,
        raw_zero_is_unavailable=raw_zero_is_unavailable,
    )
