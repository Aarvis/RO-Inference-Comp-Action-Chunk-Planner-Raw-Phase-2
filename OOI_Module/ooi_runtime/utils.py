from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def save_json(data: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_rgb_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb_image(path: str | Path, image_rgb: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(target), image_bgr):
        raise RuntimeError(f"Could not write image: {target}")


def resize_rgb_square(image_rgb: np.ndarray, size: int) -> np.ndarray:
    interpolation = cv2.INTER_LINEAR if image_rgb.shape[0] < size else cv2.INTER_AREA
    return cv2.resize(image_rgb, (size, size), interpolation=interpolation)


def preprocess_rgb_for_dino(image_rgb: np.ndarray) -> np.ndarray:
    array = image_rgb.astype(np.float32) / 255.0
    array = (array - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    return np.transpose(array, (2, 0, 1))


def cxcy_side_to_xyxy(
    cx: float,
    cy: float,
    side: float,
    image_w: int,
    image_h: int,
) -> tuple[float, float, float, float]:
    cx_px = cx * image_w
    cy_px = cy * image_h
    side_px = side * image_w
    x1 = max(0.0, cx_px - side_px * 0.5)
    y1 = max(0.0, cy_px - side_px * 0.5)
    x2 = min(float(image_w - 1), cx_px + side_px * 0.5)
    y2 = min(float(image_h - 1), cy_px + side_px * 0.5)
    return x1, y1, x2, y2


def scale_xyxy(
    xyxy: tuple[float, float, float, float],
    sx: float,
    sy: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy


def square_crop_rgb(
    image_rgb: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    output_size: int,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    side = max(x2 - x1, y2 - y1, 1.0)
    x1i = int(round(max(0.0, cx - side * 0.5)))
    y1i = int(round(max(0.0, cy - side * 0.5)))
    x2i = int(round(min(float(width), cx + side * 0.5)))
    y2i = int(round(min(float(height), cy + side * 0.5)))
    crop = image_rgb[y1i:y2i, x1i:x2i]
    if crop.size == 0:
        crop = image_rgb
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def draw_overlay_rgb(
    image_rgb: np.ndarray,
    xyxy: tuple[float, float, float, float],
    confidence: float,
) -> np.ndarray:
    overlay = image_rgb.copy()
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(
        overlay,
        (int(round(x1)), int(round(y1))),
        (int(round(x2)), int(round(y2))),
        (0, 255, 0),
        2,
    )
    cv2.putText(
        overlay,
        f"{confidence:.2f}",
        (int(round(x1)), max(15, int(round(y1)) - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    return overlay


def validate_square_rgb_uint8(image_rgb: np.ndarray, *, name: str) -> None:
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError(f"{name} must be a numpy array.")
    if image_rgb.dtype != np.uint8:
        raise TypeError(f"{name} must have dtype uint8, got {image_rgb.dtype}.")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"{name} must have shape [H, W, 3], got {image_rgb.shape}.")
    if image_rgb.shape[0] != image_rgb.shape[1]:
        raise ValueError(f"{name} must be square, got shape {image_rgb.shape}.")
