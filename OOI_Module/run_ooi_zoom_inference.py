from __future__ import annotations

import argparse
from pathlib import Path

from ooi_runtime import OOIZoomRuntime, SharedDinoRuntime
from ooi_runtime.utils import draw_overlay_rgb, ensure_dir, load_rgb_image, load_yaml, save_json, save_rgb_image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "ooi_zoom_runtime.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-frame OOI zoom inference for RO inference integration.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    paths_cfg = config["paths"]
    runtime_cfg = config.get("runtime", {})
    output_cfg = config.get("output", {})
    shared_dino = bool(runtime_cfg.get("shared_dino", True))

    output_dir = ensure_dir(paths_cfg["output_dir"])
    runtime = OOIZoomRuntime(
        checkpoint_path=paths_cfg["checkpoint"],
        device=str(runtime_cfg.get("device", "cuda")),
        precision=str(runtime_cfg.get("precision", "fp16")),
        model_cache_dir=runtime_cfg.get("model_cache_dir"),
        local_files_only=runtime_cfg.get("local_files_only", True),
        trust_remote_code=runtime_cfg.get("trust_remote_code", False),
        output_crop_size=int(runtime_cfg.get("output_crop_size", 224)),
        confidence_threshold=float(runtime_cfg.get("confidence_threshold", 0.1)),
        black_frame_below_threshold=bool(runtime_cfg.get("black_frame_below_threshold", True)),
        load_encoder=not shared_dino,
    )

    head_left_rgb = load_rgb_image(paths_cfg["head_left_image"])
    wrist_left_rgb = load_rgb_image(paths_cfg["wrist_left_image"]) if paths_cfg.get("wrist_left_image") else None
    wrist_right_rgb = load_rgb_image(paths_cfg["wrist_right_image"]) if paths_cfg.get("wrist_right_image") else None

    if shared_dino:
        head_left_model_rgb, wrist_left_model_rgb, wrist_right_model_rgb = runtime.prepare_model_images(
            head_left_rgb,
            wrist_left_rgb,
            wrist_right_rgb,
        )
        if wrist_left_model_rgb is None or wrist_right_model_rgb is None:
            raise ValueError("shared_dino mode requires head_left, wrist_left, and wrist_right images.")
        dino = SharedDinoRuntime(
            str(runtime_cfg.get("dino_model_name_or_path") or runtime.model_config["dino_model_name"]),
            device=str(runtime_cfg.get("device", "cuda")),
            precision=str(runtime_cfg.get("precision", "fp16")),
            cache_dir=runtime_cfg.get("model_cache_dir"),
            local_files_only=bool(runtime_cfg.get("local_files_only", True)),
            trust_remote_code=bool(runtime_cfg.get("trust_remote_code", False)),
        )
        tokens = dino.encode_rgb_images([head_left_model_rgb, wrist_left_model_rgb, wrist_right_model_rgb])
        result = runtime.infer_from_tokens(
            head_left_tokens=tokens[0],
            wrist_left_tokens=tokens[1],
            wrist_right_tokens=tokens[2],
            head_left_model_rgb=head_left_model_rgb,
            source_input_size=int(head_left_model_rgb.shape[0]),
            input_mode=str(runtime_cfg.get("input_mode", "auto")),
        )
    else:
        result = runtime.infer(
            head_left_rgb,
            wrist_left_rgb,
            wrist_right_rgb,
            input_mode=str(runtime_cfg.get("input_mode", "auto")),
        )

    save_json(result.to_json_dict(), output_dir / "ooi_prediction.json")
    if bool(output_cfg.get("save_crop", True)):
        save_rgb_image(output_dir / "ooi_crop_rgb.png", result.ooi_crop_rgb)
    if bool(output_cfg.get("save_overlay", True)):
        overlay = draw_overlay_rgb(
            result.head_left_model_rgb,
            result.prediction.xyxy_model,
            result.prediction.confidence,
        )
        save_rgb_image(output_dir / "head_left_overlay_model_rgb.png", overlay)
    if bool(output_cfg.get("save_resized_inputs", True)):
        save_rgb_image(output_dir / "head_left_model_rgb.png", result.head_left_model_rgb)
        if result.wrist_left_model_rgb is not None:
            save_rgb_image(output_dir / "wrist_left_model_rgb.png", result.wrist_left_model_rgb)
        if result.wrist_right_model_rgb is not None:
            save_rgb_image(output_dir / "wrist_right_model_rgb.png", result.wrist_right_model_rgb)

    print("OOI zoom inference")
    print(f"  checkpoint        : {paths_cfg['checkpoint']}")
    print(f"  head_left_image   : {paths_cfg['head_left_image']}")
    print(f"  output_dir        : {output_dir}")
    print(f"  source_input_size : {result.source_input_size}")
    print(f"  model_input_size  : {result.model_input_size}")
    print(f"  output_crop_size  : {result.output_crop_size}")
    print(f"  runtime_mode      : {'shared_dino_tokens' if shared_dino else 'standalone_dino'}")
    print(f"  threshold         : {result.confidence_threshold:.4f}")
    print(f"  confidence        : {result.prediction.confidence:.4f}")
    print(f"  target_present    : {result.target_present}")
    print(f"  used_black_frame  : {result.used_black_frame}")
    print(f"  xyxy_model        : {tuple(round(v, 2) for v in result.prediction.xyxy_model)}")
    print(f"  xyxy_source       : {tuple(round(v, 2) for v in result.prediction.xyxy_source)}")


if __name__ == "__main__":
    main()
