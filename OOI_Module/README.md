# OOI Module

OOI runtime slice for `RO-Inference-Comp-Action-Chunk-Planner-Raw`.

This module is intentionally separate from `E:/Robot-Origami-Challenge/OOI_Zoom_Module` so the final competition wrapper can be assembled without changing the training project.

## Contract

Inputs match the latest inference kit:

- `observation/image/head_left`: RGB `uint8[224, 224, 3]`
- `observation/image/wrist_left`: RGB `uint8[224, 224, 3]`
- `observation/image/wrist_right`: RGB `uint8[224, 224, 3]`

The default config points at the OOI checkpoint trained from:

- `E:/Robot-Origami-Challenge/OOI_Zoom_Module/configs/train_224_latest_inference_ooi.yaml`

The checkpoint embedded config remains authoritative for the model architecture.

## Runtime Modes

`shared_dino: true` is the path intended for the final pipeline:

1. DINO encodes `head_left`, `wrist_left`, and `wrist_right` once.
2. OOI runs from those raw DINO patch tokens.
3. The same raw DINO patch tokens can be passed to checkpoint planner.
4. If OOI is present, the OOI crop can be DINO-encoded once for checkpoint planner posterior.

`shared_dino: false` keeps a standalone smoke-test path where OOI owns its DINO encoder internally.

## Low-Confidence OOI

The runtime returns:

- `target_present`
- `confidence_threshold`
- `used_black_frame`
- `ooi_crop_rgb`

If confidence is below threshold and `black_frame_below_threshold: true`, `ooi_crop_rgb` is a black `224x224` frame. For the final checkpoint-planner integration, use `target_present=False` to select the planner prior branch instead of treating zero tokens as black-frame DINO tokens.

## Smoke Test

Fill the image paths in:

- `E:/Robot-Origami-Challenge/RO-Inference-Comp-Action-Chunk-Planner-Raw/OOI_Module/configs/ooi_zoom_runtime.yaml`

Then run:

```powershell
cd E:\Robot-Origami-Challenge\RO-Inference-Comp-Action-Chunk-Planner-Raw\OOI_Module
python .\run_ooi_zoom_inference.py --config .\configs\ooi_zoom_runtime.yaml
```

The script writes:

- `ooi_prediction.json`
- `ooi_crop_rgb.png`
- `head_left_overlay_model_rgb.png`
- model-resolution input images for inspection
