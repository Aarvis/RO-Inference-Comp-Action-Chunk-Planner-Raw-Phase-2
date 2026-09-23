# Comp Action Chunk Model Bundle

This folder is the single runtime asset root for
`RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2`.
`bundle.yaml` uses paths relative to this folder so the bundle can be moved into
a Docker image without rewriting module configs.

OpenPI runtime source is not stored in this folder. It lives in the parent
project at the profile-selected `vendor/openpi*/src` path. It must match the
exact Phase-2 source revision that trained the selected checkpoint.

Expected layout:

```text
model_bundle/
  bundle.yaml
  assets/
    paligemma_tokenizer.model
  configs/
    ooi_zoom_runtime.yaml
    checkpoint_planner_runtime_model.yaml
    openpi_comp_action_chunk_runtime.yaml
  weights/
    dino/dinov3-vits16plus-pretrain-lvd1689m/
      config.json
      model.safetensors
      preprocessor_config.json
    ooi/best.pt
    checkpoint_planner/checkpoint.pt
    checkpoint_planner/manifest/manifest.json
    openpi/pi05_origami_comp_action_chunk_phase2/
      params/
      assets/
  dataset_replay/
    episodes/...   # optional small replay set for Docker smoke tests
```

Populate this layout with `scripts/prepare_comp_action_chunk_bundle.py`, then
check it with `scripts/verify_comp_action_chunk_bundle.py`.

The `openpi_comp_action_chunk_runtime.yaml` file stores
the selected `openpi.config_name` plus a snapshot of the training-critical
OpenPI model, data, weight-loader, and optimizer settings. The action horizon
and its replay-only action stride must match the corresponding training run
exactly. See
`../README_CONFIGURATION_ALIGNMENT.md` before building an image.
