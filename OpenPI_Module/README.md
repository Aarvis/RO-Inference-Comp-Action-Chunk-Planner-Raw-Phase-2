# OpenPI Comp Action Chunk Module

Runtime wrapper for the trained `pi05_origami_comp_action_chunk` OpenPI policy.

This module expects the upstream wrapper to provide:

- latest inference-kit RGB observations for `head_left`, `wrist_left`, and `wrist_right`
- current `state` with shape `[65]`
- tactile force vector with shape `[60]`
- checkpoint-planner features from `Checkpoint_Planner_Module`
- tactile deform grid, and optional tactile raw grid

The runtime loads a trained OpenPI checkpoint through `openpi.policies.policy_config.create_trained_policy`.
That keeps the training-time transforms intact: prompt tokenization, quantile normalization,
delta-action output space, and the final `AbsoluteActions` transform.

## Raw tactile behavior

`observation/image/tactile_raw` is optional in `inferencekit_09_02_2026`.

The wrapper maps raw tactile as:

```text
valid uint8[480, 1600, 3] raw grid
    -> crop to uint8[10, 3, 224, 224]
    -> tactile_raw_available=True

missing, None, wrong dtype, wrong shape, invalid, or all-zero raw grid
    -> zero uint8[10, 3, 224, 224] raw crops
    -> tactile_raw_available=False
```

This matters because OpenPI's FTP tactile prefix encoder uses `tactile_raw_available`
to choose between the deform-only prefix and the deform+raw prefix. Zero raw images
must not be marked available.

## Config

Edit:

```text
RO-Inference-Comp-Action-Chunk-Planner-Raw/OpenPI_Module/configs/openpi_comp_action_chunk_runtime.yaml
```

Key paths:

- `paths.checkpoint_dir`: trained checkpoint directory containing `params/` and `assets/`
- `paths.tokenizer_model_path`: local `paligemma_tokenizer.model`
- `paths.asset_id`: should be `competition_paper_reprocessed_origami_comp_action_chunk`

OpenPI source is loaded from the project-vendored copy at:

```text
RO-Inference-Comp-Action-Chunk-Planner-Raw/vendor/openpi/src
```

That vendored source must match the OpenPI checkout used for the final
`pi05_origami_comp_action_chunk` training run.

For final packaging, place the trained checkpoint under:

```text
RO-Inference-Comp-Action-Chunk-Planner-Raw/OpenPI_Module/model_bundles/pi05_origami_comp_action_chunk
```

The checkpoint should include:

```text
params/
assets/competition_paper_reprocessed_origami_comp_action_chunk/norm_stats.json
```

## Smoke tests

Input-preprocessing only:

```bash
python run_openpi_smoke_test.py --input-only
```

Full policy load and inference:

```bash
python run_openpi_smoke_test.py
```

The full smoke test requires the trained OpenPI checkpoint and local OpenPI dependencies.
