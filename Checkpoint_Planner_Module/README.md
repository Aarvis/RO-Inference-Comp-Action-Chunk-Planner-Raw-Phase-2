# Checkpoint Planner Module

Token-first runtime checkpoint-planner module for `RO-Inference-Comp-Action-Chunk-Planner-Raw`.

This module sits after the shared DINO/OOI stage and before OpenPI. It does not load DINO and does not run OOI itself. The final wrapper should compute DINO patch tokens once for `head_left`, `wrist_left`, and `wrist_right`, run OOI from those tokens, compute the OOI crop tokens, then call this module.

## Runtime Contract

`CheckpointPlannerRuntime.infer_from_tokens(...)` consumes:

- `head_left_tokens`: DINOv3 ViT-S/16+ patch tokens, shape `[196, 384]`
- `wrist_left_tokens`: shape `[196, 384]`
- `wrist_right_tokens`: shape `[196, 384]`
- `ooi_tokens`: shape `[196, 384]`, or `None` as a fallback
- `state_65d`: shape `[65]`
- `tactile_60d`: shape `[60]`
- `ooi_target_present`: selects the planner branch

The runtime keeps a rolling history buffer of length `30`. At the start of an episode it left-pads the missing history with zeros and marks those rows invalid. Once more than 30 observations are seen, the oldest observation is discarded.

Branch selection:

- `ooi_target_present=True`: OpenPI features come from the `posterior` branch.
- `ooi_target_present=False`: OpenPI features come from the `prior` branch.

If OOI is absent, pass black-frame DINO crop tokens when the wrapper has them. Passing `ooi_tokens=None` is supported as a zero-token fallback, but the final wrapper should prefer the same black-frame crop-token convention used by the OOI stage.

The exported OpenPI feature dict contains:

- `planner_available`
- `planner_state_belief`, default `final` state belief, shape `[29]`
- `planner_progress_transition`, shape `[2]`
- `planner_uncertainty`, shape `[3]`
- `planner_history_latent`, shape `[512]`

## Files

- [run_checkpoint_planner_smoke_test.py](E:/Robot-Origami-Challenge/RO-Inference-Comp-Action-Chunk-Planner-Raw/Checkpoint_Planner_Module/run_checkpoint_planner_smoke_test.py): synthetic token smoke test
- [configs/checkpoint_planner_runtime_model.yaml](E:/Robot-Origami-Challenge/RO-Inference-Comp-Action-Chunk-Planner-Raw/Checkpoint_Planner_Module/configs/checkpoint_planner_runtime_model.yaml): runtime model/prior config
- [configs/checkpoint_planner_runtime_smoke_test.yaml](E:/Robot-Origami-Challenge/RO-Inference-Comp-Action-Chunk-Planner-Raw/Checkpoint_Planner_Module/configs/checkpoint_planner_runtime_smoke_test.yaml): smoke config
- [checkpoint_planner_runtime/runtime.py](E:/Robot-Origami-Challenge/RO-Inference-Comp-Action-Chunk-Planner-Raw/Checkpoint_Planner_Module/checkpoint_planner_runtime/runtime.py): reusable runtime class
- [checkpoint_planner_runtime/history.py](E:/Robot-Origami-Challenge/RO-Inference-Comp-Action-Chunk-Planner-Raw/Checkpoint_Planner_Module/checkpoint_planner_runtime/history.py): 30-step rolling history buffer

## Example

```powershell
cd E:\Robot-Origami-Challenge\RO-Inference-Comp-Action-Chunk-Planner-Raw\Checkpoint_Planner_Module
python .\run_checkpoint_planner_smoke_test.py --config .\configs\checkpoint_planner_runtime_smoke_test.yaml
```

Before running, confirm:

- `paths.planner_train_config`
- `paths.planner_checkpoint`
- `paths.planner_manifest_root`, when overriding the manifest path in the model config

## Config Sources

The runtime uses:

- planner architecture from [train_checkpoint_planner_224_headleft_tactile_distill.yaml](E:/Robot-Origami-Challenge/dataset-processing/checkpoint-planner/trainer/configs/train_checkpoint_planner_224_headleft_tactile_distill.yaml)
- tuned inference prior values from the gamma evaluation run: `gamma=10`, `gamma_future=15`, `current_ref_weight=0.3`, `near_future_weight=0.3`, `transition_completion_threshold=0.65`
- current full-dataset manifest under [Competition_Paper_Reprocessed_Dataset](E:/Robot-Origami-Challenge/Competition_Paper_Reprocessed_Dataset/metadata/checkpoint_planner_training/no_hmm_224_headleft_tactile_distill/manifest.json)
