# Phase-2 Training-to-Inference Alignment

This package deploys Phase-2 `pi05_origami_comp_action_chunk_phase2`
checkpoints without modifying the earlier inference projects.

## Planner contract

```text
planner_enabled: true
  OOI target present -> posterior branch, raw planner outputs
  OOI target absent  -> prior branch, raw planner outputs
  gamma continuity   -> disabled; no continuity-state update

planner_enabled: false
  DINO, OOI, and checkpoint planner are not constructed
  OpenPI receives the all-zero masked planner prefix from planner-dropout training
```

`prior branch` is the planner's no-OOI prediction branch. It is not the gamma
continuity prior.

## Public inference contract versus training targets

The public `origami-zenoh-v1` contract contains only:

```yaml
action_horizon: T
```

The server returns `float32[T, 65]` absolute joint-position actions. It does
not publish `action_chunk_stride`, and the robot executes all returned actions
consecutively at its normal control cadence.

`action_chunk_stride` remains important only internally: it chooses replay and
training targets at `t + j * stride`. It is never an executor instruction.

| Checkpoint profile | Public horizon | Replay/training stride | Replay target positions |
| --- | ---: | ---: | --- |
| H25/S1 | 25 | 1 | `t, t+1, ..., t+24` |
| H15/S2 | 15 | 2 | `t, t+2, ..., t+28` |

Thus an H15/S2 checkpoint intentionally outputs 15 actions that are executed
back-to-back, producing the intended time-compressed behavior.

## Select the exact Phase-2 profile

The repository includes two frozen OpenPI source profiles because historical
H25/S1 and current H15/S2 checkpoints used different registrations under the
same Phase-2 config name. Choose the profile before preparing a bundle or
building Docker:

```bash
cd /home/ubuntu/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2

# Current Phase 2: 15 actions; replay checks t, t+2, ..., t+28.
python ./scripts/select_phase2_profile.py --profile h15_s2 --bundle-root ./model_bundle

# Earlier Phase 2: 25 actions; replay checks t, t+1, ..., t+24.
python ./scripts/select_phase2_profile.py --profile h25_s1 --bundle-root ./model_bundle
```

The script changes only these alignment values:

- `openpi.config_name`, `openpi.vendored_source_root`, and source snapshot;
- `openpi.model.action_horizon` and `openpi.data.action_chunk_stride`;
- public `server.action_horizon`;
- replay `dataset_replay.action_horizon` and `dataset_replay.action_chunk_stride`.

It intentionally does not add a stride to public server metadata.

## Required checkpoint-specific edits

After selecting a profile, update the copied runtime configuration before
materialising the final bundle:

| Training property | Location | Requirement |
| --- | --- | --- |
| Saved Phase-2 checkpoint | `OpenPI_Module/configs/openpi_comp_action_chunk_runtime.yaml` → `paths.checkpoint_dir` | Full saved step directory containing `params/` and `assets/` |
| Normalization assets | inside that checkpoint | `assets/<asset_id>/norm_stats.json` must be present |
| Planner-enabled paths | `configs/dataset_replay.yaml` | DINO, OOI, planner checkpoint, planner config, and manifest |
| Planner-off test | `configs/dataset_replay_planner_off.yaml` or `runtime.planner_enabled: false` | No planner modules are loaded |
| Public output size | `server.action_horizon` | Must match selected model horizon |
| Replay comparison stride | `dataset_replay.action_chunk_stride` | Must match training target stride |

Do not substitute a model with a different action horizon. The runtime verifies
that the selected OpenPI configuration and public action horizon agree before
loading inference.

## Prepare, verify, replay, build

```bash
cd /home/ubuntu/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2

python ./scripts/prepare_comp_action_chunk_bundle.py \
  --source-config ./configs/dataset_replay.yaml \
  --bundle-root ./model_bundle \
  --copy-mode copy \
  --overwrite

python ./scripts/verify_comp_action_chunk_bundle.py --bundle-root ./model_bundle
```

Run one dataset replay after every profile/checkpoint change. This is where the
stride is validated against ground truth. Then validate the Docker server with
the inference-kit checker using `--expected-horizon 15` for H15/S2 or
`--expected-horizon 25` for H25/S1.
