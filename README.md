# RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2

Standalone integrated inference package for Phase-2
`pi05_origami_comp_action_chunk_phase2` checkpoints. It does not modify or
depend on `RO-Inference-Comp-Action-Chunk` or
`RO-Inference-Comp-Action-Chunk-Planner-Raw` at runtime.

Planner-enabled inference uses raw planner outputs only: posterior when OOI is
present and prior branch when OOI is absent. Gamma continuity post-processing
is disabled. Setting `runtime.planner_enabled: false` bypasses DINO, OOI, and
the checkpoint planner entirely and supplies the masked zero planner prefix
used in planner-dropout training.

The default profile is H15/S2. The included H25/S1 profile is for the earlier
Phase-2 checkpoints. Read [configuration alignment](README_CONFIGURATION_ALIGNMENT.md)
before changing a checkpoint, model settings, or profile.

The runtime path is:

1. Shared DINOv3 ViT-S/16+ patch tokens for `head_left`, `wrist_left`, and `wrist_right`.
2. OOI crop prediction from the shared camera tokens.
3. Checkpoint-planner inference from camera tokens, optional OOI tokens, `state_65d`, and `tactile_60d`.
4. OpenPI inference with planner prefix features, FTP tactile prefix inputs, `state_65d`, `tactile_60d`, and the prompt.
5. Output `float32[action_horizon,65]` absolute joint-position chunks in radians.

The public `origami-zenoh-v1` metadata intentionally publishes only
`action_horizon`; it does **not** publish a training action stride. The robot
executes returned actions consecutively at its normal control cadence. For
H15/S2, the model was trained against `t, t+2, ..., t+28`, then its 15 output
actions are intentionally executed as 15 consecutive control actions.

The public server accepts the latest `inferencekit_09_02_2026` `origami-zenoh-v1` observation schema, including optional `observation/image/tactile_raw`.

## Bundle Layout

`model_bundle/bundle.yaml` is the single source of runtime paths. The Docker image is built around this folder.

OpenPI runtime code is vendored in the project at:

```text
vendor/openpi/src/
  openpi/...
  openpi_client/...
  future_latent_predictor/...
```

The active source profile must match the exact Phase-2 training revision for
the checkpoint being packaged. The default current profile is
`vendor/openpi/src`; the preserved H25/S1 source is
`vendor/openpi_phase2_h25_s1/src`.

Required files:

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
      assets/competition_paper_reprocessed_origami_comp_action_chunk/norm_stats.json
  dataset_replay/
    episodes/...   # optional small replay set baked into the image
```

`model_bundle/configs/openpi_comp_action_chunk_runtime.yaml` contains the
runtime `openpi.config_name: pi05_origami_comp_action_chunk_phase2` plus a snapshot of
the training-critical OpenPI model, data, weight-loader, and optimizer config
values used for packaging checks.

### Runtime planner override

The bundle setting `runtime.planner_enabled` is the default. To switch an
already-built Docker image at launch time, set `ORIGAMI_PLANNER_ENABLED`:

```bash
# Run DINO, OOI, and the checkpoint planner; pass its features to OpenPI.
-e ORIGAMI_PLANNER_ENABLED=true

# Skip those modules; OpenPI receives the masked-zero prefix used for
# planner-dropout training.
-e ORIGAMI_PLANNER_ENABLED=false
```

Omit the variable to preserve the value in `model_bundle/bundle.yaml`.

## Prepare Bundle

Edit `configs/dataset_replay.yaml` first if the source checkpoints are at different paths. Then materialize the bundle:

```bash
cd /home/ubuntu/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2

python ./scripts/prepare_comp_action_chunk_bundle.py \
  --source-config ./configs/dataset_replay.yaml \
  --bundle-root ./model_bundle \
  --copy-mode copy \
  --overwrite
```

Use `--copy-mode hardlink` only when all source files are on the same filesystem and you want to avoid duplicating large files locally. Do not use `--copy-mode symlink` for the final Docker build, because the image must contain real files.

If the script reports missing sources, copy them manually into the expected `model_bundle/` locations above, or fix the source paths in `configs/dataset_replay.yaml` and rerun the script.

To refresh the project-vendored OpenPI source from the sibling OpenPI checkout,
for example `/home/ubuntu/RO-openpi`, run:

```bash
cd /home/ubuntu/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2

python ./scripts/prepare_comp_action_chunk_bundle.py \
  --source-config ./configs/dataset_replay.yaml \
  --bundle-root ./model_bundle \
  --copy-mode copy \
  --include-openpi-source \
  --overwrite
```

That command updates `vendor/openpi/src`; it does not add code under
`model_bundle/`.

## Add Replay Episodes

For internal replay smoke tests inside Docker, copy a small raw dataset subset into:

```text
model_bundle/dataset_replay/episodes/<episode_uid>/
  arrays/state_65d.npy
  arrays/action_65d.npy
  arrays/tactile_60d.npy
  arrays/timestamps.npy       # optional
  arrays/frame_index.npy      # optional
  videos/head_left.mp4
  videos/wrist_left.mp4
  videos/wrist_right.mp4
  videos/tactile_deform.mp4
  videos/tactile_raw.mp4      # optional, required for deform_plus_raw replay
```

Dataset replay runs three tactile modes by default:

- `deform_only`
- `deform_plus_raw`
- `mixed_50`

The replay log prints action error, training-style OpenPI `loss_base_flow`, raw tactile availability, planner availability, OOI target presence, and per-stage timings.

Replay also shows live `tqdm` progress bars for:

- replay planning by episode;
- each tactile mode's total sample count;
- each tactile mode's episode count;
- each active episode's frame/sample progress.

Server inference logs one INFO line for metadata/reset requests and, by default, one INFO line per inference request with request ID, count, action shape, raw tactile status, planner/OOI status, selected checkpoint, and inference timings. To reduce inference log volume, edit:

```yaml
server:
  log_inference_requests: true
  log_inference_every_n: 10
```

## Verify Before Docker

Run layout verification before building:

```bash
cd /home/ubuntu/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2
python ./scripts/verify_comp_action_chunk_bundle.py --bundle-root ./model_bundle
```

If the replay dataset should also be baked into the image:

```bash
python ./scripts/verify_comp_action_chunk_bundle.py --bundle-root ./model_bundle --require-dataset-replay
```

During setup only, this command is useful for checking path rewrites before heavy weights are copied:

```bash
python ./scripts/verify_comp_action_chunk_bundle.py \
  --bundle-root ./model_bundle \
  --allow-missing-model-files
```

## Test Replay Before Docker

Run a small replay locally before building the image:

```bash
cd /home/ubuntu/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python ./run_dataset_replay.py \
  --bundle-root ./model_bundle \
  --warmup-inferences 1 \
  --max-episodes 1 \
  --max-total-samples 8000 \
  --max-loss-samples 800 \
  --loss-every-n-samples 10 \
  --output-dir ./outputs/dataset_replay_smoke
```

For a dry count without model loading:

```bash
python ./run_dataset_replay.py --bundle-root ./model_bundle --dry-run
```

## Test Server Before Docker

Start a local Zenoh router using the latest inference kit instructions, then start this policy server:

```bash
cd /home/ubuntu/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2

export ORIGAMI_ZENOH_ENDPOINT=tcp/127.0.0.1:17447
export ORIGAMI_SESSION_ID=local-contract-test
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python3 ./serve_origami_comp_action_chunk_policy.py \
  --bundle-root ./model_bundle \
  --execution-mode async \
  --warmup-inferences 1
```

In another terminal, validate against the public checker:

```bash
docker rm -f origami-zenoh-router >/dev/null 2>&1 || true

docker run -d --name origami-zenoh-router \
  -p 127.0.0.1:17447:7447 \
  eclipse/zenoh:latest \
  -l tcp/0.0.0.0:7447 \
  --no-multicast-scouting \
  --cfg 'transport/shared_memory/enabled:false'
```


```bash
cd /home/ubuntu/inferencekit_09_02_2026/sharpa_north_ces_lite_sdk-main

uv sync

uv run --no-sync python ./examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:17447 \
  --session-id local-contract-test \
  --timeout 180 \
  --requests 3 \
  --expected-horizon 15
```

## Build Docker Image

Build from this directory as the Docker context:

```bash
cd /home/ubuntu/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2

docker build --build-arg EXECUTION_MODE=async -f ./docker/submission-zenoh-bundled.Dockerfile -t ro-inference-comp-action-chunk-planner-raw-phase2:async .
```

The build context must contain the populated `model_bundle/`. The final image should not rely on host mounts, host Python packages, Hugging Face cache, or internet access at runtime.

## Test Inside Docker

Verify the bundled assets inside the built image:

```bash
IMAGE='ro-inference-comp-action-chunk2:async'

docker run --rm --gpus all \
  --shm-size 8g \
  -e ORIGAMI_WARMUP_INFERENCES='10' \
  -e ORIGAMI_OPENPI_PARAM_DTYPE='checkpoint' \
  -e XLA_PYTHON_CLIENT_PREALLOCATE='true' \
  -e ORIGAMI_JAX_MEM_FRACTION='0.70' \
  "$IMAGE" \
  dataset-replay \
  --max-episodes 1 \
  --max-total-samples 800 \
  --max-loss-samples 800 \
  --loss-every-n-samples 10 \
  --output-dir /tmp/dataset_replay_validate
```

```bash
docker run --rm --gpus all "$IMAGE" verify-bundle
```

Run a small internal dataset replay from the baked-in replay episodes:

```bash
docker run --rm --gpus all \
  --shm-size 8g \
  -e ORIGAMI_JAX_MEM_FRACTION=0.60 \
  -e ORIGAMI_WARMUP_INFERENCES=1 \
  "$IMAGE" \
  dataset-replay \
  --max-episodes 1 \
  --max-total-samples 8000 \
  --max-loss-samples 800 \
  --loss-every-n-samples 10 \
  --output-dir /tmp/dataset_replay_smoke
```

The dataset replay mode does not require `ORIGAMI_ZENOH_ENDPOINT` or `ORIGAMI_SESSION_ID`.

## Test Docker Zenoh Contract

Use the router image pinned by the public inference kit:

```bash
ROUTER_IMAGE='eclipse/zenoh@sha256:157965d71e0bfd0a044d76a985ff0e5c306ad3968929168fb9678cd2a7fec23f'
IMAGE='ro-inference-comp-action-chunk-fp32:async'
SESSION='local-contract-test'

docker network create origami-contract-test

docker run -d --name origami-contract-router \
  --network origami-contract-test \
  -p 127.0.0.1:17447:7447 \
  "$ROUTER_IMAGE" \
  -l tcp/0.0.0.0:7447 \
  --no-multicast-scouting \
  --cfg 'transport/shared_memory/enabled:false'

docker run -d --name origami-contract-policy \
  --network origami-contract-test \
  --gpus all \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges=true \
  --user 65532:65532 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=4g \
  --tmpfs /run:rw,noexec,nosuid,nodev,size=64m \
  --shm-size 8g \
  --memory 32g \
  --cpus 8 \
  --pids-limit 512 \
  -e ORIGAMI_ZENOH_ENDPOINT=tcp/origami-contract-router:7447 \
  -e ORIGAMI_OPENPI_PARAM_DTYPE='checkpoint' \
  -e ORIGAMI_SESSION_ID="$SESSION" \
  -e ORIGAMI_JAX_MEM_FRACTION=0.60 \
  "$IMAGE" \
  serve
```

Watch startup:

```bash
docker logs -f origami-contract-policy
```

Run the validator:

```bash
cd /path/to/inferencekit_09_02_2026/sharpa_north_ces_lite_sdk-main

uv run --no-sync python examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:17447 \
  --session-id "$SESSION" \
  --timeout 180 \
  --requests 3 \
  --expected-horizon 15
```

Clean up:

```bash
docker rm -f origami-contract-policy origami-contract-router
docker network rm origami-contract-test
```

## Entrypoint Modes

The image entrypoint supports:

```bash
IMAGE='ro-inference-comp-action-chunk2:async'

docker run --rm --gpus all "$IMAGE" serve
docker run --rm --gpus all "$IMAGE" verify-bundle
docker run --rm --gpus all "$IMAGE" dataset-replay --dry-run
docker run --rm -it "$IMAGE" bash
```

Serving requires only:

```text
ORIGAMI_ZENOH_ENDPOINT
ORIGAMI_SESSION_ID
```

Useful optional env vars:

```text
EXECUTION_MODE=async
ORIGAMI_MODEL_BUNDLE=/app/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2/model_bundle
ORIGAMI_WARMUP_INFERENCES=1
ORIGAMI_REPLAY_WARMUP_TRAINING_LOSS=1
ORIGAMI_JAX_MEM_FRACTION=0.60
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

## Final Export

After the Docker validator passes:

```bash
IMAGE='ro-inference-comp-action-chunk2:async'
ARCHIVE='ro-inference-comp-action-chunk2.tar'

docker save -o "$ARCHIVE" "$IMAGE"
zstd -T0 -19 "$ARCHIVE"
sha256sum "${ARCHIVE}.zst" > "${ARCHIVE}.zst.sha256"
zstd -t "${ARCHIVE}.zst"
```

Keep the image tag or digest, archive, checksum, expected horizon, and the runtime resource assumptions used during validation.
