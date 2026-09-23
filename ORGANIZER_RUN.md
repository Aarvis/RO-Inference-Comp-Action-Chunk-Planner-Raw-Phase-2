# RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2 Organizer Run Guide

Runs the submitted Docker image as an `origami-zenoh-v1` policy server.

Expected output:

```text
actions: float32[15, 65]
```

## 1. Requirements

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

The host needs Docker, NVIDIA Docker GPU support, `zstd`, and the inference-kit checker if validation is required.

## 2. Load Image

```bash
zstd -dc ro-inference-comp-action-chunk-planner-raw-phase2.tar.zst | docker load
docker images | grep ro-inference-comp-action-chunk-planner-raw-phase2
```

Set the image tag:

```bash
IMAGE='ro-inference-comp-action-chunk-planner-raw-phase2:async'
SESSION='local-contract-test'
ROUTER_IMAGE='eclipse/zenoh@sha256:157965d71e0bfd0a044d76a985ff0e5c306ad3968929168fb9678cd2a7fec23f'
```

If `docker load` prints a different tag, set `IMAGE` to that exact tag.

## 3. Start Router

```bash
docker rm -f origami-contract-policy origami-contract-router >/dev/null 2>&1 || true
docker network inspect origami-contract-test >/dev/null 2>&1 || docker network create origami-contract-test

docker run -d --name origami-contract-router \
  --network origami-contract-test \
  -p 127.0.0.1:17447:7447 \
  "$ROUTER_IMAGE" \
  -l tcp/0.0.0.0:7447 \
  --no-multicast-scouting \
  --cfg 'transport/shared_memory/enabled:false'
```

## 4. Start Policy

```bash
docker run -d --name origami-contract-policy \
  --network origami-contract-test \
  --gpus all \
  --shm-size 8g \
  -e ORIGAMI_ZENOH_ENDPOINT='tcp/origami-contract-router:7447' \
  -e ORIGAMI_SESSION_ID="$SESSION" \
  -e EXECUTION_MODE='async' \
  -e ORIGAMI_WARMUP_INFERENCES='10' \
  -e ORIGAMI_OPENPI_PARAM_DTYPE='checkpoint' \
  -e XLA_PYTHON_CLIENT_PREALLOCATE='true' \
  -e ORIGAMI_JAX_MEM_FRACTION='0.70' \
  "$IMAGE" \
  serve
```

Wait for readiness:

```bash
docker logs -f origami-contract-policy
```

Ready line:

```text
READY transport=origami-zenoh-v1
```

## 5. Validate

```bash
cd /path/to/inferencekit_09_02_2026/sharpa_north_ces_lite_sdk-main

uv run --no-sync python ./examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:17447 \
  --session-id "$SESSION" \
  --timeout 300 \
  --requests 300 \
  --expected-horizon 15
```

Expected:

```text
metadata: PASS
reset: PASS
infer 1/300: PASS
...
infer 300/300: PASS
```

## 6. Cleanup

```bash
docker rm -f origami-contract-policy origami-contract-router
docker network rm origami-contract-test
```
