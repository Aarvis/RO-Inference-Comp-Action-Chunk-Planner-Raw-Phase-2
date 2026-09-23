#!/usr/bin/env bash
set -euo pipefail

mode="${1:-serve}"
if [ "$#" -gt 0 ]; then
  shift
fi

case "${mode}" in
  serve|dataset-replay|verify-bundle|bash|sh|python|python3)
    ;;
  *)
    echo "[submission][ERROR] unknown mode: ${mode}" >&2
    echo "Usage: <image> [serve|dataset-replay|verify-bundle|bash|sh|python] [args...]" >&2
    exit 2
    ;;
esac

if [ "${mode}" = "bash" ] || [ "${mode}" = "sh" ] || [ "${mode}" = "python" ] || [ "${mode}" = "python3" ]; then
  exec "${mode}" "$@"
fi

bundle_root="${ORIGAMI_MODEL_BUNDLE:-/app/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2/model_bundle}"
execution_mode="${EXECUTION_MODE:-async}"
planner_enabled="${ORIGAMI_PLANNER_ENABLED:-}"
warmup_inferences="${ORIGAMI_WARMUP_INFERENCES:-1}"
replay_warmup_training_loss="${ORIGAMI_REPLAY_WARMUP_TRAINING_LOSS:-1}"
jax_mem_fraction="${ORIGAMI_JAX_MEM_FRACTION:-${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.60}}"
openpi_param_dtype="${ORIGAMI_OPENPI_PARAM_DTYPE:-}"

case "${execution_mode}" in
  sync|async)
    ;;
  *)
    echo "[submission][ERROR] EXECUTION_MODE must be sync or async, got: ${execution_mode}" >&2
    exit 2
    ;;
esac

planner_override_args=()
if [ -n "${planner_enabled}" ]; then
  planner_enabled_lc="$(printf '%s' "${planner_enabled}" | tr '[:upper:]' '[:lower:]')"
  case "${planner_enabled_lc}" in
    true)
      planner_override_args=(--planner-enabled true)
      ;;
    false)
      planner_override_args=(--planner-enabled false)
      ;;
    *)
      echo "[submission][ERROR] ORIGAMI_PLANNER_ENABLED must be true or false, got: ${planner_enabled}" >&2
      exit 2
      ;;
  esac
fi

export HOME="${HOME:-/tmp/origami-home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/origami-cache}"
export HF_HOME="${HF_HOME:-/tmp/origami-hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"
export TORCH_HOME="${TORCH_HOME:-/tmp/origami-torch}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/origami-jax-cache}"
export TMPDIR="${TMPDIR:-/tmp/origami-tmp}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${jax_mem_fraction}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

cuda_nvcc_bin="$(python - <<'PY'
import site
from pathlib import Path

for p in site.getsitepackages():
    candidate = Path(p) / "nvidia" / "cuda_nvcc" / "bin"
    if (candidate / "ptxas").exists():
        print(candidate)
        break
PY
)"
if [ -n "${cuda_nvcc_bin}" ]; then
  export PATH="${cuda_nvcc_bin}:${PATH}"
fi

mkdir -p \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_LEROBOT_HOME}" \
  "${TORCH_HOME}" \
  "${JAX_COMPILATION_CACHE_DIR}" \
  "${TMPDIR}"

cd /app/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2

case "${mode}" in
  verify-bundle)
    echo "[submission] mode=verify-bundle bundle=${bundle_root}" >&2
    exec python scripts/verify_comp_action_chunk_bundle.py --bundle-root "${bundle_root}" "$@"
    ;;
  dataset-replay)
    echo "[submission] mode=dataset-replay bundle=${bundle_root}" >&2
    replay_warmup_args=(--warmup-inferences "${warmup_inferences}")
    replay_openpi_args=()
    if [ -n "${openpi_param_dtype}" ]; then
      replay_openpi_args+=(--openpi-param-dtype "${openpi_param_dtype}")
    fi
    replay_warmup_training_loss_lc="$(printf '%s' "${replay_warmup_training_loss}" | tr '[:upper:]' '[:lower:]')"
    if [ "${replay_warmup_training_loss_lc}" = "0" ] || [ "${replay_warmup_training_loss_lc}" = "false" ]; then
      replay_warmup_args+=(--skip-warmup-training-loss)
    fi
    exec python run_dataset_replay.py --bundle-root "${bundle_root}" "${planner_override_args[@]}" "${replay_warmup_args[@]}" "${replay_openpi_args[@]}" "$@"
    ;;
  serve)
    : "${ORIGAMI_ZENOH_ENDPOINT:?ORIGAMI_ZENOH_ENDPOINT is required}"
    : "${ORIGAMI_SESSION_ID:?ORIGAMI_SESSION_ID is required}"
    echo "[submission] mode=serve transport=origami-zenoh-v1 execution_mode=${execution_mode} planner_enabled=${planner_enabled:-bundle} endpoint=${ORIGAMI_ZENOH_ENDPOINT} session=${ORIGAMI_SESSION_ID} bundle=${bundle_root}" >&2
    python scripts/verify_comp_action_chunk_bundle.py --bundle-root "${bundle_root}"
    serve_openpi_args=()
    if [ -n "${openpi_param_dtype}" ]; then
      serve_openpi_args+=(--openpi-param-dtype "${openpi_param_dtype}")
    fi
    exec python serve_origami_comp_action_chunk_policy.py \
      --bundle-root "${bundle_root}" \
      --endpoint "${ORIGAMI_ZENOH_ENDPOINT}" \
      --session-id "${ORIGAMI_SESSION_ID}" \
      --execution-mode "${execution_mode}" \
      "${planner_override_args[@]}" \
      --jax-mem-fraction "${jax_mem_fraction}" \
      --warmup-inferences "${warmup_inferences}" \
      "${serve_openpi_args[@]}" \
      "$@"
    ;;
esac
