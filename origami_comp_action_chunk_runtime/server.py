from __future__ import annotations

import json
import logging
import math
import signal
import threading
import time
import uuid
from collections.abc import Mapping
from typing import Any

import msgpack
import numpy as np

from .config import OrigamiCompActionChunkRuntimeConfig
from .pipeline import CompActionChunkPipeline

try:
    import zenoh
except ImportError:  # pragma: no cover - only required when serving.
    zenoh = None  # type: ignore[assignment]


TRANSPORT_VERSION = "origami-zenoh-v1"
SEMANTIC_VERSION = "origami-v1"
INFERENCE_KIT = "origami-inference-kit-async"
ACTION_DIM = 65
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024

REQUIRED_IMAGE_SPECS = {
    "observation/image/head_left": (224, 224, 3),
    "observation/image/head_right": (224, 224, 3),
    "observation/image/wrist_left": (224, 224, 3),
    "observation/image/wrist_right": (224, 224, 3),
    "observation/image/tactile_deform": (480, 1200, 3),
}
OPTIONAL_IMAGE_SPECS = {
    "observation/image/tactile_raw": (480, 1600, 3),
}
VECTOR_SPECS = {
    "observation/state": (ACTION_DIM,),
    "observation/state/joint_torque": (ACTION_DIM,),
    "observation/tactile": (60,),
}


logger = logging.getLogger(__name__)


def _hand_joint_names(side: str) -> tuple[str, ...]:
    return (
        f"{side}_thumb_CMC_FE",
        f"{side}_thumb_CMC_AA",
        f"{side}_thumb_MCP_FE",
        f"{side}_thumb_MCP_AA",
        f"{side}_thumb_IP",
        f"{side}_index_MCP_FE",
        f"{side}_index_MCP_AA",
        f"{side}_index_PIP",
        f"{side}_index_DIP",
        f"{side}_middle_MCP_FE",
        f"{side}_middle_MCP_AA",
        f"{side}_middle_PIP",
        f"{side}_middle_DIP",
        f"{side}_ring_MCP_FE",
        f"{side}_ring_MCP_AA",
        f"{side}_ring_PIP",
        f"{side}_ring_DIP",
        f"{side}_pinky_CMC",
        f"{side}_pinky_MCP_FE",
        f"{side}_pinky_MCP_AA",
        f"{side}_pinky_PIP",
        f"{side}_pinky_DIP",
    )


JOINT_NAMES = (
    tuple(f"left_arm_joint_{index}" for index in range(1, 8))
    + _hand_joint_names("left")
    + tuple(f"right_arm_joint_{index}" for index in range(1, 8))
    + _hand_joint_names("right")
    + tuple(f"lower_body_joint_{index}" for index in range(1, 6))
    + ("neck_joint_1", "neck_joint_2")
)

if len(JOINT_NAMES) != ACTION_DIM or len(set(JOINT_NAMES)) != ACTION_DIM:
    raise RuntimeError("joint contract must contain 65 unique joint names")


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "V", "c"} or value.dtype.hasobject:
            raise ValueError(f"unsupported numpy dtype: {value.dtype}")
        array = np.ascontiguousarray(value)
        return {
            b"__ndarray__": True,
            b"data": array.tobytes(order="C"),
            b"dtype": array.dtype.str,
            b"shape": array.shape,
        }
    if isinstance(value, np.generic):
        if value.dtype.kind in {"O", "V", "c"} or value.dtype.hasobject:
            raise ValueError(f"unsupported numpy scalar dtype: {value.dtype}")
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _mapping_value(value: Mapping[Any, Any], key: str) -> Any:
    return value[key] if key in value else value.get(key.encode("ascii"))


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    if _mapping_value(value, "__ndarray__") is True:
        data = _mapping_value(value, "data")
        shape = _mapping_value(value, "shape")
        dtype = np.dtype(_mapping_value(value, "dtype"))
        if (
            not isinstance(data, bytes)
            or not isinstance(shape, (list, tuple))
            or len(shape) > 8
            or dtype.kind in {"O", "V", "c"}
            or dtype.hasobject
        ):
            raise ValueError("invalid numpy array payload")
        normalized_shape = tuple(int(dimension) for dimension in shape)
        if any(dimension < 0 for dimension in normalized_shape):
            raise ValueError("invalid numpy array shape")
        expected_size = math.prod(normalized_shape) * dtype.itemsize
        if expected_size > MAX_PAYLOAD_BYTES or len(data) != expected_size:
            raise ValueError("numpy array payload size does not match shape")
        return np.frombuffer(data, dtype=dtype).reshape(normalized_shape)
    if _mapping_value(value, "__npgeneric__") is True:
        return np.dtype(_mapping_value(value, "dtype")).type(_mapping_value(value, "data"))
    return value


def pack_payload(value: Any) -> bytes:
    payload = msgpack.packb(value, default=_pack_numpy, use_bin_type=True)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("response exceeds 64 MiB")
    return payload


def unpack_payload(value: Any) -> Any:
    payload = value.to_bytes() if hasattr(value, "to_bytes") else bytes(value)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("request exceeds 64 MiB")
    return msgpack.unpackb(
        payload,
        object_hook=_unpack_numpy,
        raw=False,
        strict_map_key=False,
        max_bin_len=MAX_PAYLOAD_BYTES,
        max_array_len=1_000_000,
        max_map_len=10_000,
        max_str_len=1_000_000,
    )


class OrigamiCompActionChunkZenohServer:
    def __init__(
        self,
        pipeline: CompActionChunkPipeline,
        config: OrigamiCompActionChunkRuntimeConfig,
        *,
        endpoint: str,
        session_id: str,
        action_horizon: int | None = None,
        execution_mode: str | None = None,
    ) -> None:
        action_dim = int(config.server.action_dim)
        if action_dim != ACTION_DIM:
            raise ValueError(f"server.action_dim must be {ACTION_DIM}, got {action_dim}")
        resolved_horizon = int(action_horizon or config.server.action_horizon)
        if resolved_horizon < 1 or resolved_horizon > 1024:
            raise ValueError("action_horizon must be in [1, 1024]")
        resolved_execution_mode = str(execution_mode or config.server.execution_mode)
        if resolved_execution_mode not in {"sync", "async"}:
            raise ValueError("execution_mode must be 'sync' or 'async'")

        self.pipeline = pipeline
        self.config = config
        self.endpoint = endpoint
        self.session_id = session_id
        self.action_horizon = resolved_horizon
        self.execution_mode = resolved_execution_mode
        self._policy_lock = threading.Lock()
        self._stop = threading.Event()
        self._session: Any | None = None
        self._queryables: list[Any] = []
        self._operation_counts = {operation: 0 for operation in ("metadata", "reset", "infer")}
        self.metadata = {
            "protocol_version": SEMANTIC_VERSION,
            "action_dim": ACTION_DIM,
            "action_type": "absolute_joint_position",
            "action_units": "radians",
            "action_horizon": self.action_horizon,
            "joint_names": JOINT_NAMES,
            "execution_mode": self.execution_mode,
            "inference_kit": INFERENCE_KIT,
            "policy_name": str(config.server.policy_name),
        }

    def serve_forever(self) -> None:
        if zenoh is None:
            raise RuntimeError("eclipse-zenoh is required to serve origami-zenoh-v1.")

        zenoh_config = zenoh.Config()
        zenoh_config.insert_json5("mode", json.dumps("client"))
        zenoh_config.insert_json5("connect/endpoints", json.dumps([self.endpoint]))
        zenoh_config.insert_json5("scouting/multicast/enabled", "false")
        zenoh_config.insert_json5("transport/shared_memory/enabled", "false")
        self._session = zenoh.open(zenoh_config)
        self._queryables = [
            self._session.declare_queryable(
                f"{TRANSPORT_VERSION}/{operation}",
                self._handle_query,
                complete=True,
            )
            for operation in ("metadata", "reset", "infer")
        ]
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        logger.info(
            "READY transport=%s endpoint=%s session=%s horizon=%d execution_mode=%s",
            TRANSPORT_VERSION,
            self.endpoint,
            self.session_id,
            self.action_horizon,
            self.execution_mode,
        )
        try:
            self._stop.wait()
        finally:
            for queryable in self._queryables:
                queryable.undeclare()
            if self._session is not None:
                self._session.close()
            self.pipeline.close()

    def _handle_query(self, query: Any) -> None:
        operation = str(query.key_expr).rsplit("/", 1)[-1]
        request: Any = None
        try:
            request = unpack_payload(query.payload)
            response = self.process(operation, request)
        except Exception as exc:  # noqa: BLE001 - sanitize wire-facing errors.
            error_id = uuid.uuid4().hex
            logger.error(
                "request failed operation=%s error_id=%s type=%s",
                operation,
                error_id,
                type(exc).__name__,
            )
            public_message = f"request failed; error_id={error_id}"
            if not isinstance(request, Mapping):
                query.reply_err(
                    pack_payload(
                        {
                            "error": {
                                "code": "INVALID_REQUEST",
                                "message": public_message,
                                "retryable": False,
                            }
                        }
                    ),
                    encoding="application/msgpack",
                )
                return
            response = self._envelope(operation, request)
            response["error"] = {
                "code": "INFERENCE_FAILED" if operation == "infer" else "INVALID_REQUEST",
                "message": public_message,
                "retryable": False,
            }
        query.reply(str(query.key_expr), pack_payload(response), encoding="application/msgpack")

    def process(self, operation: str, request: Any) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise ValueError("request must be a MessagePack map")
        if operation in self._operation_counts:
            self._operation_counts[operation] += 1
        request_id = str(request.get("request_id", ""))
        response = self._envelope(operation, request)
        if request.get("protocol_version") != TRANSPORT_VERSION:
            raise ValueError("invalid protocol_version")
        if request.get("operation") != operation:
            raise ValueError("operation does not match queryable key")
        if request.get("session_id") != self.session_id:
            raise ValueError("session_id does not match assigned session")
        if not isinstance(request.get("request_id"), str) or not request["request_id"]:
            raise ValueError("request_id must be a non-empty string")

        if operation == "metadata":
            response["metadata"] = self.metadata
            logger.info(
                "metadata request ok request_id=%s count=%d",
                request_id,
                self._operation_counts["metadata"],
            )
            return response
        if operation == "reset":
            with self._policy_lock:
                self.pipeline.reset()
            response["ok"] = True
            logger.info(
                "reset request ok request_id=%s count=%d",
                request_id,
                self._operation_counts["reset"],
            )
            return response
        if operation != "infer":
            raise ValueError(f"unsupported operation: {operation}")

        observation = self._validate_and_sanitize_observation(request.get("observation"))
        started = time.monotonic()
        with self._policy_lock:
            result = self.pipeline.infer(observation, prompt=observation.get("prompt"))
        actions = np.asarray(result.actions, dtype=np.float32)
        expected_shape = (self.action_horizon, ACTION_DIM)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(f"policy actions must have shape {expected_shape}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("policy actions contain NaN or Inf")
        response["actions"] = np.ascontiguousarray(actions, dtype=np.float32)
        response["server_timing"] = {
            "infer_ms": (time.monotonic() - started) * 1000.0,
            "pipeline_ms": float(result.timings_ms.get("total", 0.0)),
        }
        if self._should_log_inference():
            planner_selected = result.planner_summary.get("selected", {})
            logger.info(
                (
                    "infer request ok request_id=%s count=%d actions_shape=%s "
                    "raw_present=%s raw_available=%s planner_available=%s "
                    "ooi_present=%s selected_checkpoint=%s infer_ms=%.2f pipeline_ms=%.2f"
                ),
                request_id,
                self._operation_counts["infer"],
                tuple(actions.shape),
                "observation/image/tactile_raw" in observation,
                result.openpi_summary.get("tactile_raw_available"),
                result.planner_summary.get("planner_available"),
                result.ooi_summary.get("target_present"),
                planner_selected.get("final_checkpoint_label"),
                response["server_timing"]["infer_ms"],
                response["server_timing"]["pipeline_ms"],
            )
        return response

    def _should_log_inference(self) -> bool:
        if not self.config.server.log_inference_requests:
            return False
        every_n = max(1, int(self.config.server.log_inference_every_n))
        return self._operation_counts["infer"] == 1 or self._operation_counts["infer"] % every_n == 0

    def _envelope(self, operation: str, request: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "protocol_version": TRANSPORT_VERSION,
            "operation": operation,
            "request_id": request.get("request_id"),
            "session_id": self.session_id,
        }

    def _validate_and_sanitize_observation(self, observation: Any) -> dict[str, Any]:
        if not isinstance(observation, Mapping):
            raise ValueError("infer request must contain an observation map")

        required = {*REQUIRED_IMAGE_SPECS, *VECTOR_SPECS}
        if self.config.server.require_prompt:
            required.add("prompt")
        allowed = required | set(OPTIONAL_IMAGE_SPECS)
        if self.config.server.allow_observation_timestamp:
            allowed.add("observation_timestamp")

        missing = sorted(key for key in required if key not in observation)
        if missing:
            raise ValueError(f"observation is missing required keys: {missing}")
        unknown = sorted(str(key) for key in set(observation) - allowed)
        if unknown:
            raise ValueError(f"observation contains unsupported keys: {unknown}")

        sanitized: dict[str, Any] = {}
        for key, shape in REQUIRED_IMAGE_SPECS.items():
            sanitized[key] = self._validate_image(observation[key], key=key, shape=shape)

        raw_key = "observation/image/tactile_raw"
        if raw_key in observation:
            try:
                raw_value = observation[raw_key]
                if raw_value is None:
                    raise ValueError("optional raw tactile image is None")
                sanitized[raw_key] = self._validate_image(raw_value, key=raw_key, shape=OPTIONAL_IMAGE_SPECS[raw_key])
            except Exception:
                if not self.config.server.tolerate_invalid_optional_raw:
                    raise
                logger.warning("Dropping invalid optional tactile_raw input for this inference.")

        for key, shape in VECTOR_SPECS.items():
            sanitized[key] = self._validate_vector(observation[key], key=key, shape=shape)

        if "prompt" in observation:
            prompt = observation["prompt"]
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string")
            sanitized["prompt"] = prompt
        elif self.config.server.require_prompt:
            raise ValueError("prompt must be a string")

        if self.config.server.allow_observation_timestamp and "observation_timestamp" in observation:
            timestamp = observation["observation_timestamp"]
            if not isinstance(timestamp, (int, float, np.integer, np.floating)):
                raise ValueError("observation_timestamp must be numeric when provided")
            sanitized["observation_timestamp"] = float(timestamp)
        return sanitized

    @staticmethod
    def _validate_image(value: Any, *, key: str, shape: tuple[int, int, int]) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{key} must be a numpy array")
        if value.dtype != np.dtype(np.uint8) or tuple(value.shape) != shape:
            raise ValueError(f"{key} must be uint8{shape}, got {value.dtype}{value.shape}")
        return np.ascontiguousarray(value, dtype=np.uint8)

    @staticmethod
    def _validate_vector(value: Any, *, key: str, shape: tuple[int, ...]) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{key} must be a numpy array")
        if value.dtype != np.dtype(np.float32) or tuple(value.shape) != shape:
            raise ValueError(f"{key} must be float32{shape}, got {value.dtype}{value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite float32{shape}")
        return np.ascontiguousarray(value, dtype=np.float32)
