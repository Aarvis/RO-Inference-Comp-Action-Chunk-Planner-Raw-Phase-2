from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import cv2
import numpy as np
from tqdm.auto import tqdm

from .config import DatasetReplayConfig, OrigamiCompActionChunkRuntimeConfig, dump_json
from .pipeline import (
    HEAD_LEFT_KEY,
    STATE_KEY,
    TACTILE_DEFORM_KEY,
    TACTILE_RAW_KEY,
    TACTILE_VECTOR_KEY,
    WRIST_LEFT_KEY,
    WRIST_RIGHT_KEY,
    CompActionChunkPipeline,
)


logger = logging.getLogger(__name__)

VALID_TACTILE_MODES = {
    "deform_only",
    "deform_plus_raw",
    "mixed_50",
    "deform_plus_raw_planner_dropped",
}


@dataclasses.dataclass(frozen=True)
class ReplaySample:
    episode_uid: str
    episode_root: Path
    phase_offset: int
    frame_position: int
    frame_index: int
    target_actions: np.ndarray
    observation: dict[str, Any]
    use_raw_tactile: bool


@dataclasses.dataclass
class RunningScalar:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.count += 1
        self.total += float(value)
        self.total_sq += float(value) * float(value)
        self.min_value = min(self.min_value, float(value))
        self.max_value = max(self.max_value, float(value))

    def summary(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {
            "count": int(self.count),
            "mean": float(mean),
            "std": float(math.sqrt(variance)),
            "min": float(self.min_value),
            "max": float(self.max_value),
        }


class MetricAccumulator:
    def __init__(self, *, action_horizon: int, action_dim: int) -> None:
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.samples = 0
        self.failed_samples = 0
        self.raw_requested = 0
        self.raw_available = 0
        self.ooi_present = 0
        self.planner_available = 0
        self.loss_samples = 0

        self.action_value_count = 0
        self.action_abs_total = 0.0
        self.action_sq_total = 0.0
        self.action_max_abs = 0.0
        self.first_abs = RunningScalar()
        self.first_rmse = RunningScalar()
        self.last_abs = RunningScalar()
        self.last_rmse = RunningScalar()
        self.horizon_abs_total = np.zeros((self.action_horizon,), dtype=np.float64)
        self.horizon_sq_total = np.zeros((self.action_horizon,), dtype=np.float64)
        self.horizon_counts = np.zeros((self.action_horizon,), dtype=np.int64)
        self.dim_abs_total = np.zeros((self.action_dim,), dtype=np.float64)
        self.dim_sq_total = np.zeros((self.action_dim,), dtype=np.float64)
        self.dim_counts = np.zeros((self.action_dim,), dtype=np.int64)

        self.training_metrics: dict[str, RunningScalar] = {}
        self.timing_metrics: dict[str, RunningScalar] = {}

    def mark_failed(self) -> None:
        self.failed_samples += 1

    def add(
        self,
        *,
        result_summary: Mapping[str, Any],
        predicted_actions: np.ndarray,
        target_actions: np.ndarray,
        use_raw_tactile: bool,
        action_error_enabled: bool,
    ) -> dict[str, float | int | bool | None]:
        self.samples += 1
        if use_raw_tactile:
            self.raw_requested += 1

        ooi = result_summary.get("ooi") or {}
        planner = result_summary.get("planner") or {}
        openpi = result_summary.get("openpi") or {}
        if bool(ooi.get("target_present", False)):
            self.ooi_present += 1
        if bool(planner.get("planner_available", False)):
            self.planner_available += 1
        if bool(openpi.get("tactile_raw_available", False)):
            self.raw_available += 1

        for name, value in (result_summary.get("timings_ms") or {}).items():
            self.timing_metrics.setdefault(str(name), RunningScalar()).add(float(value))

        per_sample: dict[str, float | int | bool | None] = {
            "tactile_raw_requested": bool(use_raw_tactile),
            "tactile_raw_available": bool(openpi.get("tactile_raw_available", False)),
            "ooi_target_present": bool(ooi.get("target_present", False)),
            "planner_available": bool(planner.get("planner_available", False)),
            "action_mae": None,
            "action_rmse": None,
            "first_action_mae": None,
            "first_action_rmse": None,
        }

        if action_error_enabled:
            err = np.asarray(predicted_actions, dtype=np.float32) - np.asarray(target_actions, dtype=np.float32)
            finite = np.isfinite(err)
            if not finite.all():
                err = np.where(finite, err, 0.0)
            abs_err = np.abs(err)
            sq_err = np.square(err)
            value_count = int(finite.sum())
            if value_count > 0:
                self.action_value_count += value_count
                self.action_abs_total += float(abs_err.sum(dtype=np.float64))
                self.action_sq_total += float(sq_err.sum(dtype=np.float64))
                self.action_max_abs = max(self.action_max_abs, float(abs_err.max()))

                horizon_counts = finite.sum(axis=1)
                dim_counts = finite.sum(axis=0)
                self.horizon_counts += horizon_counts.astype(np.int64)
                self.dim_counts += dim_counts.astype(np.int64)
                self.horizon_abs_total += abs_err.sum(axis=1, dtype=np.float64)
                self.horizon_sq_total += sq_err.sum(axis=1, dtype=np.float64)
                self.dim_abs_total += abs_err.sum(axis=0, dtype=np.float64)
                self.dim_sq_total += sq_err.sum(axis=0, dtype=np.float64)

                sample_mae = float(abs_err.sum(dtype=np.float64) / value_count)
                sample_rmse = float(math.sqrt(sq_err.sum(dtype=np.float64) / value_count))
                per_sample["action_mae"] = sample_mae
                per_sample["action_rmse"] = sample_rmse
                if bool(finite[0].all()):
                    first_mae = float(np.mean(abs_err[0], dtype=np.float64))
                    first_rmse = float(math.sqrt(np.mean(sq_err[0], dtype=np.float64)))
                    self.first_abs.add(first_mae)
                    self.first_rmse.add(first_rmse)
                    per_sample["first_action_mae"] = first_mae
                    per_sample["first_action_rmse"] = first_rmse
                if bool(finite[-1].all()):
                    self.last_abs.add(float(np.mean(abs_err[-1], dtype=np.float64)))
                    self.last_rmse.add(float(math.sqrt(np.mean(sq_err[-1], dtype=np.float64))))

        loss = result_summary.get("openpi_loss")
        if isinstance(loss, Mapping):
            self.loss_samples += 1
            chunked_mean = loss.get("chunked_loss_mean")
            if chunked_mean is not None:
                self.training_metrics.setdefault("chunked_loss_mean", RunningScalar()).add(float(chunked_mean))
                per_sample["training_chunked_loss_mean"] = float(chunked_mean)
            for name, value in (loss.get("metrics") or {}).items():
                self.training_metrics.setdefault(str(name), RunningScalar()).add(float(value))
                if str(name) == "loss_base_flow":
                    per_sample["training_loss_base_flow"] = float(value)

        return per_sample

    def summary(self) -> dict[str, Any]:
        action_summary: dict[str, Any] = {
            "samples": int(self.samples),
            "values": int(self.action_value_count),
            "mae": None,
            "rmse": None,
            "max_abs": None,
            "first_step_mae": self.first_abs.summary(),
            "first_step_rmse": self.first_rmse.summary(),
            "last_step_mae": self.last_abs.summary(),
            "last_step_rmse": self.last_rmse.summary(),
            "horizon_mae": [],
            "horizon_rmse": [],
            "top_dimension_mae": [],
        }
        if self.action_value_count > 0:
            action_summary["mae"] = float(self.action_abs_total / self.action_value_count)
            action_summary["rmse"] = float(math.sqrt(self.action_sq_total / self.action_value_count))
            action_summary["max_abs"] = float(self.action_max_abs)

            horizon_mae = np.divide(
                self.horizon_abs_total,
                np.maximum(self.horizon_counts, 1),
                where=self.horizon_counts > 0,
            )
            horizon_rmse = np.sqrt(
                np.divide(self.horizon_sq_total, np.maximum(self.horizon_counts, 1), where=self.horizon_counts > 0)
            )
            action_summary["horizon_mae"] = [float(v) for v in horizon_mae.tolist()]
            action_summary["horizon_rmse"] = [float(v) for v in horizon_rmse.tolist()]

            dim_mae = np.divide(self.dim_abs_total, np.maximum(self.dim_counts, 1), where=self.dim_counts > 0)
            order = np.argsort(-dim_mae)[:10]
            action_summary["top_dimension_mae"] = [
                {"dim": int(index), "mae": float(dim_mae[index])}
                for index in order
                if self.dim_counts[index] > 0
            ]

        return {
            "samples": int(self.samples),
            "failed_samples": int(self.failed_samples),
            "tactile_raw_requested": int(self.raw_requested),
            "tactile_raw_available": int(self.raw_available),
            "ooi_target_present": int(self.ooi_present),
            "planner_available": int(self.planner_available),
            "training_loss_samples": int(self.loss_samples),
            "fractions": {
                "tactile_raw_requested": _safe_fraction(self.raw_requested, self.samples),
                "tactile_raw_available": _safe_fraction(self.raw_available, self.samples),
                "ooi_target_present": _safe_fraction(self.ooi_present, self.samples),
                "planner_available": _safe_fraction(self.planner_available, self.samples),
            },
            "action_error": action_summary,
            "training_style_loss": {
                name: metric.summary()
                for name, metric in sorted(self.training_metrics.items())
            },
            "timings_ms": {
                name: metric.summary()
                for name, metric in sorted(self.timing_metrics.items())
            },
        }


class VideoFrameReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def read(self, frame_index: int) -> np.ndarray:
        if self.frame_count > 0 and (frame_index < 0 or frame_index >= self.frame_count):
            raise IndexError(f"{self.path}: frame_index={frame_index} outside frame_count={self.frame_count}.")
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = self.capture.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"{self.path}: failed to read frame {frame_index}.")
        return np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    def close(self) -> None:
        capture = getattr(self, "capture", None)
        if capture is not None:
            capture.release()
            self.capture = None

    def __del__(self) -> None:
        close = getattr(self, "close", None)
        if close is not None:
            with contextlib.suppress(Exception):
                close()


class EpisodeReplaySource:
    def __init__(self, episode_root: Path, replay: DatasetReplayConfig) -> None:
        self.episode_root = episode_root
        self.episode_uid = episode_root.name
        self.replay = replay
        arrays_root = episode_root / replay.arrays_subdir
        videos_root = episode_root / replay.videos_subdir

        self.state = self._load_array(arrays_root / replay.state_array, expected_dim=65, name="state")
        self.actions = self._load_array(arrays_root / replay.action_array, expected_dim=65, name="actions")
        self.tactile = self._load_array(arrays_root / replay.tactile_array, expected_dim=60, name="tactile")
        self.timestamps = self._load_optional_array(arrays_root / replay.timestamps_array)
        self.frame_index = self._load_optional_array(arrays_root / replay.frame_index_array)

        self.head_left = VideoFrameReader(videos_root / replay.head_left_video)
        self.wrist_left = VideoFrameReader(videos_root / replay.wrist_left_video)
        self.wrist_right = VideoFrameReader(videos_root / replay.wrist_right_video)
        self.tactile_deform = VideoFrameReader(videos_root / replay.tactile_deform_video)
        raw_path = videos_root / replay.tactile_raw_video
        self.tactile_raw = VideoFrameReader(raw_path) if raw_path.is_file() else None

        self.length = self._effective_length()

    @staticmethod
    def _load_array(path: Path, *, expected_dim: int, name: str) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} array: {path}")
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or int(array.shape[1]) != int(expected_dim):
            raise ValueError(f"{path}: expected shape [T,{expected_dim}], got {array.shape}.")
        return array

    @staticmethod
    def _load_optional_array(path: Path) -> np.ndarray | None:
        if not path.is_file():
            return None
        return np.load(path, mmap_mode="r")

    @property
    def raw_video_present(self) -> bool:
        return self.tactile_raw is not None

    def _effective_length(self) -> int:
        lengths = [int(self.state.shape[0]), int(self.actions.shape[0]), int(self.tactile.shape[0])]
        for reader in (self.head_left, self.wrist_left, self.wrist_right, self.tactile_deform, self.tactile_raw):
            if reader is not None and int(reader.frame_count) > 0:
                lengths.append(int(reader.frame_count))
        if self.timestamps is not None:
            lengths.append(int(self.timestamps.shape[0]))
        if self.frame_index is not None:
            lengths.append(int(self.frame_index.shape[0]))
        return int(min(lengths))

    def count_phase_samples(self, phase_offset: int, replay: DatasetReplayConfig | None = None) -> int:
        replay = replay or self.replay
        max_start = self.length - 1 - (int(replay.action_horizon) - 1) * int(replay.action_chunk_stride)
        if phase_offset > max_start:
            return 0
        return 1 + (max_start - int(phase_offset)) // int(replay.step_stride)

    def count_samples(self, replay: DatasetReplayConfig | None = None) -> int:
        replay = replay or self.replay
        total = 0
        for offset in replay.resolved_phase_offsets:
            total += self.count_phase_samples(int(offset), replay)
        if replay.max_samples_per_episode is not None:
            total = min(total, int(replay.max_samples_per_episode))
        return int(total)

    def iter_phase_samples(
        self,
        *,
        phase_offset: int,
        tactile_mode: str,
        replay: DatasetReplayConfig,
        remaining_episode_samples: int | None,
    ) -> Iterator[ReplaySample]:
        max_start = self.length - 1 - (int(replay.action_horizon) - 1) * int(replay.action_chunk_stride)
        emitted = 0
        for frame_position in range(int(phase_offset), max_start + 1, int(replay.step_stride)):
            if remaining_episode_samples is not None and emitted >= remaining_episode_samples:
                break
            use_raw_tactile = resolve_use_raw_tactile(
                tactile_mode=tactile_mode,
                replay=replay,
                episode_uid=self.episode_uid,
                phase_offset=int(phase_offset),
                frame_position=int(frame_position),
                raw_video_present=self.raw_video_present,
            )
            observation = self.read_observation(
                frame_position,
                use_raw_tactile=use_raw_tactile,
                include_dataset_timestamps=bool(replay.include_dataset_timestamps),
            )
            frame_index = int(observation.get("observation_frame_index", frame_position))
            target_positions = frame_position + np.arange(int(replay.action_horizon)) * int(replay.action_chunk_stride)
            target_actions = np.asarray(self.actions[target_positions], dtype=np.float32)
            yield ReplaySample(
                episode_uid=self.episode_uid,
                episode_root=self.episode_root,
                phase_offset=int(phase_offset),
                frame_position=int(frame_position),
                frame_index=frame_index,
                target_actions=np.ascontiguousarray(target_actions, dtype=np.float32),
                observation=observation,
                use_raw_tactile=bool(use_raw_tactile),
            )
            emitted += 1

    def read_observation(
        self,
        frame_position: int,
        *,
        use_raw_tactile: bool,
        include_dataset_timestamps: bool,
    ) -> dict[str, Any]:
        observation: dict[str, Any] = {
            HEAD_LEFT_KEY: self.head_left.read(frame_position),
            WRIST_LEFT_KEY: self.wrist_left.read(frame_position),
            WRIST_RIGHT_KEY: self.wrist_right.read(frame_position),
            TACTILE_DEFORM_KEY: self.tactile_deform.read(frame_position),
            STATE_KEY: np.asarray(self.state[frame_position], dtype=np.float32),
            TACTILE_VECTOR_KEY: np.asarray(self.tactile[frame_position], dtype=np.float32),
        }
        if use_raw_tactile:
            if self.tactile_raw is None:
                raise FileNotFoundError(f"{self.episode_uid}: tactile raw video requested but missing.")
            observation[TACTILE_RAW_KEY] = self.tactile_raw.read(frame_position)
        if include_dataset_timestamps and self.timestamps is not None:
            observation["observation_timestamp"] = float(np.asarray(self.timestamps[frame_position]).reshape(()))
        if self.frame_index is not None:
            observation["observation_frame_index"] = int(np.asarray(self.frame_index[frame_position]).reshape(()))
        return observation

    def close(self) -> None:
        for reader in (self.head_left, self.wrist_left, self.wrist_right, self.tactile_deform, self.tactile_raw):
            if reader is not None:
                reader.close()

    def __enter__(self) -> EpisodeReplaySource:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()


class DatasetReplayRunner:
    def __init__(self, config: OrigamiCompActionChunkRuntimeConfig) -> None:
        self.config = config
        self.replay = config.dataset_replay
        self.output_dir = config.paths.output_dir

    def run(self, *, dry_run: bool = False) -> dict[str, Any]:
        self._validate_replay_config()
        episode_roots = discover_episode_roots(self.config.paths.dataset_root, self.replay)
        if not episode_roots:
            raise RuntimeError(f"No replay episodes found under {self.config.paths.dataset_root}.")
        logger.info(
            "dataset replay starting dataset_root=%s output_dir=%s episodes=%d tactile_modes=%s dry_run=%s",
            self.config.paths.dataset_root,
            self.output_dir,
            len(episode_roots),
            ",".join(self.replay.tactile_modes),
            dry_run,
        )

        plan = self._build_plan(episode_roots)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dump_json(self.output_dir / "dataset_replay_plan.json", plan)
        logger.info("dataset replay plan saved path=%s", self.output_dir / "dataset_replay_plan.json")
        if dry_run:
            logger.info("dataset replay dry run complete episodes=%d", plan["episode_count"])
            return {"dry_run": True, "plan_path": str(self.output_dir / "dataset_replay_plan.json"), "plan": plan}

        logger.info("dataset replay loading pipeline")
        pipeline = CompActionChunkPipeline(self.config)
        warmup_count = self._warmup_count()
        if warmup_count > 0:
            warmup_loss = bool(self.replay.compute_training_loss and self.replay.warmup_training_loss)
            logger.info(
                "dataset replay warming up pipeline count=%d training_loss=%s",
                warmup_count,
                warmup_loss,
            )
            warmup_started = time.perf_counter()
            pipeline.warmup(
                warmup_count,
                prompt=self.replay.prompt,
                compute_training_loss=warmup_loss,
                loss_noise_samples=self.replay.loss_noise_samples,
                loss_train_mode=self.replay.loss_train_mode,
            )
            logger.info(
                "dataset replay warmup complete elapsed_seconds=%.3f",
                time.perf_counter() - warmup_started,
            )
        else:
            logger.info("dataset replay warmup skipped")
        mode_summaries: dict[str, Any] = {}
        for tactile_mode in self.replay.tactile_modes:
            mode_plan = plan["modes"][tactile_mode]
            logger.info(
                "dataset replay mode starting mode=%s planned_samples=%d raw_eligible_episodes=%d",
                tactile_mode,
                int(mode_plan["samples"]),
                int(mode_plan["raw_eligible_episodes"]),
            )
            summary = self._run_mode(
                pipeline=pipeline,
                tactile_mode=tactile_mode,
                episode_roots=episode_roots,
                planned_samples=int(mode_plan["samples"]),
            )
            mode_summaries[tactile_mode] = summary
            logger.info(
                "dataset replay mode complete mode=%s processed_samples=%d failed_samples=%d samples_per_second=%.4f summary=%s",
                tactile_mode,
                int(summary["processed_samples"]),
                int(summary["metrics"]["failed_samples"]),
                float(summary["samples_per_second"]),
                summary["summary_path"],
            )

        all_summary = {
            "config_path": str(self.config.config_path),
            "dataset_root": str(self.config.paths.dataset_root),
            "output_dir": str(self.output_dir),
            "plan": plan,
            "modes": mode_summaries,
        }
        dump_json(self.output_dir / self.replay.all_modes_summary_filename, all_summary)
        logger.info("dataset replay complete summary=%s", self.output_dir / self.replay.all_modes_summary_filename)
        return all_summary

    def _validate_replay_config(self) -> None:
        unknown_modes = sorted(set(self.replay.tactile_modes) - VALID_TACTILE_MODES)
        if unknown_modes:
            raise ValueError(f"Unsupported tactile replay modes: {unknown_modes}. Valid modes: {sorted(VALID_TACTILE_MODES)}")
        if self.replay.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self.replay.action_horizon}.")
        if self.replay.action_chunk_stride <= 0:
            raise ValueError(f"action_chunk_stride must be positive, got {self.replay.action_chunk_stride}.")
        if self.replay.step_stride <= 0:
            raise ValueError(f"step_stride must be positive, got {self.replay.step_stride}.")
        for offset in self.replay.resolved_phase_offsets:
            if offset < 0 or offset >= self.replay.step_stride:
                raise ValueError(
                    f"phase offset {offset} must be in [0, step_stride={self.replay.step_stride})."
                )
        if not 0.0 <= self.replay.mixed_raw_probability <= 1.0:
            raise ValueError(f"mixed_raw_probability must be in [0,1], got {self.replay.mixed_raw_probability}.")
        if self.replay.loss_every_n_samples <= 0:
            raise ValueError(f"loss_every_n_samples must be positive, got {self.replay.loss_every_n_samples}.")

    def _build_plan(self, episode_roots: list[Path]) -> dict[str, Any]:
        episode_entries: list[dict[str, Any]] = []
        for episode_root in tqdm(episode_roots, desc="Plan replay episodes", unit="episode"):
            with EpisodeReplaySource(episode_root, self.replay) as source:
                phase_counts = {
                    str(offset): source.count_phase_samples(int(offset), self.replay)
                    for offset in self.replay.resolved_phase_offsets
                }
                episode_entries.append(
                    {
                        "episode_uid": episode_root.name,
                        "episode_root": str(episode_root),
                        "length": int(source.length),
                        "raw_video_present": bool(source.raw_video_present),
                        "phase_counts": phase_counts,
                        "samples": int(source.count_samples(self.replay)),
                    }
                )

        mode_plans: dict[str, Any] = {}
        for tactile_mode in self.replay.tactile_modes:
            raw_eligible = sum(1 for item in episode_entries if bool(item["raw_video_present"]))
            samples = sum(int(item["samples"]) for item in episode_entries)
            if self.replay.max_total_samples is not None:
                samples = min(samples, int(self.replay.max_total_samples))
            mode_plans[tactile_mode] = {
                "samples": int(samples),
                "episodes": int(len(episode_entries)),
                "raw_eligible_episodes": int(raw_eligible),
            }

        return {
            "dataset_root": str(self.config.paths.dataset_root),
            "episode_count": int(len(episode_entries)),
            "action_horizon": int(self.replay.action_horizon),
            "action_chunk_stride": int(self.replay.action_chunk_stride),
            "step_stride": int(self.replay.step_stride),
            "phase_offsets": [int(offset) for offset in self.replay.resolved_phase_offsets],
            "tactile_modes": list(self.replay.tactile_modes),
            "max_samples_per_episode": self.replay.max_samples_per_episode,
            "max_total_samples": self.replay.max_total_samples,
            "episodes": episode_entries,
            "modes": mode_plans,
        }

    def _run_mode(
        self,
        *,
        pipeline: CompActionChunkPipeline,
        tactile_mode: str,
        episode_roots: list[Path],
        planned_samples: int,
    ) -> dict[str, Any]:
        mode_output_dir = self.output_dir / tactile_mode
        mode_output_dir.mkdir(parents=True, exist_ok=True)
        per_sample_path = mode_output_dir / self.replay.per_sample_filename
        accumulator = MetricAccumulator(
            action_horizon=self.replay.action_horizon,
            action_dim=65,
        )
        started = time.perf_counter()
        global_limit = self.replay.max_total_samples
        processed = 0

        writer_cm = per_sample_path.open("w", encoding="utf-8") if self.replay.save_per_sample else contextlib.nullcontext(None)
        with writer_cm as writer:
            with tqdm(total=planned_samples, desc=f"Replay samples [{tactile_mode}]", unit="sample") as sample_bar:
                for episode_root in tqdm(episode_roots, desc=f"Replay episodes [{tactile_mode}]", unit="episode"):
                    if global_limit is not None and processed >= int(global_limit):
                        break
                    with EpisodeReplaySource(episode_root, self.replay) as source:
                        if (
                            mode_requires_raw_tactile(tactile_mode)
                            and self.replay.require_raw_for_deform_plus_raw
                            and not source.raw_video_present
                        ):
                            raise FileNotFoundError(
                                f"{source.episode_uid}: {tactile_mode} replay requested, but tactile raw video is missing."
                            )
                        episode_remaining = self.replay.max_samples_per_episode
                        episode_total = source.count_samples(self.replay)
                        if global_limit is not None:
                            episode_total = min(episode_total, max(0, int(global_limit) - processed))
                        with tqdm(
                            total=episode_total,
                            desc=f"{source.episode_uid} [{tactile_mode}]",
                            unit="sample",
                            leave=False,
                        ) as episode_bar:
                            for phase_offset in self.replay.resolved_phase_offsets:
                                if global_limit is not None and processed >= int(global_limit):
                                    break
                                if episode_remaining is not None and episode_remaining <= 0:
                                    break
                                pipeline.reset()
                                phase_remaining = episode_remaining
                                for sample in source.iter_phase_samples(
                                    phase_offset=int(phase_offset),
                                    tactile_mode=tactile_mode,
                                    replay=self.replay,
                                    remaining_episode_samples=phase_remaining,
                                ):
                                    if global_limit is not None and processed >= int(global_limit):
                                        break
                                    compute_loss = self._should_compute_loss(processed, accumulator.loss_samples)
                                    seed = stable_seed(
                                        self.replay.loss_rng_seed,
                                        tactile_mode,
                                        sample.episode_uid,
                                        sample.phase_offset,
                                        sample.frame_position,
                                    )
                                    try:
                                        result = pipeline.infer(
                                            sample.observation,
                                            prompt=self.replay.prompt,
                                            use_raw_tactile=sample.use_raw_tactile,
                                            require_raw_tactile=(
                                                mode_requires_raw_tactile(tactile_mode)
                                                and self.replay.require_raw_for_deform_plus_raw
                                            ),
                                            planner_prefix_enabled=mode_uses_planner_prefix(tactile_mode),
                                            target_actions_65d=sample.target_actions,
                                            compute_training_loss=compute_loss,
                                            loss_rng_seed=seed,
                                            loss_noise_samples=self.replay.loss_noise_samples,
                                            loss_train_mode=self.replay.loss_train_mode,
                                        )
                                        summary = result.to_summary_dict()
                                        per_sample_metrics = accumulator.add(
                                            result_summary=summary,
                                            predicted_actions=result.actions,
                                            target_actions=sample.target_actions,
                                            use_raw_tactile=sample.use_raw_tactile,
                                            action_error_enabled=self.replay.compute_action_error,
                                        )
                                        record = {
                                            "episode_uid": sample.episode_uid,
                                            "phase_offset": int(sample.phase_offset),
                                            "frame_position": int(sample.frame_position),
                                            "frame_index": int(sample.frame_index),
                                            "target_end_frame_position": int(
                                                sample.frame_position
                                                + (self.replay.action_horizon - 1) * self.replay.action_chunk_stride
                                            ),
                                            "tactile_mode": tactile_mode,
                                            "summary": {
                                                "selected_checkpoint": (
                                                    summary.get("planner", {})
                                                    .get("selected", {})
                                                    .get("final_checkpoint_label")
                                                ),
                                                "selected_branch": summary.get("planner", {}).get("selected_branch"),
                                                "raw_status": summary.get("openpi", {}).get("tactile_raw_status"),
                                            },
                                            "metrics": per_sample_metrics,
                                            "loss_metrics": (
                                                None
                                                if summary.get("openpi_loss") is None
                                                else summary["openpi_loss"].get("metrics")
                                            ),
                                            "timings_ms": summary.get("timings_ms"),
                                        }
                                        if writer is not None:
                                            writer.write(json.dumps(_json_ready(record), separators=(",", ":")) + "\n")
                                    except Exception as exc:
                                        accumulator.mark_failed()
                                        logger.exception(
                                            "dataset replay sample failed mode=%s episode=%s phase_offset=%d frame_position=%d frame_index=%d",
                                            tactile_mode,
                                            sample.episode_uid,
                                            int(sample.phase_offset),
                                            int(sample.frame_position),
                                            int(sample.frame_index),
                                        )
                                        if writer is not None:
                                            writer.write(
                                                json.dumps(
                                                    _json_ready(
                                                        {
                                                            "episode_uid": sample.episode_uid,
                                                            "phase_offset": int(sample.phase_offset),
                                                            "frame_position": int(sample.frame_position),
                                                            "frame_index": int(sample.frame_index),
                                                            "tactile_mode": tactile_mode,
                                                            "error": repr(exc),
                                                        }
                                                    ),
                                                    separators=(",", ":"),
                                                )
                                                + "\n"
                                            )
                                        raise
                                    finally:
                                        processed += 1
                                        if episode_remaining is not None:
                                            episode_remaining -= 1
                                        sample_bar.update(1)
                                        episode_bar.update(1)

        elapsed = time.perf_counter() - started
        summary_path = mode_output_dir / self.replay.summary_filename
        summary = {
            "tactile_mode": tactile_mode,
            "elapsed_seconds": float(elapsed),
            "samples_per_second": float(processed / elapsed) if elapsed > 0 else None,
            "planned_samples": int(planned_samples),
            "processed_samples": int(processed),
            "per_sample_path": str(per_sample_path) if self.replay.save_per_sample else None,
            "summary_path": str(summary_path),
            "metrics": accumulator.summary(),
        }
        dump_json(summary_path, summary)
        return summary

    def _should_compute_loss(self, processed_samples: int, loss_samples: int) -> bool:
        if not self.replay.compute_training_loss:
            return False
        if self.replay.max_loss_samples is not None and loss_samples >= int(self.replay.max_loss_samples):
            return False
        return processed_samples % int(self.replay.loss_every_n_samples) == 0

    def _warmup_count(self) -> int:
        if self.replay.warmup_inferences is not None:
            return max(0, int(self.replay.warmup_inferences))
        return max(0, int(self.config.server.warmup_inferences))


def discover_episode_roots(dataset_root: Path, replay: DatasetReplayConfig) -> list[Path]:
    root = Path(dataset_root)
    if (root / replay.arrays_subdir).is_dir() and (root / replay.videos_subdir).is_dir():
        candidates = [root]
    elif (root / "episodes").is_dir():
        candidates = sorted(path for path in (root / "episodes").iterdir() if path.is_dir())
    else:
        candidates = sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and (path / replay.arrays_subdir).is_dir() and (path / replay.videos_subdir).is_dir()
        )

    if replay.episode_uids:
        requested = set(replay.episode_uids)
        candidates = [path for path in candidates if path.name in requested]
        missing = sorted(requested - {path.name for path in candidates})
        if missing:
            raise FileNotFoundError(f"Requested replay episodes not found under {dataset_root}: {missing[:10]}")

    if replay.max_episodes is not None:
        candidates = candidates[: int(replay.max_episodes)]
    return candidates


def resolve_use_raw_tactile(
    *,
    tactile_mode: str,
    replay: DatasetReplayConfig,
    episode_uid: str,
    phase_offset: int,
    frame_position: int,
    raw_video_present: bool,
) -> bool:
    if tactile_mode == "deform_only":
        return False
    if mode_requires_raw_tactile(tactile_mode):
        return True
    if tactile_mode == "mixed_50":
        if not raw_video_present:
            return False
        unit = stable_unit_interval(
            replay.mixed_seed,
            tactile_mode,
            episode_uid,
            int(phase_offset),
            int(frame_position),
        )
        return bool(unit < float(replay.mixed_raw_probability))
    raise ValueError(f"Unsupported tactile_mode={tactile_mode!r}.")


def mode_requires_raw_tactile(tactile_mode: str) -> bool:
    return tactile_mode in {"deform_plus_raw", "deform_plus_raw_planner_dropped"}


def mode_uses_planner_prefix(tactile_mode: str) -> bool:
    if tactile_mode not in VALID_TACTILE_MODES:
        raise ValueError(f"Unsupported tactile_mode={tactile_mode!r}.")
    return tactile_mode != "deform_plus_raw_planner_dropped"


def stable_unit_interval(*parts: Any) -> float:
    digest = hashlib.blake2b("|".join(str(part) for part in parts).encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    return float(value / float(2**64 - 1))


def stable_seed(*parts: Any) -> int:
    digest = hashlib.blake2b("|".join(str(part) for part in parts).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _safe_fraction(numer: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return float(numer / denom)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value
