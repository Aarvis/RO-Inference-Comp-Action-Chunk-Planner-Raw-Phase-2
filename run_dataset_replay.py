from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from pathlib import Path

from origami_comp_action_chunk_runtime import (
    DatasetReplayRunner,
    load_runtime_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay raw origami episodes through the comp-action-chunk inference pipeline.")
    parser.add_argument("--config", default="configs/dataset_replay.yaml", help="Path to dataset replay YAML config.")
    parser.add_argument(
        "--bundle-root",
        default=None,
        help="Model bundle directory or bundle.yaml path. When set, it replaces --config for runtime/model paths.",
    )
    parser.add_argument("--dataset-root", default=None, help="Override paths.dataset_root from the config.")
    parser.add_argument("--output-dir", default=None, help="Override paths.output_dir from the config.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit the number of replay episodes.")
    parser.add_argument("--max-total-samples", type=int, default=None, help="Limit replay samples per tactile mode.")
    parser.add_argument("--max-loss-samples", type=int, default=None, help="Limit diffusion-loss samples per tactile mode.")
    parser.add_argument("--loss-every-n-samples", type=int, default=None, help="Compute diffusion loss every N replay samples.")
    parser.add_argument(
        "--warmup-inferences",
        type=int,
        default=None,
        help="Number of startup warmup inferences before collecting replay metrics.",
    )
    parser.add_argument("--skip-warmup", action="store_true", help="Do not run startup warmup inferences.")
    parser.add_argument(
        "--skip-warmup-training-loss",
        action="store_true",
        help="Do not warm up the optional OpenPI training-style loss path.",
    )
    parser.add_argument("--tactile-modes", nargs="+", default=None, help="Subset/order of tactile modes to run.")
    parser.add_argument(
        "--phase-offsets",
        nargs="+",
        type=int,
        default=None,
        help="Replay only these phase offsets. Use 0 for starts 0,10,20,... with step_stride=10.",
    )
    parser.add_argument(
        "--openpi-param-dtype",
        choices=("bfloat16", "bf16", "float32", "fp32", "float16", "fp16", "checkpoint", "native", "none"),
        default=os.environ.get("ORIGAMI_OPENPI_PARAM_DTYPE"),
        help="Override OpenPI checkpoint restore dtype for replay/server testing. Defaults to config/env bfloat16.",
    )
    parser.add_argument(
        "--planner-enabled",
        choices=("true", "false"),
        default=None,
        help=(
            "Override runtime.planner_enabled. false skips DINO/OOI/checkpoint-planner "
            "and supplies OpenPI's masked zero planner-dropout prefix."
        ),
    )
    parser.add_argument("--skip-action-error", action="store_true", help="Do not compute action rollout error.")
    parser.add_argument("--skip-training-loss", action="store_true", help="Do not compute OpenPI training-style loss.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate/count replay episodes; do not load models.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def _summary_value(mapping: dict, *keys: str) -> object:
    value: object = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _fmt(value: object, *, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _print_runtime_summary(result: dict) -> None:
    modes = result.get("modes", {})
    if not isinstance(modes, dict):
        return
    print("Dataset replay result summary")
    for tactile_mode, mode_summary in modes.items():
        if not isinstance(mode_summary, dict):
            continue
        metrics = mode_summary.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        action_error = metrics.get("action_error", {})
        if not isinstance(action_error, dict):
            action_error = {}
        fractions = metrics.get("fractions", {})
        if not isinstance(fractions, dict):
            fractions = {}
        training_loss = metrics.get("training_style_loss", {})
        if not isinstance(training_loss, dict):
            training_loss = {}
        timings = metrics.get("timings_ms", {})
        if not isinstance(timings, dict):
            timings = {}

        print(
            f"{tactile_mode}: "
            f"processed={_fmt(mode_summary.get('processed_samples'))}/"
            f"{_fmt(mode_summary.get('planned_samples'))} "
            f"failed={_fmt(metrics.get('failed_samples'))} "
            f"samples_per_second={_fmt(mode_summary.get('samples_per_second'))}"
        )
        print(
            "  action_error: "
            f"mae={_fmt(action_error.get('mae'))} "
            f"rmse={_fmt(action_error.get('rmse'))} "
            f"max_abs={_fmt(action_error.get('max_abs'))} "
            f"first_step_mae={_fmt(_summary_value(action_error, 'first_step_mae', 'mean'))} "
            f"last_step_mae={_fmt(_summary_value(action_error, 'last_step_mae', 'mean'))}"
        )
        print(
            "  training_loss: "
            f"samples={_fmt(metrics.get('training_loss_samples'))} "
            f"loss_base_flow={_fmt(_summary_value(training_loss, 'loss_base_flow', 'mean'))} "
            f"chunked_loss_mean={_fmt(_summary_value(training_loss, 'chunked_loss_mean', 'mean'))}"
        )
        print(
            "  availability: "
            f"raw_requested={_fmt(fractions.get('tactile_raw_requested'))} "
            f"raw_available={_fmt(fractions.get('tactile_raw_available'))} "
            f"planner_available={_fmt(fractions.get('planner_available'))} "
            f"ooi_target_present={_fmt(fractions.get('ooi_target_present'))}"
        )
        print(
            "  timing_ms: "
            f"total={_fmt(_summary_value(timings, 'total', 'mean'))} "
            f"dino_camera={_fmt(_summary_value(timings, 'dino_camera', 'mean'))} "
            f"planner={_fmt(_summary_value(timings, 'planner', 'mean'))} "
            f"openpi={_fmt(_summary_value(timings, 'openpi', 'mean'))} "
            f"training_loss={_fmt(_summary_value(timings, 'training_loss', 'mean'))}"
        )
        per_sample_path = mode_summary.get("per_sample_path")
        if per_sample_path:
            print(f"  per_sample: {per_sample_path}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()), format="%(asctime)s %(levelname)s %(message)s")
    if args.openpi_param_dtype is not None:
        os.environ["ORIGAMI_OPENPI_PARAM_DTYPE"] = str(args.openpi_param_dtype)
        logging.info("Overriding OpenPI checkpoint restore dtype=%s", args.openpi_param_dtype)
    config = load_runtime_config(config_path=args.config, bundle_root=args.bundle_root)

    if args.planner_enabled is not None:
        runtime = dataclasses.replace(
            config.runtime,
            planner_enabled=(args.planner_enabled == "true"),
        )
        config = dataclasses.replace(config, runtime=runtime)
        logging.info("Overriding runtime.planner_enabled=%s", runtime.planner_enabled)

    paths = config.paths
    if args.dataset_root is not None:
        paths = dataclasses.replace(paths, dataset_root=Path(args.dataset_root).resolve())
    if args.output_dir is not None:
        paths = dataclasses.replace(paths, output_dir=Path(args.output_dir).resolve())

    replay = config.dataset_replay
    if args.max_episodes is not None:
        replay = dataclasses.replace(replay, max_episodes=int(args.max_episodes))
    if args.max_total_samples is not None:
        replay = dataclasses.replace(replay, max_total_samples=int(args.max_total_samples))
    if args.max_loss_samples is not None:
        replay = dataclasses.replace(replay, max_loss_samples=int(args.max_loss_samples))
    if args.loss_every_n_samples is not None:
        replay = dataclasses.replace(replay, loss_every_n_samples=int(args.loss_every_n_samples))
    if args.warmup_inferences is not None:
        replay = dataclasses.replace(replay, warmup_inferences=int(args.warmup_inferences))
    if args.skip_warmup:
        replay = dataclasses.replace(replay, warmup_inferences=0)
    if args.skip_warmup_training_loss:
        replay = dataclasses.replace(replay, warmup_training_loss=False)
    if args.tactile_modes is not None:
        replay = dataclasses.replace(replay, tactile_modes=tuple(str(mode) for mode in args.tactile_modes))
    if args.phase_offsets is not None:
        replay = dataclasses.replace(replay, phase_offsets=tuple(int(offset) for offset in args.phase_offsets))
    if args.skip_action_error:
        replay = dataclasses.replace(replay, compute_action_error=False)
    if args.skip_training_loss:
        replay = dataclasses.replace(replay, compute_training_loss=False)

    config = dataclasses.replace(config, paths=paths, dataset_replay=replay)
    result = DatasetReplayRunner(config).run(dry_run=bool(args.dry_run))

    summary_path = config.paths.output_dir / (
        "dataset_replay_plan.json" if args.dry_run else config.dataset_replay.all_modes_summary_filename
    )
    print(f"Saved dataset replay {'plan' if args.dry_run else 'summary'}: {summary_path}")
    if args.dry_run:
        plan = result["plan"]
    else:
        plan = result["plan"]
    print(f"episodes: {plan['episode_count']}")
    for mode, mode_plan in plan["modes"].items():
        print(f"{mode}: planned_samples={mode_plan['samples']} raw_eligible_episodes={mode_plan['raw_eligible_episodes']}")
    if not args.dry_run:
        _print_runtime_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
