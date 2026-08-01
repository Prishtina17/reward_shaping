"""Aggregate seeded TensorBoard runs and render publication-ready summaries."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


LOGGER = logging.getLogger(__name__)
STRUCTURED_RUN_RE = re.compile(
    r"^env=(?P<env>.+?)__map=(?P<map>.+?)__seed=(?P<seed>\d+)"
    r"__timestamp=(?P<timestamp>\d{8}_\d{6})$"
)
LEGACY_TIMESTAMP_RE = re.compile(r"\d{8}_\d{6}$")
T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


@dataclass(frozen=True)
class RunIdentity:
    env: str
    map_name: str
    seed: Optional[int]
    timestamp: str


@dataclass
class ScalarSeries:
    steps: List[int]
    values: List[float]


@dataclass
class TensorBoardRun:
    identity: RunIdentity
    path: Path
    scalars: Dict[str, ScalarSeries]


def t_critical_95(df: int) -> float:
    """Two-sided 95% Student-t critical value, with normal fallback."""
    if df < 1:
        raise ValueError("degrees of freedom must be positive")
    return T_CRITICAL_975.get(df, 1.96)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate seeded TensorBoard runs, plot mean/95% CI, and write CSV summaries."
    )
    parser.add_argument(
        "--log-root",
        default=Path("results/final_run/tb_logs"),
        type=Path,
        help="TensorBoard root produced by run_all_shapings.sh.",
    )
    parser.add_argument(
        "--output-root",
        default=Path("results/final_run/metrics"),
        type=Path,
        help="Destination for figures and CSV summaries.",
    )
    parser.add_argument(
        "--primary-metric",
        default="test_battle_won_mean",
        help="Metric used for AUC, final value, and steps-to-threshold summaries.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        default=(0.50, 0.80, 0.95),
        type=float,
        help="Win-rate thresholds for sample-efficiency reporting.",
    )
    parser.add_argument(
        "--threshold-window",
        default=3,
        type=int,
        help="Consecutive evaluations that must stay above a threshold.",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include old unseeded <env>_<map>_<timestamp> directories.",
    )
    parser.add_argument("--dpi", default=150, type=int)
    parser.add_argument(
        "--figsize", nargs=2, default=(10.0, 6.0), type=float, metavar=("WIDTH", "HEIGHT")
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s - %(message)s",
    )


def parse_run_identity(
    run_name: str, legacy_map_name: Optional[str] = None
) -> Optional[RunIdentity]:
    match = STRUCTURED_RUN_RE.match(run_name)
    if match:
        return RunIdentity(
            env=match.group("env"),
            map_name=match.group("map"),
            seed=int(match.group("seed")),
            timestamp=match.group("timestamp"),
        )

    if not legacy_map_name:
        return None
    separator = f"_{legacy_map_name}_"
    if separator not in run_name:
        return None
    env, _, timestamp = run_name.partition(separator)
    if not env or not LEGACY_TIMESTAMP_RE.fullmatch(timestamp):
        return None
    return RunIdentity(env=env, map_name=legacy_map_name, seed=None, timestamp=timestamp)


def contains_tensorboard_events(path: Path) -> bool:
    return any(path.glob("events.out.tfevents.*"))


def discover_run_directories(
    log_root: Path, include_legacy: bool = False
) -> List[Tuple[RunIdentity, Path]]:
    discovered: List[Tuple[RunIdentity, Path]] = []
    candidates = [log_root] + [path for path in log_root.rglob("*") if path.is_dir()]
    for path in candidates:
        if not contains_tensorboard_events(path):
            continue
        identity = parse_run_identity(path.name)
        if identity is None and include_legacy:
            identity = parse_run_identity(path.name, path.parent.name)
        if identity is None:
            LOGGER.debug("Ignoring unrecognised run directory: %s", path)
            continue
        discovered.append((identity, path))
    return sorted(
        discovered,
        key=lambda item: (
            item[0].map_name,
            item[0].env,
            -1 if item[0].seed is None else item[0].seed,
            item[0].timestamp,
        ),
    )


def load_scalars(run_dir: Path) -> Dict[str, ScalarSeries]:
    # TensorBoard versions used by the original PyMARL2 environment reference
    # the pre-NumPy-2 alias. Keep legacy log analysis usable on a newer host.
    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_  # type: ignore[attr-defined]
    if not hasattr(np, "string_"):
        np.string_ = np.bytes_  # type: ignore[attr-defined]
    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_  # type: ignore[attr-defined]
    # EventAccumulator only needs TensorBoard's lightweight compatibility stub.
    # Force it so an unrelated, ABI-incompatible TensorFlow installation on the
    # analysis host cannot break reading event files.
    sys.modules.setdefault(
        "tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf")
    )
    from tensorboard.backend.event_processing import event_accumulator

    accumulator = event_accumulator.EventAccumulator(str(run_dir))
    accumulator.Reload()
    result: Dict[str, ScalarSeries] = {}
    for tag in accumulator.Tags().get("scalars", []):
        points = accumulator.Scalars(tag)
        if not points:
            continue
        # TensorBoard can contain repeated steps after resumed writes. Keep the
        # final value for each step so interpolation receives a strict x-axis.
        by_step = {int(point.step): float(point.value) for point in points}
        steps = sorted(by_step)
        result[tag] = ScalarSeries(steps=steps, values=[by_step[step] for step in steps])
    return result


def load_runs(log_root: Path, include_legacy: bool = False) -> List[TensorBoardRun]:
    runs: List[TensorBoardRun] = []
    for identity, path in discover_run_directories(log_root, include_legacy):
        try:
            scalars = load_scalars(path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping %s: %s", path, exc)
            continue
        runs.append(TensorBoardRun(identity=identity, path=path, scalars=scalars))
        LOGGER.info(
            "Loaded %d metrics: map=%s env=%s seed=%s",
            len(scalars),
            identity.map_name,
            identity.env,
            identity.seed,
        )
    return runs


def aggregate_series(
    series_list: Sequence[ScalarSeries],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    usable = [series for series in series_list if series.steps and series.values]
    if not usable:
        return np.asarray([]), np.asarray([]), None

    overlap_start = max(series.steps[0] for series in usable)
    overlap_end = min(series.steps[-1] for series in usable)
    if overlap_end < overlap_start:
        return np.asarray([]), np.asarray([]), None

    grid = sorted(
        {
            step
            for series in usable
            for step in series.steps
            if overlap_start <= step <= overlap_end
        }
    )
    x = np.asarray(grid, dtype=np.float64)
    values = np.vstack(
        [
            np.interp(
                x,
                np.asarray(series.steps, dtype=np.float64),
                np.asarray(series.values, dtype=np.float64),
            )
            for series in usable
        ]
    )
    mean = np.mean(values, axis=0)
    if len(usable) < 2:
        return x, mean, None

    sem = np.std(values, axis=0, ddof=1) / np.sqrt(float(len(usable)))
    critical = t_critical_95(len(usable) - 1)
    return x, mean, critical * sem


def label_for_env(env: str) -> str:
    prefix = "melee_range_control_"
    return env[len(prefix) :] if env.startswith(prefix) else env


def sanitize_metric_name(metric: str) -> str:
    return metric.replace("/", "__")


def plot_map_metrics(
    map_name: str,
    runs: Sequence[TensorBoardRun],
    output_root: Path,
    figsize: Tuple[float, float],
    dpi: int,
) -> None:
    tags = sorted({tag for run in runs for tag in run.scalars})
    by_env: Dict[str, List[TensorBoardRun]] = defaultdict(list)
    for run in runs:
        by_env[run.identity.env].append(run)

    map_output = output_root / map_name
    map_output.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        plotted = False
        plt.figure(figsize=figsize)
        for env, env_runs in sorted(by_env.items()):
            series = [run.scalars[tag] for run in env_runs if tag in run.scalars]
            x, mean, ci = aggregate_series(series)
            if x.size == 0:
                continue
            plotted = True
            label = f"{label_for_env(env)} (n={len(series)})"
            (line,) = plt.plot(x, mean, label=label)
            if ci is not None:
                plt.fill_between(x, mean - ci, mean + ci, color=line.get_color(), alpha=0.18)

        if not plotted:
            plt.close()
            continue
        plt.title(f"{map_name} — {tag}")
        plt.xlabel("Environment steps")
        plt.ylabel(tag)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        destination = map_output / f"{sanitize_metric_name(tag)}.png"
        plt.savefig(destination, dpi=dpi)
        plt.close()
        LOGGER.info("Saved %s", destination)


def normalized_auc(series: ScalarSeries) -> float:
    if not series.values:
        return float("nan")
    if len(series.values) == 1 or series.steps[-1] == series.steps[0]:
        return float(series.values[-1])
    x = np.asarray(series.steps, dtype=np.float64)
    y = np.asarray(series.values, dtype=np.float64)
    area = np.sum((y[1:] + y[:-1]) * (x[1:] - x[:-1]) * 0.5)
    return float(area / (x[-1] - x[0]))


def sustained_threshold_step(
    series: ScalarSeries, threshold: float, window: int
) -> Optional[float]:
    if window < 1:
        raise ValueError("threshold window must be at least 1")
    values = np.asarray(series.values, dtype=np.float64)
    for index in range(0, len(values) - window + 1):
        if np.all(values[index : index + window] >= threshold):
            return float(series.steps[index])
    return None


def confidence_interval(values: Sequence[float]) -> Tuple[float, float, float]:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(clean))
    if clean.size == 1:
        return mean, float("nan"), float("nan")
    std = float(np.std(clean, ddof=1))
    half_width = t_critical_95(int(clean.size - 1)) * std / np.sqrt(clean.size)
    return mean, std, half_width


def write_summaries(
    runs: Sequence[TensorBoardRun],
    output_root: Path,
    metric: str,
    thresholds: Sequence[float],
    threshold_window: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    threshold_columns = [f"steps_to_{int(round(value * 100))}" for value in thresholds]
    fieldnames = [
        "map",
        "env",
        "seed",
        "metric",
        "points",
        "last_step",
        "final_value",
        "normalized_auc",
        *threshold_columns,
        "run_path",
    ]

    rows: List[Dict[str, object]] = []
    for run in runs:
        series = run.scalars.get(metric)
        if series is None or not series.values:
            LOGGER.warning("Metric %s is missing in %s", metric, run.path)
            continue
        row: Dict[str, object] = {
            "map": run.identity.map_name,
            "env": run.identity.env,
            "seed": "" if run.identity.seed is None else run.identity.seed,
            "metric": metric,
            "points": len(series.values),
            "last_step": series.steps[-1],
            "final_value": series.values[-1],
            "normalized_auc": normalized_auc(series),
            "run_path": str(run.path),
        }
        for threshold, column in zip(thresholds, threshold_columns):
            step = sustained_threshold_step(series, threshold, threshold_window)
            row[column] = "" if step is None else step
        rows.append(row)

    with (output_root / "run_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_fields = [
        "map",
        "env",
        "n_seeds",
        "final_mean",
        "final_std",
        "final_ci95",
        "auc_mean",
        "auc_std",
        "auc_ci95",
    ]
    for column in threshold_columns:
        aggregate_fields.extend([f"{column}_mean", f"{column}_reached"])

    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["map"]), str(row["env"]))].append(row)

    aggregate_rows: List[Dict[str, object]] = []
    for (map_name, env), group in sorted(grouped.items()):
        final_mean, final_std, final_ci = confidence_interval(
            [float(row["final_value"]) for row in group]
        )
        auc_mean, auc_std, auc_ci = confidence_interval(
            [float(row["normalized_auc"]) for row in group]
        )
        aggregate_row: Dict[str, object] = {
            "map": map_name,
            "env": env,
            "n_seeds": len(group),
            "final_mean": final_mean,
            "final_std": final_std,
            "final_ci95": final_ci,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "auc_ci95": auc_ci,
        }
        for column in threshold_columns:
            reached = [float(row[column]) for row in group if row[column] != ""]
            aggregate_row[f"{column}_mean"] = (
                float(np.mean(reached)) if reached else ""
            )
            aggregate_row[f"{column}_reached"] = f"{len(reached)}/{len(group)}"
        aggregate_rows.append(aggregate_row)

    with (output_root / "aggregate_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    LOGGER.info("Saved per-run and aggregate summaries under %s", output_root)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    if not args.log_root.exists():
        raise FileNotFoundError(f"Log root does not exist: {args.log_root}")
    if args.threshold_window < 1:
        raise ValueError("--threshold-window must be at least 1")

    runs = load_runs(args.log_root, include_legacy=args.include_legacy)
    if not runs:
        legacy_hint = " Add --include-legacy for old unseeded logs." if not args.include_legacy else ""
        raise RuntimeError(f"No recognised TensorBoard runs found under {args.log_root}.{legacy_hint}")

    by_map: Dict[str, List[TensorBoardRun]] = defaultdict(list)
    for run in runs:
        by_map[run.identity.map_name].append(run)
    for map_name, map_runs in sorted(by_map.items()):
        plot_map_metrics(
            map_name,
            map_runs,
            args.output_root,
            tuple(args.figsize),
            args.dpi,
        )

    write_summaries(
        runs,
        args.output_root,
        args.primary_metric,
        args.thresholds,
        args.threshold_window,
    )


if __name__ == "__main__":
    main()
