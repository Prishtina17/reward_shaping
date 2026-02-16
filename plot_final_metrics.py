"""
Generate Matplotlib plots from TensorBoard logs stored under ``final_tb_logs``.

For each map, the script selects the most recent run per shaping configuration and
produces one figure per scalar metric with all shapings and the baseline SC2 curve.
Figures are saved under ``final_metrics/<map>/<metric>.png``.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing import event_accumulator  # noqa: E402


LOGGER = logging.getLogger(__name__)


@dataclass
class ScalarSeries:
    label: str
    steps: List[int]
    values: List[float]


TIMESTAMP_RE = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})(?:_(?P<suffix>.*))?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics from TensorBoard logs.")
    parser.add_argument(
        "--log-root",
        default="final_tb_logs",
        type=Path,
        help="Directory containing map subdirectories with TensorBoard runs.",
    )
    parser.add_argument(
        "--output-root",
        default="final_metrics",
        type=Path,
        help="Destination directory for the generated figures.",
    )
    parser.add_argument(
        "--dpi",
        default=150,
        type=int,
        help="Figure DPI to use when saving.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        default=(10, 6),
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")


def find_latest_runs(map_dir: Path, map_name: str) -> Dict[str, Path]:
    """Select the most recent run directory per shaping prefix."""
    latest: Dict[str, Tuple[str, Path]] = {}
    pattern = f"_{map_name}_"
    for entry in sorted(map_dir.iterdir()):
        if not entry.is_dir():
            continue
        run_name = entry.name
        prefix, timestamp = split_prefix_and_timestamp(run_name, pattern)
        key = prefix or run_name
        current = latest.get(key)
        if current is None or timestamp > current[0]:
            latest[key] = (timestamp, entry)
    LOGGER.debug("Map %s -> selected runs: %s", map_name, {k: v[1].name for k, v in latest.items()})
    return {k: v[1] for k, v in latest.items()}


def split_prefix_and_timestamp(run_name: str, separator: str) -> Tuple[str, str]:
    if separator in run_name:
        prefix, _, suffix = run_name.partition(separator)
    else:
        prefix, suffix = run_name, ""
    timestamp = suffix if TIMESTAMP_RE.match(suffix) else suffix
    return prefix, timestamp


def label_for_prefix(prefix: str) -> str:
    if prefix.startswith("melee_range_control_"):
        return prefix[len("melee_range_control_") :]
    return prefix


def load_scalars(run_dir: Path) -> Dict[str, ScalarSeries]:
    accumulator = event_accumulator.EventAccumulator(str(run_dir))
    accumulator.Reload()
    scalars = accumulator.Tags().get("scalars", [])
    series: Dict[str, ScalarSeries] = {}
    for tag in scalars:
        data = accumulator.Scalars(tag)
        if not data:
            continue
        steps = [point.step for point in data]
        values = [point.value for point in data]
        series[tag] = ScalarSeries(label="", steps=steps, values=values)
    return series


def collect_series(map_dir: Path, map_name: str) -> Dict[str, List[ScalarSeries]]:
    runs = find_latest_runs(map_dir, map_name)
    metric_series: Dict[str, List[ScalarSeries]] = defaultdict(list)
    for prefix, run_path in sorted(runs.items()):
        label = label_for_prefix(prefix)
        try:
            scalars = load_scalars(run_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping run %s due to error: %s", run_path.name, exc)
            continue
        for metric, series in scalars.items():
            metric_series[metric].append(
                ScalarSeries(label=label, steps=series.steps, values=series.values)
            )
        LOGGER.info("Loaded %s metrics for %s/%s", len(scalars), map_name, run_path.name)
    return metric_series


def sanitize_metric_name(metric: str) -> str:
    return metric.replace("/", "__")


def plot_metric(
    map_name: str,
    metric: str,
    series_list: Iterable[ScalarSeries],
    output_dir: Path,
    figsize: Tuple[float, float],
    dpi: int,
) -> None:
    usable = [series for series in series_list if series.steps and series.values]
    if not usable:
        LOGGER.debug("Skipping empty metric %s for map %s", metric, map_name)
        return

    plt.figure(figsize=figsize)
    for series in usable:
        plt.plot(series.steps, series.values, label=series.label)
    plt.title(f"{map_name} — {metric}")
    plt.xlabel("Training Step")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = sanitize_metric_name(metric) + ".png"
    path = output_dir / filename
    plt.savefig(path, dpi=dpi)
    plt.close()
    LOGGER.info("Saved %s", path)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if not args.log_root.exists():
        raise FileNotFoundError(f"Log root does not exist: {args.log_root}")

    maps = sorted(entry for entry in args.log_root.iterdir() if entry.is_dir())
    LOGGER.info("Found %d maps under %s", len(maps), args.log_root)

    for map_path in maps:
        metric_series = collect_series(map_path, map_path.name)
        if not metric_series:
            LOGGER.warning("No metrics found for map %s; skipping.", map_path.name)
            continue
        LOGGER.info(
            "Rendering %d metrics for map %s", len(metric_series), map_path.name
        )
        out_dir = args.output_root / map_path.name
        for metric in sorted(metric_series):
            plot_metric(
                map_name=map_path.name,
                metric=metric,
                series_list=metric_series[metric],
                output_dir=out_dir,
                figsize=tuple(args.figsize),
                dpi=args.dpi,
            )


if __name__ == "__main__":
    main()
