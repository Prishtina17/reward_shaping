from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plot_final_metrics import (  # noqa: E402
    ScalarSeries,
    aggregate_series,
    normalized_auc,
    parse_run_identity,
    sustained_threshold_step,
)


def test_structured_run_identity_keeps_seed_and_underscores():
    identity = parse_run_identity(
        "env=melee_range_control_pb__map=3s_vs_5z__seed=44__timestamp=20260802_120000"
    )
    assert identity is not None
    assert identity.env == "melee_range_control_pb"
    assert identity.map_name == "3s_vs_5z"
    assert identity.seed == 44


def test_sustained_threshold_rejects_single_spike():
    series = ScalarSeries(
        steps=[0, 10, 20, 30, 40, 50],
        values=[0.1, 0.85, 0.2, 0.81, 0.82, 0.84],
    )
    assert sustained_threshold_step(series, 0.8, window=3) == 30.0
    assert sustained_threshold_step(series, 0.9, window=2) is None


def test_aggregate_series_interpolates_seed_curves_on_overlap():
    first = ScalarSeries(steps=[0, 10, 20], values=[0.0, 0.5, 1.0])
    second = ScalarSeries(steps=[0, 20], values=[0.0, 0.8])
    x, mean, ci = aggregate_series([first, second])
    assert x.tolist() == [0.0, 10.0, 20.0]
    assert mean.tolist() == pytest.approx([0.0, 0.45, 0.9])
    assert ci is not None
    assert np.all(ci >= 0.0)


def test_normalized_auc_uses_the_observed_step_axis():
    series = ScalarSeries(steps=[0, 10, 20], values=[0.0, 0.5, 1.0])
    assert normalized_auc(series) == pytest.approx(0.5)
