from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_shaping_utils():
    path = ROOT / "src" / "envs" / "starcraft" / "utils.py"
    spec = importlib.util.spec_from_file_location("reward_shaping_utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


UTILS = load_shaping_utils()


class DummyUnit:
    def __init__(self, x, y, *, unit_type=73, cooldown=0.0):
        self.pos = SimpleNamespace(x=float(x), y=float(y))
        self.unit_type = unit_type
        self.weapon_cooldown = float(cooldown)
        self.health = 100.0
        self.shield = 0.0


class DummyEnv:
    n_agents = 1
    n_enemies = 1
    n_actions_no_attack = 6

    def __init__(self, distance, cooldown=0.0):
        self.agents = {0: DummyUnit(0.0, 0.0, unit_type=74, cooldown=cooldown)}
        self.enemies = {0: DummyUnit(distance, 0.0, unit_type=73)}


@pytest.mark.parametrize(
    ("unit_type", "expected"),
    [(9, True), (73, True), (105, True), (48, False), (74, False), (107, False)],
)
def test_melee_taxonomy(unit_type, expected):
    assert UTILS._is_melee(unit_type) is expected


@pytest.mark.parametrize(
    ("distance", "cooldown", "action", "expected"),
    [
        (2.0, 0.0, 5, 1.0),  # too close: move west, away from an eastern target
        (5.0, 0.0, 6, 1.0),  # in range and ready: attack
        (5.0, 4.0, 5, 1.0),  # in range on cooldown: kite away
        (8.0, 0.0, 4, 1.0),  # out of range: move east, toward the target
        (8.0, 0.0, 5, 0.0),  # moving farther away must not be rewarded
    ],
)
def test_kiting_action_advice(distance, cooldown, action, expected):
    bonus, cache = UTILS.compute_kiting_action_bonus(
        DummyEnv(distance, cooldown),
        [action],
        weight=1.0,
        melee_only=True,
        melee_range=3.0,
        shoot_range=6.0,
    )
    assert bonus == pytest.approx(expected)
    assert cache["dmins"] == pytest.approx([distance])
    assert cache["raw_bonus"] == pytest.approx(expected)
    assert cache["action_match_fraction"] == pytest.approx(expected)


def test_action_weight_scales_reward_but_not_match_diagnostic():
    bonus, cache = UTILS.compute_kiting_action_bonus(
        DummyEnv(5.0),
        [6],
        weight=0.25,
        melee_only=True,
        melee_range=3.0,
        shoot_range=6.0,
    )
    assert bonus == pytest.approx(0.25)
    assert cache["raw_bonus"] == pytest.approx(1.0)
    assert cache["action_match_fraction"] == pytest.approx(1.0)


def test_ring_score_is_signed_around_target_band():
    score = lambda distance: UTILS.ring_function(
        distance, center=4.5, half_width=1.5
    )
    assert score(2.0) < 0.0
    assert score(3.0) == pytest.approx(0.0, abs=1e-12)
    assert score(4.5) > 0.8
    assert score(6.0) == pytest.approx(0.0, abs=1e-12)
    assert score(7.0) < 0.0
    assert -1.0 <= score(100.0) <= 1.0


def test_potential_bonus_telescopes_with_terminal_zero():
    gamma = 0.99
    phi_0, phi_1, phi_terminal = -0.4, 0.7, 0.0
    first = UTILS.potential_shaping_bonus(phi_0, phi_1, gamma)
    second = UTILS.potential_shaping_bonus(phi_1, phi_terminal, gamma)
    assert first + gamma * second == pytest.approx(-phi_0)


def test_unit_cap_preserves_single_normalized_components():
    assert UTILS.clip_shaping_bonus(0.25, 1.0) == pytest.approx(0.25)
    assert UTILS.clip_shaping_bonus(-0.75, 1.0) == pytest.approx(-0.75)
    assert UTILS.clip_shaping_bonus(1.4, 1.0) == pytest.approx(1.0)
    assert UTILS.clip_shaping_bonus(-1.4, 1.0) == pytest.approx(-1.0)


def test_episode_metrics_accumulate_components_and_reconcile_terminal_reward():
    metrics = UTILS.RewardShapingEpisodeMetrics()
    metrics.record(
        base=0.5,
        action=0.1,
        state=-0.2,
        potential=0.05,
        total_applied=-0.05,
        action_match=1.0,
        state_score=-0.2,
        potential_value=0.4,
    )
    metrics.record(
        base=0.25,
        action=0.4,
        total_applied=0.3,
        action_match=0.0,
    )
    metrics.reconcile_last_transition(base=2.25, applied=0.3)

    info = metrics.to_info()
    assert info["rc/episode_steps"] == 2
    assert info["rc/base_return"] == pytest.approx(2.75)
    assert info["rc/action_return"] == pytest.approx(0.5)
    assert info["rc/state_return"] == pytest.approx(-0.2)
    assert info["rc/potential_return"] == pytest.approx(0.05)
    assert info["rc/total_raw_return"] == pytest.approx(0.35)
    assert info["rc/total_applied_return"] == pytest.approx(0.25)
    assert info["rc/negative_fraction"] == pytest.approx(0.5)
    assert info["rc/clipped_fraction"] == pytest.approx(0.5)
    assert info["rc/action_match_fraction"] == pytest.approx(0.5)

    metrics.reset()
    assert metrics.to_info()["rc/episode_steps"] == 0


def test_all_env_configs_share_movement_and_explicit_ranges():
    config_dir = ROOT / "src" / "config" / "envs"
    baseline_args = yaml.safe_load(
        (config_dir / "sc2.yaml").read_text(encoding="utf-8")
    )["env_args"]
    for path in sorted(config_dir.glob("melee_range_control_*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        args = config["env_args"]
        assert (
            args["move_amount"] == baseline_args["move_amount"] == 2
        ), path.name
        assert args["rc_melee_r_default"] == pytest.approx(3.0), path.name
        assert args["rc_shoot_r_default"] == pytest.approx(6.0), path.name
        assert args["rc_weight"] == pytest.approx(1.0), path.name
        assert args["max_shaping_abs"] == pytest.approx(1.0), path.name
        assert "max_shaping_ratio" not in args, path.name


def test_shaping_classes_do_not_override_move_amount():
    source_dir = ROOT / "src" / "envs" / "starcraft"
    for path in sorted(source_dir.glob("melee_range_control_*.py")):
        source = path.read_text(encoding="utf-8")
        assert "kwargs['move_amount']" not in source, path.name
        assert 'kwargs["move_amount"]' not in source, path.name


def test_final_qmix_protocol_uses_fixed_step_budget():
    config = yaml.safe_load(
        (ROOT / "src" / "config" / "algs" / "qmix.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["training_stop_mode"] == "steps"


def test_potential_configs_match_learner_gamma():
    learner_gamma = yaml.safe_load(
        (ROOT / "src" / "config" / "default.yaml").read_text(encoding="utf-8")
    )["gamma"]
    config_dir = ROOT / "src" / "config" / "envs"
    for suffix in ("pb", "ap", "sp", "asp"):
        args = yaml.safe_load(
            (config_dir / f"melee_range_control_{suffix}.yaml").read_text(
                encoding="utf-8"
            )
        )["env_args"]
        assert args["rc_pb_gamma"] == pytest.approx(learner_gamma), suffix
