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


def test_all_env_configs_share_movement_and_explicit_ranges():
    config_dir = ROOT / "src" / "config" / "envs"
    for path in sorted(config_dir.glob("melee_range_control_*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        args = config["env_args"]
        assert args["move_amount"] == 2, path.name
        assert args["rc_melee_r_default"] == pytest.approx(3.0), path.name
        assert args["rc_shoot_r_default"] == pytest.approx(6.0), path.name


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
