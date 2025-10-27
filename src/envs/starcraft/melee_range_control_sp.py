# envs/starcraft/melee_range_control_sp.py
from __future__ import annotations
from typing import Dict, List

import math
import numpy as np

from .utils import (
    ShapingMetrics,
    _is_melee,
    _nearest_enemy,
    ring_function,
    update_shaping_metrics,
    ally_damage_step,
    enemy_damage_step,
    count_alive_allies,
    count_alive_enemies,
)

from .StarCraft2Env import StarCraft2Env


class Starcraft2EnvRewardShaping(StarCraft2Env):
    """State + Potential shaping.
    State shaping uses ring_function(d_min) average; potential adds discounted phi-delta.
    """

    def __init__(
        self,
        *args,
        rc_weight: float = 0.7,
        max_shaping_ratio: float = 0.30,
        log_shaping: bool = True,
        rc_pb_gamma: float = 0.99,
        rc_melee_r_default: float = 3.0,
        rc_shoot_r_default: float = 6.0,
        rc_melee_only: bool = True,
        **kwargs,
    ):
        kwargs['move_amount'] = 3
        super().__init__(*args, **kwargs)

        self._rc_weight = float(rc_weight)
        self._max_ratio = float(max_shaping_ratio)
        self._log = bool(log_shaping)
        self._rc_r_melee_def = float(rc_melee_r_default)
        self._rc_r_shoot_def = float(rc_shoot_r_default)
        self._rc_melee_only = bool(rc_melee_only)

        self.metrics = ShapingMetrics()
        self._pending_state_bonus: float = 0.0
        self.rc_pb_gamma = float(rc_pb_gamma)
        self.phi_prev = 0.0
        self._shaping_cache: Dict[str, object] = {}
        self._first_allied_killed_step = -1.0
        self._first_enemy_killed_step = -1.0

    def reset(self):
        self.metrics.reset()
        self._pending_state_bonus = 0.0
        self.phi_prev = 0.0
        self._shaping_cache.clear()
        self._first_allied_killed_step = -1.0
        self._first_enemy_killed_step = -1.0
        return super().reset()

    def step(self, actions):
        reward, terminated, info = super().step(actions)
        if info is None:
            info = {}
        self._compute_state_bonus()
        if self._log:
            info.update(self.metrics.to_dict())
        if terminated:
            info.update(self._episode_metrics_payload(max(1, self._episode_steps)))
        return reward, terminated, info

    def reward_battle(self) -> float:
        base = super().reward_battle()

        # Raw components
        rc_bonus_state_raw = float(self._pending_state_bonus)
        phi_curr = self._compute_phi_only()
        shaped_delta_raw = float(self._rc_weight * ((self.rc_pb_gamma * phi_curr) - self.phi_prev))

        # Cap only the sum
        cap = self._max_ratio * max(1.0, abs(float(base)))
        total_bonus = rc_bonus_state_raw + shaped_delta_raw
        total_bonus = float(np.clip(total_bonus, -cap, +cap))

        shaped = float(base) + total_bonus

        cache = dict(self._shaping_cache) if isinstance(self._shaping_cache, dict) else {}

        alive_allies = count_alive_allies(self)
        alive_enemies = count_alive_enemies(self)
        cache["ally_alive"] = float(alive_allies)
        cache["enemy_alive"] = float(alive_enemies)

        dmg_to_enemies, enemy_kills = enemy_damage_step(self)
        ally_dmg_total = float(np.sum(dmg_to_enemies)) if getattr(dmg_to_enemies, "size", 0) > 0 else 0.0
        cache["ally_dmg"] = ally_dmg_total

        _, ally_deaths = ally_damage_step(self)
        if ally_deaths > 0 and self._first_allied_killed_step < 0:
            self._first_allied_killed_step = float(self._episode_steps)
        if enemy_kills > 0 and self._first_enemy_killed_step < 0:
            self._first_enemy_killed_step = float(self._episode_steps)

        update_shaping_metrics(
            self.metrics,
            base=float(base),
            delta=float(total_bonus),
            cache=cache,
        )
        self._shaping_cache = {}
        self._pending_state_bonus = 0.0
        self.phi_prev = float(phi_curr)
        return shaped

    def _compute_state_bonus(self) -> None:
        # Use ring_function(dmin) average as a state-based bonus
        melee_ids = []
        rc_dmins: List[float] = []
        ring_raw_sum = 0.0
        cache: Dict[str, object] = {"dmins": [], "raw_bonus": 0.0}
        for j in range(self.n_enemies):
            e = self.enemies.get(j, None)
            if e is None:
                continue
            if (float(getattr(e, "health", 0.0)) + float(getattr(e, "shield", 0.0))) <= 1e-6:
                continue
            if (not self._rc_melee_only) or _is_melee(e.unit_type):
                melee_ids.append(j)
        if len(melee_ids) == 0:
            self._pending_state_bonus = 0.0
            cache["cooldown"] = 0.0
            cache["ally_alive"] = float(count_alive_allies(self))
            cache["enemy_alive"] = float(count_alive_enemies(self))
            cache["ally_dmg"] = 0.0
            self._shaping_cache = cache
            return
        n_alive = 0
        cooldown_sum = 0.0
        for i, ally in self.agents.items():
            if float(getattr(ally, "health", 0.0)) <= 1e-6:
                continue
            n_alive += 1
            dmin, _ = _nearest_enemy(self, ally, melee_ids)
            rc_dmins.append(dmin)
            ring_raw_sum += ring_function(dmin)
            cooldown_sum += float(getattr(ally, "weapon_cooldown", 0.0))
        if n_alive == 0:
            self._pending_state_bonus = 0.0
            cache["cooldown"] = 0.0
            cache["ally_alive"] = float(count_alive_allies(self))
            cache["enemy_alive"] = float(count_alive_enemies(self))
            cache["ally_dmg"] = 0.0
            self._shaping_cache = cache
            return
        ring_raw_mean = ring_raw_sum / n_alive
        self._pending_state_bonus = float(self._rc_weight * ring_raw_mean)
        cache["dmins"] = list(rc_dmins)
        cache["raw_bonus"] = float(ring_raw_mean)
        cache["cooldown"] = float(cooldown_sum / n_alive) if n_alive > 0 else 0.0
        cache["ally_alive"] = float(count_alive_allies(self))
        cache["enemy_alive"] = float(count_alive_enemies(self))
        cache["ally_dmg"] = 0.0
        self._shaping_cache = cache

    def _compute_phi_only(self) -> float:
        # same as state bonus pre-weight
        melee_ids = []
        ring_raw_sum = 0.0
        n_alive = 0
        for j in range(self.n_enemies):
            e = self.enemies.get(j, None)
            if e is None:
                continue
            if (float(getattr(e, "health", 0.0)) + float(getattr(e, "shield", 0.0))) <= 1e-6:
                continue
            if (not self._rc_melee_only) or _is_melee(e.unit_type):
                melee_ids.append(j)
        if len(melee_ids) == 0:
            return 0.0
        for i, ally in self.agents.items():
            if float(getattr(ally, "health", 0.0)) <= 1e-6:
                continue
            n_alive += 1
            dmin, _ = _nearest_enemy(self, ally, melee_ids)
            ring_raw_sum += ring_function(dmin)
        if n_alive == 0:
            return 0.0
        return float(ring_raw_sum / n_alive)

    def _episode_metrics_payload(self, steps: int) -> Dict[str, float]:
        fa = self._first_allied_killed_step if self._first_allied_killed_step >= 0 else 0.0
        fe = self._first_enemy_killed_step if self._first_enemy_killed_step >= 0 else 0.0
        return {
            "shaping/first_allied_killed": float(fa),
            "shaping/first_enemy_killed": float(fe),
        }
