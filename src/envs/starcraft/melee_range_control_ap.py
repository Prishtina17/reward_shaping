# envs/starcraft/melee_range_control_ap.py
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


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Starcraft2EnvRewardShaping(StarCraft2Env):
    """Action + Potential shaping.
    Action shaping follows a simple rule-based controller toward/away shooting zone.
    Potential shaping uses ring-based state potential phi(d_min) with discount rc_pb_gamma.
    """

    def __init__(
        self,
        *args,
        rc_weight: float = 0.7,
        max_shaping_ratio: float = 0.30,
        log_shaping: bool = True,
        rc_pb_gamma: float = 0.99,
        # geometry
        rc_alpha: float = 10,
        rc_m_melee: float = 0.10,
        rc_m_shoot: float = 0.10,
        rc_ring_gain: float = 4,
        rc_melee_barrier: float = 0.7,
        rc_far_cost: float = 0.10,
        rc_close_cost: float = 0.25,
        rc_close_zone_shrink: float = 0.10,
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

        self._rc_alpha = float(rc_alpha)
        self._rc_m_melee = float(rc_m_melee)
        self._rc_m_shoot = float(rc_m_shoot)
        self._rc_ring_gain = float(rc_ring_gain)
        self._rc_barrier = float(rc_melee_barrier)
        self._rc_far = float(rc_far_cost)
        self._rc_close_cost = float(rc_close_cost)
        self._rc_close_zone_shrink = float(rc_close_zone_shrink)

        self._rc_r_melee_def = float(rc_melee_r_default)
        self._rc_r_shoot_def = float(rc_shoot_r_default)
        self._rc_melee_only = bool(rc_melee_only)

        self.metrics = ShapingMetrics()
        self._pending_action_bonus: float = 0.0
        self._shaping_cache: Dict[str, object] = {}

        self.prev_dmin = 0.0
        self.rc_pb_gamma = float(rc_pb_gamma)
        self.phi_prev = 0.0
        self._first_allied_killed_step = -1.0
        self._first_enemy_killed_step = -1.0

    def reset(self):
        self.metrics.reset()
        self._pending_action_bonus = 0.0
        self.prev_dmin = 0.0
        self.phi_prev = 0.0
        self._shaping_cache.clear()
        self._first_allied_killed_step = -1.0
        self._first_enemy_killed_step = -1.0
        return super().reset()

    def step(self, actions):
        reward, terminated, info = super().step(actions)
        if info is None:
            info = {}
        # Prepare action-based shaping for next reward tick
        self._compute_action_bonus(actions)
        if self._log:
            info.update(self.metrics.to_dict())
        if terminated:
            info.update(self._episode_metrics_payload(max(1, self._episode_steps)))
        return reward, terminated, info

    def reward_battle(self) -> float:
        base = super().reward_battle()

        # Raw components (no per-component clipping)
        rc_bonus_act_raw = float(self._pending_action_bonus)
        phi_curr = self._compute_rc_phi()
        shaped_delta_raw = float(self._rc_weight * ((self.rc_pb_gamma * phi_curr) - self.phi_prev))

        # Cap only the sum of all bonuses
        cap = self._max_ratio * max(1.0, abs(float(base)))
        total_bonus = rc_bonus_act_raw + shaped_delta_raw
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
        self._pending_action_bonus = 0.0
        self.phi_prev = float(phi_curr)
        return shaped

    def _compute_rc_phi(self) -> float:
        ring_raw_sum = 0.0
        rc_dmins: List[float] = []
        cache: Dict[str, object] = {"dmins": [], "raw_bonus": 0.0, "cooldown": 0.0}
        cooldown_sum = 0.0

        melee_ids = []
        for j in range(self.n_enemies):
            e = self.enemies.get(j, None)
            if e is None:
                continue
            if (float(getattr(e, "health", 0.0)) + float(getattr(e, "shield", 0.0))) <= 1e-6:
                continue
            if (not self._rc_melee_only) or _is_melee(e.unit_type):
                melee_ids.append(j)
        if len(melee_ids) == 0:
            cache["ally_alive"] = float(count_alive_allies(self))
            cache["enemy_alive"] = float(count_alive_enemies(self))
            cache["ally_dmg"] = 0.0
            self._shaping_cache = cache
            return 0.0

        n_alive = 0
        for i, ally in self.agents.items():
            if float(getattr(ally, "health", 0.0)) <= 1e-6:
                continue
            n_alive += 1
            dmin, _ = _nearest_enemy(self, ally, melee_ids)
            rc_dmins.append(dmin)
            ring_score = ring_function(dmin)
            ring_raw_sum += ring_score
            cooldown_sum += float(getattr(ally, "weapon_cooldown", 0.0))

        if n_alive == 0:
            cache["ally_alive"] = float(count_alive_allies(self))
            cache["enemy_alive"] = float(count_alive_enemies(self))
            cache["ally_dmg"] = 0.0
            self._shaping_cache = cache
            return 0.0
        ring_raw_mean = ring_raw_sum / n_alive
        cache["dmins"] = list(rc_dmins)
        cache["raw_bonus"] = float(ring_raw_mean)
        cache["cooldown"] = float(cooldown_sum / n_alive) if n_alive > 0 else 0.0
        cache["ally_alive"] = float(count_alive_allies(self))
        cache["enemy_alive"] = float(count_alive_enemies(self))
        cache["ally_dmg"] = 0.0
        self._shaping_cache = cache
        return float(ring_raw_mean)

    def _compute_action_bonus(self, actions) -> None:
        actions_int = [int(a) for a in actions]
        rc_dmins: List[float] = []
        cooldown_sum = 0.0
        melee_ids = []
        for j in range(self.n_enemies):
            e = self.enemies.get(j, None)
            if e is None:
                continue
            if (float(getattr(e, "health", 0.0)) + float(getattr(e, "shield", 0.0))) <= 1e-6:
                continue
            if (not self._rc_melee_only) or _is_melee(e.unit_type):
                melee_ids.append(j)
        if len(melee_ids) == 0:
            self._pending_action_bonus = 0.0
            self._shaping_cache.setdefault("cooldown", 0.0)
            self._shaping_cache["cooldown"] = 0.0
            self._shaping_cache["ally_alive"] = float(count_alive_allies(self))
            self._shaping_cache["enemy_alive"] = float(count_alive_enemies(self))
            self._shaping_cache["ally_dmg"] = 0.0
            return

        n_alive = 0
        score_sum = 0.0
        for i, action in enumerate(actions_int):
            ally = self.agents.get(i, None)
            if ally is None or float(getattr(ally, "health", 0.0)) <= 1e-6:
                continue
            n_alive += 1
            cooldown_sum += float(getattr(ally, "weapon_cooldown", 0.0))
            dmin, j_near = _nearest_enemy(self, ally, melee_ids)
            rc_dmins.append(dmin)
            target = self.enemies.get(j_near)
            if target is None:
                continue
            dx = target.pos.x - ally.pos.x
            dy = target.pos.y - ally.pos.y

            action_num = 0
            if abs(dx) > abs(dy):
                action_num = 4 if dx < 0 else 5
            else:
                action_num = 2 if dy < 0 else 3

            score = 0
            if 3.0 <= dmin <= 7.0:
                if getattr(ally, "weapon_cooldown", 0.0) > 0:
                    if action_num == action:
                        score = 1
                else:
                    if action > 5:
                        score = 1
            else:
                if action_num == action:
                    score = 1

            score_sum += score

        if n_alive == 0:
            self._pending_action_bonus = 0.0
            self._shaping_cache.setdefault("cooldown", 0.0)
            self._shaping_cache["ally_alive"] = float(count_alive_allies(self))
            self._shaping_cache["enemy_alive"] = float(count_alive_enemies(self))
            self._shaping_cache["ally_dmg"] = 0.0
            return
        rc_raw_mean = (score_sum / n_alive) * float(self._rc_weight)
        self._pending_action_bonus = float(rc_raw_mean)
        self._shaping_cache.setdefault("cooldown", 0.0)
        self._shaping_cache["cooldown"] = float(cooldown_sum / n_alive) if n_alive > 0 else 0.0
        self._shaping_cache["ally_alive"] = float(count_alive_allies(self))
        self._shaping_cache["enemy_alive"] = float(count_alive_enemies(self))
        self._shaping_cache["ally_dmg"] = 0.0

    def _episode_metrics_payload(self, steps: int) -> Dict[str, float]:
        fa = self._first_allied_killed_step if self._first_allied_killed_step >= 0 else 0.0
        fe = self._first_enemy_killed_step if self._first_enemy_killed_step >= 0 else 0.0
        return {
            "shaping/first_allied_killed": float(fa),
            "shaping/first_enemy_killed": float(fe),
        }
