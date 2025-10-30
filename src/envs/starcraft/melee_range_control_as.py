# envs/starcraft/melee_range_control_as.py
from __future__ import annotations
from typing import Dict, List

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
    """Action + State reward shaping without potential term."""

    def __init__(
        self,
        *args,
        rc_weight: float = 0.7,
        max_shaping_ratio: float = 0.30,
        log_shaping: bool = True,
        rc_melee_only: bool = True,
        **kwargs,
    ):
        kwargs["move_amount"] = 3
        super().__init__(*args, **kwargs)

        self._rc_weight = float(rc_weight)
        self._max_ratio = float(max_shaping_ratio)
        self._log = bool(log_shaping)
        self._rc_melee_only = bool(rc_melee_only)

        self.metrics = ShapingMetrics()
        self._pending_action_bonus: float = 0.0
        self._pending_state_bonus: float = 0.0
        self._action_cache: Dict[str, object] = {}
        self._state_cache: Dict[str, object] = {}
        self._first_allied_killed_step = -1.0
        self._first_enemy_killed_step = -1.0

    def reset(self):
        self.metrics.reset()
        self._pending_action_bonus = 0.0
        self._pending_state_bonus = 0.0
        self._action_cache = {}
        self._state_cache = {}
        self._first_allied_killed_step = -1.0
        self._first_enemy_killed_step = -1.0
        return super().reset()

    def step(self, actions):
        self._compute_action_bonus(actions)

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

        bonus_action = float(self._pending_action_bonus)
        bonus_state = float(self._pending_state_bonus)
        total_bonus = bonus_action + bonus_state

        cap = self._max_ratio * max(1.0, abs(float(base)))
        total_bonus = float(np.clip(total_bonus, -cap, +cap))

        shaped = float(base) + total_bonus

        cache: Dict[str, object] = {}
        if self._state_cache:
            cache.update(self._state_cache)
        if "dmins" not in cache and self._action_cache.get("dmins"):
            cache["dmins"] = list(self._action_cache["dmins"])
        if "cooldown" not in cache and self._action_cache.get("cooldown") is not None:
            cache["cooldown"] = float(self._action_cache["cooldown"])

        cache["raw_bonus_action"] = float(self._action_cache.get("raw_bonus_action", 0.0))
        cache["raw_bonus_state"] = float(self._state_cache.get("raw_bonus_state", 0.0))
        cache["raw_bonus"] = cache["raw_bonus_action"] + cache["raw_bonus_state"]

        cache["ally_alive"] = float(count_alive_allies(self))
        cache["enemy_alive"] = float(count_alive_enemies(self))

        dmg_to_enemies, enemy_kills = enemy_damage_step(self)
        cache["ally_dmg"] = float(np.sum(dmg_to_enemies)) if getattr(dmg_to_enemies, "size", 0) > 0 else 0.0

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

        self._pending_action_bonus = 0.0
        self._pending_state_bonus = 0.0
        self._action_cache = {}
        self._state_cache = {}
        return shaped

    def _compute_action_bonus(self, actions) -> None:
        self._action_cache = {}
        actions_int = [int(a) for a in actions]
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
            self._action_cache = {"raw_bonus_action": 0.0}
            return

        score_sum = 0.0
        n_alive = 0
        cooldown_sum = 0.0
        rc_dmins: List[float] = []
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
            self._action_cache = {
                "raw_bonus_action": 0.0,
                "dmins": list(rc_dmins),
                "cooldown": 0.0,
            }
            return

        raw_mean = float(score_sum / n_alive)
        self._pending_action_bonus = float(self._rc_weight * raw_mean)
        self._action_cache = {
            "raw_bonus_action": raw_mean,
            "dmins": list(rc_dmins),
            "cooldown": float(cooldown_sum / n_alive) if n_alive > 0 else 0.0,
        }

    def _compute_state_bonus(self) -> None:
        self._state_cache = {}
        melee_ids = []
        rc_dmins: List[float] = []
        ring_raw_sum = 0.0
        cooldown_sum = 0.0
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
            self._state_cache = {
                "dmins": [],
                "raw_bonus_state": 0.0,
                "cooldown": 0.0,
            }
            return

        n_alive = 0
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
            self._state_cache = {
                "dmins": list(rc_dmins),
                "raw_bonus_state": 0.0,
                "cooldown": 0.0,
            }
            return

        ring_raw_mean = ring_raw_sum / n_alive
        self._pending_state_bonus = float(self._rc_weight * ring_raw_mean)
        self._state_cache = {
            "dmins": list(rc_dmins),
            "raw_bonus_state": float(ring_raw_mean),
            "cooldown": float(cooldown_sum / n_alive) if n_alive > 0 else 0.0,
        }

    def _episode_metrics_payload(self, steps: int) -> Dict[str, float]:
        fa = self._first_allied_killed_step if self._first_allied_killed_step >= 0 else 0.0
        fe = self._first_enemy_killed_step if self._first_enemy_killed_step >= 0 else 0.0
        return {
            "shaping/first_allied_killed": float(fa),
            "shaping/first_enemy_killed": float(fe),
        }

