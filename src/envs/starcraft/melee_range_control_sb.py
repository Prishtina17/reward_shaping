# envs/starcraft/Starcraft2EnvRewardShaping.py
from __future__ import annotations
from typing import Dict, List

import numpy as np

from .utils import (
    ShapingMetrics,
    compute_ring_bonus_from_state,
    update_shaping_metrics,
    ally_damage_step,
    enemy_damage_step,
    count_alive_allies,
    count_alive_enemies,
)

from .StarCraft2Env import StarCraft2Env


class StateBasedRewardShaping(StarCraft2Env):
    """
    Чисто позиционный reward shaping («бублик») под сценарий: мы — рейндж, враги — melee.
    Бонус максимален в sweet-зоне между мили-радиусом врага и нашим шут-рейнджем.
    Мягкий штраф — за вход в мили и за уход слишком далеко. Доп. мягкий штраф — если
    внутри рабочей зоны агент СБЛИЖАЕТСЯ с ближайшим врагом (по dr = d_t - d_{t-1}).
    """

    def __init__(
        self,
        *args,
        rc_weight: float = 0.7,
        max_shaping_ratio: float = 0.30,
        log_shaping: bool = True,
        rc_melee_r_default: float = 3.0,
        rc_shoot_r_default: float = 6.0,
        rc_melee_only: bool = True,
        **kwargs,
    ):
        kwargs['move_amount'] = 3
        super().__init__(*args, **kwargs)

        # параметры shaping
        self._rc_weight = float(rc_weight)
        self._max_ratio = float(max_shaping_ratio)
        self._log = bool(log_shaping)

        # дефолтные рейнджи
        self._rc_r_melee_def = float(rc_melee_r_default)
        self._rc_r_shoot_def = float(rc_shoot_r_default)

        self._rc_melee_only = bool(rc_melee_only)

        # метрики и временные хранилища
        self.metrics = ShapingMetrics()
        self._pending_action_bonus: float = 0.0
        self._shaping_cache: Dict[str, object] = {}

        # эпизодные агрегаты
        self._ep_steps = 0
        self._ep_rc_ring_time_sum = 0.0
        self._ep_rc_oor_time_sum = 0.0
        self._ep_rc_dmin_sum = 0.0
        self._ep_rc_entries = 0

        self._first_allied_killed_step = -1.0
        self._first_enemy_killed_step = -1.0

    # ---------- базовые хуки ----------
    def reset(self):
        self.metrics.reset()
        self._pending_action_bonus = 0.0

        self._ep_steps = 0
        self._ep_rc_ring_time_sum = 0.0
        self._ep_rc_oor_time_sum = 0.0
        self._ep_rc_dmin_sum = 0.0
        self._ep_rc_entries = 0

        self._shaping_cache.clear()
        self._first_allied_killed_step = -1.0
        self._first_enemy_killed_step = -1.0
        return super().reset()

    def step(self, actions):
        reward, terminated, info = super().step(actions)
        if info is None:
            info = {}
        if self._log:
            info.update(self.metrics.to_dict())
        if terminated:
            info.update(self._episode_metrics_payload(max(1, self._ep_steps)))
        return reward, terminated, info

    def reward_battle(self) -> float:
        base = super().reward_battle()
        center = (self._rc_r_melee_def + self._rc_r_shoot_def) / 2.0
        half_width = max(1e-3, (self._rc_r_shoot_def - self._rc_r_melee_def) / 2.0)
        rc_bonus, cache = compute_ring_bonus_from_state(
            self, self._rc_weight, self._rc_melee_only, center, half_width
        )
        cap = self._max_ratio * max(1.0, abs(float(base)))
        if abs(rc_bonus) > cap:
            rc_bonus = float(np.sign(rc_bonus)) * cap
        shaped = float(base) + float(rc_bonus)
        self._ep_steps += 1

        cache["ally_alive"] = float(count_alive_allies(self))
        cache["enemy_alive"] = float(count_alive_enemies(self))
        dmg_to_enemies, enemy_kills = enemy_damage_step(self)
        cache["ally_dmg"] = float(np.sum(dmg_to_enemies)) if getattr(dmg_to_enemies, "size", 0) > 0 else 0.0
        _, ally_deaths = ally_damage_step(self)
        if ally_deaths > 0 and self._first_allied_killed_step < 0:
            self._first_allied_killed_step = float(self._episode_steps)
        if enemy_kills > 0 and self._first_enemy_killed_step < 0:
            self._first_enemy_killed_step = float(self._episode_steps)
        cache["first_allied_killed"] = self._first_allied_killed_step if self._first_allied_killed_step >= 0 else -1.0
        cache["first_enemy_killed"] = self._first_enemy_killed_step if self._first_enemy_killed_step >= 0 else -1.0
        update_shaping_metrics(self.metrics, base=float(base), delta=float(rc_bonus), cache=cache)
        return shaped

    def _episode_metrics_payload(self, steps: int) -> Dict[str, float]:
        fa = self._first_allied_killed_step if self._first_allied_killed_step >= 0 else 0.0
        fe = self._first_enemy_killed_step if self._first_enemy_killed_step >= 0 else 0.0
        return {
            "shaping/first_allied_killed": float(fa),
            "shaping/first_enemy_killed": float(fe),
        }
