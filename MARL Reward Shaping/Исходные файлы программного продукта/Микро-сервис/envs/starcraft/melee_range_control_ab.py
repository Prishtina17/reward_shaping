# envs/starcraft/Starcraft2EnvRewardShaping.py
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .utils import (
    ShapingMetrics,
    _is_melee,
    _nearest_enemy,
    update_shaping_metrics,
    ally_damage_step,
    enemy_damage_step,
    count_alive_allies,
    count_alive_enemies,
)

from .StarCraft2Env import StarCraft2Env


class Starcraft2EnvRewardShaping(StarCraft2Env):
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
        rc_melee_only: bool = True,
        **kwargs,
    ):
        kwargs['move_amount'] = 3
        super().__init__(*args, **kwargs)

        # параметры shaping
        self._rc_weight = float(rc_weight)
        self._max_ratio = float(max_shaping_ratio)
        self._log = bool(log_shaping)

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
        self._ep_cooldown_sum = 0.0
        self._ep_cooldown_count = 0

        self._first_allied_killed_step = -1.0
        self._first_enemy_killed_step = -1.0

    # ---------- базовые хуки ----------
    def reset(self):
        self.metrics.reset()
        self._pending_action_bonus = 0.0
        self._shaping_cache.clear()

        self._ep_steps = 0
        self._ep_rc_ring_time_sum = 0.0
        self._ep_rc_oor_time_sum = 0.0
        self._ep_rc_dmin_sum = 0.0
        self._ep_rc_entries = 0
        self._ep_cooldown_sum = 0.0
        self._ep_cooldown_count = 0

        self._first_allied_killed_step = -1.0
        self._first_enemy_killed_step = -1.0
        return super().reset()

    def step(self, actions):
        # 1) позиционный бонус
        self._compute_rc_bonus(actions)

        # 2) шаг среды
        reward, terminated, info = super().step(actions)
        if info is None:
            info = {}

        # 3) логи
        if self._log:
            info.update(self.metrics.to_dict())

        if terminated:
            info.update(self._episode_metrics_payload(max(1, self._ep_steps)))

        return reward, terminated, info

    # ---------- добавляем бонус к базовому вознаграждению ----------
    def reward_battle(self) -> float:
        base = super().reward_battle()

        rc_bonus = float(self._pending_action_bonus)
        cap = self._max_ratio * max(1.0, abs(float(base)))
        if abs(rc_bonus) > cap:
            rc_bonus = float(np.sign(rc_bonus)) * cap

        shaped = float(base) + rc_bonus
        self._ep_steps += 1

        cache = dict(self._shaping_cache) if isinstance(self._shaping_cache, dict) else {}

        # Refresh diagnostics after the SC2 step so counts reflect current state.
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
            delta=float(rc_bonus),
            cache=cache,
        )
        self._shaping_cache = {}
        self._pending_action_bonus = 0.0
        return shaped

    # ---------- основной расчёт позиционного бонуса ----------
    def _compute_rc_bonus(self, actions) -> None:
        actions_int = [int(a) for a in actions]

        cache: Dict[str, object] = {"dmins": [], "raw_bonus": 0.0, "cooldown": 0.0}
        rc_dmins: List[float] = []
        rc_in_ring_count = 0
        rc_oor_count = 0
        score_sum = 0.0
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
            cache["cooldown"] = 0.0
            cache["ally_alive"] = float(count_alive_allies(self))
            cache["enemy_alive"] = float(count_alive_enemies(self))
            cache["ally_dmg"] = 0.0
            self._shaping_cache = cache
            return

        n_alive = 0
        for i, action in enumerate(actions_int):
            ally = self.agents.get(i, None)
            if ally is None or float(getattr(ally, "health", 0.0)) <= 1e-6:
                continue

            n_alive += 1
            dmin, j_near = _nearest_enemy(self, ally, melee_ids)
            rc_dmins.append(dmin)

            target = self.enemies.get(j_near)
            if target is None:
                continue

            delta_x = target.pos.x - ally.pos.x
            delta_y = target.pos.y - ally.pos.y

            if abs(delta_x) > abs(delta_y):
                action_num = 4 if delta_x < 0 else 5
            else:
                action_num = 2 if delta_y < 0 else 3

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

            cooldown_sum += float(getattr(ally, "weapon_cooldown", 0.0))

        if n_alive == 0:
            self._pending_action_bonus = 0.0
            cache["cooldown"] = 0.0
            cache["ally_alive"] = float(count_alive_allies(self))
            cache["enemy_alive"] = float(count_alive_enemies(self))
            cache["ally_dmg"] = 0.0
            self._shaping_cache = cache
            return

        rc_raw_mean = (score_sum / n_alive) * float(self._rc_weight)
        self._pending_action_bonus = float(rc_raw_mean)

        cache["dmins"] = list(rc_dmins)
        cache["raw_bonus"] = float(rc_raw_mean)
        if n_alive > 0:
            cooldown_val = float(cooldown_sum / n_alive)
            cache["cooldown"] = cooldown_val
            self._ep_cooldown_sum += cooldown_val
            self._ep_cooldown_count += 1
        cache["ally_alive"] = float(count_alive_allies(self))
        cache["enemy_alive"] = float(count_alive_enemies(self))
        cache["ally_dmg"] = 0.0
        self._shaping_cache = cache

        self._ep_rc_ring_time_sum += float(rc_in_ring_count / max(1, n_alive))
        self._ep_rc_oor_time_sum += float(rc_oor_count / max(1, n_alive))
        self._ep_rc_dmin_sum += float(np.mean(rc_dmins)) if rc_dmins else 0.0

    def _episode_metrics_payload(self, steps: int) -> Dict[str, float]:
        # Emit first_* only once per episode under shaping/ namespace.
        fa = self._first_allied_killed_step if self._first_allied_killed_step >= 0 else 0.0
        fe = self._first_enemy_killed_step if self._first_enemy_killed_step >= 0 else 0.0
        return {
            "shaping/first_allied_killed": float(fa),
            "shaping/first_enemy_killed": float(fe),
        }
