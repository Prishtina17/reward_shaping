# envs/starcraft/Starcraft2EnvRewardShaping.py
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
    """
    Чисто позиционный reward shaping («бублик») под сценарий: мы — рейндж, враги — melee.
    Бонус максимален в sweet-зоне между мили-радиусом врага и нашим шут-рейнджем.
    Мягкий штраф — за вход в мили и за уход слишком далеко. Доп. мягкий штраф — если
    внутри рабочей зоны агент СБЛИЖАЕТСЯ с ближайшим врагом (по dr = d_t - d_{t-1}).
    """

    def __init__(
        self,
        *args,
        rc_weight: float = 0.7,           # фиксированный вес shaping
        max_shaping_ratio: float = 0.30,
        log_shaping: bool = True,
        rc_pb_gamma = 0.99,

        # геометрия «бублика»
        rc_alpha: float = 10,            # крутизна сигмоид (6..12 обычно ок)
        rc_m_melee: float = 0.10,         # небольшой запас к мили-радиусу врага
        rc_m_shoot: float = 0.10,         # небольшой запас к нашему рейнджу (внутрь)
        rc_ring_gain: float = 4,        # усиление позитивного бонуса в sweet-зоне
        rc_melee_barrier: float = 0.7,    # мягкий штраф внутри мили
        rc_far_cost: float = 0.10,        # мягкий штраф слишком далеко
        # анти-сближение в рабочей зоне:
        rc_close_cost: float = 0.25,      # штраф за dr<0 (приближение) внутри зоны
        rc_close_zone_shrink: float = 0.10, # включать анти-сближение если d < Rs - shrink

        # дефолтные рейнджи
        rc_melee_r_default: float = 3.0,
        rc_shoot_r_default: float = 6.0,

        rc_melee_only: bool = True,       # если False, все враги считаются угрозой
        **kwargs,
    ):
        kwargs['move_amount'] = 3
        super().__init__(*args, **kwargs)

        # параметры shaping
        self._rc_weight = float(rc_weight)
        self._max_ratio = float(max_shaping_ratio)
        self._log = bool(log_shaping)

        # геометрия и штрафы
        self._rc_alpha = float(rc_alpha)
        self._rc_m_melee = float(rc_m_melee)
        self._rc_m_shoot = float(rc_m_shoot)
        self._rc_ring_gain = float(rc_ring_gain)
        self._rc_barrier = float(rc_melee_barrier)
        self._rc_far = float(rc_far_cost)
        self._rc_close_cost = float(rc_close_cost)
        self._rc_close_zone_shrink = float(rc_close_zone_shrink)

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

        # для pay-on-crossing и dr
        self._rc_prev_inside_melee: Dict[int, bool] = {}
        self._rc_prev_dmin: Dict[int, float] = {}

        self.prev_dmin = 0

        self.rc_pb_gamma = rc_pb_gamma
        self.phi_prev = 0
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

        self._rc_prev_inside_melee.clear()
        self._rc_prev_dmin.clear()
        self.phi_prev = 0
        self._shaping_cache.clear()
        self._first_allied_killed_step = -1.0
        self._first_enemy_killed_step = -1.0

        return super().reset()

    def step(self, actions):
        # 1) позиционный бонус

        # 2) шаг среды
        reward, terminated, info = super().step(actions)
        if info is None:
            info = {}

        
        # 3) логи
        if self._log:
            info.update(self.metrics.to_dict())


        if terminated:
            self.phi_prev = 0
            info.update(self._episode_metrics_payload(max(1, self._ep_steps)))
        

        return reward, terminated, info

    # ---------- добавляем бонус к базовому вознаграждению ----------
    def reward_battle(self) -> float:
        base = super().reward_battle()

        phi_curr = self._compute_rc_phi()

        delta = (self.rc_pb_gamma * phi_curr) - self.phi_prev
        shaped_delta = self._rc_weight * delta

        cap = self._max_ratio * max(1.0, abs(float(base)))
        clipped = float(np.clip(shaped_delta, -cap, +cap))
        reward = float(base) + clipped

        self.phi_prev = float(phi_curr)


        
        self._ep_steps += 1

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
            delta=float(clipped),
            cache=cache,
        )
        self._shaping_cache = {}
        return reward

    # ---------- основной расчёт позиционного бонуса ----------
    def _compute_rc_phi(self) -> float:
        rc_raw_sum = 0.0
        rc_ring_scores: List[float] = []
        rc_dmins: List[float] = []
        rc_in_ring_count = 0
        rc_oor_count = 0
        close_penalties: List[float] = []
        ring_raw_sum = 0
        cooldown_sum = 0.0
        cache: Dict[str, object] = {"dmins": [], "raw_bonus": 0.0, "cooldown": 0.0}

        # активные враги (по умолчанию — только melee)
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
            cooldown_sum += float(getattr(ally, "weapon_cooldown", 0.0))


            dmin, j_near = _nearest_enemy(self, ally, melee_ids)
            rc_dmins.append(dmin)
            
            ring_score = ring_function(dmin)

            Rm = self._rc_r_melee_def
            Rs = self._rc_r_shoot_def

            low = Rm 
            high = Rs

            if high < low:  # на всякий случай
                high = low + 1e-3

            
            ring_raw_sum += ring_score
            
            


            '''# «бублик» (гладкий, даёт плотный + внутри [low, high])
            left = _sigmoid(self._rc_alpha * (dmin - low))
            right = _sigmoid(self._rc_alpha * (high - dmin))
            ring01 = left * right                # 0..1
            ring_centered = self._rc_ring_gain * (2.0 * ring01 - 1.0)  # в [-gain, +gain]
            rc_ring_scores.append(ring_centered)

            # время в sweet-зоне и вне рейнджа (для диагностики)
            if (dmin >= low) and (dmin <= high):
                rc_in_ring_count += 1
            if dmin > (Rs + 1e-6):
                rc_oor_count += 1

            # pay-on-crossing (метрика)
            inside_melee = (dmin < Rm)
            prev_inside = self._rc_prev_inside_melee.get(i, False)
            if (not prev_inside) and inside_melee:
                self._ep_rc_entries += 1
            self._rc_prev_inside_melee[i] = inside_melee

            # мягкие барьеры
            barrier = - self._rc_barrier * float(inside_melee)
            far_pen = - self._rc_far * float(dmin > (Rs + self._rc_m_shoot))

            # анти-сближение: если в рабочей зоне (чуть ужатой) и dr<0 → мягкий штраф
            prev_d = self._rc_prev_dmin.get(i, dmin)
            dr = dmin - prev_d  # >0 отдаляемся, <0 приближаемся
            self._rc_prev_dmin[i] = dmin

            close_zone = (dmin < (Rs - self._rc_close_zone_shrink)) and (dmin > Rm)
            width = max(0.5, Rs - Rm)  # нормировка шага
            close_pen = - self._rc_close_cost * max(0.0, -dr) / width if close_zone else 0.0
            close_penalties.append(close_pen)

            rc_raw_sum += (ring_centered + barrier + far_pen + close_pen)'''

        if n_alive == 0:
            cache["ally_alive"] = float(count_alive_allies(self))
            cache["enemy_alive"] = float(count_alive_enemies(self))
            cache["ally_dmg"] = 0.0
            self._shaping_cache = cache
            return 0.0

        ring_raw_mean = ring_raw_sum / n_alive

        last_dmin = rc_dmins[-1] if rc_dmins else self.prev_dmin
        self.prev_dmin = float(last_dmin)

        cache["dmins"] = list(rc_dmins)
        cache["raw_bonus"] = float(ring_raw_mean)
        cache["cooldown"] = float(cooldown_sum / n_alive) if n_alive > 0 else 0.0
        cache["ally_alive"] = float(count_alive_allies(self))
        cache["enemy_alive"] = float(count_alive_enemies(self))
        cache["ally_dmg"] = 0.0
        self._shaping_cache = cache

        return float(ring_raw_mean)


    

    def _episode_metrics_payload(self, steps: int) -> Dict[str, float]:
        fa = self._first_allied_killed_step if self._first_allied_killed_step >= 0 else 0.0
        fe = self._first_enemy_killed_step if self._first_enemy_killed_step >= 0 else 0.0
        return {
            "shaping/first_allied_killed": float(fa),
            "shaping/first_enemy_killed": float(fe),
        }
