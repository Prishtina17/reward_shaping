from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

import math
import numpy as np

# Aggregates helper routines shared across StarCraft shaping environments.


@dataclass
class ShapingMetrics:
    """Lightweight container for per-step shaping diagnostics."""

    base: float = 0.0
    delta: float = 0.0
    ratio_abs: float = 0.0
    raw_bonus: float = 0.0
    dmin_mean: float = 0.0
    dmin_last: float = 0.0
    cooldown: float = 0.0
    ally_alive: float = 0.0
    ally_dmg: float = 0.0
    enemy_alive: float = 0.0
    first_allied_killed: float = 0.0
    first_enemy_killed: float = 0.0

    def reset(self) -> None:
        self.base = 0.0
        self.delta = 0.0
        self.ratio_abs = 0.0
        self.raw_bonus = 0.0
        self.dmin_mean = 0.0
        self.dmin_last = 0.0
        self.cooldown = 0.0
        self.ally_alive = 0.0
        self.ally_dmg = 0.0
        self.enemy_alive = 0.0
        self.first_allied_killed = 0.0
        self.first_enemy_killed = 0.0

    def to_dict(self) -> Dict[str, float]:
        payload = {
            "shaping/base": self.base,
            "shaping/delta": self.delta,
            "shaping/ratio_abs": self.ratio_abs,
            "shaping/raw_bonus": self.raw_bonus,
            "shaping/dmin_mean": self.dmin_mean,
            "shaping/cooldown": self.cooldown,
            "shaping/ally_alive": self.ally_alive,
            "shaping/ally_dmg": self.ally_dmg,
            "shaping/enemy_alive": self.enemy_alive,
            "shaping/first_allied_killed": self.first_allied_killed,
            "shaping/first_enemy_killed": self.first_enemy_killed,
        }
        return payload


def update_shaping_metrics(
    metrics: ShapingMetrics,
    *,
    base: float,
    delta: float,
    cache: Optional[Dict[str, Any]] = None,
) -> None:
    """Populate diagnostic metrics used by StarCraft shaping environments."""
    metrics.base = float(base)
    metrics.delta = float(delta)

    denom = max(1.0, abs(float(base)))
    metrics.ratio_abs = float(abs(float(delta)) / denom) if denom > 1e-9 else 0.0

    cache = cache or {}

    raw_bonus = cache.get("raw_bonus", 0.0)
    metrics.raw_bonus = float(raw_bonus)

    dmins = cache.get("dmins", None)
    if isinstance(dmins, (list, tuple, np.ndarray)) and len(dmins) > 0:
        finite_vals = [float(d) for d in dmins if np.isfinite(d)]
        if finite_vals:
            metrics.dmin_mean = float(np.mean(finite_vals))
            metrics.dmin_last = float(finite_vals[-1])
        else:
            metrics.dmin_mean = 0.0
            metrics.dmin_last = 0.0
    else:
        metrics.dmin_mean = 0.0
        metrics.dmin_last = 0.0

    cooldown = cache.get("cooldown", None)
    metrics.cooldown = float(cooldown) if cooldown is not None else 0.0

    ally_alive = cache.get("ally_alive")
    metrics.ally_alive = float(ally_alive) if ally_alive is not None else 0.0

    ally_dmg = cache.get("ally_dmg")
    if ally_dmg is not None:
        metrics.ally_dmg += float(ally_dmg)

    enemy_alive = cache.get("enemy_alive")
    metrics.enemy_alive = float(enemy_alive) if enemy_alive is not None else 0.0

    first_ally = cache.get("first_allied_killed")
    try:
        fa = float(first_ally) if first_ally is not None else 0.0
    except Exception:
        fa = 0.0
    metrics.first_allied_killed = fa if fa >= 0.0 else 0.0

    first_enemy = cache.get("first_enemy_killed")
    try:
        fe = float(first_enemy) if first_enemy is not None else 0.0
    except Exception:
        fe = 0.0
    metrics.first_enemy_killed = fe if fe >= 0.0 else 0.0

def extract_attack_targets(env: Any, actions) -> Tuple[List[int], List[bool]]:
    """Return targets chosen by agents and eligibility flags for focus-fire shaping."""
    targets: List[int] = []
    eligible: List[bool] = []
    n_agents = getattr(env, "n_agents", len(getattr(env, "agents", {})))

    for i in range(n_agents):
        if isinstance(actions, (list, tuple, np.ndarray)):
            act = actions[i]
        elif hasattr(actions, "get"):
            act = actions.get(i, 0)
        else:
            try:
                act = actions[i]
            except Exception:
                act = 0
        try:
            avail = env.get_avail_agent_actions(i)
            n_actions = int(len(avail))
            n_enemies = int(getattr(env, "n_enemies", 0))
            attack_start = n_actions - n_enemies
            if isinstance(act, (int, np.integer)) and 0 <= act < n_actions:
                if act >= attack_start:
                    local_enemy_idx = int(act - attack_start)
                    enemy_id = local_enemy_idx
                    enemy = env.enemies.get(enemy_id, None)
                    if enemy is not None and (enemy.health + enemy.shield) > 1e-6:
                        targets.append(enemy_id)
                        eligible.append(True)
                    else:
                        targets.append(-1)
                        eligible.append(False)
                else:
                    targets.append(-1)
                    eligible.append(False)
            else:
                targets.append(-1)
                eligible.append(False)
        except Exception:
            targets.append(-1)
            eligible.append(False)

    return targets, eligible


def enemy_damage_step(env: Any) -> Tuple[np.ndarray, int]:
    """Calculate per-enemy damage and kill count for the current step."""
    prev = getattr(env, "previous_enemy_units", None)
    if prev is None or len(prev) == 0:
        return np.zeros(env.n_enemies, dtype=np.float32), 0

    dmg = np.zeros(env.n_enemies, dtype=np.float32)
    kills = 0
    for j in range(env.n_enemies):
        cur = env.enemies.get(j, None)
        prv = prev.get(j, None)
        if cur is None or prv is None:
            continue
        cur_hp = float(cur.health) + float(cur.shield)
        prv_hp = float(prv.health) + float(prv.shield)
        dealt = max(0.0, prv_hp - cur_hp)
        if dealt > 1e-6:
            dmg[j] = dealt
        if prv_hp > 1e-6 and cur_hp <= 1e-6:
            kills += 1
    return dmg, kills


def ally_damage_step(env: Any) -> Tuple[np.ndarray, int]:
    """Calculate per-ally damage and deaths for the current step."""
    prev = getattr(env, "previous_ally_units", None)
    n_agents = getattr(env, "n_agents", len(getattr(env, "agents", {})))
    if prev is None or n_agents == 0:
        return np.zeros(n_agents, dtype=np.float32), 0

    dmg = np.zeros(n_agents, dtype=np.float32)
    deaths = 0
    for i in range(n_agents):
        cur = env.agents.get(i, None)
        prv = prev.get(i, None)
        if cur is None or prv is None:
            continue
        cur_hp = float(getattr(cur, "health", 0.0)) + float(getattr(cur, "shield", 0.0))
        prv_hp = float(getattr(prv, "health", 0.0)) + float(getattr(prv, "shield", 0.0))
        taken = max(0.0, prv_hp - cur_hp)
        if taken > 1e-6:
            dmg[i] = taken
        if prv_hp > 1e-6 and cur_hp <= 1e-6:
            deaths += 1
    return dmg, deaths


def alive_enemy_ehp_vec(env: Any) -> Tuple[List[int], np.ndarray]:
    """Return ids of alive enemies and their effective HP."""
    alive_ids: List[int] = []
    ehp: List[float] = []
    for j in range(env.n_enemies):
        enemy = env.enemies.get(j, None)
        if enemy is None:
            continue
        hp = float(enemy.health) + float(enemy.shield)
        if hp > 1e-6:
            alive_ids.append(j)
            ehp.append(hp)
    if len(ehp) == 0:
        return [], np.zeros((0,), dtype=np.float32)
    return alive_ids, np.asarray(ehp, dtype=np.float32)


def count_alive_enemies(env: Any) -> int:
    """Count how many enemies are still alive."""
    c = 0
    for j in range(env.n_enemies):
        enemy = env.enemies.get(j, None)
        if enemy is None:
            continue
        if (float(enemy.health) + float(enemy.shield)) > 1e-6:
            c += 1
    return c

def count_alive_allies(self) -> int:
    n_alive = 0
    for i, ally in self.agents.items():
        if float(getattr(ally, "health", 0.0)) <= 1e-6:
            continue
        n_alive += 1
    return n_alive

def weapon_idling_mean(self) -> float:
    """Mean fraction of alive, weapon-capable allies whose weapons are ready
    (cooldown == 0) but they did not select an attack action this step.

    - Excludes non-combat units (e.g., Medivac on MMM).
    - Counts only allies that have at least one attack action available.
    """
    try:
        n_actions_no_attack = int(getattr(self, "n_actions_no_attack", 0))
    except Exception:
        n_actions_no_attack = 0

    last_action = getattr(self, "last_action", None)
    idle = 0
    denom = 0

    for i, ally in getattr(self, "agents", {}).items():
        # alive
        hp = float(getattr(ally, "health", 0.0)) + float(getattr(ally, "shield", 0.0))
        if hp <= 1e-6:
            continue
        # exclude medivac on MMM (healer, no weapon)
        if getattr(self, "map_type", None) == "MMM" and getattr(ally, "unit_type", None) == getattr(self, "medivac_id", None):
            continue

        # can attack now (any attack action available)
        can_attack = True
        try:
            avail = self.get_avail_agent_actions(i)
            if isinstance(avail, (list, tuple, np.ndarray)) and len(avail) > n_actions_no_attack:
                can_attack = (np.sum(avail[n_actions_no_attack:]) > 0)
        except Exception:
            pass
        if not can_attack:
            continue

        denom += 1

        cd = float(getattr(ally, "weapon_cooldown", 0.0))
        ready = cd <= 1e-6

        attacked = False
        try:
            if last_action is not None:
                a_idx = int(np.argmax(last_action[i]))
                attacked = (a_idx >= n_actions_no_attack)
        except Exception:
            attacked = False

        if ready and not attacked:
            idle += 1

    if denom == 0:
        return 0.0
    return float(idle) / float(denom)

def ring_function(
    d: float,
    center: float = 5.25,
    half_width: float = 0.75,
    slope: float = 0.15,
) -> float:
    """
    Бублик с плато и квадратичным спадом.
    По умолчанию зона [center - half_width, center + half_width].
    Для melee/shoot можно задать center=(Rm+Rs)/2, half_width=(Rs-Rm)/2.

    :param d: расстояние до врага
    :param center: центр плато (середина sweet spot)
    :param half_width: половина ширины плато
    :param slope: скорость квадратичного спада
    :return: скор ∈ [-1, 1]
    """
    if half_width <= 0:
        half_width = 1e-3
    if abs(d - center) <= half_width:
        return 1.0
    dist = abs(d - center) - half_width
    val = 1.0 - slope * (dist ** 1.5)
    return max(-1.0, val)


def compute_ring_bonus_from_state(
    env: Any,
    rc_weight: float,
    rc_melee_only: bool,
    center: float,
    half_width: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Вычисляет позиционный бонус (ring) по текущему состоянию окружения.
    Возвращает (weighted_bonus, cache) для использования в reward_battle и update_shaping_metrics.
    """
    cache: Dict[str, Any] = {"dmins": [], "raw_bonus": 0.0, "cooldown": 0.0}
    melee_ids: List[int] = []
    for j in range(env.n_enemies):
        e = env.enemies.get(j, None)
        if e is None:
            continue
        if (float(getattr(e, "health", 0.0)) + float(getattr(e, "shield", 0.0))) <= 1e-6:
            continue
        if (not rc_melee_only) or _is_melee(getattr(e, "unit_type", 0)):
            melee_ids.append(j)
    if len(melee_ids) == 0:
        cache["ally_alive"] = float(count_alive_allies(env))
        cache["enemy_alive"] = float(count_alive_enemies(env))
        cache["ally_dmg"] = 0.0
        return 0.0, cache
    rc_dmins: List[float] = []
    ring_raw_sum = 0.0
    cooldown_sum = 0.0
    n_alive = 0
    for i, ally in env.agents.items():
        if float(getattr(ally, "health", 0.0)) + float(getattr(ally, "shield", 0.0)) <= 1e-6:
            continue
        n_alive += 1
        cooldown_sum += float(getattr(ally, "weapon_cooldown", 0.0))
        dmin, _ = _nearest_enemy(env, ally, melee_ids)
        rc_dmins.append(dmin)
        ring_raw_sum += ring_function(dmin, center=center, half_width=half_width)
    if n_alive == 0:
        cache["ally_alive"] = float(count_alive_allies(env))
        cache["enemy_alive"] = float(count_alive_enemies(env))
        cache["ally_dmg"] = 0.0
        return 0.0, cache
    ring_raw_mean = ring_raw_sum / n_alive
    weighted = float(rc_weight) * float(ring_raw_mean)
    cache["dmins"] = list(rc_dmins)
    cache["raw_bonus"] = float(ring_raw_mean)
    cache["cooldown"] = float(cooldown_sum / n_alive)
    cache["ally_alive"] = float(count_alive_allies(env))
    cache["enemy_alive"] = float(count_alive_enemies(env))
    cache["ally_dmg"] = 0.0
    return weighted, cache


# ---------- утилиты ----------
def _unit_xy(u) -> Optional[Tuple[float, float]]:
    p = getattr(u, "pos", None)
    if p is not None:
        try:
            return float(getattr(p, "x")), float(getattr(p, "y"))
        except Exception:
            pass
    for a, b in (("pos_x", "pos_y"), ("x", "y")):
        try:
            return float(getattr(u, a)), float(getattr(u, b))
        except Exception:
            continue
    return None

def _dist_units(a, b) -> float:
    pa, pb = _unit_xy(a), _unit_xy(b)
    if (pa is None) or (pb is None):
        return 1e9
    dx, dy = pa[0] - pb[0], pa[1] - pb[1]
    return float(math.hypot(dx, dy))

def _nearest_enemy(env: Any, ally, enemy_ids: List[int]) -> Tuple[float, Optional[int]]:
    dmin, idx = 1e9, None
    for j in enemy_ids:
        e = env.enemies.get(j, None)
        if e is None:
            continue
        if (float(getattr(e, "health", 0.0)) + float(getattr(e, "shield", 0.0))) <= 1e-6:
            continue
        d = _dist_units(ally, e)
        if d < dmin:
            dmin, idx = d, j
    return dmin, idx

def _is_melee(unit_type: int) -> bool:
    return True

def dmin_mean(self) -> float:
    """Average min distance from each alive ally to the nearest alive enemy.
    Safe across env variants (does not assume melee-only flags).
    """
    alive_enemy_ids = []
    for j in range(getattr(self, "n_enemies", 0)):
        e = self.enemies.get(j, None)
        if e is None:
            continue
        if (float(getattr(e, "health", 0.0)) + float(getattr(e, "shield", 0.0))) <= 1e-6:
            continue
        alive_enemy_ids.append(j)

    if len(alive_enemy_ids) == 0:
        return 0.0

    rc_dmins = []
    for i, ally in getattr(self, "agents", {}).items():
        hp = float(getattr(ally, "health", 0.0)) + float(getattr(ally, "shield", 0.0))
        if hp <= 1e-6:
            continue
        dmin, _ = _nearest_enemy(self, ally, alive_enemy_ids)
        if np.isfinite(dmin):
            rc_dmins.append(dmin)
    return float(np.mean(rc_dmins)) if rc_dmins else 0.0
