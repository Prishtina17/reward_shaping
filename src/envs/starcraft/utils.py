from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

import math
import numpy as np

# Aggregates helper routines shared across StarCraft shaping environments.


@dataclass
class RewardShapingEpisodeMetrics:
    """Episode-level diagnostics for decomposed reward shaping components.

    PyMARL2 only aggregates ``info`` from terminal transitions.  Keeping the
    accumulator in the environment makes every exported value describe the
    complete episode rather than the final transition.
    """

    steps: int = 0
    base_return: float = 0.0
    base_abs_sum: float = 0.0
    action_return: float = 0.0
    state_return: float = 0.0
    potential_return: float = 0.0
    total_raw_return: float = 0.0
    total_applied_return: float = 0.0
    total_applied_abs_sum: float = 0.0
    negative_steps: int = 0
    clipped_steps: int = 0
    action_match_sum: float = 0.0
    action_match_count: int = 0
    state_score_sum: float = 0.0
    state_score_count: int = 0
    potential_value_sum: float = 0.0
    potential_value_count: int = 0
    _last_base: float = 0.0
    _last_raw: float = 0.0
    _last_applied: float = 0.0

    def reset(self) -> None:
        self.steps = 0
        self.base_return = 0.0
        self.base_abs_sum = 0.0
        self.action_return = 0.0
        self.state_return = 0.0
        self.potential_return = 0.0
        self.total_raw_return = 0.0
        self.total_applied_return = 0.0
        self.total_applied_abs_sum = 0.0
        self.negative_steps = 0
        self.clipped_steps = 0
        self.action_match_sum = 0.0
        self.action_match_count = 0
        self.state_score_sum = 0.0
        self.state_score_count = 0
        self.potential_value_sum = 0.0
        self.potential_value_count = 0
        self._last_base = 0.0
        self._last_raw = 0.0
        self._last_applied = 0.0

    def record(
        self,
        *,
        base: float,
        action: float = 0.0,
        state: float = 0.0,
        potential: float = 0.0,
        total_raw: Optional[float] = None,
        total_applied: Optional[float] = None,
        action_match: Optional[float] = None,
        state_score: Optional[float] = None,
        potential_value: Optional[float] = None,
    ) -> None:
        action = float(action)
        state = float(state)
        potential = float(potential)
        raw = action + state + potential if total_raw is None else float(total_raw)
        applied = raw if total_applied is None else float(total_applied)
        base = float(base)

        self.steps += 1
        self.base_return += base
        self.base_abs_sum += abs(base)
        self.action_return += action
        self.state_return += state
        self.potential_return += potential
        self.total_raw_return += raw
        self.total_applied_return += applied
        self.total_applied_abs_sum += abs(applied)
        self.negative_steps += int(applied < -1e-12)
        self.clipped_steps += int(abs(applied - raw) > 1e-12)
        self._last_base = base
        self._last_raw = raw
        self._last_applied = applied

        if action_match is not None:
            self.action_match_sum += float(action_match)
            self.action_match_count += 1
        if state_score is not None:
            self.state_score_sum += float(state_score)
            self.state_score_count += 1
        if potential_value is not None:
            self.potential_value_sum += float(potential_value)
            self.potential_value_count += 1

    @property
    def last_applied(self) -> float:
        return float(self._last_applied)

    def reconcile_last_transition(self, *, base: float, applied: float) -> None:
        """Replace the last transition totals after terminal reward handling.

        ``StarCraft2Env.step`` adds win/defeat rewards and applies reward
        scaling after ``reward_battle`` returns.  This reconciliation keeps
        the episode decomposition aligned with the reward actually returned
        to the learner while retaining unscaled units in the diagnostics.
        """
        if self.steps <= 0:
            return

        base = float(base)
        applied = float(applied)
        self.base_return += base - self._last_base
        self.base_abs_sum += abs(base) - abs(self._last_base)
        self.total_applied_return += applied - self._last_applied
        self.total_applied_abs_sum += abs(applied) - abs(self._last_applied)
        self.negative_steps += int(applied < -1e-12) - int(
            self._last_applied < -1e-12
        )
        self.clipped_steps += int(abs(applied - self._last_raw) > 1e-12) - int(
            abs(self._last_applied - self._last_raw) > 1e-12
        )
        self._last_base = base
        self._last_applied = applied

    def to_info(self) -> Dict[str, float]:
        steps = max(1, self.steps)
        return {
            "rc/episode_steps": float(self.steps),
            "rc/base_return": float(self.base_return),
            "rc/action_return": float(self.action_return),
            "rc/state_return": float(self.state_return),
            "rc/potential_return": float(self.potential_return),
            "rc/total_raw_return": float(self.total_raw_return),
            "rc/total_applied_return": float(self.total_applied_return),
            "rc/action_per_step": float(self.action_return / steps),
            "rc/state_per_step": float(self.state_return / steps),
            "rc/potential_per_step": float(self.potential_return / steps),
            "rc/total_applied_per_step": float(self.total_applied_return / steps),
            "rc/applied_abs_per_step": float(self.total_applied_abs_sum / steps),
            "rc/negative_fraction": float(self.negative_steps / steps),
            "rc/clipped_fraction": float(self.clipped_steps / steps),
            "rc/applied_abs_to_base_abs_ratio": float(
                self.total_applied_abs_sum / max(1.0, self.base_abs_sum)
            ),
            "rc/action_match_fraction": float(
                self.action_match_sum / max(1, self.action_match_count)
            ),
            "rc/state_score_per_step": float(
                self.state_score_sum / max(1, self.state_score_count)
            ),
            "rc/potential_value_per_step": float(
                self.potential_value_sum / max(1, self.potential_value_count)
            ),
        }


def clip_shaping_bonus(value: float, max_abs: float) -> float:
    """Apply a fixed per-transition cap independent of sparse base rewards."""
    max_abs = max(0.0, float(max_abs))
    return float(np.clip(float(value), -max_abs, max_abs))


def potential_shaping_bonus(
    phi_prev: float,
    phi_curr: float,
    gamma: float,
    weight: float = 1.0,
) -> float:
    """Return the policy-invariant potential difference in reward units."""
    return float(weight) * (
        float(gamma) * float(phi_curr) - float(phi_prev)
    )


def reconcile_terminal_reward(
    env: Any,
    metrics: RewardShapingEpisodeMetrics,
    reward: float,
    terminated: bool,
    info: Optional[Dict[str, Any]],
) -> None:
    """Reconcile episode metrics with terminal outcome and reward scaling."""
    if not terminated or metrics.steps <= 0:
        return

    scale = 1.0
    if bool(getattr(env, "reward_scale", False)):
        scale = float(env.max_reward) / float(env.reward_scale_rate)
    unscaled_reward = float(reward) * scale

    info = info or {}
    battle_over = bool(info.get("battle_won", False)) or (
        count_alive_allies(env) == 0 or count_alive_enemies(env) == 0
    )
    applied = metrics.last_applied
    if bool(getattr(env, "reward_sparse", False)) and battle_over:
        # Sparse terminal rewards replace, rather than augment, reward_battle.
        applied = 0.0
    metrics.reconcile_last_transition(
        base=unscaled_reward - applied,
        applied=applied,
    )

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
    slope: float = 1.0,
) -> float:
    """Smooth signed score for staying between melee and shooting ranges.

    The score is positive strictly inside the band, zero on its boundaries,
    and negative outside it.  The bounded tanh product avoids the previous
    behaviour where states deep inside melee range still received a bonus.
    """
    half_width = max(1e-3, float(half_width))
    slope = max(1e-6, float(slope))
    lower = float(center) - half_width
    upper = float(center) + half_width
    return float(
        math.tanh(slope * (float(d) - lower))
        * math.tanh(slope * (upper - float(d)))
    )


def compute_ring_bonus_from_state(
    env: Any,
    rc_weight: float,
    rc_melee_only: bool,
    center: float,
    half_width: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Вычисляет позиционный бонус (ring) по текущему состоянию окружения.
    Возвращает (weighted_bonus, cache) для reward_battle и episode-метрик.
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

MELEE_UNIT_TYPE_IDS = frozenset(
    {
        9,    # Baneling
        73,   # Zealot
        105,  # Zergling
    }
)


def _is_melee(unit_type: int) -> bool:
    """Return whether a raw SC2 unit type is melee in the supported SMAC set."""
    try:
        return int(unit_type) in MELEE_UNIT_TYPE_IDS
    except (TypeError, ValueError):
        return False


def compute_kiting_action_bonus(
    env: Any,
    actions,
    *,
    weight: float,
    melee_only: bool,
    melee_range: float,
    shoot_range: float,
) -> Tuple[float, Dict[str, Any]]:
    """Score action advice for a ranged-vs-melee kiting transition.

    The advice is deliberately simple and deterministic:
    move away below melee range, attack inside shooting range when the weapon
    is ready, otherwise kite away, and close the distance when out of range.
    """
    actions_int = [int(action) for action in actions]
    cache: Dict[str, Any] = {
        "dmins": [],
        "raw_bonus": 0.0,
        "cooldown": 0.0,
    }

    enemy_ids: List[int] = []
    for enemy_id in range(getattr(env, "n_enemies", 0)):
        enemy = env.enemies.get(enemy_id, None)
        if enemy is None:
            continue
        hp = float(getattr(enemy, "health", 0.0)) + float(
            getattr(enemy, "shield", 0.0)
        )
        if hp <= 1e-6:
            continue
        if (not melee_only) or _is_melee(getattr(enemy, "unit_type", None)):
            enemy_ids.append(enemy_id)

    if not enemy_ids:
        cache["ally_alive"] = float(count_alive_allies(env))
        cache["enemy_alive"] = float(count_alive_enemies(env))
        cache["ally_dmg"] = 0.0
        return 0.0, cache

    score_sum = 0.0
    cooldown_sum = 0.0
    alive = 0
    n_actions_no_attack = int(getattr(env, "n_actions_no_attack", 6))

    for agent_id, action in enumerate(actions_int):
        ally = env.agents.get(agent_id, None)
        if ally is None:
            continue
        ally_hp = float(getattr(ally, "health", 0.0)) + float(
            getattr(ally, "shield", 0.0)
        )
        if ally_hp <= 1e-6:
            continue

        distance, nearest_id = _nearest_enemy(env, ally, enemy_ids)
        target = env.enemies.get(nearest_id, None)
        if target is None:
            continue

        alive += 1
        cache["dmins"].append(float(distance))
        cooldown = float(getattr(ally, "weapon_cooldown", 0.0))
        cooldown_sum += cooldown

        dx = float(target.pos.x) - float(ally.pos.x)
        dy = float(target.pos.y) - float(ally.pos.y)
        if abs(dx) > abs(dy):
            toward_action = 4 if dx > 0 else 5
            away_action = 5 if dx > 0 else 4
        else:
            toward_action = 2 if dy > 0 else 3
            away_action = 3 if dy > 0 else 2

        if distance < float(melee_range):
            desired = action == away_action
        elif distance <= float(shoot_range):
            desired = (
                action == away_action
                if cooldown > 1e-6
                else action >= n_actions_no_attack
            )
        else:
            desired = action == toward_action

        score_sum += float(desired)

    if alive == 0:
        cache["ally_alive"] = float(count_alive_allies(env))
        cache["enemy_alive"] = float(count_alive_enemies(env))
        cache["ally_dmg"] = 0.0
        return 0.0, cache

    raw_mean = score_sum / float(alive)
    weighted_bonus = float(weight) * raw_mean
    cache["raw_bonus"] = raw_mean
    cache["action_match_fraction"] = raw_mean
    cache["cooldown"] = cooldown_sum / float(alive)
    cache["ally_alive"] = float(count_alive_allies(env))
    cache["enemy_alive"] = float(count_alive_enemies(env))
    cache["ally_dmg"] = 0.0
    return weighted_bonus, cache

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
