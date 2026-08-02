from __future__ import annotations

from .StarCraft2Env import StarCraft2Env
from .utils import (
    RewardShapingEpisodeMetrics,
    clip_shaping_bonus,
    compute_ring_bonus_from_state,
    potential_shaping_bonus,
    reconcile_terminal_reward,
)


class Starcraft2EnvRewardShaping(StarCraft2Env):
    """Policy-invariant potential-based shaping for distance control."""

    def __init__(
        self,
        *args,
        rc_weight: float = 1.0,
        max_shaping_abs: float = 1.0,
        max_shaping_ratio=None,
        log_shaping: bool = True,
        rc_pb_gamma: float = 0.99,
        rc_melee_r_default: float = 3.0,
        rc_shoot_r_default: float = 6.0,
        rc_melee_only: bool = True,
        clip_potential_shaping: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._rc_weight = float(rc_weight)
        self._max_shaping_abs = float(
            max_shaping_abs if max_shaping_ratio is None else max_shaping_ratio
        )
        self._log = bool(log_shaping)
        self.rc_pb_gamma = float(rc_pb_gamma)
        self._rc_r_melee_def = float(rc_melee_r_default)
        self._rc_r_shoot_def = float(rc_shoot_r_default)
        self._rc_melee_only = bool(rc_melee_only)
        self._clip_potential_shaping = bool(clip_potential_shaping)

        self._rc_episode = RewardShapingEpisodeMetrics()
        self.phi_prev = 0.0

    def reset(self):
        self._rc_episode.reset()
        result = super().reset()
        self.phi_prev = self._potential(terminal=False)[0]
        return result

    def step(self, actions):
        reward, terminated, info = super().step(actions)
        info = {} if info is None else info
        reconcile_terminal_reward(self, self._rc_episode, reward, terminated, info)
        if terminated:
            self.phi_prev = 0.0
            if self._log:
                info.update(self._rc_episode.to_info())
        return reward, terminated, info

    def reward_battle(self) -> float:
        base = float(super().reward_battle())
        terminal = self._episode_steps >= self.episode_limit
        phi_curr, _ = self._potential(terminal=terminal)
        potential_bonus = potential_shaping_bonus(
            self.phi_prev,
            phi_curr,
            self.rc_pb_gamma,
            self._rc_weight,
        )
        applied = float(potential_bonus)
        if self._clip_potential_shaping:
            applied = clip_shaping_bonus(applied, self._max_shaping_abs)

        self._rc_episode.record(
            base=base,
            potential=potential_bonus,
            total_raw=potential_bonus,
            total_applied=applied,
            potential_value=phi_curr,
        )
        self.phi_prev = float(phi_curr)
        return base + applied

    def _potential(self, *, terminal: bool):
        if terminal:
            return 0.0, {}
        center = (self._rc_r_melee_def + self._rc_r_shoot_def) / 2.0
        half_width = max(
            1e-3, (self._rc_r_shoot_def - self._rc_r_melee_def) / 2.0
        )
        phi, cache = compute_ring_bonus_from_state(
            self,
            1.0,
            self._rc_melee_only,
            center,
            half_width,
        )
        return float(phi), cache
