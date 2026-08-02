from __future__ import annotations

from .StarCraft2Env import StarCraft2Env
from .utils import (
    RewardShapingEpisodeMetrics,
    clip_shaping_bonus,
    compute_ring_bonus_from_state,
    reconcile_terminal_reward,
)


class StateBasedRewardShaping(StarCraft2Env):
    """Signed state-based reward for the ranged-vs-melee distance band."""

    def __init__(
        self,
        *args,
        rc_weight: float = 1.0,
        max_shaping_abs: float = 1.0,
        max_shaping_ratio=None,
        log_shaping: bool = True,
        rc_melee_r_default: float = 3.0,
        rc_shoot_r_default: float = 6.0,
        rc_melee_only: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._rc_weight = float(rc_weight)
        self._max_shaping_abs = float(
            max_shaping_abs if max_shaping_ratio is None else max_shaping_ratio
        )
        self._log = bool(log_shaping)
        self._rc_r_melee_def = float(rc_melee_r_default)
        self._rc_r_shoot_def = float(rc_shoot_r_default)
        self._rc_melee_only = bool(rc_melee_only)
        self._rc_episode = RewardShapingEpisodeMetrics()

    def reset(self):
        self._rc_episode.reset()
        return super().reset()

    def step(self, actions):
        reward, terminated, info = super().step(actions)
        info = {} if info is None else info
        reconcile_terminal_reward(self, self._rc_episode, reward, terminated, info)
        if terminated and self._log:
            info.update(self._rc_episode.to_info())
        return reward, terminated, info

    def reward_battle(self) -> float:
        base = float(super().reward_battle())
        center, half_width = self._distance_band()
        state_bonus, cache = compute_ring_bonus_from_state(
            self,
            self._rc_weight,
            self._rc_melee_only,
            center,
            half_width,
        )
        applied = clip_shaping_bonus(state_bonus, self._max_shaping_abs)
        self._rc_episode.record(
            base=base,
            state=state_bonus,
            total_raw=state_bonus,
            total_applied=applied,
            state_score=cache.get("raw_bonus"),
        )
        return base + applied

    def _distance_band(self):
        center = (self._rc_r_melee_def + self._rc_r_shoot_def) / 2.0
        half_width = max(
            1e-3, (self._rc_r_shoot_def - self._rc_r_melee_def) / 2.0
        )
        return center, half_width
