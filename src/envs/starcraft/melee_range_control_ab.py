from __future__ import annotations

from .StarCraft2Env import StarCraft2Env
from .utils import (
    RewardShapingEpisodeMetrics,
    clip_shaping_bonus,
    compute_kiting_action_bonus,
    reconcile_terminal_reward,
)


class Starcraft2EnvRewardShaping(StarCraft2Env):
    """Action-based kiting advice with bounded episode-level diagnostics."""

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
        self._pending_action_bonus = 0.0
        self._pending_action_match = None

    def reset(self):
        self._rc_episode.reset()
        self._pending_action_bonus = 0.0
        self._pending_action_match = None
        return super().reset()

    def step(self, actions):
        bonus, cache = compute_kiting_action_bonus(
            self,
            actions,
            weight=self._rc_weight,
            melee_only=self._rc_melee_only,
            melee_range=self._rc_r_melee_def,
            shoot_range=self._rc_r_shoot_def,
        )
        self._pending_action_bonus = float(bonus)
        self._pending_action_match = cache.get("action_match_fraction")

        reward, terminated, info = super().step(actions)
        info = {} if info is None else info
        reconcile_terminal_reward(self, self._rc_episode, reward, terminated, info)
        if terminated and self._log:
            info.update(self._rc_episode.to_info())
        return reward, terminated, info

    def reward_battle(self) -> float:
        base = float(super().reward_battle())
        action_bonus = float(self._pending_action_bonus)
        applied = clip_shaping_bonus(action_bonus, self._max_shaping_abs)
        self._rc_episode.record(
            base=base,
            action=action_bonus,
            total_raw=action_bonus,
            total_applied=applied,
            action_match=self._pending_action_match,
        )
        self._pending_action_bonus = 0.0
        self._pending_action_match = None
        return base + applied
