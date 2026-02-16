from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env
from .starcraft.melee_range_control_sp import (
    Starcraft2EnvRewardShaping as MeleeRangeControlSPEnv,
)
from .starcraft.melee_range_control_sb import (
    StateBasedRewardShaping as MeleeRangeControlSBEnv,
)
from .starcraft.melee_range_control_ap import (
    Starcraft2EnvRewardShaping as MeleeRangeControlAPEnv,
)
from .starcraft.melee_range_control_ab import (
    Starcraft2EnvRewardShaping as MeleeRangeControlABEnv,
)
from .starcraft.melee_range_control_pb import (
    Starcraft2EnvRewardShaping as MeleeRangeControlPBEnv,
)
from .starcraft.melee_range_control_asp import (
    Starcraft2EnvRewardShaping as MeleeRangeControlASPEnv,
)
from .starcraft.melee_range_control_as import (
    Starcraft2EnvRewardShaping as MeleeRangeControlASEnv,
)
from .matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except Exception as e:
    gfootball = False
    print(e)

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["melee_range_control_sp"] = partial(env_fn, env=MeleeRangeControlSPEnv)
REGISTRY["melee_range_control_sb"] = partial(env_fn, env=MeleeRangeControlSBEnv)
REGISTRY["melee_range_control_ap"] = partial(env_fn, env=MeleeRangeControlAPEnv)
REGISTRY["melee_range_control_ab"] = partial(env_fn, env=MeleeRangeControlABEnv)
REGISTRY["melee_range_control_pb"] = partial(env_fn, env=MeleeRangeControlPBEnv)
REGISTRY["melee_range_control_asp"] = partial(env_fn, env=MeleeRangeControlASPEnv)
REGISTRY["melee_range_control_as"] = partial(env_fn, env=MeleeRangeControlASEnv)

if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
