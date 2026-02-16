# Known issues / TODO

## Reward shaping (melee_range_control)

### AB: action bonus does not check available actions

In `melee_range_control_ab.py`, in `_compute_rc_bonus`, the check `if action > 5` is used to treat an action as "attack". This does not use `get_avail_agent_actions`, so the agent can be given a bonus for "attacking" when the attack action was not actually available (e.g. wrong map, different action space). This can incorrectly reward or penalise steps.

**Suggested fix:** use `n_actions_no_attack` (or equivalent from the env) and `get_avail_agent_actions(i)` to determine whether the chosen action is an attack and whether attack was available, and only then apply the action-based bonus.
