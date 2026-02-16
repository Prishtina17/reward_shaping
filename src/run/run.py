import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from smac.env import StarCraft2Env

def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    env_cfg_name = getattr(args, "env_config_name", None) or getattr(args, "env", "env")
    map_name = args.env_args.get("map_name", "map") if isinstance(args.env_args, dict) else "map"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_token = f"{env_cfg_name}_{map_name}_{timestamp}"
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    stop_mode = getattr(args, "training_stop_mode", "steps").lower()
    win_rate_threshold = float(getattr(args, "training_win_rate_threshold", 0.95))
    early_stop_triggered = False

    replay_winrate_thresholds = [0.95]  # save replays only at 0.95 win rate
    saved_replay_markers = set()

    env_name_for_replay = getattr(args, "env", "env")
    env_cfg_name = getattr(args, "env_config_name", None)
    env_token = env_cfg_name or env_name_for_replay
    if isinstance(args.env_args, dict):
        map_token = args.env_args.get("map_name", "map")
    else:
        map_token = getattr(args.env_args, "map_name", "map")

    # How many winning episodes to collect in the replay session before saving
    replay_min_wins = int(getattr(args, "replay_min_wins", 50))
    replay_max_episodes = int(getattr(args, "replay_max_episodes", 800))

    def _save_training_replay(win_rate_value: float) -> None:
        """Save a replay that contains several winning episodes. Uses full_restart so the
        replay file is a fresh SC2 session; runs episodes until replay_min_wins wins collected."""
        timestamp_label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{env_token}_{map_token}_{win_rate_value:.2f}_{timestamp_label}"

        if not hasattr(runner, "save_replay"):
            logger.console_logger.warning("Replay save requested but runner has no save_replay method.")
            return

        # SC2 replay = entire game session. We full_restart env[0] to get a fresh session,
        # then run episodes until env[0] accumulates enough wins, then save env[0]'s replay.
        is_parallel = hasattr(runner, "full_restart_replay_env")
        is_episode = hasattr(runner, "env") and hasattr(runner.env, "full_restart")

        if is_parallel:
            # ParallelRunner: full_restart env[0], run batches, count env[0] wins
            runner.full_restart_replay_env()
            wins_in_session = 0
            episodes_in_session = 0
            while wins_in_session < replay_min_wins and episodes_in_session < replay_max_episodes:
                with th.no_grad():
                    episode_batch = runner.run(test_mode=False)
                buffer.insert_episode_batch(episode_batch)
                episodes_in_session += 1
                if runner.last_battle_won_per_env[0]:
                    wins_in_session += 1
            logger.console_logger.info(
                "Replay at {:.0f}%: env[0] session has {} wins in {} batches, saving.".format(
                    win_rate_value * 100.0, wins_in_session, episodes_in_session,
                )
            )
        elif is_episode:
            # EpisodeRunner: full_restart, run single episodes, count wins
            runner.env.full_restart()
            wins_in_session = 0
            episodes_in_session = 0
            while wins_in_session < replay_min_wins and episodes_in_session < replay_max_episodes:
                with th.no_grad():
                    episode_batch = runner.run(test_mode=False)
                buffer.insert_episode_batch(episode_batch)
                episodes_in_session += 1
                if getattr(runner, "last_episode_won", False):
                    wins_in_session += 1
            logger.console_logger.info(
                "Replay at {:.0f}%: session has {} wins in {} episodes, saving.".format(
                    win_rate_value * 100.0, wins_in_session, episodes_in_session,
                )
            )

        result = runner.save_replay(prefix=prefix)
        if isinstance(result, dict) and not result.get("ok", True):
            logger.console_logger.warning(
                "Replay save failed for win rate {:.2f}%: {}".format(
                    win_rate_value * 100.0,
                    result.get("error", "unknown error"),
                )
            )
            return

        logger.console_logger.info(
            "Replay saved for win rate {:.2f}% with prefix '{}'.".format(win_rate_value * 100.0, prefix)
        )

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time

        with th.no_grad():
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)

        if stop_mode == "win_rate":
            stats_dict = getattr(runner, "train_stats", {}) or {}
            wins = stats_dict.get("battle_won", 0)
            episodes_count = stats_dict.get("n_episodes", 0)

            battle_won_mean = None
            if episodes_count:
                battle_won_mean = float(wins) / float(episodes_count)

            if battle_won_mean is not None:
                for threshold in replay_winrate_thresholds:
                    if threshold in saved_replay_markers:
                        continue
                    if battle_won_mean >= threshold:
                        _save_training_replay(threshold)
                        saved_replay_markers.add(threshold)

            if battle_won_mean is not None and battle_won_mean >= win_rate_threshold:
                logger.log_stat("battle_won_mean", battle_won_mean, runner.t_env)
                logger.console_logger.info(
                    "Early stop triggered: win rate {:.2f}% (threshold {:.2f}%).".format(
                        battle_won_mean * 100.0,
                        win_rate_threshold * 100.0,
                    )
                )
                episode += args.batch_size_run
                early_stop_triggered = True
                break

        if buffer.can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue

            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)
            del episode_sample

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

        if early_stop_triggered:
            break

    if stop_mode == "win_rate" and early_stop_triggered:
        logger.console_logger.info(
            "Training stopped early after reaching the target win rate. Final t_env: {}, episodes: {}.".format(
                runner.t_env, episode
            )
        )

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    config.setdefault("training_stop_mode", "steps")
    config.setdefault("training_win_rate_threshold", 0.95)

    return config
