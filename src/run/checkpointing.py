import os
import random
import shutil

import numpy as np
import torch as th


CHECKPOINT_MARKER = "checkpoint_complete"
TRAINING_STATE_FILE = "training_state.th"


def latest_complete_checkpoint(checkpoint_root):
    if not checkpoint_root or not os.path.isdir(checkpoint_root):
        return None

    candidates = []
    for name in os.listdir(checkpoint_root):
        checkpoint_path = os.path.join(checkpoint_root, name)
        if not name.isdigit() or not os.path.isdir(checkpoint_path):
            continue
        if not os.path.isfile(os.path.join(checkpoint_path, CHECKPOINT_MARKER)):
            continue
        candidates.append((int(name), checkpoint_path))

    return max(candidates, default=None, key=lambda item: item[0])


def capture_training_state(
    buffer,
    runner_t_env,
    episode,
    last_test_t,
    last_log_t,
    model_save_time,
):
    stored_episodes = int(buffer.episodes_in_buffer)
    return {
        "runner_t_env": int(runner_t_env),
        "episode": int(episode),
        "last_test_t": int(last_test_t),
        "last_log_t": int(last_log_t),
        "model_save_time": int(model_save_time),
        "buffer_index": int(buffer.buffer_index),
        "episodes_in_buffer": int(buffer.episodes_in_buffer),
        "buffer_transition_data": {
            key: value[:stored_episodes].detach().cpu()
            for key, value in buffer.data.transition_data.items()
        },
        "buffer_episode_data": {
            key: value[:stored_episodes].detach().cpu()
            for key, value in buffer.data.episode_data.items()
        },
        "numpy_random_state": np.random.get_state(),
        "python_random_state": random.getstate(),
        "torch_random_state": th.get_rng_state(),
    }


def restore_training_state(checkpoint_path, buffer):
    state_path = os.path.join(checkpoint_path, TRAINING_STATE_FILE)
    if not os.path.isfile(state_path):
        return None

    state = th.load(state_path, map_location="cpu")
    for key, value in state["buffer_transition_data"].items():
        target = buffer.data.transition_data[key]
        if target.shape[1:] != value.shape[1:] or target.shape[0] < value.shape[0]:
            raise ValueError(
                "Replay-buffer transition shape mismatch for {}: {} cannot restore into {}".format(
                    key, tuple(value.shape), tuple(target.shape)
                )
            )
        target.zero_()
        target[: value.shape[0]].copy_(value.to(target.device))

    for key, value in state["buffer_episode_data"].items():
        target = buffer.data.episode_data[key]
        if target.shape[1:] != value.shape[1:] or target.shape[0] < value.shape[0]:
            raise ValueError(
                "Replay-buffer episode shape mismatch for {}: {} cannot restore into {}".format(
                    key, tuple(value.shape), tuple(target.shape)
                )
            )
        target.zero_()
        target[: value.shape[0]].copy_(value.to(target.device))

    buffer.buffer_index = int(state["buffer_index"])
    buffer.episodes_in_buffer = int(state["episodes_in_buffer"])
    np.random.set_state(state["numpy_random_state"])
    random.setstate(state["python_random_state"])
    th.set_rng_state(state["torch_random_state"])
    return state


def write_resume_checkpoint(checkpoint_root, step, learner, training_state):
    os.makedirs(checkpoint_root, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_root, str(int(step)))
    if os.path.isfile(os.path.join(checkpoint_path, CHECKPOINT_MARKER)):
        return checkpoint_path

    temporary_path = os.path.join(
        checkpoint_root, ".{}.tmp-{}".format(int(step), os.getpid())
    )
    if os.path.isdir(temporary_path):
        shutil.rmtree(temporary_path)
    os.makedirs(temporary_path)

    learner.save_models(temporary_path)
    th.save(training_state, os.path.join(temporary_path, TRAINING_STATE_FILE))
    with open(os.path.join(temporary_path, CHECKPOINT_MARKER), "w") as marker:
        marker.write("ok\n")
    os.replace(temporary_path, checkpoint_path)

    for name in os.listdir(checkpoint_root):
        old_path = os.path.join(checkpoint_root, name)
        if name.isdigit() and name != str(int(step)) and os.path.isdir(old_path):
            shutil.rmtree(old_path)

    return checkpoint_path
