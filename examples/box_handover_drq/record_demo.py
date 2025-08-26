import gymnasium as gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os
import threading
from pynput import keyboard
from pprint import pprint

from ur_env.envs.camera_env.box_picking_camera_env import UR5Env
from ur_env.envs.dual_wrappers import DualToMrpWrapper, DualNormalizationWrapper
from ur_env.envs.handover_env.box_handover_env import UR5Handover90Degrees, UR5HandoverEnv
from ur_env.envs.plot_wrapper import PlotWrapper
from ur_env.envs.relative_env import DualRelativeFrame

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from ur_env.envs.handover_env import UR5DualCameraConfigLeft, UR5DualCameraConfigRight, UR5DualCameraConfig90DegreesLeft, UR5DualCameraConfig90DegreesRight
from ur_env.envs.wrappers import DualSpaceMouseIntervention
import ur_env


exit_program = threading.Event()


def on_space(key, info_dict):
    if key == keyboard.Key.space:
        for key, item in info_dict.items():
            print(f'{key}:  {item}', end='   ')
        print()


def on_esc(key):
    if key == keyboard.Key.esc:
        exit_program.set()


def plot_poses(data):
    import matplotlib

    matplotlib.use('TkAgg')  # or 'QtAgg' if you have PyQt5/PySide installed
    import matplotlib.pyplot as plt

    plt.ion()

    target = np.asarray(data[0])
    actual = np.asarray(data[1])

    labels = ['X Position', 'Y Position', 'Z Position']
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i in range(3):
        axes[i].plot(np.arange(target.shape[0]), target[:, i], label='Target', linestyle='--')
        axes[i].plot(np.arange(actual.shape[0]), actual[:, i], label='Actual', linestyle='-')
        axes[i].set_ylabel(labels[i])
        axes[i].legend(loc='upper right')
        axes[i].grid(True)

    axes[2].set_xlabel('Sample Index')

    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    fake_env = False
    camera_mode = "pointcloud"
    use_90_degrees = True

    config_left = UR5DualCameraConfigLeft if not use_90_degrees else UR5DualCameraConfig90DegreesLeft
    config_right = UR5DualCameraConfigRight if not use_90_degrees else UR5DualCameraConfig90DegreesRight
    left_env = UR5Env(
        fake_env = fake_env,
        config = config_left,
        camera_mode=camera_mode,
        visualize_camera_mode=False,
    )
    right_env = UR5Env(
        fake_env = fake_env,
        config = config_right,
        camera_mode=camera_mode,
        visualize_camera_mode=False,
    )

    handover_env = UR5HandoverEnv if not use_90_degrees else UR5Handover90Degrees
    env = handover_env(
        env_left=left_env,
        env_right=right_env,
    )

    env = DualRelativeFrame(env)
    env = DualToMrpWrapper(env)
    env = PlotWrapper(env)
    env = DualNormalizationWrapper(env)

    if not fake_env:
        env = DualSpaceMouseIntervention(env)

    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    obs, _ = env.reset()

    transitions = []
    success_count = 0
    success_needed = 10
    total_count = 0
    pbar = tqdm(total=success_needed)

    info_dict = {'state': env.unwrapped.env_left.curr_pos, 'gripper_state': env.unwrapped.env_left.gripper_state,
                 'force': env.unwrapped.env_left.curr_force, 'reset_pose': env.unwrapped.env_left.curr_reset_pose}
    listener_1 = keyboard.Listener(daemon=True, on_press=lambda event: on_space(event, info_dict=info_dict))
    listener_1.start()

    listener_2 = keyboard.Listener(on_press=on_esc, daemon=True)
    listener_2.start()

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"box_picking_{success_needed}_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    try:
        running_reward = 0.
        while success_count < success_needed:
            if exit_program.is_set():
                raise KeyboardInterrupt  # stop program, but clean up before

            action = np.array([0., 0., 0., 0., 0., 0., 0.])
            action = np.concatenate([action, action])
            next_obs, rew, done, truncated, info = env.step(action)

            if "intervene_action" in info:
                action = info["intervene_action"]

            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=action,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                )
            )
            transitions.append(transition)
            # pprint(next_obs["state"])

            obs = next_obs
            running_reward += rew

            if done or truncated:
                success_count += int(rew > 0.99)
                total_count += 1
                print(
                    f"{rew}\tGot {success_count} successes of {total_count} trials. {success_needed} successes needed."
                )
                pbar.update(int(rew > 0.99))
                obs, _ = env.reset()
                print("Reward total:", running_reward)
                running_reward = 0.

        with open(file_path, "wb") as f:
            pkl.dump(transitions, f)
            print(f"saved {success_needed} demos to {file_path}")

    except KeyboardInterrupt as e:
        print(f'\nProgram was interrupted, cleaning up...  ', e.__str__())

    finally:
        # data_left = left_env.controller._poses
        # data_right = right_env.controller._poses

        # plot_poses(data_left)
        # plot_poses(data_right)

        pbar.close()
        env.close()
        listener_1.stop()
        listener_2.stop()
