import tensorflow_datasets as tfds
import numpy as np


def get_numpy_from_tensor(tensor):
    """
    Convert a tensor to numpy
    """
    if isinstance(tensor, dict):
        return {k: get_numpy_from_tensor(v) for k, v in tensor.items()}
    return tensor.numpy()


name_to_obs = {'l2r/tcp_pose': (0, 6), 'l2r/tcp_vel': (6, 12), 'left/action': (12, 19), 'left/gripper_state': (19, 21),
               'left/tcp_force': (21, 24), 'left/tcp_pose': (24, 30), 'left/tcp_torque': (30, 33),
               'left/tcp_vel': (33, 39), 'left/time_diff': (39, 40), 'r2l/tcp_pose': (40, 46), 'r2l/tcp_vel': (46, 52),
               'right/action': (52, 59), 'right/gripper_state': (59, 61), 'right/tcp_force': (61, 64),
               'right/tcp_pose': (64, 70), 'right/tcp_torque': (70, 73), 'right/tcp_vel': (73, 79),
               'right/time_diff': (79, 80)}

def name_obs(obs):
    return {key: obs[low:high] for key, (low, high) in name_to_obs.items()}


if __name__ == "__main__":
    RLDS_Path = "/home/nico/real-world-rl/serl/examples/box_handover_drq/rlds"
    dataset = tfds.builder_from_directory(RLDS_Path).as_dataset(split="all")

    observations = []
    i = 0
    for j, episode in enumerate(dataset):
        if j < 100:
            continue
        steps = episode["steps"]
        print(i)
        for i, step in enumerate(steps):
            # 'action', 'discount', 'is_first', 'is_last', 'is_terminal', 'observation', 'reward'
            observations.append(get_numpy_from_tensor(step["observation"]).flatten())

    print("obs len  ", len(observations))
    named_obs = [name_obs(obs) for obs in observations]
    print("named obs len  ", len(named_obs))

    # list of dicts to dict of lists
    obs_info = {key: [] for key in name_to_obs.keys()}
    for obs in named_obs:
        for key, value in obs.items():
            obs_info[key].append(value)

    for key, value in obs_info.items():
        print(f"{key}  -> {np.mean(value, axis=0)}   {np.std(value, axis=0)}")
