import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

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

def print_norm_infos(dataset):
    observations = []
    for j, episode in enumerate(dataset):
        steps = episode["steps"]
        if len(steps) > 6 and j > 50:
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

    constants = {}
    obs_info = {key: np.asarray(value) for key, value in obs_info.items()}
    for key, value in obs_info.items():
        if value.shape[-1] == 6:
            print(f"{key} [:3] -> {np.mean(value[..., :3], axis=(0, 1))}   {np.std(value[..., :3], axis=(0, 1))}")
            print(f"{key} [3:] -> {np.mean(value[..., 3:], axis=(0, 1))}   {np.std(value[..., 3:], axis=(0, 1))}")
            constants[key + "/pos"] = [np.mean(value[..., :3], axis=(0, 1)), np.std(value[..., :3], axis=(0, 1))]
            constants[key + "/rot"] = [np.mean(value[..., 3:], axis=(0, 1)), np.std(value[..., 3:], axis=(0, 1))]
        else:
            print(f"{key}  -> {np.mean(value, axis=(0, 1))}   {np.std(value, axis=(0, 1))}")
            constants[key] = [np.mean(value, axis=(0, 1)), np.std(value, axis=(0, 1))]


def check_consistency(dataset):
    observations = []
    for episode in dataset:
        steps = episode["steps"]
        observation = []
        for i, step in enumerate(steps):
            # 'action', 'discount', 'is_first', 'is_last', 'is_terminal', 'observation', 'reward'
            obs = get_numpy_from_tensor(step["observation"]).flatten()
            observation.append(name_obs(obs))
        observations.append(observation)

    def check(obs, new_obs):
        diff = new_obs["right/tcp_pose"][:3] - obs["right/tcp_pose"][:3]
        return np.all(np.sign(diff) * np.sign(obs["right/action"][:3]) >= 0.)

    for j, episode in enumerate(observations):
        yes = 0
        for i in range(len(episode)-1):
            ob, n_ob = episode[i], episode[i+1]
            yes += check(ob, n_ob)
        print(f"{j}: {yes} / {len(episode)}")

    # for obs in observations[55]:
    #     print(f"a: {obs['left/action'][:3]}   s:{obs['left/tcp_pose'][:3]} ")

    action = np.asarray([o["left/action"][:3] for o in observations[55]]) * 0.05
    pose = np.asarray([o["left/tcp_pose"][:3] for o in observations[55]])
    print(f"{action.shape}, {pose.shape}")

    action_sum = np.asarray([np.sum(action[:i, :], axis=0) for i in range(len(action))])
    print(action, action_sum)

    plt.figure(0)
    l = np.linspace(0, 1, action.shape[0])
    plt.plot(l, action_sum[:, 1], "r")
    plt.plot(l, pose[:, 1], "b")
    plt.show()


if __name__ == "__main__":
    RLDS_Path = "/home/nico/real-world-rl/serl/examples/box_handover_drq/rlds"
    dataset = tfds.builder_from_directory(RLDS_Path).as_dataset(split="all")

    check_consistency(dataset)