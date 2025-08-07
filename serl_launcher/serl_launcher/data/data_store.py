from threading import Lock
from typing import Union, Iterable
import threading
import queue
import copy

import gym
import jax
from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.data.memory_efficient_replay_buffer import (
    MemoryEfficientReplayBuffer,
)

from agentlace.data.data_store import DataStoreBase

from typing import List, Optional, TypeVar
from concurrent.futures import ThreadPoolExecutor

# import oxe_envlogger if it is installed
try:
    from oxe_envlogger.rlds_logger import RLDSLogger, RLDSStepType
except ImportError:
    print(
        "rlds logger is not installed, install it if required: "
        "https://github.com/rail-berkeley/oxe_envlogger "
    )
    RLDSLogger = TypeVar("RLDSLogger")


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            capacity: int,
            rlds_logger: Optional[RLDSLogger] = None,
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()
        self._logger = None

        if rlds_logger:
            self.step_type = RLDSStepType.TERMINATION  # to init the state for restart
            self._logger = rlds_logger

    # ensure thread safety
    def insert(self, data):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(data)

            # add data to the rlds logger
            if self._logger:
                if self.step_type in {
                    RLDSStepType.TERMINATION,
                    RLDSStepType.TRUNCATION,
                }:
                    self.step_type = RLDSStepType.RESTART
                elif not data["masks"]:  # 0 is done, 1 is not done
                    self.step_type = RLDSStepType.TERMINATION
                elif data["dones"]:
                    self.step_type = RLDSStepType.TRUNCATION
                else:
                    self.step_type = RLDSStepType.TRANSITION

                self._logger(
                    action=data["actions"],
                    obs=data["next_observations"],  # TODO: check if this is correct
                    reward=data["rewards"],
                    step_type=self.step_type,
                )

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO

    def __del__(self):
        if self._logger:
            self._logger.close()
            print("[ReplayBufferDataStore] RLDS logger closed successfully")


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer, DataStoreBase):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            capacity: int,
            image_keys: Iterable[str] = ("image",),
            rlds_logger: Optional[RLDSLogger] = None,
            use_deep_copy: bool = True,
    ):
        MemoryEfficientReplayBuffer.__init__(
            self, observation_space, action_space, capacity, pixel_keys=image_keys
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()
        self._logger = None
        self._logger_queue = None
        self._logger_thread = None
        self._shutdown_event = None
        self._use_deep_copy = use_deep_copy

        if rlds_logger:
            self.step_type = RLDSStepType.TERMINATION  # to init the state for restart
            self._logger = rlds_logger
            self._setup_async_logging()

    def _setup_async_logging(self):
        """Setup completely async logging with queue and dedicated thread."""
        self._logger_queue = queue.Queue(maxsize=1000)  # Buffer for logging operations
        self._shutdown_event = threading.Event()
        
        def logger_worker():
            """Dedicated worker thread for logging operations."""
            while not self._shutdown_event.is_set():
                try:
                    log_data = self._logger_queue.get(timeout=0.5)
                    if log_data is None:  # Shutdown signal
                        break
                    
                    action, obs, reward, step_type = log_data
                    self._logger(
                        action=action,
                        obs=obs,
                        reward=reward,
                        step_type=step_type,
                    )
                    self._logger_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"RLDS Logger thread error: {e}")
                    continue

        self._logger_thread = threading.Thread(target=logger_worker, daemon=True)
        self._logger_thread.start()

    def _copy_data_for_logging(self, data):
        """Copy data for logging with configurable strategy."""
        if self._use_deep_copy:
            # Deep copy for maximum safety (slower)
            return (
                copy.deepcopy(data["actions"]),
                copy.deepcopy(data["next_observations"]),
                copy.deepcopy(data["rewards"]),
                self.step_type,
            )
        else:
            # Shallow copy for better performance (faster)
            return (
                data["actions"].copy() if hasattr(data["actions"], 'copy') else data["actions"],
                data["next_observations"].copy() if hasattr(data["next_observations"], 'copy') else data["next_observations"],
                data["rewards"].copy() if hasattr(data["rewards"], 'copy') else data["rewards"],
                self.step_type,
            )

    # ensure thread safety
    def insert(self, data):
        with self._lock:
            super(MemoryEfficientReplayBufferDataStore, self).insert(data)

            if self._logger:
                # handle restart when it was done before
                if self.step_type in {
                    RLDSStepType.TERMINATION,
                    RLDSStepType.TRUNCATION,
                }:
                    self.step_type = RLDSStepType.RESTART
                elif self.step_type == RLDSStepType.TRUNCATION:
                    self.step_type = RLDSStepType.RESTART
                elif not data["masks"]:  # 0 is done, 1 is not done
                    self.step_type = RLDSStepType.TERMINATION
                elif data["dones"]:
                    self.step_type = RLDSStepType.TRUNCATION
                else:
                    self.step_type = RLDSStepType.TRANSITION

                try:
                    log_data = self._copy_data_for_logging(data)
                    self._logger_queue.put_nowait(log_data)
                except queue.Full:
                    print("Warning: Logger queue full, dropping log entry")
                except Exception as e:
                    print(f"Error queuing log data: {e}")

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(MemoryEfficientReplayBufferDataStore, self).sample(
                *args, **kwargs
            )

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO

    def __del__(self):
        if self._shutdown_event:
            self._shutdown_event.set()
        if self._logger_queue:
            self._logger_queue.put(None)
        if self._logger_thread:
            self._logger_thread.join(timeout=5.0)
        if self._logger:
            self._logger.close()
            print("[MemoryEfficientReplayBufferDataStore] RLDS logger closed successfully")


def populate_data_store(
        data_store: DataStoreBase,
        demos_path: str,
        reward_scaling: int = 1,
):
    """
    Utility function to populate demonstrations data into data_store.
    :return data_store
    """
    import pickle as pkl

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                transition["rewards"] *= reward_scaling  # apply reward scaling
                data_store.insert(transition)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store


def populate_data_store_with_z_axis_only(
        data_store: DataStoreBase,
        demos_path: str,
):
    """
    Utility function to populate demonstrations data into data_store.
    This will remove the x and y cartesian coordinates from the state.
    :return data_store
    """
    import pickle as pkl
    import numpy as np
    from copy import deepcopy

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                tmp = deepcopy(transition)
                tmp["observations"]["state"] = np.concatenate(
                    (
                        tmp["observations"]["state"][:, :4],
                        tmp["observations"]["state"][:, 6][None, ...],
                        tmp["observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                tmp["next_observations"]["state"] = np.concatenate(
                    (
                        tmp["next_observations"]["state"][:, :4],
                        tmp["next_observations"]["state"][:, 6][None, ...],
                        tmp["next_observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                data_store.insert(tmp)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store
