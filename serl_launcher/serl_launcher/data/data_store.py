from threading import Lock, Event
from typing import Union, Iterable
import threading
import queue
import copy
import time

import gymnasium as gym
import jax
from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.data.memory_efficient_replay_buffer import (
    MemoryEfficientReplayBuffer,
)

from agentlace.data.data_store import DataStoreBase

from typing import List, Optional, TypeVar

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
        self._logger_lock = Lock()
        self._logger_flush = Event()
        self._shutdown_event = None
        self._use_deep_copy = use_deep_copy
        self._episode_data_queue = queue.Queue(maxsize=500)  # Queue for episode data

        if rlds_logger:
            self.step_type = RLDSStepType.TERMINATION  # to init the state for restart
            self._logger = rlds_logger
            self._setup_threaded_logging()

    def _setup_threaded_logging(self):
        """Setup completely threaded logging with queue and dedicated thread."""
        self._logger_queue = queue.Queue(maxsize=1000)  # Buffer for logging operations
        self._shutdown_event = threading.Event()
        
        def logger_worker():
            """Dedicated worker thread for logging operations."""
            while not self._shutdown_event.is_set():
                if self._logger_flush.is_set():
                    self._flush_episode_data()
                    self._logger_flush.clear()
                time.sleep(0.1)
                
        self._logger_thread = threading.Thread(target=logger_worker, daemon=True)
        self._logger_thread.start()

    def _queue_episode_data(self, data):
        """Queue episode data for later logging (outside of main lock)."""
        if self._logger:
            with self._logger_lock:
                try:
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

                    # Copy data and queue for later logging
                    log_data = (data["actions"], data["next_observations"], data["rewards"], self.step_type)
                    self._episode_data_queue.put_nowait(log_data)
                    
                    if self.step_type == RLDSStepType.RESTART:
                        self._logger_flush.set()

                except queue.Full:
                    print("Warning: Episode data queue full, dropping log entry")
                except Exception as e:
                    print(f"Error queuing episode data: {e}")

    def _flush_episode_data(self):
        """Flush all queued episode data to the logger."""
        if not self._logger:
            return
        
        start_time = time.time()
        with self._logger_lock:
            # Process all queued episode data
            while not self._episode_data_queue.empty():
                try:
                    log_data = self._episode_data_queue.get_nowait()
                    action, obs, reward, step_type = log_data
                    self._logger(
                        action=action,
                        obs=obs,
                        reward=reward,
                        step_type=step_type,
                    )
                    self._episode_data_queue.task_done()
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"Error logging episode data: {e}")
                    continue
            # print(f"Flushed episode data in {time.time() - start_time:.3f} seconds")

    # ensure thread safety for insert, not logging
    def insert(self, data):
        with self._lock:
            super(MemoryEfficientReplayBufferDataStore, self).insert(data)

        self._queue_episode_data(data)

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

    def close_logger(self):
        if self._logger:
            self._shutdown_event.set()
            self._logger_thread.join(timeout=1.0)
            self._logger.close()
            del self._logger
            print("[MemoryEfficientReplayBufferDataStore] RLDS logger closed successfully")
        else:
            print("[MemoryEfficientReplayBufferDataStore] No RLDS logger to close.")

    def __del__(self):
        self.close_logger()


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
