from sys import getsizeof

import zmq
import json
import time
import threading
import asyncio
import numpy as np

from ur_env.utils.vacuum_gripper import VacuumGripper


class ControllerClientWithGripper(threading.Thread):
    def __init__(self, robot_ip, config=None, *args, **kwargs):
        super(ControllerClientWithGripper, self).__init__(*args, **kwargs)
        self._stop = threading.Event()
        self._reset = threading.Event()
        self._release_gripper = threading.Event()
        self._is_ready = threading.Event()
        self._is_truncated = threading.Event()
        self.lock = threading.Lock()
        self.plot_pose = False

        if self.plot_pose:
            self._poses = ([], [])

        self.robot_ip = robot_ip
        self.frequency = config.CONTROLLER_HZ
        self.config = config

        self.gripper_timeout = {"timeout": config.GRIPPER_TIMEOUT, "last_grip": time.monotonic() - 1e6}
        self.target_grip = np.zeros((1,), dtype=np.float32)
        self.gripper_state = np.zeros((2,), dtype=np.float32)

        self.reset_angles = np.array([np.pi / 2., -np.pi / 2., np.pi / 2., -np.pi / 2., -np.pi / 2., 0.], dtype=np.float32)

        self.controller = None
        self.gripper = None
        self._release_gripper.set()

    def start(self):
        super().start()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.is_set()

    def is_ready(self):
        return self._is_ready.is_set()

    def is_reset(self):
        return not self._reset.is_set()

    def set_gripper_pos(self, target_grip: np.ndarray):
        with self.lock:
            self.target_grip[:] = target_grip

    def set_target_pose(self, target_pose: np.ndarray):
        self.controller.send_target_pose(target_pose)

    def set_reset_angles(self, reset_pose: np.ndarray):
        with self.lock:
            self.reset_angles[:] = reset_pose
        self._reset.set()

    def reset_forces(self):
        self.controller.send_force_reset_command()

    def get_height(self):
        return self.controller.get_state()["pos"][2]

    def is_moving(self):
        return np.linalg.norm(self.get_state()["vel"], 2) > 0.001

    def _truncate_check(self):
        max_force = np.linalg.norm(self.get_state()["force"]) > 100.
        if max_force:  # TODO add better criteria
            self._is_truncated.set()

    def auto_release_gripper(self, yes=True):
        if yes:
            self._release_gripper.set()
        else:
            self._release_gripper.clear()

    def is_truncated(self):
        return self._is_truncated.is_set()

    def run(self):
        try:
            asyncio.run(self.run_async())
        finally:
            self.stop()

    async def start_controllers(self):
        self.controller = ControllerClient(p_port=self.config.ZEROMQ_PUBLISHER_PORT, s_port=self.config.ZEROMQ_SUBSCRIBER_PORT)
        self.gripper = VacuumGripper(self.robot_ip)
        await self.gripper.connect()
        await self.gripper.activate()

    async def send_gripper_command(self, force_release=False):
        if force_release:
            await self.gripper.automatic_release()
            self.target_grip[0] = 0.0
            return

        timeout_exceeded = (time.monotonic() - self.gripper_timeout["last_grip"]) * 1000 > self.gripper_timeout[
            "timeout"]
        # target grip above threshold and timeout exceeded and not gripping something already
        if self.target_grip[0] > 0.5 and timeout_exceeded and self.gripper_state[1] < 0.5:
            await self.gripper.automatic_grip()
            self.target_grip[0] = 0.0
            self.gripper_timeout["last_grip"] = time.monotonic()

        # release if below neg threshold and gripper activated (grip_status not zero)
        elif self.target_grip[0] < -0.5 and abs(self.gripper_state[1]) > 0.5:
            await self.gripper.automatic_release()
            self.target_grip[0] = 0.0
            # print("release")

    async def _go_to_reset_pose(self):
        # first disable vaccum gripper
        if self.gripper and self._release_gripper.is_set():      # TODO how to handle this
            await self.send_gripper_command(force_release=True)
            time.sleep(0.01)
        self.controller.send_reset_joint_angles(self.reset_angles)

        # wait for the controller to finish
        time.sleep(0.5)
        while np.linalg.norm(np.asarray(self.get_state()["Qd"])) > 0.01:
            time.sleep(1./self.frequency)
        self._is_truncated.clear()
        self._reset.clear()

    async def _update_gripper_state(self):
        pressure = await self.gripper.get_current_pressure()
        obj_status = await self.gripper.get_object_status()
        grip_status = [-1., 1., 1., 0.][obj_status.value]
        pressure = pressure if pressure < 99 else 0     # 100 no obj, 99 sucking empty, so they are ignored
        # grip status, 0->neutral, -1->bad (sucking but no obj), 1-> good (sucking and obj)
        grip_status = 1. if pressure > 0 else grip_status
        pressure /= 98.  # pressure between [0, 1]
        with self.lock:
            self.gripper_state[:] = [pressure, grip_status]

    def get_state(self):
        with self.lock:
            state = self.controller.get_state()
            state["gripper"] = self.gripper_state
            if state["is_truncated"]:
                self._is_truncated.set()
            elif self.is_truncated():
                state["is_truncated"] = 1
        return state

    async def run_async(self):
        await self.start_controllers()
        time.sleep(0.5)     # wait for controller

        try:
            self._is_ready.set()
            t_next = time.monotonic()
            while not self.stopped():
                t_next += 1. / self.frequency
                await asyncio.sleep(t_next - time.monotonic())

                await self._update_gripper_state()
                await self.send_gripper_command()
                self._truncate_check()

                if self.is_truncated():
                    self._reset.set()

                if self._reset.is_set():
                    await self._go_to_reset_pose()

                if self.plot_pose:
                    self._poses[0].append(self.controller.target_pose.copy())
                    self._poses[1].append(self.controller.get_state()["pos"])

        finally:
            # release gripper, controller stays open
            if self.gripper:
                await self.send_gripper_command(force_release=True)
                time.sleep(0.05)


class ControllerClient:
    """
    Sends target poses to the C++ UR5 robot controller using ZeroMQ.
    The CLient is non-blocking, so it does not need another thread.

    Args:
        ip (str): The IP address of the ZeroMQ server (default: "127.0.0.1").
        p_port (int): The port number of the ZeroMQ publisher (default: 5555).
        s_port (int): The port number of the ZeroMQ subscriber (default: 5555).
    """
    def __init__(self, ip="127.0.0.1", p_port=5555, s_port=5556):
        context = zmq.Context()
        self.publisher = context.socket(zmq.PUB)
        self.publisher.bind(f"tcp://{ip}:{p_port}")

        self.subscriber = context.socket(zmq.SUB)
        self.subscriber.setsockopt(zmq.CONFLATE, 1)  # Keep only the latest message (newest state)
        self.subscriber.connect(f"tcp://{ip}:{s_port}")
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")        # sub to all

        self.target_pose = np.zeros((7,))

    def send_target_pose(self, pose: np.ndarray):
        assert pose.shape == (7,)
        self.target_pose[:] = pose.flatten()
        cmd = {"target_ee_pose": pose.astype(float).tolist()}
        self.publisher.send_json(cmd)

    def send_force_reset_command(self):
        self.publisher.send_json({"reset_force_sensor": True})

    def send_reset_joint_angles(self, joint_angles: np.ndarray):
        assert joint_angles.shape == (6, )
        cmd = {"target_q": joint_angles.astype(float).tolist()}
        self.publisher.send_json(cmd)

    def get_state(self):
        return self.subscriber.recv_json()
