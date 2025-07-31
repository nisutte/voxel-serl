import threading
import time
import copy
import numpy as np
import robotic as ry

from scipy.spatial.transform import Rotation as R


def urtde_to_rai(Q):
    # not sure why they are offset, and this should probably be fixed in the kinematics, not here
    tmp = copy.deepcopy(Q)
    tmp[1] += np.pi / 2
    tmp[3] += np.pi / 2
    return tmp


class ThreadedCollisionDetector:
    def __init__(self, robot_left_transform, robot_right_transform, distance_margin=0.005, frequency=10, headless=False):
        """
        Initialize the collision detector with two robots.

        :param robot_left_transform: 4x4 transformation matrix for the left robot.
        :param robot_right_transform: 4x4 transformation matrix for the right robot.
        """
        # Create the scene
        self.C = ry.Config()
        self.C.addFrame("floor").setPosition([0, 0, 0.0]).setShape(
            ry.ST.box, size=[20, 20, 0.02]
        ).setColor([0.9, 0.9, 0.9]).setContact(0)

        self.robot_left_name = "robot_left"
        self._add_robot(self.robot_left_name, robot_left_transform)
        self.robot_right_name = "robot_right"
        self._add_robot(self.robot_right_name, robot_right_transform)

        self.robot_joint_names = {
            self.robot_left_name: self._get_robot_joints(self.robot_left_name),
            self.robot_right_name: self._get_robot_joints(self.robot_right_name),
        }

        self.robot_joint_states = {
            self.robot_left_name: np.zeros(len(self.robot_joint_names[self.robot_left_name])),
            self.robot_right_name: np.zeros(len(self.robot_joint_names[self.robot_right_name])),
        }

        self.distance_margin = distance_margin
        self.frequency = frequency
        self.headless = headless
        self.collision_msg = ""

        self.collision_free = True
        self.stop = threading.Event()
        self.thread = None

    def _add_robot(self, robot_name, transform):
        """
        Add a robot to the scene.

        :param robot_name: Name of the robot.
        :param transform: 4x4 transformation matrix for the robot.
        """
        robot_path = "/home/nico/robot_ipc_control/pose_estimation/ur5/ur5_vacuum_inflated.g"
        pos = transform[:3, 3]
        x, y, z, w = R.from_matrix(transform[:3, :3]).as_quat()
        quat_wfirst = (w, x, y, z)

        self.C.addFile(robot_path, namePrefix=f"{robot_name}_").setParent(
            self.C.getFrame("floor")
        ).setRelativePosition(pos).setRelativeQuaternion(quat_wfirst).setJoint(ry.JT.rigid)

    def _get_robot_joints(self, prefix):
        """
        Get joint names for a robot.

        :param prefix: Prefix of the robot's joint names.
        :return: List of joint names.
        """
        links = []
        for name in self.C.getJointNames():
            if prefix in name:
                name = name.split(":")[0]
                if name not in links:
                    links.append(name)
        return links

    def update_joint_state(self, robot_name, joint_state):
        """
        Update the joint state of a robot.

        :param robot_name: Name of the robot.
        :param joint_state: List of joint values.
        """
        if robot_name in self.robot_joint_states:
            self.robot_joint_states[robot_name] = urtde_to_rai(joint_state)

    def _collision_detection_loop(self):
        """
        threaded collision detection loop
        """
        while not self.stop.is_set():
            t_start = time.time()
            for robot, joint_state in self.robot_joint_states.items():
                self.C.setJointState(joint_state, self.robot_joint_names[robot])

            # Check for collisions
            collisions = self.C.getCollisions(belowMargin=self.distance_margin)
            # ignore gripper to gripper collisions
            for i in range(len(collisions) - 1, -1, -1):
                names = ["ur_gripper_fill_coll", "ur_vacuum_coll", "_ur_gripper"]
                if any(name in collisions[i][0] for name in names) and any(name in collisions[i][1] for name in names):
                    del collisions[i]

            self.collision_free = len(collisions) < 1
            if not self.collision_free:
                for c in collisions:
                    self.collision_msg = f"collision detected between {c[0]} and {c[1]} with distance {c[2] * 100} cm"
                    # print(f"collision detected between {c[0]} and {c[1]} with distance {c[2] * 100} cm")

            if not self.headless:
                self.C.view(False)
            time.sleep(max(1. / self.frequency - (time.time() - t_start), 0.))

    def start(self):
        if not self.stop.is_set():
            self.thread = threading.Thread(target=self._collision_detection_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.stop.set()
        self.thread.join()

    def is_collision_free(self):
        return self.collision_free


if __name__ == "__main__":
    # Example transformation matrices for the robots
    cam2left = np.load("/home/nico/robot_ipc_control/calibration/T_cam_to_robot_base_left_20250523_153325.npy")
    cam2right = np.load("/home/nico/robot_ipc_control/calibration/T_cam_to_robot_base_right_20250523_152927.npy")
    left2right = np.linalg.inv(cam2left) @ cam2right

    np.set_printoptions(precision=3, suppress=1)
    print(left2right)

    # Initialize the collision detector
    collision_detector = ThreadedCollisionDetector(np.eye(4), left2right)

    # Start the collision detection thread
    collision_detector.start()

    try:
        # Simulate joint updates
        for i in range(100):
            # Create some joint positions that might cause collision
            joints_left = np.array([-1.5708 + i * 0.05, np.pi / 3, 0, -1.5708 + np.pi / 2, -1.5708, 0.0])
            joints_right = np.array(
                [-1.5708 - i * 0.04, np.pi / 3, 0, -1.5708 + np.pi / 2, -1.5708, 0.0])

            collision_detector.update_joint_state("robot_left", joints_left)
            collision_detector.update_joint_state("robot_right", joints_right)

            print(f"Collision-free: {collision_detector.is_collision_free()}")
            time.sleep(0.5)
    finally:
        # Stop the collision detection thread
        collision_detector.stop()
