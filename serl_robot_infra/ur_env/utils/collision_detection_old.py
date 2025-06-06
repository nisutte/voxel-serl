import numpy as np
import itertools


"""
simple collision detection for the dual UR5 robot
it uses transformations and straightforward sphere to sphere collision detection
(not tested yet, using the solution from valentin)
"""

UR5 = [
    # [a_{i-1} (m),    d_i (m),       alpha_{i-1} (rad)]
    [0.0,           0.089159,      np.pi/2],     # Joint 1
    [-0.425,         0.0,           0.0],         # Joint 2
    [-0.39225,       0.0,           0.0],         # Joint 3
    [0.0,           0.10915,       np.pi/2],     # Joint 4
    [0.0,           0.09465,      -np.pi/2],     # Joint 5
    [0.0,           0.08230,       0.0]          # Joint 6
]

radius = 0.05  # 5 cm
sphere_points = {
    # "name": [joint, radius [m], offset (x, y, z) [m]]
    "suction cup": [5, radius, [0.0, 0.0, 0.16]],
    "ee": [5, radius, [0., 0., 0.]],
    "link 4": [4, radius, [0., 0., 0.]],
    "link 3": [3, radius, [0., 0., 0.]],
    "link 2": [2, radius, [0., 0., 0.]],
}


class Robot:
    def __init__(self, dh_parameters, name, base_transform=None):
        self.params = dh_parameters
        self.name = name
        self.joints = np.zeros(6, dtype=np.float32)

        self.base_transform = np.eye(4, dtype=np.float32)
        if base_transform is not None:
            assert base_transform.shape == (4, 4)
            self.base_transform = base_transform

    def _get_T(self, joint: int, theta: float):
        a, d = self.params[joint][:2]
        ct, st = np.cos(theta), np.sin(theta)
        ca = np.cos(self.params[joint][2])
        sa = np.sin(self.params[joint][2])
        return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    def get_transformation(self, link) -> np.ndarray:
        assert link < self.joints.shape[0]
        T = self.base_transform.copy()
        for i in range(link + 1):
            T = T @ self._get_T(i, self.joints[i])
        return T

    def fk(self) -> np.ndarray:
        return self.get_transformation(5)

    def update_joints(self, joints: np.ndarray):
        assert joints.shape == self.joints.shape
        self.joints = joints


class CollisionDetection:
    def __init__(self, robots: list[Robot], spheres: dict, collision_distance: float = 0.01):
        self.robots = robots
        self.spheres = spheres
        self.collision_distance = collision_distance

    def detect_collision(self, robot1, robot2):
        closest = [np.inf, "none"]
        for name_1, points_1 in self.spheres.items():
            T_trans_1 = np.eye(4)
            T_trans_1[:3, 3] = points_1[2]
            T_1 = robot1.get_transformation(points_1[0])
            T_1 = T_1 @ T_trans_1

            for name_2, points_2 in self.spheres.items():
                T_trans_2 = np.eye(4)
                T_trans_2[:3, 3] = points_2[2]
                T_2 = robot2.get_transformation(points_2[0])
                T_2 = T_2 @ T_trans_2

                # check distance
                dist = np.linalg.norm((np.linalg.inv(T_1) @ T_2)[:3, 3])

                if dist < closest[0]:
                    closest = [dist, f"[{robot1.name}] {name_1} <-> [{robot2.name}] {name_2}   dist: {dist:.4f} m"]

                if dist < points_1[1] + points_2[1] + self.collision_distance:
                    print(f"collision detected: [{robot1.name}] {name_1} <-> [{robot2.name}] {name_2}    dist: {dist:.4f} m")
                    return True

        print(closest[1])
        return False

    def check_collisions(self):
        assert len(self.robots) > 1
        collision = False
        for r1, r2 in itertools.combinations(self.robots, 2):       # check every robot combination
            collision &= self.detect_collision(r1, r2)
        return collision
