import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import threading
from typing import Any


def finetune_pointcloud_fusion(pc1: np.ndarray, pc2: np.ndarray):
    pcd1, pcd2 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    pcd2.points = o3d.utility.Vector3dVector(pc2)
    pcd1.estimate_normals()
    pcd2.estimate_normals()

    def pairwise_registration(source, target, max_correspondence_distance):
        # see https://www.open3d.org/docs/latest/tutorial/Advanced/multiway_registration.html
        icp = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        transformation_icp = icp.transformation
        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance,
            icp.transformation)
        return transformation_icp, information_icp

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        transformation, info = pairwise_registration(pcd1, pcd2, max_correspondence_distance=1e-3)

    r = R.from_matrix(transformation[:3, :3].copy()).as_euler("xyz")
    t = transformation[:3, 3].copy().flatten()
    print(f"fusion result--> r: {r}   t: {t}")
    return transformation


def pointcloud_to_voxel_grid(points: np.ndarray, voxel_size: float, min_bounds: np.ndarray, max_bounds: np.ndarray, grid_dimensions: np.ndarray):
    points_filtered = crop_pointcloud(points, min_bounds, max_bounds)
    voxel_indices = ((points_filtered[:, :3] - min_bounds) / voxel_size).astype(np.uint8)
    voxel_grid = None

    if points.shape[1] == 3:    # uncolored
        voxel_grid = np.zeros(grid_dimensions, dtype=np.bool_)
        voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True

    if points.shape[1] == 6:
        voxel_grid = np.zeros(np.concatenate((grid_dimensions, [4])), dtype=np.uint8)
        voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2], 0] = 255  # occupancy
        voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2], 1:] = points_filtered[:, 3:]  # color

    return voxel_grid, np.concatenate((voxel_indices, points_filtered[:, 3:]), axis=-1)


def crop_pointcloud(points: np.ndarray, min_bounds: np.ndarray, max_bounds: np.ndarray):
    mask = (points[:, 0] > min_bounds[0]) & (points[:, 1] > min_bounds[1]) & (points[:, 2] > min_bounds[2])
    mask &= (points[:, 0] < max_bounds[0]) & (points[:, 1] < max_bounds[1]) & (points[:, 2] < max_bounds[2])
    return points[mask]


def transform_point_cloud(points: np.ndarray, transform_matrix):
    if points.shape[1] == 3 or points.shape[1] == 6:
        points = np.hstack([points[:, :3], np.ones((points.shape[0], 1)), points[:, 3:]])

    transformed_points = np.dot(points[:, :4], transform_matrix.T)

    if transformed_points.shape[1] == 4:
        transformed_points = transformed_points[:, :3]

    if points.shape[1] > 4:    # add color if there
        transformed_points = np.concatenate((transformed_points, points[:, -3:]), axis=-1)

    return transformed_points


class PointCloudFusion:
    def __init__(self, camera_params, voxel_params):
        assert len(camera_params.keys()) < 3
        assert "voxel_box_size" in voxel_params and "voxel_grid_shape" in voxel_params

        self.min_bounds = -np.array(voxel_params["voxel_box_size"]) / 2.
        self.max_bounds = -self.min_bounds

        self.grid_dimensions = voxel_params["voxel_grid_shape"]
        vox_size = (self.max_bounds - self.min_bounds) / self.grid_dimensions
        assert np.all(np.isclose(vox_size, vox_size[0]))
        self.voxel_size: float = float(vox_size[0])

        self.original_pcds = []
        self._is_transformed = False

        self.keys = list(camera_params.keys())
        self.pcd: dict[str: np.ndarray] = {}
        self.t = {}         # transformation
        for key, params in camera_params.items():
            t = np.eye(4)
            t[:3, :3] = R.from_euler("xyz", params["angle"], degrees=True).as_matrix()
            t[:3, 3]  = params["center_offset"]
            self.t[key] = t

    def get_voxelgrid_shape(self):
        return np.ceil((self.max_bounds - self.min_bounds) / self.voxel_size).astype(int)

    def append(self, pcd: np.ndarray, key):
        assert key in self.keys
        self.pcd[key] = pcd
        self.original_pcds.append(pcd)

    def clear(self):
        self.pcd = {}
        self.original_pcds = []
        self._is_transformed = False

    def _transform(self):
        assert not self.is_empty()
        if self._is_transformed:
            return

        for key in self.pcd.keys():
            self.pcd[key] = transform_point_cloud(points=self.pcd[key], transform_matrix=self.t[key])
        self._is_transformed = True

    def voxelize(self, points: np.ndarray):
        grid, indices = pointcloud_to_voxel_grid(points, voxel_size=self.voxel_size, min_bounds=self.min_bounds,
                                                 max_bounds=self.max_bounds, grid_dimensions=self.grid_dimensions)
        return grid, indices

    def crop(self, points: np.ndarray):
        return crop_pointcloud(points=points, min_bounds=self.min_bounds, max_bounds=self.max_bounds)

    def get_pointcloud_representation(self, voxelize=True, crop=True):
        assert self.is_complete()
        self._transform()

        if len(self.keys) == 1:
            pcd = self.pcd[self.keys[0]]
            return self.voxelize(pcd) if voxelize else (self.crop(pcd) if crop else pcd)

        else:       # len 2
            swap = lambda x: np.moveaxis(x, 0, 1)
            pcd1, pcd2 = self.pcd.values()
            fused = swap(np.hstack([swap(pcd1), swap(pcd2)]))
            return self.voxelize(fused) if voxelize else (self.crop(fused) if crop else fused)

    def get_original_pcds(self):
        if len(self.original_pcds) == 1:
            return self.original_pcds[0]
        else:
            return self.original_pcds

    def is_complete(self):
        return len(self.pcd) == len(self.keys)

    def is_empty(self):
        return len(self.pcd) == 0

    def calibrate_fusion(self):
        assert self.is_complete()
        # rough transform
        self._transform()

        # then calibrate
        pcd1, pcd2 = self.pcd.values()
        t = finetune_pointcloud_fusion(pc1=pcd1, pc2=pcd2)
        return t


class CalibrationTread(threading.Thread):
    def __init__(self, pc_fusion: PointCloudFusion, num_samples=20, verbose=False, *args, **kwargs):
        super(CalibrationTread, self).__init__(*args, **kwargs)
        self.pc_fusion = pc_fusion
        self.samples = np.zeros((num_samples, 4, 4))  # transformation matrix samples
        self.pc_backlog = []
        self.verbose = verbose

    def start(self):
        super().start()
        if self.verbose:
            print(f"Calibration Thread started at {self.native_id}")

    def append_backlog(self, pc1, pc2):
        self.pc_backlog.append([pc1, pc2])
        assert self.samples.shape[0] >= len(self.pc_backlog)

    def calibrate(self, visualize=False):
        print(f"calibrating for {len(self.pc_backlog)} samples...")
        for i, (pc1, pc2) in enumerate(self.pc_backlog):
            self.pc_fusion.clear()
            self.pc_fusion.append(pc1, self.pc_fusion.keys[0])
            self.pc_fusion.append(pc2, self.pc_fusion.keys[1])

            self.samples[i, ...] = self.pc_fusion.calibrate_fusion()

            if visualize:
                # visualize for testing
                pc, pc2 = self.pc_fusion.pcd.values()
                pc = transform_point_cloud(points=pc.copy(), transform_matrix=self.samples[i])  # transform

                swap = lambda x: np.moveaxis(x, 0, 1)
                fused = swap(np.hstack([swap(pc), swap(pc2)]))

                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(fused)
                o3d.visualization.draw_geometries([pc])

        rotations = R.from_matrix(self.samples[:, :3, :3])
        mean_rot = rotations.mean()
        mean_translation = np.mean(self.samples[:, :3, 3], axis=0)

        print(f"mean translation from {self.pc_fusion.keys[0]} to {self.pc_fusion.keys[1]} "
              f"is: {mean_translation}   and mean rotation: {mean_rot.as_euler('xyz', degrees=True)}")

        t = mean_translation.copy() / 2.  # half the translation
        rot = np.zeros((2, 3, 3))
        rot[0, ...] = mean_rot.as_matrix()
        rot[1, ...] = np.eye(3)
        r = R.from_matrix(rot).mean()  # half the rotation

        t1_fine = np.eye(4)
        t1_fine[:3, :3] = r.as_matrix()
        t1_fine[:3, 3] = t
        self.pc_fusion.t[self.pc_fusion.keys[0]] = np.dot(self.pc_fusion.t[self.pc_fusion.keys[0]], t1_fine)

        t2_fine = np.eye(4)
        t2_fine[:3, :3] = r.inv().as_matrix()
        t2_fine[:3, 3] = -t
        self.pc_fusion.t[self.pc_fusion.keys[1]] = np.dot(self.pc_fusion.t[self.pc_fusion.keys[1]], t1_fine)

        t1, t2 = self.pc_fusion.t.values()
        print(f"change the params to: {self.pc_fusion.keys[0]}: 'angle': {R.from_matrix(t1[:3, :3]).as_euler('xyz')}"
              f" 'center_offset': {t1[:3, 3]}")
        print(f"change the params to: {self.pc_fusion.keys[1]}: 'angle': {R.from_matrix(t2[:3, :3]).as_euler('xyz')}"
              f" 'center_offset': {t2[:3, 3]}")
