import numpy as np
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
import time


class RSCapture:
    def get_device_serial_numbers(self):
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    def __init__(self, name, serial_number, dim=(640, 480), fps=30, rgb=True, depth=False, pointcloud=False,
                 rgb_pointcloud=False):
        self.name = name
        # print(self.get_device_serial_numbers())
        assert serial_number in self.get_device_serial_numbers()
        self.serial_number = serial_number
        self.rgb = rgb
        self.depth = depth
        self.pointcloud = pointcloud
        self.rgb_pc = rgb_pointcloud

        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)

        assert self.rgb or self.depth or self.pointcloud or self.rgb_pc
        if self.rgb or self.rgb_pc:
            self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        if self.depth or self.pointcloud or self.rgb_pc:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)

        self.profile = self.pipe.start(self.cfg)

        if self.depth:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            self.max_clipping_distance = 0.2 / depth_scale  # 0.15m max clipping distance

        if self.pointcloud or self.rgb_pc:
            self.pc = rs.pointcloud()
            self.threshold_filter = rs.threshold_filter(min_dist=0., max_dist=0.25)
            self.decimation_filter = rs.decimation_filter(magnitude=2.)  # 2 or 4
            self.temporal_filter = rs.temporal_filter(smooth_alpha=0.53, smooth_delta=24.,
                                                      persistence_control=2)  # standard values

        # for some weird reason, these values have to be set in order for the image to appear with good lightning
        # for firmware >5.13, auto_exposure False & auto_white_balance True, below only auto_exposure True
        for sensor in self.profile.get_device().query_sensors():
            sensor.set_option(rs.option.enable_auto_exposure, True)
            # sensor.set_option(rs.option.enable_auto_white_balance, True)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def read(self):
        t = time.time()
        frames = self.pipe.wait_for_frames(timeout_ms=5000)
        tdiff = time.time() - t
        if tdiff > 0.2:
            print(f"wait for frames took {tdiff:.4f} seconds")
        image, depth, pointcloud, colors = [None] * 4

        timestamp = frames.get_timestamp()  # Retrieve the timestamp of the frames

        if self.rgb or self.rgb_pc:
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()

            if color_frame.is_video_frame():
                image = np.asarray(color_frame.get_data())

            if self.rgb_pc:
                self.pc.map_to(color_frame)

        if self.depth:
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()

            if depth_frame.is_depth_frame():
                depth = np.asanyarray(depth_frame.get_data())
                # clip max
                depth = np.where((depth > self.max_clipping_distance), 0., self.max_clipping_distance - depth)

                depth = (depth * (256. / self.max_clipping_distance)).astype(np.uint8)
                depth = depth[..., None]

        if self.pointcloud or self.rgb_pc:
            depth_frame = self.decimation_filter.process(frames.get_depth_frame())
            depth_frame = self.threshold_filter.process(depth_frame)
            depth_frame = self.temporal_filter.process(depth_frame)
            if depth_frame.is_depth_frame():
                points = self.pc.calculate(depth_frame)
                pointcloud = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

                if self.rgb_pc:
                    tex_coords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

                    # (u, v) to pixel indices
                    h, w, _ = image.shape
                    x = np.clip((tex_coords[:, 0] * w).astype(int), 0, w - 1)
                    y = np.clip((tex_coords[:, 1] * h).astype(int), 0, h - 1)
                    colors = image[y, x]
                    image = None # so it does not get returned

        if isinstance(image, np.ndarray) and isinstance(depth, np.ndarray):
            return True, np.concatenate((image, depth), axis=-1), timestamp
        elif isinstance(image, np.ndarray):
            return True, image, timestamp
        elif isinstance(depth, np.ndarray):
            return True, depth, timestamp
        elif isinstance(pointcloud, np.ndarray) and isinstance(colors, np.ndarray):
            return True, np.concatenate((pointcloud, colors[:, ::-1]), axis=-1), timestamp  # (w, h, p+c)
        elif isinstance(pointcloud, np.ndarray):
            return True, pointcloud, timestamp
        else:
            return False, None, 0

    def close(self):
        self.pipe.stop()
        self.cfg.disable_all_streams()
