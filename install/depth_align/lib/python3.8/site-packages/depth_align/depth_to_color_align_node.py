#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import tf2_ros
from message_filters import Subscriber, ApproximateTimeSynchronizer


def quat_to_rot(qx, qy, qz, qw):
    # Returns 3x3 rotation matrix
    # Assumes quaternion is normalized (TF usually is).
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    R = np.array([
        [1.0 - 2.0*(yy + zz),       2.0*(xy - wz),       2.0*(xz + wy)],
        [      2.0*(xy + wz), 1.0 - 2.0*(xx + zz),       2.0*(yz - wx)],
        [      2.0*(xz - wy),       2.0*(yz + wx), 1.0 - 2.0*(xx + yy)]
    ], dtype=np.float32)
    return R


class DepthToColorAlign(Node):
    """
    Publishes depth registered to color camera frame:
      /camera/aligned_depth_to_color/image_raw (32FC1 meters)
      /camera/aligned_depth_to_color/camera_info (copied from color camera_info)
    """

    def __init__(self):
        super().__init__('depth_to_color_align')

        # ---- Parameters (edit defaults if your topic names differ) ----
        self.declare_parameter('depth_image_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_info_topic',  '/camera/depth/camera_info')
        self.declare_parameter('color_info_topic',  '/camera/color/camera_info')
        self.declare_parameter('aligned_depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('aligned_info_topic',  '/camera/aligned_depth_to_color/camera_info')

        self.declare_parameter('queue_size', 10)
        self.declare_parameter('slop_sec', 0.05)
        self.declare_parameter('tf_timeout_sec', 0.2)
        self.declare_parameter('min_depth_m', 0.1)
        self.declare_parameter('max_depth_m', 10.0)

        self.bridge = CvBridge()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers (no need for RGB image; only color CameraInfo)
        depth_img_sub = Subscriber(self, Image, self.get_parameter('depth_image_topic').value)
        depth_info_sub = Subscriber(self, CameraInfo, self.get_parameter('depth_info_topic').value)
        color_info_sub = Subscriber(self, CameraInfo, self.get_parameter('color_info_topic').value)

        queue_size = int(self.get_parameter('queue_size').value)
        slop = float(self.get_parameter('slop_sec').value)
        self.sync = ApproximateTimeSynchronizer(
            [depth_img_sub, depth_info_sub, color_info_sub],
            queue_size=queue_size,
            slop=slop
        )
        self.sync.registerCallback(self.cb)

        # Publishers
        self.pub_depth = self.create_publisher(Image, self.get_parameter('aligned_depth_topic').value, 10)
        self.pub_info  = self.create_publisher(CameraInfo, self.get_parameter('aligned_info_topic').value, 10)

        self.get_logger().info("Depth->Color alignment node started.")

    def cb(self, depth_msg: Image, depth_info: CameraInfo, color_info: CameraInfo):
        # Frames
        depth_frame = depth_info.header.frame_id or depth_msg.header.frame_id
        color_frame = color_info.header.frame_id

        if not depth_frame or not color_frame:
            self.get_logger().warn("Missing frame_id in camera_info/header; cannot align.")
            return

        # Lookup TF depth->color at this timestamp
        tf_timeout = rclpy.duration.Duration(seconds=float(self.get_parameter('tf_timeout_sec').value))
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame=color_frame,
                source_frame=depth_frame,
                time=depth_msg.header.stamp,
                timeout=tf_timeout
            )
        except Exception as e:
            # TF can be momentarily unavailable; keep it quiet-ish
            self.get_logger().warn(f"TF lookup failed ({depth_frame} -> {color_frame}): {e}")
            return

        # Build transform matrix
        t = tf.transform.translation
        q = tf.transform.rotation
        R = quat_to_rot(q.x, q.y, q.z, q.w)  # 3x3
        T = np.array([t.x, t.y, t.z], dtype=np.float32)  # 3

        # Intrinsics
        fx_d, fy_d = float(depth_info.k[0]), float(depth_info.k[4])
        cx_d, cy_d = float(depth_info.k[2]), float(depth_info.k[5])

        fx_c, fy_c = float(color_info.k[0]), float(color_info.k[4])
        cx_c, cy_c = float(color_info.k[2]), float(color_info.k[5])

        # Output size = color size
        Wc, Hc = int(color_info.width), int(color_info.height)

        # Convert depth image to numpy
        try:
            depth_np = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # Handle encodings: 16UC1(mm) or 32FC1(m)
        if depth_msg.encoding == '16UC1':
            depth_m = depth_np.astype(np.float32) * 0.001  # mm -> m
        elif depth_msg.encoding == '32FC1':
            depth_m = depth_np.astype(np.float32)
        else:
            # Try anyway
            depth_m = depth_np.astype(np.float32)

        Hd, Wd = depth_m.shape[:2]

        # Create pixel grid for depth image
        u = np.arange(Wd, dtype=np.float32)
        v = np.arange(Hd, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        Z = depth_m
        min_z = float(self.get_parameter('min_depth_m').value)
        max_z = float(self.get_parameter('max_depth_m').value)

        valid = np.isfinite(Z) & (Z > min_z) & (Z < max_z)

        if not np.any(valid):
            # Publish empty aligned depth anyway (optional)
            aligned = np.zeros((Hc, Wc), dtype=np.float32)
            self.publish(aligned, color_info, depth_msg.header.stamp)
            return

        uu = uu[valid]
        vv = vv[valid]
        Z  = Z[valid]

        # Backproject to depth camera 3D
        Xd = (uu - cx_d) * Z / fx_d
        Yd = (vv - cy_d) * Z / fy_d
        Pd = np.stack([Xd, Yd, Z], axis=0)  # 3xN

        # Transform to color camera: Pc = R*Pd + T
        Pc = (R @ Pd) + T.reshape(3, 1)
        Xc3, Yc3, Zc3 = Pc[0], Pc[1], Pc[2]

        # Only points in front of color camera
        front = Zc3 > 1e-6
        Xc3, Yc3, Zc3 = Xc3[front], Yc3[front], Zc3[front]

        if Zc3.size == 0:
            aligned = np.zeros((Hc, Wc), dtype=np.float32)
            self.publish(aligned, color_info, depth_msg.header.stamp)
            return

        # Project into color image
        uc = (fx_c * (Xc3 / Zc3) + cx_c)
        vc = (fy_c * (Yc3 / Zc3) + cy_c)

        ui = np.round(uc).astype(np.int32)
        vi = np.round(vc).astype(np.int32)

        inside = (ui >= 0) & (ui < Wc) & (vi >= 0) & (vi < Hc)
        ui = ui[inside]
        vi = vi[inside]
        zc = Zc3[inside]

        # Z-buffer into aligned depth image
        aligned = np.zeros((Hc, Wc), dtype=np.float32)
        # Initialize with +inf for min reduce, then set 0 for missing
        tmp = np.full((Hc, Wc), np.inf, dtype=np.float32)
        # For repeated indices, keep minimum depth
        np.minimum.at(tmp, (vi, ui), zc)
        aligned = np.where(np.isfinite(tmp), tmp, 0.0).astype(np.float32)

        self.publish(aligned, color_info, depth_msg.header.stamp)

    def publish(self, aligned_depth_m: np.ndarray, color_info: CameraInfo, stamp):
        # Publish depth (32FC1 meters) aligned to color
        out = self.bridge.cv2_to_imgmsg(aligned_depth_m, encoding='32FC1')
        out.header.stamp = stamp
        out.header.frame_id = color_info.header.frame_id

        info = CameraInfo()
        info = color_info  # copy
        info.header.stamp = stamp
        info.header.frame_id = color_info.header.frame_id

        self.pub_depth.publish(out)
        self.pub_info.publish(info)


def main():
    rclpy.init()
    node = DepthToColorAlign()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
