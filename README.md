# Depth to Color Alignment (ROS2)

This repository contains a ROS2 node that aligns depth images to the color camera frame and publishes the aligned depth stream.

It enables synchronized RGB and depth data which can be used for robotics perception, computer vision, and machine learning pipelines.

---

## Repository Structure

```
align_ws/
│
├── src/                     # ROS2 source packages
├── build/                   # Build artifacts
├── install/                 # Installed ROS2 workspace
├── log/                     # ROS2 logs
├── imgs/                    # Images used in README
│   ├── Post 1.png
│   └── Post 2.png
└── README.md
```

## Running the Node

First source the ROS2 workspace:

```bash
source install/setup.bash
```

Run the alignment node:

```bash
ros2 run depth_align depth_to_color_align
```

---

## Recording Data from the Node

To record aligned depth and color streams into a ROS2 bag file:

```bash
ros2 bag record \
  /camera/color/image_raw \
  /camera/color/camera_info \
  /camera/aligned_depth_to_color/image_raw \
  /camera/aligned_depth_to_color/camera_info
```

This command records synchronized RGB and aligned depth data for later playback and analysis.

![Aligned Depth Output](imgs/Post%201.png)

![Visualization](imgs/Post%202.png)