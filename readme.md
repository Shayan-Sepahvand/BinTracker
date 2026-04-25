
# Pre-requisits
# 🚀 CUDA-Based Project

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img src="https://img.shields.io/badge/Ubuntu-20.04-orange.svg">
  <img src="https://img.shields.io/badge/CUDA-12.4-green.svg">
  <img src="https://img.shields.io/badge/GPU-RTX%203060-76B900.svg">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg">
</p>

---

## 📖 Overview

This repository contains the implementation for the assignment.  
It is designed to run on GPU-accelerated systems using CUDA and provides a reproducible environment for experimentation.

---

## ⚙️ Requirements

| Component       | Version / Details           |
|-----------------|-----------------------------|
| OS              | Ubuntu 20.04                |
| Python          | 3.10.14                     |
| CUDA Toolkit    | 12.4                        |
| NVIDIA Driver   | 570+                        |
| GPU             | NVIDIA RTX 3060 (12GB VRAM) |



This package requires pyenv installation:

```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```
restart the termianl

```bash
pyenv install 3.10.14
sudo apt install -y python3-venv
```


---

## Question (1) - Part (a): Model Selection and Motivation

For this task, I selected the YOLOv10 architecture due to its strong balance between detection accuracy and real-time inference performance. A key advantage of YOLOv10 is its post-processing overhead reduction which leads to lower latency—critical for meeting the <100 ms/frame constraint. Initial experiments were conducted using COCO-pretrained weights, which include a rash can class. However, these models did not generalize well to the provided video. Specifically:

- The garbage bin was frequently misclassified as a "person" (Class ID 0)
- In several frames, the bin was not detected at all, particularly under:
  - Motion blur  
  - Partial occlusion (e.g., workers in front of the bin)  
  - Scale variations due to change in the distance from the optical center 

Given these limitations, I opted to ine-tune the model on a more relevant dataset. I used a public dataset Google Open Images v7 collected from the that closely matches the visual characteristics of the target environment. The "Waste container" class detections serves as the lables. This dataset provides:

- Real-world industrial settings  
- Similar lighting and background clutter  
- Adequate sample diversity for robust training  

To improve generalization and robustness, I applied the following techniques:

- **Spatial Augmentation**
  - Random scaling and translation which required attention on not to overscale the object.
  - Simulates varying camera distances and viewpoints  

- **Occlusion Modeling**
  - Random erasing / cutout  
  - Mimics real-world occlusions from workers or equipment  

- **Iterlative Learning**
  - The model has been trained several time in a consequtive manner, each time the wieghts of the previously trained model are used as the initial weights. 


These augmentations were critical for improving detection consistency under challenging conditions. Authors interested more details of the training.

### 📊 Performance Criteria

| Metric                     | Value                          |
|--------------------------|-------------------------------|
| Detection Rate           | 95.20%                        |
| Inference Time (GPU)     | 5.10 ms                       |
| Inference Time (CPU)     | 130.36 ms                        |
| IoU (vs Ground Truth)    | Seems OK (NO GT Provided) |

 
Furthermore, the bounding box coordinates are stored in the /results/1a.csv

<img src="./results/inference_perf.png" alt="Bin Trajectory" width="500"/>

---
<!-- ---================================================================================================================== -->

## Question 1 - Part B: Occlusion continuity

The detection images are included in the results directory. As appears in the following figures, the detection is maintained in presence of partial occlusion. 
This is thanks to the reach training dataset that included augemented partially occluded images.

The directory for the images after the execusion is /results/detection

<!-- ---================================================================================================================== -->

---

## Question 1 - Part C: Model choice justification

To evaluate the impact of our custom dataset and fine-tuning process, we compared the resulting model against the baseline pre-trained weights. The fine-tuned model demonstrates significant quantitative gains across all primary detection metrics.

|### Performance Comparison

| Metric | Baseline (Pre-trained) | Fine-Tuned (Ours) | Gain (Delta) |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.671 | 0.865 | **+0.194** |
| **Recall** | 0.550 | 0.852 | **+0.302** |
| **mAP@0.5** | 0.655 | 0.904 | **+0.249** |
| **mAP@0.5:0.95** | 0.410 | 0.655 | **+0.245** |


The dataset is splited into %80 training and %20 validation. The follwoings are several examples of these image.

| Early Training (Batch 2) | Late Training (Batch 408) |
| :---: | :---: |
| <img src="./results/train_batch2.jpg" width="400" /> | <img src="./results/train_batch408.jpg" width="400" /> |


<!-- ---================================================================================================================== -->
---
## Question 2 - Part A: Distance estimation from bounding box

```python
def build_extrinsic(cam_h: float, tilt_rad: float):

    """
    Construct the extrinsics of the camera using the give fixed static tf.
    Returns:
      R and t
    """
    # --- Axis mapping: camera -> world ---
    # each of the columns is the result of the projection of the camera axis on the world coordinates, e.g. the camera x axis lies on the world -y axis.
    Rs = np.array([
        [ 0,  0,  1],   # world_X = cam_Z
        [-1,  0,  0],   # world_Y = -cam_X
        [ 0, -1,  0],   # world_Z = -cam_Y
    ], dtype=np.float64)


    # --- This is a basic roation around the camera local x-axis, which is given to be -15 degrees. 
    c, s = np.cos(tilt_rad), np.sin(tilt_rad)
    Rx = np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c],
    ], dtype=np.float64)

    # The rotation matrix that transforms from camera to the spatial (world frame).
    # The rotation from the tilted camera {frame 2} to untilted camera {frame 1} is Rx,
    # The rotation from the untilted camera {frame 1} to the world frame {frame 0} is static here.
    # The rotation from the the tilted camera frame {frame 2} to the world frame {frame 0} is Rs @ Rx.
    # The world postion vector should be described w.r.t world frame, thus, Pworld = Rs @ Rx @ Pcam + t 
    R = Rs @ Rx
    t = np.array([0.0, 0.0, cam_h]) #this is the postion vec. that starts from the world and ends at the camera optical centre

    return R, t
```


Once the extrinsics are found, it is possible to reconstruct the world coordinates up to a scale due to as the camera projection is an affine transformation on the homogenous coordinates. Accordingly, the world coordinates are obtained up to a sclase using the following formulation:

the first thing that happens in the pin-hole camera model.
---
<!-- ---================================================================================================================== -->

## Question 2 - Part B: position in camera frame

The requested file can be found in the following directory after the execution: /resutls/2b.csv. Here are some examples of the output.

```csv
frame_id,timestamp_ms,x_cam,y_cam,z_cam,confidence
0000,0,-0.04,0.20,3.02,0.85
0001,33,-0.03,0.20,2.99,0.84
0002,67,-0.03,0.20,2.93,0.82
0003,100,-0.03,0.20,2.93,0.81
0004,133,-0.02,0.20,2.90,0.84
0005,167,-0.01,0.20,2.92,0.86
0006,200,0.00,0.20,2.90,0.86
0007,234,0.01,0.20,2.87,0.85
0008,267,0.01,0.20,2.87,0.85
0009,300,0.01,0.21,2.86,0.78
0010,334,0.02,0.21,2.86,0.80
```
---
<!-- ---================================================================================================================== -->

## Question 2 - Part C: Transform to world frame

This also have been achieved. The file is stored in the /resutls/2c.csv. The follwoings examplify the world coordinate frame as a timestamped csv.

```csv
frame_id,t_ms,x_cam,y_cam,z_cam,x_world,y_world,z_world,conf
0000,0,-0.04,0.20,3.02,2.86,0.04,0.38,0.85
0001,33,-0.03,0.20,2.99,2.83,0.03,0.38,0.84
0002,67,-0.03,0.20,2.93,2.78,0.03,0.40,0.82
0003,100,-0.03,0.20,2.93,2.78,0.03,0.40,0.81
0004,133,-0.02,0.20,2.90,2.75,0.02,0.40,0.84
0005,167,-0.01,0.20,2.92,2.77,0.01,0.40,0.86
0006,200,0.00,0.20,2.90,2.75,-0.00,0.40,0.86
0007,234,0.01,0.20,2.87,2.72,-0.01,0.41,0.85
0008,267,0.01,0.20,2.87,2.72,-0.01,0.41,0.85
0009,300,0.01,0.21,2.86,2.71,-0.01,0.41,0.78
```
---
<!-- ---================================================================================================================== -->

## Question 2 - Part D: Error analysis vs. ground-truth waypoints

The waypoint.json data is extracted using this function:

```python
def load_waypoints(path: str):
    """
    Load 2D ground-truth pixel coordinates and their corresponding frame index from waypoint.json
    Returns:
        An np array containing (3 by 4)  [pixel_u, pixel_v, order, approx_frame]
    """
    with open(path, "r") as f:
        data = json.load(f)
    markers = data["markers"]
    waypoints = []
    for i, marker in enumerate(markers):
        u = int(marker["pixel_u"])
        v = int(marker["pixel_v"])
        frame_idx = int(marker["approx_frame"])
        waypoints.append([u, v, i, frame_idx]) 
    return np.array(waypoints, dtype=np.float64)

```


Next, the estimated ground-truth stops are extracted and given as:

A [2.2227, -1.4127e-08, 0.77307] (m)
B [3.957, -0.67358, 0.60103] (m)
C [3.7013, 0.38184, 0.78677] (m)


The computed RMSE between the GT and the estimated stop points are:

```bash
=============================================
[run.sh] RMSE per axis (m): x=0.15, y=1.06, z=0.44
=============================================
```

---

<!-- ---================================================================================================================== -->

## Question 3 - Part A: Live coordinate stream

This has been completed and a few of the generated output are as follows:

```bash
frame [297] bin @ world (3.81, 0.68, 0.19) m conf=0.77 dt=148ms
frame [298] bin @ world (3.73, 0.65, 0.21) m conf=0.80 dt=147ms
frame [299] bin @ world (3.81, 0.63, 0.19) m conf=0.82 dt=151ms
frame [300] bin @ world (3.81, 0.61, 0.19) m conf=0.79 dt=151ms
frame [301] bin @ world (3.81, 0.61, 0.19) m conf=0.79 dt=149ms
frame [302] bin @ world (3.73, 0.57, 0.20) m conf=0.62 dt=148ms
frame [303] bin @ world (3.65, 0.54, 0.22) m conf=0.75 dt=148ms
frame [304] bin @ world (3.66, 0.54, 0.22) m conf=0.73 dt=147ms
frame [305] OCCLUDED - last known(3.66, 0.54, 0.22) m  age=1fr
frame [306] OCCLUDED - last known(3.66, 0.54, 0.22) m  age=2fr
frame [307] bin @ world (3.53, 0.49, 0.25) m conf=0.59 dt=145ms
frame [308] bin @ world (3.69, 0.49, 0.21) m conf=0.76 dt=145ms
frame [309] bin @ world (3.71, 0.46, 0.21) m conf=0.77 dt=144ms

```
---

<!-- ---================================================================================================================== -->

## Question 3 - Part B: Trajectory visualisation

A top-down view of the bin trajecory in the world frame along with the three stop points, start and stop positions are included and stored in results/trajectory.png.

<img src="./results/trajectory.png" alt="Bin Trajectory" width="500"/>


<!-- ---================================================================================================================== -->

## Question 3 - Part C: Kalman filter smoothing

The state vector includes the the [x y z vx vy vz], yet the observation vector is limited to the postiion.The xyz positino graph of the measured vs the filtered signals  are illustrated in the follwoing figure. 

<img src="./results/trajectory_kf_vs_raw.png" alt="KF Results vs. raw measurements" width="500"/>


The result of the jitter reduction in the stationary readings are as follows:

```bash
=============================================
JITTER REDUCTION (Frames 250 to 280)
=============================================
Axis            | Raw Std    | Filt Std   | Reduction
---------------------------------------------
X (Forward)     | 0.0416     | 0.0267     | 35.8%
Y (Left)        | 0.1090     | 0.1093     | -0.3%
Z (Up)          | 0.0101     | 0.0052     | 48.6%
=============================================
```
---
<!-- ---================================================================================================================== -->

## Question 3 - Part D: Edge deployment notes (Jetson Orin NX)

TBC.
---


<!-- ---================================================================================================================== -->

## Demo screen recording

TBC.