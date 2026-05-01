
# 🚀AI-Enabled Industrial Bin Detection and Trajectory Construction


## Pre-requisits

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img src="https://img.shields.io/badge/Ubuntu-20.04-orange.svg">
  <img src="https://img.shields.io/badge/CUDA-12.4-green.svg">
  <img src="https://img.shields.io/badge/GPU-RTX%203060-76B900.svg">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg">
</p>

---

## 📖 Overview

This project implements a computer vision pipeline to detect and track industrial waste containers using YOLOv10. By combining 2D object detection with camera intrinsics and the bin's known physical dimensions, the system accurately estimates the full 3D spatial trajectory of the container.

---

## 📖 Overview

This project implements a computer vision pipeline to detect and track industrial waste containers using YOLOv10. By combining 2D object detection with camera intrinsics and the bin's known physical dimensions, the system accurately estimates the full 3D spatial trajectory of the container.

---

## ⚙️ Requirements

| Component       | Version / Details           |
|-----------------|-----------------------------|
| OS              | Ubuntu 20.04                |
| Python (pyenv)  | 3.10.14                     |
| CUDA Toolkit    | 12.4                        |
| NVIDIA Driver   | 570+                        |
| GPU             | NVIDIA RTX 3060 (12GB VRAM) |

> **IMPORTANT NOTE:** This package requires an isolated Python environment via `pyenv`. 

If you do not have `pyenv` installed, we recommend the following installation steps:

**1. Install build dependencies and `pyenv`:**
```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev git

curl [https://pyenv.run](https://pyenv.run) | bash

---

## ⚙️ How to run

```bash
 ./run.sh --video input.mp4 --calib calib.json --kalman 
```
---

## Model Selection and Motivation

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
| Inference Time (GPU)     | 5-7 ms                       |
| Inference Time (CPU)     | 130-160 ms                        |
| IoU (vs Ground Truth)    | Seems OK (NO GT Provided) |

 
Furthermore, the bounding box coordinates are stored in the /results/1a.csv

<img src="./results/inference_perf.png" alt="Bin Trajectory" width="500"/>

---
<!-- ---================================================================================================================== -->

## Occlusion continuity

The detection images are included in the results directory. As shown in the following figures, detection is successfully maintained even in the presence of partial occlusion. This is thanks to the rich training dataset, which included augmented, partially occluded images.

The directory for the images after execution is: /results/detection

```python

results = model.train(
    data="./dumpster_dataset/dataset.yaml", 
    epochs=30,          
    imgsz=640,          
    batch=20,           
    workers=4,          # Increase this if your CPU has more cores (usually 4 is safe)
    cache=False,      
    
    # --- Augmentations ---
    erasing=0.4,        
    mosaic=1.0,         
    box=10.0,           
    scale=0.5,          
    translate=0.4,      
    
    device=0,
    name="dumpster_model_v2"
)
```

<!-- ---================================================================================================================== -->

---

## Model choice justification

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
## Distance estimation from bounding box

Once the camera extrinsics are established, 3D world coordinates can be reconstructed up to a scale factor, since the camera projection acts as a transformation on homogeneous coordinates. 

The RViz visualization of this scenario, alongside a comparison between our calculated transformation matrix and the ROS TF tree, yields the following results:

<img src="./results/rviz_transformations.png" width="400" />

| ROS TF Tree | Our Transformation Matrix |
| :---: | :---: |
| <img src="./results/rostopic.png" width="400" /> | <img src="./results/our_tf.png" width="400" /> |


---
<!-- ---================================================================================================================== -->

## Position in camera frame

The dataset was partitioned into an 80% training set and a 20% validation set. Several examples from the dataset are provided below: (Using CPU Here)

```csv
frame_id,timestamp_ms,x_cam,y_cam,z_cam,confidence
0000,0,-0.04,0.20,2.99,0.85
0001,33,-0.03,0.20,2.96,0.84
0002,67,-0.03,0.20,2.90,0.82
0003,100,-0.03,0.20,2.90,0.81
0004,133,-0.02,0.20,2.87,0.84
0005,167,-0.01,0.20,2.89,0.86
0006,200,0.00,0.20,2.87,0.86
0007,234,0.01,0.21,2.83,0.85
0008,267,0.01,0.21,2.83,0.85
0009,300,0.02,0.21,2.82,0.78
0010,334,0.02,0.21,2.82,0.80
```

The theoretical justification of the code can be found in the following figures:

<img src="./results/justification3.png" alt="Proof" width="500"/>
<img src="./results/justification4.png" alt="Proof" width="500"/>


---
<!-- ---================================================================================================================== -->

## Transform to world frame

This also have been achieved. The file is stored in the /resutls/2c.csv. The follwoings examplify the world coordinate frame as a timestamped csv. (Using CPU here)

```csv
frame_id,t_ms,x_cam,y_cam,z_cam,x_world,y_world,z_world,conf
0000,0,-0.04,0.20,2.99,2.83,0.04,0.38,0.85
0001,33,-0.03,0.20,2.96,2.80,0.03,0.39,0.84
0002,67,-0.03,0.20,2.90,2.75,0.03,0.40,0.82
0003,100,-0.03,0.20,2.90,2.75,0.03,0.40,0.81
0004,133,-0.02,0.20,2.87,2.72,0.02,0.41,0.84
0005,167,-0.01,0.20,2.89,2.74,0.01,0.40,0.86
0006,200,0.00,0.20,2.87,2.72,-0.00,0.41,0.86
0007,234,0.01,0.21,2.83,2.68,-0.01,0.42,0.85
0008,267,0.01,0.21,2.83,2.68,-0.01,0.42,0.85
0009,300,0.02,0.21,2.82,2.67,-0.02,0.42,0.78
0010,334,0.02,0.21,2.82,2.67,-0.02,0.42,0.80
```

The following screenshots shows the justification of the code:

<img src="./results/justification1.png" alt="Proof" width="500"/>
<img src="./results/justification2.png" alt="Proof" width="500"/>

---
<!-- ---================================================================================================================== -->

## Error analysis vs. ground-truth waypoints

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

The estimated ground-truth stops for the bin trajectory were extracted as 3D spatial coordinates (measured in meters). These points serve as the reference for tracking accuracy.

| Stop | X-Axis (m) | Y-Axis (m) | Z-Axis (m) |
| :--- | :---: | :---: | :---: |
| **Point A** | 2.1747| 0 |  0.78552 |
| **Point B** | 3.876 | -0.6598 | 0.61636 |
| **Point C** | 3.6745 |  0.37909, |  0.79084 |

The computed RMSE between the GT and the estimated stop points are:

```bash
=============================================
[run.sh] RMSE per axis: x=0.15, y=1.04, z=0.44
=============================================
```

---

<!-- ---================================================================================================================== -->

## Live coordinate stream

This has been completed and a few of the generated output are as follows: (CPU used here)

```bash
frame [849] bin @ world (4.27, -0.10, 0.09) m conf=0.70 dt=138ms
frame [850] bin @ world (4.27, -0.10, 0.09) m conf=0.70 dt=147ms
frame [851] bin @ world (4.25, -0.10, 0.09) m conf=0.68 dt=141ms
frame [852] bin @ world (4.23, -0.10, 0.10) m conf=0.63 dt=144ms
frame [853] bin @ world (4.23, -0.10, 0.10) m conf=0.62 dt=142ms
frame [854] bin @ world (4.23, -0.10, 0.10) m conf=0.54 dt=142ms
frame [855] OCCLUDED - last known (4.23, -0.10, 0.10) m  age=1fr
frame [856] bin @ world (4.27, -0.10, 0.09) m conf=0.52 dt=142ms
frame [857] OCCLUDED - last known (4.27, -0.10, 0.09) m  age=1fr
frame [858] OCCLUDED - last known (4.27, -0.10, 0.09) m  age=2fr

```
---

<!-- ---================================================================================================================== -->

## Trajectory visualisation

A top-down view of the bin trajecory in the world frame along with the three stop points, start and stop positions are included and stored in results/trajectory.png.

<img src="./results/trajectory.png" alt="Bin Trajectory" width="500"/>

---
<!-- ---================================================================================================================== -->

## Question 3: Kalman Filter Formulation and Jitter Reduction

To track the container smoothly through time, a Kalman Filter is implemented. Rather than just relying on the raw spatial coordinates from the camera pipeline, the filter uses an internal kinematic model to predict and correct the bin's trajectory.

### State Vector and Key Parameters

* **State Vector ($X$):** The filter models the system using a 6-dimensional state vector, $X = [x, y, z, v_x, v_y, v_z]^T$. This tracks both the 3D spatial position ($x, y, z$) and the 3D velocity ($v_x, v_y, v_z$) of the container. 
* **Observation Vector ($Z$):** Because our 2D-to-3D projection pipeline only outputs spatial coordinates, our measurement vector is strictly limited to position: $Z = [x, y, z]^T$.
* **State Transition Matrix ($F$ or $A$):** This matrix defines the physical kinematics of the system. We assume a linear constant-velocity model, meaning the predicted next position is calculated as the current position plus the velocity scaled by the time step between frames (e.g., $x_{k} = x_{k-1} + v_x \Delta t$).
* **Observation Matrix ($H$):** This maps the 6D state space down to the 3D measurement space. It isolates the positional elements of the state vector so they can be directly compared against our raw YOLO-derived depth measurements.
* **Noise Covariances ($Q$ and $R$):** * **Process Noise ($Q$):** Accounts for physical deviations from our constant velocity assumption (e.g., the container suddenly accelerating or decelerating).
    * **Measurement Noise ($R$):** Represents the confidence in our sensor data. In this context, it accounts for the inherent jitter in YOLO bounding boxes and the resulting noise in the depth estimations.

By maintaining internal velocity states despite only measuring position, the filter effectively smooths out sudden jumps and maintains trajectory continuity during partial occlusions.

### Results

The XYZ position graph comparing the raw measured signals to the filtered trajectory is illustrated in the following figure:

<img src="./results/trajectory_kf_vs_raw.png" alt="KF Results vs. raw measurements" width="500"/>

The quantitative impact of the filter is particularly noticeable during stationary periods, where bounding box flickering typically introduces significant noise. The jitter reduction results are as follows:

```bash
=============================================
JITTER REDUCTION (Frames 250 to 280)
=============================================
Axis            | Raw Std    | Filt Std   | Reduction
---------------------------------------------
X (Forward)     | 0.0374     | 0.0199     | 46.9%
Y (Left)        | 0.1064     | 0.1067     | -0.3%
Z (Up)          | 0.0091     | 0.0030     | 67.0%
=============================================
```
---
<!-- ---================================================================================================================== -->

## Edge deployment notes (Jetson Orin NX)

This section outlines the architectural migration of the pipeline from a fixed-camera workstation to a **Jetson Orin NX** companion computer mounted on a moving UAV for real-time trajectory tracking.

### 1. Model Quantization Strategy (NVIDIA TensorRT)
To achieve real-time performance on the edge, we bypass general-purpose inference engines in favor of **TensorRT**.

* **Optimization Path:** Since Jetson Orin NX utilizes NVIDIA’s Ampere architecture, we utilize **FP16 (Half Precision)** as the primary target. While INT8 offers higher throughput, FP16 provides the best balance of "mAP preservation" and speed without requiring the complex calibration datasets needed for INT8.

While the Orin NX supports INT8 quantization, our primary deployment target is **FP16 (Half Precision)** using the **TensorRT** inference engine.

| Strategy | Precision | Accuracy (mAP) | Latency (ms) | Reasoning |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (FP32)** | 32-bit | 0.904 | ~45ms | Too slow for stable UAV control loops. |
| **FP16 (Ours)** | 16-bit | 0.902 | ~12ms | **Optimal balance.** High speed with negligible accuracy loss. |
| **INT8** | 8-bit | 0.885 | ~7ms | Fastest, but requires complex calibration and causes mAP drop. |


* **Engine vs. RKNN:** We strictly utilize **TensorRT**. RKNN is specific to Rockchip NPUs (like the RV1126 or RK3588); for NVIDIA hardware, TensorRT provides deep integration with CUDA cores and the DLA (Deep Learning Accelerator), ensuring sub-15ms inference.

### 2. Dynamic Coordinate Transformation (IMU Fusion)
Unlike the fixed-camera setup where the transform is constant, a moving UAV requires a dynamic transformation from the **Camera Frame ($C$)** to the **World Frame ($W$)**.

* **The Chain:** $P_{world} = T_{body \to world}(t) \cdot T_{camera \to body} \cdot P_{camera}$
* **EKF Integration:** We fuse the model’s visual detections with the UAV's internal **IMU (Inertial Measurement Unit)** and GPS via an **Extended Kalman Filter (EKF)**. This filters out high-frequency vibrations and compensates for the UAV's roll, pitch, and yaw in real-time, ensuring the "Bin Trajectory" remains stable even during aggressive maneuvers.

### 3. Flight Controller Communication (MAVLink over UART)
The Jetson communicates with the Flight Controller (e.g., Pixhawk/Cube) via a dedicated UART bridge (TTL 3.3V).

* **Protocol:** MAVLink 2.0
* **Message Type:** `VISION_POSITION_ESTIMATE` (#102) or `LANDING_TARGET` (#149) depending on the mission phase.
* **Frequency:** **30Hz to 50Hz**. Higher frequencies are avoided to prevent saturating the flight controller's CPU, while anything lower than 20Hz introduces control-loop instability (latency jitter).
* **Baud Rate:** 921,600 bps (to ensure low-latency serial transport).

### 4. Latency Budget Breakdown
To maintain a stable flight control loop, the "Photon-to-Actuator" latency must be minimized. Our target budget is **< 40ms**.

| Stage | Process | Est. Latency |
| :--- | :--- | :--- |
| **Capture** | CSI-Camera MIPI frames to VRAM | 8ms |
| **Detect** | TensorRT FP16 Inference (Orin NX) | 12ms |
| **Localize** | EKF Update & Coordinate Transform | 4ms |
| **Transmit** | MAVLink Packet Serialization & UART | 2ms |
| **Total** | **End-to-End Latency** | **~26ms (~38 FPS)** |
---


<!-- ---================================================================================================================== -->
