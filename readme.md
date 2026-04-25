
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

| Component        | Version / Details              |
|-----------------|------------------------------|
| OS              | Ubuntu 20.04                |
| Python          | 3.10+                       |
| CUDA Toolkit    | 12.4                        |
| NVIDIA Driver   | 570+                        |
| GPU             | NVIDIA RTX 3060 (12GB VRAM) |


---

## Question (1) - Part (a): Model Selection and Motivation

For this task, I selected the YOLOv10 architecture due to its strong balance between detection accuracy and real-time inference performance. A key advantage of YOLOv10 is its NMS-free training paradigm, which reduces post-processing overhead and leads to lower latency—critical for meeting the <100 ms/frame constraint. Initial experiments were conducted using COCO-pretrained weights, which include a rash can class. However, these models did not generalize well to the provided video. Specifically:

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
<!-- ---================================================================================================================== -->

## Question 1 - Part B: Occlusion continuity

The detection images are included in the results directory. As appears in the following figures, the detection is maintained in presence of partial occlusion. 
This is thanks to the reach training dataset that included augemented partially occluded images.

The directory for the images after the execusion is /results/detection

<!-- ---================================================================================================================== -->

---

## Question 1 - Part C: Model choice justification

Include the training graphs here. 

<!-- ---================================================================================================================== -->

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

<!-- ---================================================================================================================== -->