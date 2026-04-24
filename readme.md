
Question (1) - Part (a)

Question 1 - Part A: Model Selection and Motivation
Enhanced Version:
For this task, I utilized the YOLOv10 architecture. The motivation behind this choice is its state-of-the-art balance between inference speed and accuracy, specifically its "NMS-free" training which reduces latency.

During initial testing, I observed that the pre-trained COCO weights frequently misclassified the garbage bin as a "person" (Class ID 0) or ignored it entirely. To resolve this, I performed fine-tuning on a targeted dataset. My objective was to move beyond generic detection and specifically account for environmental factors present in the test data, such as motion blur, partial occlusion by workers, and significant scale variations.

I selected the [Insert Dataset Name Here] public industrial dumpster dataset because its imagery closely mirrors the target environment and provides sufficient volume for convergence. To improve model robustness, I implemented a data augmentation pipeline focusing on:

Spatial Augmentation: Scaling and translation to handle varying camera distances.

Occlusion Modeling: Introducing random erasing to simulate bins partially blocked by personnel or equipment.

The efficacy of this approach is demonstrated in the processed output video, where the tracking remains consistent and the bounding box remains stable despite the challenging conditions.

For the inference time the average and the time are plotted in RES/inference_perf.png which shows and average around 7ms<90ms

A gpu with the following specification including model, driver, VRAM, and the CUDA version are used to complete this assignment:

```bash
nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```
```bash
nvidia-smi
Fri Apr 24 10:32:58 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060        Off |   00000000:0B:00.0  On |                  N/A |
|  0%   46C    P8             13W /  170W |     604MiB /  12288MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```















Question 1 - Part B: Training Strategy and Implementation
Enhanced Version:
The core strategy involved enriching the training distribution through heavy geometric and occlusion-based augmentations. This ensures the model generalizes well to "edge cases" where the bin is partially off-screen or obscured.

The training was configured as follows:

results = model.train(
    data="./dumpster_dataset/dataset.yaml", 
    epochs=30,          
    imgsz=640,          
    batch=20,           
    workers=1,        
    cache=False,      
    
    # --- OCCLUSION & BOUNDARY ROBUSTNESS ---
    erasing=0.4,        # Randomly masks parts of the image to simulate occlusions.
    mosaic=1.0,         # Combines 4 images into one to help the model learn small-scale objects.
    box=10.0,           # Increases the weight of the box loss for more precise coordinates.
    
    # --- SCALE & POSITION ROBUSTNESS ---
    scale=0.5,          # Randomly scales images by ±50% to handle varying camera-to-bin distances.
    translate=0.4,      # Shifts the image by 40% to handle bins partially cut off at the frame edges.
    
    device=0,
    name="dumpster_model_v2"
)
