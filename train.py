import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.random as foor

# 1. Download the dataset
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train", # This refers to the OpenImages source split
    label_types=["detections"],
    classes=["Waste container"],
    max_samples=500
)

# 2. Split the local dataset: 80% for training, 20% for validation
foor.random_split(dataset, {"train": 0.8, "val": 0.2})

# 3. Export with the split
dataset.export(
    export_dir="./dumpster_dataset",
    dataset_type=fo.types.YOLOv5Dataset,
    classes=["Waste container"],
    split="train" # Exports the 80%
)

dataset.export(
    export_dir="./dumpster_dataset",
    dataset_type=fo.types.YOLOv5Dataset,
    classes=["Waste container"],
    split="val"   # Exports the 20%
)

# ... (after your random_split call) ...

model = YOLO("./weights/best.pt")

if torch.cuda.is_available():
    print("GPU found! Training on CUDA.")
else:
    print("GPU not found. Training on CPU.")

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