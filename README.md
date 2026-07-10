# YOLO-DBN (Diverse Branch Network inside YOLO)

This repository contains the implementation of **YOLO-DBN**, integrating Diverse Branch Networks/Blocks into the YOLO architecture to improve feature representation while maintaining inference efficiency through structural reparameterization.

## Getting Started / Execution Guide

Follow these steps to set up the environment, download the dataset, obtain the model configuration, and execute training and evaluation.

### 1. Set Up Virtual Environment

Create a virtual environment and activate it to manage dependencies in isolation:

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows (Command Prompt):
venv\Scripts\activate
# On Windows (PowerShell):
.\venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate
```

### 2. Install the Modified Ultralytics Library

Install the custom version of the `ultralytics` library containing the YOLO-DBN implementations from our GitHub repository:

```bash
pip install git+https://github.com/N786h/YOLO-DBN.git
```

### 3. Download the Dataset

Download the preprocessed and augmented publicly available mulberry leaf disease dataset from Roboflow:

#### a. Install Roboflow
```bash
pip install roboflow
```

#### b. Download the Mulberry Leaf Disease Dataset
Run the following Python script to download the dataset in YOLO format:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="H2A65FaRZ08GLCs3xt5A")
project = rf.workspace("village").project("mulberry-nit0r")
version = project.version(2)
dataset = version.download("yolov11")
```

### 4. Download Custom Model Configuration

Download the customized YAML YOLOv10-DBN file:

```bash
wget -q -O yolov10-DBN.yaml https://raw.githubusercontent.com/N786h/YOLO-DBN/main/ultralytics/cfg/models/v10/yolov10-DBN.yaml
```

### 5. Execute Training

Execute the following training script in Python to initialize the model with pre-trained weights and begin training:

```python
import os
from ultralytics import YOLO

# Asset 'bus.jpg' which is required for the AMP check
assets_path = '/usr/local/lib/python3.12/dist-packages/ultralytics/assets/'
os.makedirs(assets_path, exist_ok=True)
if not os.path.exists(os.path.join(assets_path, 'bus.jpg')):
    # In Jupyter/Colab use !wget, or run via terminal without '!'
    os.system(f"wget -q -O {assets_path}bus.jpg https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg")

# Load the model configuration and transfer weights
model = YOLO('/content/yolov10m-DBN.yaml').load('yolov10m.pt')

# Start training
result = model.train(data='/content/Mulberry-2/data.yaml', epochs=100, batch=16)
```

### 6. Evaluate Test Set Results

Evaluate the trained model on the test dataset split using the command-line interface:

```bash
yolo task=detect mode=val model=./runs/detect/train/weights/best.pt data=Mulberry-2/data.yaml split=test
```
