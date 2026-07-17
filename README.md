# YOLO-DBN (Diverse Branch Network inside YOLO)

This repository contains the implementation of **YOLO-DBN**, integrating Diverse Branch Networks/Blocks into the YOLO architecture to improve feature representation while maintaining inference efficiency through structural reparameterization.

---

## 🚀 Live Demo & Web App

You can test our trained model directly through our web application:
👉 **[Mulberry Leaf Disease Detector Web App](https://mulberry.streamlit.app/)**

---

## 💻 Quick Start with Google Colab

You can run the training and evaluation workflows directly in your browser using Google Colab:

👉 **[YOLOv10m-DBN Training Notebook](https://colab.research.google.com/drive/1rdFFJnAjjFdDngo3Gnh3qyyBSJ8BL7pk?usp=sharing)**
👉 **[YOLOv10m-DBN Generate Test Results Notebook](https://colab.research.google.com/drive/1Z456qiDNsnn58N0xdRBPZCblZelhWdFB?usp=sharing)**

---

## Local Execution Guide

Follow the steps below to set up the environment, download the dataset, configure the model, and execute either training or testing locally.

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

Download the preprocessed and augmented publicly available mulberry leaf disease dataset from Roboflow. The dataset download script loads the API key from the `.env` file included in the repository.

#### a. Install Required Libraries

```bash
pip install roboflow python-dotenv
```

#### b. Set Up Environment Variables (`.env`)

Make sure a `.env` file exists in the root directory of your project with the API key:

#### c. Download the Mulberry Leaf Disease Dataset

Run the following Python script to download the dataset in YOLO format:

```python
import os
from dotenv import load_dotenv
from roboflow import Roboflow

# Load credentials from .env
load_dotenv()

# Initialize Roboflow
api_key = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=api_key)

project = rf.workspace("village").project("mulberry-nit0r")
version = project.version(2)
dataset = version.download("yolov11")
```

---

## Training a Model from Scratch

If you want to train a new model from scratch, proceed with the following steps:

### 4. Download Custom Model Configuration

Download the customized YAML YOLOv10-DBN file:

```bash
wget -q -O yolov10-DBN.yaml https://raw.githubusercontent.com/N786h/YOLO-DBN/main/ultralytics/cfg/models/v10/yolov10-DBN.yaml
```

### 5. Download Pre-trained Weights

Download the pre-trained weights for transfer learning:

```bash
wget -q -O yolov10m.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10m.pt
```

### 6. Execute Training

Execute the following training script in Python to initialize the model with pre-trained weights and begin training:

```python
from ultralytics import YOLO

# Load the model configuration and transfer weights
model = YOLO('yolov10m-DBN.yaml').load('yolov10m.pt')

# Start training
result = model.train(data='Mulberry-2/data.yaml', epochs=100, batch=16, amp=False)
```

---

## Evaluating and Generating Test Results

If you want to evaluate an already trained model using our pre-trained weights:

### 7. Download Pre-trained Weights

Download the pre-trained weights (`best.pt`) directly from our repository:

```bash
wget -q -O best.pt https://github.com/N786h/YOLO-DBN/raw/main/models/best.pt
```

### 8. Run Test Set Evaluation

Evaluate the model on the test dataset split using the command-line interface:

```bash
yolo task=detect mode=val model=best.pt data=Mulberry-2/data.yaml split=test
```
