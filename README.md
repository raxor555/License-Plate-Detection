# License Plate Detection using YOLOv8

## Overview
This project focuses on detecting license plates using the **YOLOv8 model**. It is implemented in **Google Colab** and **Jupyter Notebook**, providing an efficient deep learning-based solution for real-time license plate recognition.

## Features
- **Uses YOLOv8**: A state-of-the-art object detection model.
- **Google Colab & Jupyter Notebook**: Enables easy training and inference.
- **Real-time Detection**: Can process live feeds or video files.
- **Custom Dataset Support**: Can be trained on custom datasets for different regions.

## Installation
### Clone the Repository
```bash
git clone https://github.com/raxor555/License-Plate-Detection.git
cd license-plate-detection
```
### Install Dependencies
```bash
pip install ultralytics opencv-python numpy matplotlib
```

## Training the Model
1. **Download the YOLOv8 model**:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   ```
2. **Train the model on a custom dataset**:
   ```python
   model.train(data="dataset.yaml", epochs=50, imgsz=640)
   ```

## Dataset Preparation
Ensure that your dataset follows the YOLO format:
- **Images** stored in `images/train/` and `images/val/`.
- **Labels** stored in `labels/train/` and `labels/val/`.
- `dataset.yaml` should include:
  ```yaml
  train: path/to/train/images
  val: path/to/val/images
  nc: 1
  names: ['license_plate']
  ```

## Results
- The model outputs **bounding boxes** around detected license plates.
- **Evaluation metrics**: mAP, precision, recall, and F1-score.

## Acknowledgments
- **Ultralytics YOLOv8**: https://github.com/ultralytics/ultralytics
- **Google Colab** for cloud-based execution.

## License
This project is open-source under the MIT License. Feel free to contribute and improve!
