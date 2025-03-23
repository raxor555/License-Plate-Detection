# License Plate Detection using YOLOv8

## 📌 Project Overview
This project focuses on detecting license plates in images and videos using the **YOLOv8** model. It is implemented in **Google Colab** and **Jupyter Notebook**, leveraging **Ultralytics YOLOv8** for object detection.

---
## 🚀 Features
- **Train YOLOv8** on a custom **License Plate Dataset**
- **Visualize Training Progress** (Loss and mAP Metrics)
- **Test on Images** with Bounding Boxes
- **Run Inference on Videos** for License Plate Detection
- **Video Compression** using FFmpeg for optimized output

---
## 🛠 Requirements
Ensure you have the following installed:

```bash
pip install ultralytics opencv-python matplotlib seaborn pandas PyYAML
```
Alternatively, run the project in **Google Colab**, where dependencies are installed automatically.

---
## 📂 Dataset Setup
The dataset should be structured as follows:
```
License-Plate-Dataset/
│── archive/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   ├── dataset.yaml
```

The `dataset.yaml` file is generated automatically and should contain:
```yaml
train: /path/to/train
val: /path/to/val

nc: 1
names: ['license_plate']
```

---
## 🔥 Training the Model
Train YOLOv8 on the dataset using the following command:
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(
    data="/content/drive/MyDrive/License-Plate-Dataset/archive/dataset.yaml",
    epochs=50,
    imgsz=640,
    lr0=0.0005,
    batch=32,
    lrf=0.1,
    augment=True
)
```

---
## 📊 Visualizing Training Progress
After training, visualize the results:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/content/runs/detect/train/results.csv")

plt.figure(figsize=(12, 5))
sns.set_style("whitegrid")

# Plot Training & Validation Loss
plt.subplot(1, 2, 1)
plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", color="red")
plt.plot(df["epoch"], df["val/box_loss"], label="Validation Box Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

# Plot mAP@50
plt.subplot(1, 2, 2)
plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@50", color="green")
plt.xlabel("Epochs")
plt.ylabel("mAP")
plt.title("Mean Average Precision (mAP)")
plt.legend()

plt.tight_layout()
plt.show()
```

---
## 📷 Running Inference on Images
Test the model on sample images:
```python
import random, cv2, os, glob
import matplotlib.pyplot as plt

all_images = glob.glob("/content/drive/MyDrive/License-Plate-Dataset/archive/images/train/*.jpg")
test_images = random.sample(all_images, min(6, len(all_images)))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for img_path, ax in zip(test_images, axes.flatten()):
    results = model(img_path)
    result_img = results[0].plot()
    img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.axis("off")
    ax.set_title(os.path.basename(img_path))
plt.tight_layout()
plt.show()
```

---
## 🎥 Running Inference on Videos
```python
import cv2, os

model_path = "/content/runs/detect/train/weights/best.pt"
model = YOLO(model_path)

input_video = "/content/drive/MyDrive/License-Plate-Dataset/sample_video.mp4"
output_video = "/content/runs/output_video.mp4"

cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, verbose=False)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = f"Plate {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Detection completed!")
```

---
## 🎯 Compressing Video Output
```python
import subprocess

compressed_video = "/content/runs/output_video_compressed.mp4"
ffmpeg_command = [
    "ffmpeg", "-i", output_video, "-vcodec", "libx264", "-crf", "28", "-preset", "fast", compressed_video
]
subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print("✅ Compression done!")
```

---
## 🖥 Display Video in Jupyter Notebook
```python
from IPython.display import HTML
from base64 import b64encode

def play_video(file_path, width=800):
    with open(file_path, "rb") as video_file:
        video_base64 = b64encode(video_file.read()).decode()
    return HTML(f"<video width='{width}' controls autoplay loop>"
                f"<source src='data:video/mp4;base64,{video_base64}' type='video/mp4'>"
                f"Your browser does not support the video tag.</video>")

play_video(compressed_video, width=1000)
```

---
## 📜 License
This project is licensed under the **MIT License**.

---
## 🎯 Conclusion
This project successfully detects **license plates** using **YOLOv8**, visualizes results, and processes videos efficiently. 🚀 Happy Coding!

