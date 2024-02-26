# License Plate Detection with YOLOv5

This project uses a custom trained YOLOv5 object detection model to detect license plates in images and videos. It provides options to filter license plates by color (red or blue) and displays the results in real-time using OpenCV. When you run script, you will be required to pick a color (red or blue). This color will represent your opponent's color. This project draws a box around your opponents plate as a means of identification.

# Demo Video

*Watchout for the red boxes on the robot plates*

<p align="center">
  <img src="https://github.com/IJAMUL1/RTDETR-Tracking-Detection/assets/60096099/b6619da3-78a5-4a82-aa35-f67df451874f" alt="Your Image Description" width="600">
</p>


## Prerequisites

- Python 3.x
- PyTorch
- OpenCV

## Installation

- Clone this repository to your local machine
```
bash
git clone https://github.com/IJAMUL1/License-Plate-Object-detection-CV-Application.git
```

- Download the YOLOv5 model and place them in the root directory.

## Note
you will have to upload your pre trained deep learning model to detect plates. This code will help identify only plates of specific colors (red or blue).

## Usage

Run the script named: plate_detection_code.py








