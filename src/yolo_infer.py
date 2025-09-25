from ultralytics import YOLO
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True)
args = parser.parse_args()

model = YOLO('yolov8n.pt')
results = model(args.image)
results[0].show()  # Opens a window with detections; save or display as needed
