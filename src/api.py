from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2

app = Flask(__name__)
model = YOLO('yolov8n.pt')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    arr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    results = model(img)
    boxes = results[0].boxes.xyxy.tolist()
    return jsonify({'boxes': boxes})

if __name__ == "__main__":
    app.run(debug=True)
