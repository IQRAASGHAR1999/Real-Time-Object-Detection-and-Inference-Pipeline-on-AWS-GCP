"""
For custom training, see Ultralytics YOLOv8 docs:
https://docs.ultralytics.com/
Example:
    yolo train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
"""
from ultralytics import YOLO

def main():
    # Path to your data YAML file (update this as needed)
    data_path = '../data/data.yaml'  # Adjust path as needed

    # Initialize model (YOLOv8 nano, can be changed to yolov8s.pt, yolov8m.pt, etc.)
    model = YOLO('yolov8n.pt')  # or 'yolov8s.pt', etc.

    # Train the model
    results = model.train(
        data=data_path,
        epochs=50,
        imgsz=640,
        batch=16,           # Adjust batch size as needed
        project='../results',  # Output directory
        name='yolov8n_exp',    # Experiment name
        device=0 if model.device.type == 'cuda' else 'cpu'  # Use GPU if available
    )

    print("Training complete. Results saved in ../results/yolov8n_exp")

if __name__ == "__main__":
    main()
