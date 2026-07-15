from ultralytics import YOLO

if __name__ == '__main__':
    # Load a YOLOv8n PyTorch model
    model = YOLO("test_models/vision_transformer_final.pth")

    # Export the model
    model.export(format="openvino")