import os
from ultralytics import YOLO


def main():
    # Load the pre-trained nano architecture (best for edge devices/drones)
    model = YOLO("yolov8n.pt")
    print("Starting optimized training pipeline with automated Early Stopping...")

    model.train(
        data="./unified_dataset_80_10_10/data.yaml",
        epochs=150,
        imgsz=640,
        batch=16,       # Change to 8 or 4 if your system runs out of memory (VRAM)
        device=0,       # Change to device='cpu' if you do not have an NVIDIA GPU
        workers=4,
        project="Ariel_Project",
        name="drone_model_final",
        save=True,      # Tells YOLO to save the absolute best weights as 'best.pt'
        plots=True,

        # --- Automated Best Epoch Selection ---
        patience=15,    # EARLY STOPPING: If the model's validation accuracy (mAP)
                        # does not improve for 15 straight epochs, training stops
                        # automatically and locks in the best historical epoch.

        # --- Advanced Aerial Optimizations ---
        box=7.5,        # Enforces strict edge-to-edge bounding constraints
        cls=1.5,        # Higher weight on exact object classifications
        cos_lr=True,    # Smooth learning decay profile
    )

    print("Training complete! Loading the absolute BEST epoch to export...")

    # Reload the absolute best weights explicitly before running export
    best_model_path = os.path.join("Ariel_Project", "drone_model_final", "weights", "best.pt")
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        best_model.export(format="onnx")
        print("Best epoch ONNX model exported successfully!")
    else:
        model.export(format="onnx")


if __name__ == "__main__":
    main()