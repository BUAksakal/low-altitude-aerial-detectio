"""
Fine-tune YOLOv8 on the custom low-altitude aerial dataset.

Starts from COCO-pretrained weights and fine-tunes for the target classes
(person, car) detected in drone footage at 3-9 m altitude.

Usage:
    python scripts/train.py --data data/splits/dataset.yaml --device mps
    python scripts/train.py --data data/splits/dataset.yaml --device 0   # CUDA
    python scripts/train.py --data data/splits/dataset.yaml --device cpu
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train(data: Path, model_size: str, epochs: int, device: str) -> None:
    model = YOLO(f"yolov8{model_size}.pt")
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=device,
        patience=20,
        project="runs",
        name="low_altitude_detector",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   type=Path, default=Path("data/splits/dataset.yaml"))
    parser.add_argument("--model",  type=str,  default="n", choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 model size: n=nano, s=small, m=medium")
    parser.add_argument("--epochs", type=int,  default=100)
    parser.add_argument("--device", type=str,  default="mps",
                        help="Device: mps (Apple Silicon), 0 (CUDA GPU), cpu")
    args = parser.parse_args()
    train(args.data, args.model, args.epochs, args.device)
