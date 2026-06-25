"""
Pre-labeling script for person detection.

Uses YOLOv8x (COCO pretrained) to automatically generate bounding box
annotations for the 'person' class on extracted frames. Output .txt files
follow YOLO format: <class_id> <x_center> <y_center> <width> <height>
(all values normalized 0–1).

These auto-labels are intended as a starting point; manual correction in
X-AnyLabeling or similar tool is required before training.

Usage:
    python scripts/pre_label.py --frames_dir data/frames --conf 0.15
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

COCO_PERSON_CLASS = 0  # person index in COCO


def pre_label(frames_dir: Path, conf: float) -> None:
    model = YOLO("yolov8x.pt")

    images = sorted(frames_dir.rglob("*.jpg"))
    if not images:
        print(f"No .jpg files found under {frames_dir}")
        return

    print(f"Found {len(images)} images. Running pre-labeling (conf={conf})...")

    total_boxes = 0
    for i, img_path in enumerate(images, 1):
        results = model(img_path, conf=conf, verbose=False)[0]
        lines = []
        for box in results.boxes:
            if int(box.cls) != COCO_PERSON_CLASS:
                continue
            x, y, w, h = box.xywhn[0].tolist()
            lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        img_path.with_suffix(".txt").write_text("\n".join(lines))
        total_boxes += len(lines)
        if i % 100 == 0:
            print(f"  {i}/{len(images)} processed...")

    (frames_dir / "classes.txt").write_text("person\n")
    print(f"\nDone. {len(images)} images → {total_boxes} person boxes written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label person class with YOLOv8x")
    parser.add_argument(
        "--frames_dir",
        type=Path,
        default=Path("data/frames"),
        help="Root directory containing extracted frame images",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold (lower = more detections, more false positives)",
    )
    args = parser.parse_args()
    pre_label(args.frames_dir, args.conf)
