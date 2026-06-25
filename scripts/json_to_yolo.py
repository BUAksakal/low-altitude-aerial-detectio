"""
Convert X-AnyLabeling JSON annotations to YOLO .txt format.

X-AnyLabeling saves corrections as .json files. This script reads those
files and writes YOLO-format .txt label files alongside each image.

Output format per line: <class_id> <x_center> <y_center> <width> <height>
All coordinates are normalized to [0, 1].

Usage:
    python scripts/json_to_yolo.py --frames_dir data/frames
"""

import argparse
import json
from pathlib import Path

CLASS_NAMES = ["person", "car"]


def convert(frames_dir: Path) -> None:
    images = sorted(frames_dir.rglob("*.jpg"))
    print(f"Found {len(images)} images. Converting JSON → YOLO .txt ...")

    n_imgs, n_boxes = 0, 0
    for img_path in images:
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text())
        w, h = data["imageWidth"], data["imageHeight"]
        lines = []
        for shape in data.get("shapes", []):
            if shape["shape_type"] != "rectangle":
                continue
            label = shape["label"]
            cls_id = CLASS_NAMES.index(label) if label in CLASS_NAMES else 0
            pts = shape["points"]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = ((min(xs) + max(xs)) / 2) / w
            cy = ((min(ys) + max(ys)) / 2) / h
            bw = (max(xs) - min(xs)) / w
            bh = (max(ys) - min(ys)) / h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_boxes += 1
        img_path.with_suffix(".txt").write_text("\n".join(lines))
        n_imgs += 1

    print(f"Done. {n_imgs} files written, {n_boxes} total boxes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", type=Path, default=Path("data/frames"))
    args = parser.parse_args()
    convert(args.frames_dir)
