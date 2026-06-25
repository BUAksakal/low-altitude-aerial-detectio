"""
Label visualization script for quick quality inspection.

Draws bounding boxes from YOLO .txt annotation files onto a random sample
of images and saves the results to an output directory. Useful for verifying
pre-label quality before manual correction.

Usage:
    python scripts/visualize_labels.py --frames_dir data/frames --n 24 --out data/preview
"""

import argparse
import random
from pathlib import Path

import cv2


def visualize(frames_dir: Path, n: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(frames_dir.rglob("*.jpg"))
    if not images:
        print(f"No .jpg files found under {frames_dir}")
        return

    sample = random.sample(images, min(n, len(images)))
    print(f"Visualizing {len(sample)} random frames → {out_dir}")

    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        txt = img_path.with_suffix(".txt")
        box_count = 0
        if txt.exists():
            for line in txt.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                box_count += 1
        out_name = f"[{box_count}persons]_{img_path.name}"
        cv2.imwrite(str(out_dir / out_name), img)

    print(f"Done. {len(sample)} preview images saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO labels on sample frames")
    parser.add_argument("--frames_dir", type=Path, default=Path("data/frames"))
    parser.add_argument("--n", type=int, default=24, help="Number of random frames to sample")
    parser.add_argument("--out", type=Path, default=Path("data/preview"))
    args = parser.parse_args()
    visualize(args.frames_dir, args.n, args.out)
