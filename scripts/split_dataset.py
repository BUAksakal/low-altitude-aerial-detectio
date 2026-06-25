"""
Split extracted frames into train / val / test sets.

Each video is assigned entirely to one split to prevent data leakage
(frames from the same video are nearly identical, so mixing them across
splits would inflate validation/test metrics).

Default mapping (edit SPLITS below to match your videos):
    train → video1
    val   → video3
    test  → video5

Usage:
    python scripts/split_dataset.py --frames_dir data/frames --dataset_dir data/splits
"""

import argparse
import shutil
from pathlib import Path

SPLITS = {
    "train": "video1",
    "val":   "video3",
    "test":  "video5",
}


def split(frames_dir: Path, dataset_dir: Path) -> None:
    for split_name, video_name in SPLITS.items():
        img_out = dataset_dir / "images" / split_name
        lbl_out = dataset_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        images = sorted((frames_dir / video_name).glob("*.jpg"))
        for img_path in images:
            shutil.copy(img_path, img_out / img_path.name)
            txt_path = img_path.with_suffix(".txt")
            dest_txt = lbl_out / txt_path.name
            if txt_path.exists():
                shutil.copy(txt_path, dest_txt)
            else:
                dest_txt.write_text("")

        print(f"{split_name:5s}: {len(images)} images ({video_name})")

    print(f"\nDataset ready at: {dataset_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir",  type=Path, default=Path("data/frames"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("data/splits"))
    args = parser.parse_args()
    split(args.frames_dir, args.dataset_dir)
