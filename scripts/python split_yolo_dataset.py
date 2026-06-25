from pathlib import Path
import shutil
import re
from collections import Counter, defaultdict

# ==========================
# CONFIG
# ==========================

ROOT = Path(r"D:\FULL_DATA")
OUT_ROOT = Path(r"D:\FULL_DATA_SPLIT")

CLASS_NAMES = ["Human", "Vehicle", "Bicycle"]  # class_0, class_1, class_2

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# If True, deletes old OUT_ROOT and recreates it
CLEAR_OUTPUT = True


# ==========================
# SPLIT PLAN
# ==========================

# Full video folders copied directly
FULL_VIDEO_SPLIT = {
    "train": [
        ("DK_backyard", "v1"),
        ("DK_backyard", "v3"),
        ("THI_Grass", "v1"),
        ("THI_Grass", "v2"),
        ("THI_Bikepark", "v1"),
    ],

    "valid": [
        ("DK_backyard", "v5"),
        ("THI_Grass", "v3"),
        ("THI_Bikepark", "v3"),
    ],

    "test": [
        ("THI_Bikepark", "v4"),
        ("DK_parking", "v4"),
    ],
}

# Partial video split
# DK_parking/v3 first 80% -> train, last 20% -> valid
PARTIAL_VIDEO_SPLIT = [
    {
        "scene": "DK_parking",
        "video": "v3",
        "train_ratio": 0.80,
        "first_split": "train",
        "second_split": "valid",
    }
]


# ==========================
# HELPERS
# ==========================

def natural_key(path):
    """
    Sort frame files naturally:
    frame_2.jpg before frame_10.jpg
    """
    text = path.stem
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", text)]


def get_image_files(images_dir):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(images_dir.glob(f"*{ext}"))
        files.extend(images_dir.glob(f"*{ext.upper()}"))
    return sorted(files, key=natural_key)


def prepare_output_dirs():
    if OUT_ROOT.exists() and CLEAR_OUTPUT:
        shutil.rmtree(OUT_ROOT)

    for split in ["train", "valid", "test"]:
        (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_image_and_label(image_path, source_video_dir, split):
    scene_name = source_video_dir.parent.name
    video_name = source_video_dir.name

    labels_dir = source_video_dir / "labels"
    label_path = labels_dir / f"{image_path.stem}.txt"

    # Unique filename to avoid overwriting same frame names from different videos
    new_stem = f"{scene_name}_{video_name}__{image_path.stem}"

    new_image_path = OUT_ROOT / split / "images" / f"{new_stem}{image_path.suffix.lower()}"
    new_label_path = OUT_ROOT / split / "labels" / f"{new_stem}.txt"

    shutil.copy2(image_path, new_image_path)

    if label_path.exists():
        shutil.copy2(label_path, new_label_path)
    else:
        # YOLO allows empty label files for background images
        new_label_path.write_text("", encoding="utf-8")


def copy_full_video(scene, video, split):
    video_dir = ROOT / scene / video
    images_dir = video_dir / "images"
    labels_dir = video_dir / "labels"

    if not video_dir.exists():
        print(f"WARNING: missing video folder: {video_dir}")
        return

    if not images_dir.exists() or not labels_dir.exists():
        print(f"WARNING: missing images/labels in: {video_dir}")
        return

    image_files = get_image_files(images_dir)

    print(f"Copying FULL {scene}/{video} -> {split}: {len(image_files)} images")

    for image_path in image_files:
        copy_image_and_label(image_path, video_dir, split)


def copy_partial_video(scene, video, train_ratio, first_split, second_split):
    video_dir = ROOT / scene / video
    images_dir = video_dir / "images"
    labels_dir = video_dir / "labels"

    if not video_dir.exists():
        print(f"WARNING: missing video folder: {video_dir}")
        return

    if not images_dir.exists() or not labels_dir.exists():
        print(f"WARNING: missing images/labels in: {video_dir}")
        return

    image_files = get_image_files(images_dir)

    split_index = int(len(image_files) * train_ratio)

    first_part = image_files[:split_index]
    second_part = image_files[split_index:]

    print(f"Splitting PARTIAL {scene}/{video}:")
    print(f"  {first_split}: {len(first_part)} images")
    print(f"  {second_split}: {len(second_part)} images")

    for image_path in first_part:
        copy_image_and_label(image_path, video_dir, first_split)

    for image_path in second_part:
        copy_image_and_label(image_path, video_dir, second_split)


def write_yaml():
    yaml_path = OUT_ROOT / "data.yaml"

    names_text = ", ".join([f"'{name}'" for name in CLASS_NAMES])

    content = f"""train: train/images
val: valid/images
test: test/images

nc: {len(CLASS_NAMES)}
names: [{names_text}]
"""

    yaml_path.write_text(content, encoding="utf-8")
    print(f"\nSaved YAML: {yaml_path}")


def count_split_classes():
    print("\n==============================")
    print("FINAL SPLIT COUNTS")
    print("==============================")

    global_counter = Counter()

    for split in ["train", "valid", "test"]:
        labels_dir = OUT_ROOT / split / "labels"
        split_counter = Counter()
        image_count = len(get_image_files(OUT_ROOT / split / "images"))
        label_count = len(list(labels_dir.glob("*.txt")))

        for txt_file in labels_dir.glob("*.txt"):
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()

                    if len(parts) == 0:
                        continue

                    class_id = parts[0]
                    split_counter[class_id] += 1
                    global_counter[class_id] += 1

        print(f"\n{split.upper()}")
        print(f"Images: {image_count}")
        print(f"Labels: {label_count}")
        print(f"Total objects: {sum(split_counter.values())}")

        for class_id in sorted(split_counter.keys(), key=lambda x: int(x)):
            print(f"Class {class_id}: {split_counter[class_id]}")

    print("\nGLOBAL")
    print(f"Total objects: {sum(global_counter.values())}")
    for class_id in sorted(global_counter.keys(), key=lambda x: int(x)):
        print(f"Class {class_id}: {global_counter[class_id]}")


# ==========================
# RUN
# ==========================

prepare_output_dirs()

for split, folders in FULL_VIDEO_SPLIT.items():
    for scene, video in folders:
        copy_full_video(scene, video, split)

for item in PARTIAL_VIDEO_SPLIT:
    copy_partial_video(
        scene=item["scene"],
        video=item["video"],
        train_ratio=item["train_ratio"],
        first_split=item["first_split"],
        second_split=item["second_split"],
    )

write_yaml()
count_split_classes()

print("\nDone.")
print(f"Final dataset saved to: {OUT_ROOT}")