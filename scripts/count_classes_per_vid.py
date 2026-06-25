from pathlib import Path
from collections import Counter, defaultdict
import csv

ROOT = Path(r"D:\FULL_DATA")

video_folders = []

for scene_dir in ROOT.iterdir():
    if not scene_dir.is_dir():
        continue

    for video_dir in scene_dir.iterdir():
        if not video_dir.is_dir():
            continue

        images_dir = video_dir / "images"
        labels_dir = video_dir / "labels"

        if images_dir.exists() and labels_dir.exists():
            video_folders.append(video_dir)

video_folders = sorted(video_folders)

print("Found video folders:")
for folder in video_folders:
    print(folder)

global_counter = Counter()
scene_counter = defaultdict(Counter)
rows = []


def sort_class_ids(class_ids):
    return sorted(class_ids, key=lambda x: int(x) if x.isdigit() else x)


def print_table(rows, columns, title=None):
    if title:
        print("\n" + title)
        print("=" * len(title))

    if not rows:
        print("No data found.")
        return

    col_widths = {}

    for col in columns:
        max_width = len(col)
        for row in rows:
            max_width = max(max_width, len(str(row.get(col, ""))))
        col_widths[col] = max_width

    separator = "+-" + "-+-".join("-" * col_widths[col] for col in columns) + "-+"

    print(separator)
    print("| " + " | ".join(col.ljust(col_widths[col]) for col in columns) + " |")
    print(separator)

    for row in rows:
        print("| " + " | ".join(str(row.get(col, "")).ljust(col_widths[col]) for col in columns) + " |")

    print(separator)


for video_dir in video_folders:
    images_dir = video_dir / "images"
    labels_dir = video_dir / "labels"

    scene_name = video_dir.parent.name
    video_name = video_dir.name

    video_counter = Counter()

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        image_files.extend(images_dir.glob(ext))

    label_files = list(labels_dir.glob("*.txt"))

    empty_label_files = 0

    for txt_file in label_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) == 0:
            empty_label_files += 1
            continue

        for line in lines:
            parts = line.strip().split()

            if len(parts) == 0:
                continue

            class_id = parts[0]

            video_counter[class_id] += 1
            scene_counter[scene_name][class_id] += 1
            global_counter[class_id] += 1

    row = {
        "scene": scene_name,
        "video": video_name,
        "images": len(image_files),
        "label_files": len(label_files),
        "empty_labels": empty_label_files,
        "total_objects": sum(video_counter.values()),
    }

    for class_id in video_counter:
        row[f"class_{class_id}"] = video_counter[class_id]

    rows.append(row)


# Get all class IDs found globally
all_class_ids = sort_class_ids(global_counter.keys())

# Add missing class columns as 0
for row in rows:
    for class_id in all_class_ids:
        row.setdefault(f"class_{class_id}", 0)


video_columns = [
    "scene",
    "video",
    "images",
    "label_files",
    "empty_labels",
    "total_objects",
] + [f"class_{class_id}" for class_id in all_class_ids]

print_table(rows, video_columns, title="COUNTS PER VIDEO")


# Scene summary table
scene_rows = []

for scene_name, counter in scene_counter.items():
    row = {
        "scene": scene_name,
        "total_objects": sum(counter.values()),
    }

    for class_id in all_class_ids:
        row[f"class_{class_id}"] = counter[class_id]

    scene_rows.append(row)

scene_columns = ["scene", "total_objects"] + [f"class_{class_id}" for class_id in all_class_ids]

print_table(scene_rows, scene_columns, title="TOTAL PER SCENE")


# Global summary table
global_row = {
    "dataset": "GLOBAL",
    "total_objects": sum(global_counter.values()),
}

for class_id in all_class_ids:
    global_row[f"class_{class_id}"] = global_counter[class_id]

global_columns = ["dataset", "total_objects"] + [f"class_{class_id}" for class_id in all_class_ids]

print_table([global_row], global_columns, title="GLOBAL TOTAL")


# Save results to CSV
fieldnames = video_columns
csv_path = ROOT / "class_counts_per_video.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nCSV saved to: {csv_path}")