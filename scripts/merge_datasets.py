import os
import shutil

input_folder = "merged_DK_parking"
output_folder = "DK_parking_final"
os.makedirs(output_folder, exist_ok=True)

images = []
for f in os.listdir(input_folder):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
        images.append(f)

images.sort()

for i, img in enumerate(images):
    ext = os.path.splitext(img)[1]
    new_name = f"DK_parking_frame_{i+1:04d}{ext}"
    shutil.copy(
        os.path.join(input_folder, img),
        os.path.join(output_folder, new_name)
    )
    print(f"{img} -> {new_name}")

print(f"\nTotal {len(images)} images renamed -> {output_folder}")
