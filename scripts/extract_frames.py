import cv2
from pathlib import Path


VIDEO_BASE = Path("/Users/macbookairm2/Desktop/data")

PIXEL_CHANGE_RATIO = 0.10
INTENSITY_THRESHOLD = 25
MIN_FRAME_GAP = 10


def extract_frames(video_path):
    # Her video için kendi klasörünü oluştur
    video_folder = VIDEO_BASE / video_path.stem
    images_dir = video_folder / "images"
    annotations_dir = video_folder / "annotations"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n📹 {video_path.stem} | FPS: {fps} | Total: {total_frames}")

    saved_count = 0
    frame_index = 0
    last_saved_gray = None
    last_saved_index = -999999

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        save_frame = False

        if last_saved_gray is None:
            save_frame = True
        else:
            diff = cv2.absdiff(gray, last_saved_gray)
            changed_pixels = diff > INTENSITY_THRESHOLD
            change_ratio = changed_pixels.sum() / changed_pixels.size
            if change_ratio >= PIXEL_CHANGE_RATIO:
                if frame_index - last_saved_index >= MIN_FRAME_GAP:
                    save_frame = True

        if save_frame:
            output_path = images_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            last_saved_gray = gray
            last_saved_index = frame_index
            saved_count += 1
            print(f"  Saved frame {frame_index} | total: {saved_count}")

        frame_index += 1

    cap.release()
    print(f"✅ {video_path.stem}: {saved_count} frames → {images_dir}")

for video_file in sorted(VIDEO_BASE.glob("*.MP4")):
    extract_frames(video_file)

print("\n🎉 Done!")
