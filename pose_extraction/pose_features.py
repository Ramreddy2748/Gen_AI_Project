import os
import json

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import cv2
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

DATA_DIR = "data"
OUTPUT_DIR = "data/pose_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = "pose_landmarker_lite.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading pose landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


def process_sequence(seq_path):
    frames = sorted(os.listdir(seq_path))

    # take 1 frame every 20 frames
    sampled_frames = frames[::20]

    features = []

    base_options = mp_tasks.BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=mp_tasks.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False
    )
    try:
        pose = mp_vision.PoseLandmarker.create_from_options(options)
    except RuntimeError as exc:
        raise RuntimeError(
            "PoseLandmarker could not initialize. If you are running in a "
            "restricted or headless environment, rerun the script in a normal "
            "local terminal so MediaPipe can create its graphics context."
        ) from exc

    for f in sampled_frames:
        img_path = os.path.join(seq_path, f)

        if not f.endswith(".png"):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        results = pose.detect(mp_image)
        landmarks = results.pose_landmarks[0] if results.pose_landmarks else None

        if landmarks:
            lm = landmarks  # Assuming single person

            features.append({
                # Positions
                "hip_x": lm[23].x,
                "hip_y": lm[23].y,
                
                "shoulder_x": lm[11].x,
                "shoulder_y": lm[11].y,
                
                "nose_x": lm[0].x,
                "nose_y": lm[0].y,

                # Derived feature (VERY IMPORTANT)
                "torso_vertical_diff": abs(lm[11].y - lm[23].y)
            })

    close_method = getattr(pose, "close", None)
    if close_method is not None:
        close_method()
    return features


def main():
    for category in ["falls", "adl"]:
        cat_path = os.path.join(DATA_DIR, category)

        for seq in os.listdir(cat_path):
            seq_path = os.path.join(cat_path, seq)

            # 🔥 HANDLE FALL (nested)
            if category == "falls":
                inner_dirs = [
                    d for d in os.listdir(seq_path)
                    if "cam0-rgb" in d
                ]

                if not inner_dirs:
                    continue

                seq_path = os.path.join(seq_path, inner_dirs[0])

            # ADL is already direct
            if os.path.isdir(seq_path):
                feats = process_sequence(seq_path)

                out_file = os.path.join(OUTPUT_DIR, f"{seq}.json")

                with open(out_file, "w") as f:
                    json.dump(feats, f, indent=2)

                print("Processed:", seq)


if __name__ == "__main__":
    main()
