import argparse
import http.cookiejar
import json
import os
import re
import urllib.request
from urllib.parse import parse_qs, urlparse
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = Path("pose_landmarker_lite.task")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}


def ensure_model() -> None:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return
    print("Downloading pose landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def is_google_drive_url(value: str) -> bool:
    parsed = urlparse(value)
    return "drive.google.com" in parsed.netloc


def extract_drive_file_id(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "id" in query and query["id"]:
        return query["id"][0]

    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    raise ValueError("Could not extract a Google Drive file ID from the provided URL.")


def build_url_opener() -> urllib.request.OpenerDirector:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    return opener


def choose_download_name(file_id: str, response_url: str, response_bytes: bytes) -> str:
    html = response_bytes.decode("utf-8", "ignore")
    title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = re.sub(r"\s*-\s*Google Drive\s*$", "", title_match.group(1)).strip()
        if title:
            return title

    parsed = urlparse(response_url)
    query = parse_qs(parsed.query)
    if "filename" in query and query["filename"]:
        return query["filename"][0]

    return f"{file_id}.mp4"


def find_confirm_token(html: str) -> str:
    patterns = [
        r'confirm=([0-9A-Za-z_-]+)',
        r'name="confirm" value="([0-9A-Za-z_-]+)"',
        r'"confirm":"([0-9A-Za-z_-]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            return match.group(1)
    return ""


def stream_download(
    opener: urllib.request.OpenerDirector,
    url: str,
    destination: Path,
) -> Path:
    with opener.open(url, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "")
        data = response.read()

    if "text/html" in content_type.lower():
        html = data.decode("utf-8", "ignore")
        lowered = html.lower()
        if "unauthorized" in lowered or "you need access" in lowered or "sign in" in lowered:
            raise PermissionError(
                "This Google Drive file is not publicly accessible. "
                "Set sharing to 'Anyone with the link' or download it locally first."
            )

        confirm_token = find_confirm_token(html)
        if confirm_token:
            separator = "&" if "?" in url else "?"
            return stream_download(opener, f"{url}{separator}confirm={confirm_token}", destination)

        raise RuntimeError("Google Drive returned an HTML page instead of the video file.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as file_obj:
        file_obj.write(data)
    return destination


def download_google_drive_file(url: str, download_dir: Path) -> Path:
    file_id = extract_drive_file_id(url)
    opener = build_url_opener()
    preview_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    with opener.open(preview_url, timeout=60) as response:
        preview_bytes = response.read()
        preview_url_after_redirects = response.geturl()
        content_type = response.headers.get("Content-Type", "")

    if "text/html" not in content_type.lower():
        guessed_name = f"{file_id}.mp4"
        destination = download_dir / guessed_name
        with open(destination, "wb") as file_obj:
            file_obj.write(preview_bytes)
        return destination

    download_name = choose_download_name(file_id, preview_url_after_redirects, preview_bytes)
    suffix = Path(download_name).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        download_name = f"{Path(download_name).stem or file_id}.mp4"

    destination = download_dir / download_name
    return stream_download(opener, preview_url, destination)


def resolve_video_input(video_input: str, output_dir: Path) -> Path:
    if is_google_drive_url(video_input):
        downloads_dir = output_dir / "downloads"
        print("Downloading video from Google Drive...")
        downloaded_path = download_google_drive_file(video_input, downloads_dir)
        print(f"Downloaded Drive file to: {downloaded_path}")
        return downloaded_path

    video_path = Path(video_input)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    return video_path


def build_landmarker() -> mp_vision.PoseLandmarker:
    base_options = mp_tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH),
        delegate=mp_tasks.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def landmark_to_dict(index: int, landmark: object) -> Dict[str, float]:
    return {
        "index": index,
        "x": float(landmark.x),
        "y": float(landmark.y),
        "z": float(landmark.z),
        "visibility": float(getattr(landmark, "visibility", 0.0)),
        "presence": float(getattr(landmark, "presence", 0.0)),
    }


def detect_pose(
    landmarker: mp_vision.PoseLandmarker,
    frame_bgr: object,
) -> Dict[str, object]:
    rgb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return {
            "pose_detected": False,
            "landmarks": [],
            "world_landmarks": [],
        }

    landmarks = [landmark_to_dict(i, lm) for i, lm in enumerate(result.pose_landmarks[0])]
    world_landmarks = [
        landmark_to_dict(i, lm) for i, lm in enumerate(result.pose_world_landmarks[0])
    ] if result.pose_world_landmarks else []

    return {
        "pose_detected": True,
        "landmarks": landmarks,
        "world_landmarks": world_landmarks,
    }


def draw_pose(frame_bgr: object, frame_result: Dict[str, object]) -> object:
    if not frame_result["pose_detected"]:
        return frame_bgr

    frame_height, frame_width = frame_bgr.shape[:2]
    normalized_landmarks = []
    for landmark in frame_result["landmarks"]:
        normalized_landmarks.append(
            landmark_pb2.NormalizedLandmark(
                x=landmark["x"],
                y=landmark["y"],
                z=landmark["z"],
                visibility=landmark["visibility"],
                presence=landmark["presence"],
            )
        )

    landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=normalized_landmarks)

    annotated = frame_bgr.copy()
    mp.solutions.drawing_utils.draw_landmarks(
        annotated,
        landmark_list,
        mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
    )

    cv2.putText(
        annotated,
        f"Frame size: {frame_width}x{frame_height}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def get_video_metadata(capture: cv2.VideoCapture) -> Dict[str, float]:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_seconds = (frame_count / fps) if fps > 0 else 0.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }


def process_video(
    video_path: Path,
    output_dir: Path,
    sample_every_n: int,
    max_frames: int,
    save_annotated: bool,
) -> Path:
    frames_dir = output_dir / "sampled_frames"
    annotated_dir = output_dir / "annotated_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    metadata = get_video_metadata(capture)
    results: List[Dict[str, object]] = []
    frame_index = 0
    saved_count = 0

    landmarker = build_landmarker()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            should_sample = frame_index % sample_every_n == 0
            if should_sample:
                frame_name = f"frame_{frame_index:06d}.jpg"
                frame_path = frames_dir / frame_name
                cv2.imwrite(str(frame_path), frame)

                frame_result = detect_pose(landmarker, frame)
                frame_result.update(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": (
                            frame_index / metadata["fps"] if metadata["fps"] > 0 else None
                        ),
                        "frame_file": str(frame_path),
                    }
                )
                results.append(frame_result)

                if save_annotated:
                    annotated_frame = draw_pose(frame, frame_result)
                    annotated_path = annotated_dir / frame_name
                    cv2.imwrite(str(annotated_path), annotated_frame)
                    frame_result["annotated_frame_file"] = str(annotated_path)

                saved_count += 1
                if max_frames > 0 and saved_count >= max_frames:
                    break

            frame_index += 1
    finally:
        capture.release()
        landmarker.close()

    summary = {
        "video_path": str(video_path),
        "sampling": {
            "sample_every_n": sample_every_n,
            "max_frames": max_frames,
            "sampled_frame_count": len(results),
        },
        "video_metadata": metadata,
        "pose_detected_frame_count": sum(1 for item in results if item["pose_detected"]),
        "frames": results,
    }

    json_path = output_dir / "landmarks.json"
    with open(json_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample frames from a video and extract MediaPipe pose landmarks."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--output-dir",
        default="data/video_pose_output",
        help="Directory where sampled frames and landmark JSON will be saved.",
    )
    parser.add_argument(
        "--sample-every-n",
        type=int,
        default=10,
        help="Keep one frame every N frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on sampled frames. Use 0 for no cap.",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Also save copies of sampled frames with pose landmarks drawn on them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_every_n <= 0:
        raise ValueError("--sample-every-n must be greater than 0")
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative")

    ensure_model()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = resolve_video_input(args.video, output_dir)

    json_path = process_video(
        video_path=video_path,
        output_dir=output_dir,
        sample_every_n=args.sample_every_n,
        max_frames=args.max_frames,
        save_annotated=args.save_annotated,
    )
    print(f"Saved sampled frames and landmarks to: {output_dir}")
    print(f"Landmark JSON: {json_path}")


if __name__ == "__main__":
    main()
