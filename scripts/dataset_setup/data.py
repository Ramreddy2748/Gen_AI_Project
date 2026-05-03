import argparse
import os
from pathlib import Path
from typing import Dict, List
from urllib.parse import urljoin
from zipfile import ZipFile

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DATASET_PAGE_CANDIDATES = [
    "https://fenix.ur.edu.pl/~mkepski/ds/uf.html",
    "http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html",
]

REQUEST_TIMEOUT = 20
CHUNK_SIZE = 1024 * 64
DEFAULT_LIMIT = 0

IMAGE_OUTPUT_ROOT = Path("data")
VIDEO_OUTPUT_ROOT = Path("data/raw/urfd_videos")


def build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_dataset_page(session: requests.Session) -> requests.Response:
    for candidate_url in DATASET_PAGE_CANDIDATES:
        try:
            response = session.get(candidate_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            print(f"Using dataset page: {candidate_url}")
            return response
        except requests.RequestException as exc:
            print(f"Failed to reach {candidate_url}: {exc}")

    raise requests.ConnectionError(
        "Unable to reach any dataset page URL. Check internet access or update URLs."
    )


def classify_link(href: str) -> str:
    filename = href.split("/")[-1]
    if filename.endswith(".mp4"):
        return "video"
    if filename.endswith(".zip"):
        return "zip"
    return "other"


def fetch_dataset_links(
    session: requests.Session,
    include_zips: bool,
    include_videos: bool,
    limit: int,
) -> List[str]:
    print("Fetching dataset page...")
    response = fetch_dataset_page(session)
    soup = BeautifulSoup(response.text, "html.parser")

    selected_urls: List[str] = []
    seen = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        kind = classify_link(href)
        if kind == "zip" and not include_zips:
            continue
        if kind == "video" and not include_videos:
            continue
        if kind == "other":
            continue

        full_url = urljoin(response.url, href)
        if full_url in seen:
            continue

        selected_urls.append(full_url)
        seen.add(full_url)

    selected_urls = sorted(selected_urls)
    if limit > 0:
        selected_urls = selected_urls[:limit]
    return selected_urls


def image_output_path(filename: str) -> Path:
    if filename.startswith("adl-"):
        return IMAGE_OUTPUT_ROOT / "adl" / filename
    if filename.startswith("fall-"):
        seq_id = filename.split("-")[1]
        seq_dir = IMAGE_OUTPUT_ROOT / "falls" / f"fall-{int(seq_id):02d}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        return seq_dir / filename
    raise ValueError(f"Unsupported image filename: {filename}")


def video_output_path(filename: str) -> Path:
    parts = filename.replace(".mp4", "").split("-")
    label = parts[0]
    seq_id = parts[1]
    camera = parts[2]

    if label == "adl":
        target_dir = VIDEO_OUTPUT_ROOT / "adl" / f"adl-{int(seq_id):02d}" / camera
    elif label == "fall":
        target_dir = VIDEO_OUTPUT_ROOT / "fall" / f"fall-{int(seq_id):02d}" / camera
    else:
        raise ValueError(f"Unsupported video filename: {filename}")

    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / filename


def save_stream_to_file(
    session: requests.Session,
    url: str,
    output_path: Path,
) -> None:
    with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as file_obj:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    file_obj.write(chunk)


def handle_zip_file(path: Path) -> None:
    extract_to = path.parent
    with ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    path.unlink()
    print(f"Extracted and removed zip: {path.name}")


def download_files(session: requests.Session, urls: List[str], extract_zips: bool) -> Dict[str, int]:
    stats = {"zip_files": 0, "video_files": 0}

    for url in urls:
        filename = url.split("/")[-1]
        if filename.endswith(".zip"):
            output_path = image_output_path(filename)
            stats["zip_files"] += 1
        elif filename.endswith(".mp4"):
            output_path = video_output_path(filename)
            stats["video_files"] += 1
        else:
            continue

        if output_path.exists():
            print(f"Skipping existing file: {output_path}")
            continue

        print(f"Downloading: {filename}")
        try:
            save_stream_to_file(session, url, output_path)
            print(f"Saved: {output_path}")
            if extract_zips and output_path.suffix.lower() == ".zip":
                handle_zip_file(output_path)
        except requests.RequestException as exc:
            print(f"Download failed for {filename}: {exc}")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download URFD image archives and/or rendered videos."
    )
    parser.add_argument(
        "--mode",
        choices=["images", "videos", "both"],
        default="images",
        help="Download image zips, video files, or both.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Optional limit on number of files to download. Use 0 for all.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Keep zip files without extracting them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = build_session()
    include_zips = args.mode in {"images", "both"}
    include_videos = args.mode in {"videos", "both"}

    urls = fetch_dataset_links(
        session=session,
        include_zips=include_zips,
        include_videos=include_videos,
        limit=args.limit,
    )
    print(f"Found {len(urls)} dataset files for mode={args.mode}")

    stats = download_files(session=session, urls=urls, extract_zips=not args.skip_extract)
    print(
        "Done. "
        f"Image zip files selected: {stats['zip_files']}, "
        f"video files selected: {stats['video_files']}"
    )


if __name__ == "__main__":
    main()
