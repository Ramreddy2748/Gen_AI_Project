from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


DATASET_PAGE = "https://fenix.ur.edu.pl/~mkepski/ds/uf.html"
FALLS_ROOT = Path("data/falls")
TIMEOUT = 45
CHUNK_SIZE = 1024 * 256

EXPECTED_SUFFIXES = [
    "-cam0-d.zip",
    "-cam1-d.zip",
    "-cam0-rgb.zip",
    "-cam1-rgb.zip",
    "-data.csv",
    "-acc.csv",
]


def is_valid_file(path: Path) -> Tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    size = path.stat().st_size
    if size == 0:
        return False, "empty file"

    if path.suffix == ".zip":
        with path.open("rb") as f:
            sig = f.read(4)
        if sig != b"PK\x03\x04":
            return False, "invalid ZIP signature"
        return True, "ok zip"

    if path.suffix == ".csv":
        with path.open("rb") as f:
            head = f.read(256).lower()
        if b"<html" in head or b"<!doctype" in head:
            return False, "html content instead of csv"
        return True, "ok csv"

    return True, "unknown extension"


def fetch_links(session: requests.Session) -> Dict[str, str]:
    resp = session.get(DATASET_PAGE, timeout=TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links: Dict[str, str] = {}

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        name = Path(href).name
        if re.match(r"^fall-\d{2}-(cam0|cam1)-(d|rgb)\.zip$", name) or re.match(
            r"^fall-\d{2}-(data|acc)\.csv$", name
        ):
            links[name] = urljoin(resp.url, href)
    return links


def targets(start: int, end: int) -> Iterable[Tuple[str, Path]]:
    for n in range(start, end + 1):
        seq = f"fall-{n:02d}"
        folder = FALLS_ROOT / seq
        folder.mkdir(parents=True, exist_ok=True)
        for suffix in EXPECTED_SUFFIXES:
            yield f"{seq}{suffix}", folder / f"{seq}{suffix}"


def download_file(session: requests.Session, url: str, dest: Path) -> Tuple[bool, str]:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with session.get(url, timeout=TIMEOUT, stream=True) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)

        ok, reason = is_valid_file(tmp)
        if not ok:
            tmp.unlink(missing_ok=True)
            return False, reason

        tmp.replace(dest)
        return True, "downloaded"
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair missing/corrupt files in data/falls by re-downloading from URFD."
    )
    parser.add_argument("--start", type=int, default=1, help="First fall index (default: 1)")
    parser.add_argument("--end", type=int, default=8, help="Last fall index (default: 8)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file currently looks valid.",
    )
    args = parser.parse_args()

    session = requests.Session()
    link_map = fetch_links(session)
    print(f"Loaded {len(link_map)} fall links from dataset page.")

    downloaded = 0
    skipped = 0
    failed = 0

    for name, path in targets(args.start, args.end):
        url = link_map.get(name)
        if not url:
            print(f"FAIL {name}: no URL found on dataset page")
            failed += 1
            continue

        if not args.force:
            ok, _ = is_valid_file(path)
            if ok:
                print(f"SKIP {name}: already valid")
                skipped += 1
                continue

        ok, msg = download_file(session, url, path)
        if ok:
            print(f"OK   {name}: {msg}")
            downloaded += 1
        else:
            print(f"FAIL {name}: {msg}")
            failed += 1

    print(f"\nSummary: downloaded={downloaded}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
