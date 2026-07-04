#!/usr/bin/env python3
"""
Scrape Pinterest search results and upload images to the Pose Search App index.

Usage:
    python pinterest_upload.py "yoga poses" [--count 20] [--similarity-threshold 0.05] [--api-url http://localhost:8000]
"""

import argparse
import json
import sys
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import numpy as np
import requests
from PIL import Image
from playwright.sync_api import sync_playwright

# Add parent directory to path to import app.main
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
from main import extract_pose_vector


def load_current_index():
    data_dir = Path(__file__).parent.parent / "app" / "data"
    index_path = data_dir / "pose_index.npz"
    db_path = data_dir / "db_files.json"

    if not index_path.exists() or not db_path.exists():
        return None, []

    try:
        index_data = np.load(index_path)
        pose_index = index_data["vectors"]
        with open(db_path, "r", encoding="utf-8") as f:
            file_paths = json.load(f)
        return pose_index, file_paths
    except Exception:
        return None, []


def is_duplicate(vector, existing_vectors, threshold=0.05):
    if existing_vectors is None or len(existing_vectors) == 0:
        return False, float("inf"), -1
    distances = np.linalg.norm(existing_vectors - vector, axis=1)
    min_distance = np.min(distances)
    closest_idx = int(np.argmin(distances))
    return min_distance < threshold, min_distance, closest_idx


def normalize_image_url(url: str) -> str:
    if not url:
        return url
    parsed = urlparse(url)
    clean = parsed.path
    if parsed.path:
        return parsed.scheme + "://" + parsed.netloc + clean
    return url


def scrape_pinterest_images(query, count=20, scroll_rounds=10, wait_ms=1200):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        )
        search_url = f"https://www.pinterest.com/search/pins/?q={quote_plus(query)}"
        page.goto(search_url, timeout=60000)
        page.wait_for_timeout(2500)

        image_urls = []
        seen = set()

        for _ in range(scroll_rounds):
            page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
            page.wait_for_timeout(wait_ms)
            urls = page.evaluate(
                "() => {"
                "  const imgs = Array.from(document.querySelectorAll('img'));"
                "  const results = [];"
                "  const attrs = ['currentSrc','src','data-src','data-image-src','data-image-url','srcset'];"
                "  for (const img of imgs) {"
                "    for (const attr of attrs) {"
                "      const value = img[attr] || img.getAttribute(attr);"
                "      if (!value) continue;"
                "      for (const fragment of value.split(',')) {"
                "        const url = fragment.trim().split(' ')[0];"
                "        if (url && url.startsWith('http') && url.includes('pinimg.com')) {"
                "          results.push(url.split('?')[0]);"
                "        }"
                "      }"
                "    }"
                "  }"
                "  return Array.from(new Set(results));"
                "}"
            )
            for url in urls:
                if url not in seen:
                    seen.add(url)
                    image_urls.append(url)
            if len(image_urls) >= count:
                break

        browser.close()

    return image_urls[:count]


def download_image(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.pinterest.com/",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.content


def upload_image_bytes(image_bytes, filename, api_url):
    content_type = "image/jpeg"
    if filename.lower().endswith(".png"):
        content_type = "image/png"
    elif filename.lower().endswith(".webp"):
        content_type = "image/webp"
    elif filename.lower().endswith(".gif"):
        content_type = "image/gif"

    files = {
        "image": (filename, BytesIO(image_bytes), content_type),
    }
    response = requests.post(f"{api_url}/api/upload", files=files, timeout=60)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Pinterest search results and upload them into the Pose Search App index"
    )
    parser.add_argument("query", help="Pinterest search query")
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Maximum number of Pinterest images to scrape",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.05,
        help="Duplicate pose similarity threshold (lower = stricter)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Pose Search App API URL",
    )

    args = parser.parse_args()

    print(f"Searching Pinterest for '{args.query}'...")
    image_urls = scrape_pinterest_images(args.query, count=args.count)
    if not image_urls:
        print("No Pinterest image URLs were found.")
        return

    print(f"Found {len(image_urls)} Pinterest image URLs.")

    pose_index, file_paths = load_current_index()
    uploaded = 0
    duplicates = 0
    failed = 0

    for idx, image_url in enumerate(image_urls, start=1):
        print(f"[{idx}/{len(image_urls)}] Downloading: {image_url}")
        try:
            image_bytes = download_image(image_url)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            vector = extract_pose_vector(image)
            if vector is None:
                print("  SKIPPED - no pose detected")
                failed += 1
                continue

            is_dup, distance, closest_idx = is_duplicate(vector, pose_index, args.similarity_threshold)
            if is_dup:
                closest_file = file_paths[closest_idx] if closest_idx >= 0 else "unknown"
                print(f"  SKIPPED - duplicate (distance={distance:.4f}, similar to={closest_file})")
                duplicates += 1
                continue

            parsed = urlparse(image_url)
            ext = Path(parsed.path).suffix or ".jpg"
            filename = f"pinterest_{idx}{ext}"
            upload_image_bytes(image_bytes, filename, args.api_url)
            print("  UPLOADED")
            uploaded += 1

            if pose_index is None:
                pose_index = vector.reshape(1, -1)
            else:
                pose_index = np.vstack([pose_index, vector])
            file_paths.append(filename)
        except Exception as exc:
            print(f"  FAILED - {exc}")
            failed += 1

    print("\nSummary")
    print(f"  Uploaded:   {uploaded}")
    print(f"  Duplicates: {duplicates}")
    print(f"  Failed:     {failed}")
    print(f"  Total seen: {len(image_urls)}")


if __name__ == "__main__":
    main()
