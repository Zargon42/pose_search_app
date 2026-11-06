#!/usr/bin/env python3
"""Stream local PDFs, image directories, or image archives into a review queue."""

import argparse
import sys
import time
import uuid
from pathlib import Path

import requests

EXTENSIONS = {".pdf", ".zip", ".cbz", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def checked(response):
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json()


def multipart(paths, root):
    boundary = uuid.uuid4().hex
    parts = []
    for path in paths:
        name = path.relative_to(root).as_posix().replace('"', '_').replace('\r', '_').replace('\n', '_')
        header = (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
                  'Content-Type: application/octet-stream\r\n\r\n').encode('utf-8')
        parts.append((header, path))
    ending = f"--{boundary}--\r\n".encode()
    length = sum(len(header) + path.stat().st_size + 2 for header, path in parts) + len(ending)

    def stream():
        for header, path in parts:
            yield header
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    yield chunk
            yield b"\r\n"
        yield ending

    class Body:
        def __iter__(self):
            return stream()

        def __len__(self):
            # requests needs this to avoid also adding Transfer-Encoding: chunked.
            return length

    return Body(), {"Content-Type": f"multipart/form-data; boundary={boundary}", "Content-Length": str(length)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="A PDF, image, CBZ/ZIP file, or directory (recursive).")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset-id", help="Import into an existing collection.")
    parser.add_argument("--name", help="Name for a new collection; defaults to the source name.")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--license", default="")
    parser.add_argument("--attribution", default="")
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.exists():
        parser.error("Source does not exist.")
    paths = sorted((path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS), key=lambda path: path.as_posix()) if source.is_dir() else [source]
    if not paths or any(path.suffix.lower() not in EXTENSIONS for path in paths):
        parser.error("No supported files selected.")
    if len(paths) > 20_000 or any(path.stat().st_size > 1024**3 for path in paths) or sum(path.stat().st_size for path in paths) > 4 * 1024**3:
        parser.error("Split this import into batches of at most 20,000 files, 1 GiB per file, and 4 GiB per batch.")
    base = args.api_url.rstrip("/")
    job_id = None
    try:
        with requests.Session() as session:
            dataset_id = args.dataset_id
            if not dataset_id:
                dataset_id = checked(session.post(f"{base}/api/datasets", json={"name": args.name or source.stem,
                    "source_url": args.source_url, "license": args.license, "attribution": args.attribution}, timeout=30))["id"]
            body, headers = multipart(paths, source if source.is_dir() else source.parent)
            job_id = checked(session.post(f"{base}/api/datasets/{dataset_id}/imports", data=body, headers=headers, timeout=600))["id"]
            print(f"Import {job_id} · collection {dataset_id}")
            previous = None
            while True:
                job = checked(session.get(f"{base}/api/imports/{job_id}", timeout=30))
                progress = (job["status"], job["files_done"], job["pages_done"], job["panel_count"], job["current"])
                if progress != previous:
                    print(f"{job['status']}: {job['files_done']}/{job['total_files']} files, {job['pages_done']} pages, {job['panel_count']} panels. {job['current']}")
                    previous = progress
                if job["status"] not in {"queued", "running", "cancelling"}:
                    for warning in job["warnings"]:
                        print(f"Warning: {warning}", file=sys.stderr)
                    if job["error"]:
                        print(job["error"], file=sys.stderr)
                    print(f"Review panels at {base}/library?dataset={dataset_id}")
                    return 0 if job["status"] == "completed" else 1
                time.sleep(1)
    except KeyboardInterrupt:
        if job_id:
            checked(requests.post(f"{base}/api/imports/{job_id}/cancel", timeout=30))
            print("\nCancellation requested. Completed pages remain available for review.")
        return 130
    except (requests.RequestException, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
