"""One background worker bounds PDF memory use and makes cancellation predictable."""

import io
import logging
import re
import shutil
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from PIL import Image, ImageOps

from .catalog import uid
from .page_segmentation import segment_panels
from .pose_detector import extract_pose_details

logger = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf", ".zip", ".cbz"}
MAX_ENTRY_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_PAGES = 20_000
MAX_PDF_PAGES = 10_000


def natural_key(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def archive_entries(archive):
    entries = sorted((entry for entry in archive.infolist()
                      if not entry.is_dir() and Path(entry.filename).suffix.lower() in IMAGE_EXTENSIONS),
                     key=lambda entry: natural_key(entry.filename))
    if len(entries) > MAX_ARCHIVE_PAGES:
        raise ValueError(f"Archive exceeds {MAX_ARCHIVE_PAGES} image pages.")
    if not entries:
        raise ValueError("Archive contains no supported page images.")
    return entries


def open_image(source):
    with Image.open(source) as image:
        if image.width * image.height > 40_000_000:
            raise ValueError("Page image exceeds 40 megapixels.")
        return ImageOps.exif_transpose(image).convert("RGB")


def panel_candidates(image, bounds_list=None, cancelled=lambda: False):
    candidates = []
    for bounds in bounds_list if bounds_list is not None else segment_panels(image):
        if cancelled():
            raise ImportCancelled()
        left, top, right, bottom = bounds
        crop = image.crop(bounds)
        poses = []
        for pose in extract_pose_details(crop, manga=True):
            pose["joints"] = [[max(0, min(image.width, left + x * crop.width)),
                               max(0, min(image.height, top + y * crop.height))] for x, y in pose["joints"]]
            x0, y0, x1, y1 = pose["bounds"]
            pose["bounds"] = [left + x0 * crop.width, top + y0 * crop.height,
                              left + x1 * crop.width, top + y1 * crop.height]
            poses.append(pose)
        # Keep panels with no detection too: manga detectors miss figures; review can add them.
        candidates.append({"bounds": list(bounds), "poses": poses})
    return candidates


class ImportCancelled(Exception):
    pass


class ImportManager:
    def __init__(self, catalog):
        self.catalog = catalog
        self.staging = catalog.root / "staging"
        self.staging.mkdir(parents=True, exist_ok=True)
        # Uploads from an interrupted process cannot be resumed; rendered pages are already durable.
        for directory in self.staging.iterdir():
            if directory.is_dir() and len(directory.name) == 32 and all(char in "0123456789abcdef" for char in directory.name):
                if directory.resolve().parent == self.staging.resolve():
                    shutil.rmtree(directory)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="corpus-import")
        self.events = {}
        self.lock = threading.Lock()

    def start(self, job_id, files, directory):
        event = threading.Event()
        with self.lock:
            self.events[job_id] = event
        self.executor.submit(self._run, job_id, files, directory, event)

    def cancel(self, job_id):
        with self.lock:
            event = self.events.get(job_id)
            if event and self.catalog.job(job_id)["status"] in {"queued", "running", "cancelling"}:
                self.catalog.update_job(job_id, status="cancelling")
                event.set()
        return self.catalog.job(job_id)

    def close(self):
        with self.lock:
            for event in self.events.values():
                event.set()
        self.executor.shutdown(wait=True)

    def _run(self, job_id, files, directory, event):
        catalog = self.catalog
        job = catalog.job(job_id)
        dataset_id = job["dataset_id"]
        warnings = []
        pages_done = total_pages = panel_count = 0

        def check():
            if event.is_set():
                raise ImportCancelled()

        def warn(message):
            warnings.append(message)
            # Bound persisted diagnostics; a bad file must not inflate each progress response.
            catalog.update_job(job_id, warnings=warnings[-100:])

        def finish(status, error=""):
            with self.lock:
                catalog.update_job(job_id, status="cancelled" if event.is_set() else status, error=error, current="")

        def process(image, source_name, number):
            nonlocal pages_done, panel_count
            check()
            catalog.update_job(job_id, current=f"{source_name} · page {number}")
            try:
                candidates = panel_candidates(image, cancelled=event.is_set)
                check()
                catalog.add_page(dataset_id, source_name, number, image, candidates)
                pages_done += 1
                panel_count += len(candidates)
                catalog.update_job(job_id, pages_done=pages_done, panel_count=panel_count)
            finally:
                image.close()

        try:
            catalog.update_job(job_id, status="running")
            for file_index, (path, name) in enumerate(files):
                check()
                try:
                    extension = Path(name).suffix.lower()
                    if extension == ".pdf":
                        import pypdfium2 as pdfium

                        with closing(pdfium.PdfDocument(str(path))) as document:
                            if len(document) > MAX_PDF_PAGES:
                                raise ValueError(f"PDF exceeds {MAX_PDF_PAGES} pages.")
                            total_pages += len(document)
                            catalog.update_job(job_id, total_pages=total_pages)
                            for number in range(len(document)):
                                check()
                                try:
                                    with closing(document[number]) as page:
                                        width, height = page.get_size()
                                        if min(width, height) <= 0:
                                            raise ValueError("Invalid PDF page size.")
                                        with closing(page.render(scale=min(2.0, 2000 / max(width, height)))) as bitmap:
                                            image = bitmap.to_pil().convert("RGB")
                                    process(image, name, number + 1)
                                except ImportCancelled:
                                    raise
                                except Exception as exc:
                                    warn(f"{name}, page {number + 1}: {exc}")
                    elif extension in {".zip", ".cbz"}:
                        with zipfile.ZipFile(path) as archive:
                            entries = archive_entries(archive)
                            total_pages += len(entries)
                            catalog.update_job(job_id, total_pages=total_pages)
                            for number, entry in enumerate(entries, 1):
                                check()
                                try:
                                    if entry.file_size > MAX_ENTRY_BYTES:
                                        raise ValueError("Archive image exceeds 100 MiB.")
                                    # Never extract archive paths onto disk, including ../ or absolute paths.
                                    with archive.open(entry) as stream:
                                        raw = stream.read(MAX_ENTRY_BYTES + 1)
                                    if len(raw) > MAX_ENTRY_BYTES:
                                        raise ValueError("Archive image exceeds 100 MiB.")
                                    process(open_image(io.BytesIO(raw)), f"{name} / {entry.filename}", number)
                                except ImportCancelled:
                                    raise
                                except Exception as exc:
                                    warn(f"{name} / {entry.filename}: {exc}")
                    else:
                        total_pages += 1
                        catalog.update_job(job_id, total_pages=total_pages)
                        process(open_image(path), name, 1)
                except ImportCancelled:
                    raise
                except Exception as exc:
                    warn(f"{name}: {exc}")
                catalog.update_job(job_id, files_done=file_index + 1)
            check()
            finish("completed" if pages_done else "failed", "" if pages_done else "No pages imported. Check the warnings.")
        except ImportCancelled:
            finish("cancelled")
        except Exception as exc:
            logger.exception("Import %s failed", job_id)
            finish("failed", str(exc))
        finally:
            # Only a UUID-named child of the staging root created by the upload handler.
            directory = Path(directory).resolve()
            if directory.parent == self.staging.resolve():
                shutil.rmtree(directory)
            with self.lock:
                self.events.pop(job_id, None)
