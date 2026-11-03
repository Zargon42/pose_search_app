import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps, UnidentifiedImageError

from .page_segmentation import MAX_PANELS, segment_panels
from .pose_detector import _pose_detector, extract_pose_detections
from .pose_detector import normalize_pose as _normalize_pose


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "data" / "pose_index.npz"
DB_PATH = BASE_DIR / "data" / "db_files.json"
POSE_RECORDS_PATH = BASE_DIR / "data" / "pose_records.json"
IMAGES_DIR = BASE_DIR / "data" / "images"
POSE_SCHEMA_VERSION = 1

pose_index: np.ndarray | None = None
file_paths: list[str] = []
pose_records: list[dict[str, Any]] = []
index_schema_version = POSE_SCHEMA_VERSION


def load_index() -> None:
    global pose_index, file_paths, pose_records, index_schema_version
    if not INDEX_PATH.exists() or not DB_PATH.exists():
        pose_index = None
        file_paths = []
        pose_records = []
        index_schema_version = POSE_SCHEMA_VERSION
        return

    with np.load(INDEX_PATH) as data:
        pose_index = data["vectors"].astype(np.float32, copy=False)
        index_schema_version = (
            int(data["schema_version"]) if "schema_version" in data else POSE_SCHEMA_VERSION
        )
    with DB_PATH.open("r", encoding="utf-8") as file:
        file_paths = json.load(file)

    if POSE_RECORDS_PATH.exists():
        with POSE_RECORDS_PATH.open("r", encoding="utf-8") as file:
            pose_records = json.load(file)
    else:
        pose_records = [
            {"panel_bounds": None, "pose_bounds": None, "schema_version": POSE_SCHEMA_VERSION}
            for _ in file_paths
        ]

    if len(file_paths) != len(pose_index) or len(pose_records) != len(file_paths):
        raise ValueError("Pose index, image paths, and pose metadata have different lengths.")


load_index()


def extract_pose_vector(image: Image.Image) -> np.ndarray | None:
    """Compatibility wrapper returning the first detected pose as a vector."""
    detections = extract_pose_detections(image)
    return detections[0][0] if detections else None


def normalize_pose(coords: np.ndarray) -> np.ndarray:
    return _normalize_pose(coords)


def _read_image(upload: UploadFile) -> Image.Image:
    try:
        with Image.open(upload.file) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc


def _pixel_bounds(
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
) -> list[int]:
    left, top, right, bottom = bounds
    return [
        round(left * width),
        round(top * height),
        round(right * width),
        round(bottom * height),
    ]


def _save_image(image: Image.Image, prefix: str = "") -> str:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}{uuid.uuid4()}.png"
    image.save(IMAGES_DIR / filename, format="PNG")
    return f"/images/{filename}"


def _validate_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    vector_array = np.asarray(vectors, dtype=np.float32)
    if vector_array.ndim != 2 or not len(vector_array):
        raise ValueError("At least one valid pose vector is required.")
    if pose_index is not None:
        _ensure_index()
    if pose_index is not None and vector_array.shape[1] != pose_index.shape[1]:
        raise HTTPException(status_code=409, detail="Pose vector size mismatch; rebuild the index with one model.")
    return vector_array


def _save_index(
    vector_array: np.ndarray,
    image_url: str,
    records: list[dict[str, Any]],
) -> None:
    global pose_index, file_paths, pose_records
    updated_index = vector_array if pose_index is None else np.vstack((pose_index, vector_array))
    updated_paths = file_paths + [image_url] * len(vector_array)
    updated_records = pose_records + records

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(INDEX_PATH, vectors=updated_index, schema_version=POSE_SCHEMA_VERSION)
    with DB_PATH.open("w", encoding="utf-8") as file:
        json.dump(updated_paths, file)
    with POSE_RECORDS_PATH.open("w", encoding="utf-8") as file:
        json.dump(updated_records, file)
    load_index()


def _pose_record(
    panel_bounds: list[int] | None = None,
    pose_bounds: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "panel_bounds": panel_bounds,
        "pose_bounds": pose_bounds,
        "schema_version": POSE_SCHEMA_VERSION,
    }


def _page_detections(
    image: Image.Image,
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[tuple[int, int, int, int]]]:
    panels = segment_panels(image)
    vectors: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for panel_index, bounds in enumerate(panels):
        left, top, right, bottom = bounds
        crop = image.crop(bounds)
        for vector, local_bounds in extract_pose_detections(crop):
            x0, y0, x1, y1 = local_bounds
            pose_bounds = [
                left + round(x0 * (right - left)),
                top + round(y0 * (bottom - top)),
                left + round(x1 * (right - left)),
                top + round(y1 * (bottom - top)),
            ]
            vectors.append(vector)
            records.append(
                {
                    **_pose_record(list(bounds), pose_bounds),
                    "panel_index": panel_index,
                }
            )
    return vectors, records, panels


def _match_pose(vector: np.ndarray) -> tuple[int, float]:
    _ensure_index()
    assert pose_index is not None
    if vector.shape[0] != pose_index.shape[1]:
        raise HTTPException(status_code=409, detail="Pose vector size mismatch; rebuild the index with one model.")
    distances = np.linalg.norm(pose_index - vector, axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def _ensure_index() -> None:
    if pose_index is None or not len(file_paths):
        raise HTTPException(status_code=400, detail="Index is empty. Upload pose images first.")
    if index_schema_version != POSE_SCHEMA_VERSION:
        raise HTTPException(status_code=409, detail="Pose index schema is incompatible; rebuild the index.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Warming up MediaPipe pose detector")
    _pose_detector.warm_up()
    yield
    _pose_detector.close()


app = FastAPI(title="Pose Search App", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/upload")
def upload_page(request: Request):
    return templates.TemplateResponse(request=request, name="upload.html")


@app.get("/full-page")
def full_page(request: Request):
    return templates.TemplateResponse(request=request, name="full_page.html")


@app.post("/api/search")
def search(image: UploadFile = File(...)):
    _ensure_index()
    image_data = _read_image(image)
    detections = extract_pose_detections(image_data)
    if not detections:
        raise HTTPException(status_code=400, detail="No pose detected in the image.")

    match_index = -1
    distance = float("inf")
    query_index = 0
    query_bounds = detections[0][1]
    for index, (vector, bounds) in enumerate(detections):
        candidate_index, candidate_distance = _match_pose(vector)
        if candidate_distance < distance:
            match_index = candidate_index
            distance = candidate_distance
            query_index = index
            query_bounds = bounds
    uploaded_url = _save_image(image_data, prefix="search_")
    return JSONResponse(
        {
            "uploaded_image": uploaded_url,
            "match": file_paths[match_index],
            "distance": distance,
            "query_pose_count": len(detections),
            "query_pose_index": query_index,
            "query_pose_bounds": _pixel_bounds(query_bounds, image_data.width, image_data.height),
            "match_panel_bounds": pose_records[match_index].get("panel_bounds"),
            "match_pose_bounds": pose_records[match_index].get("pose_bounds"),
        }
    )


@app.post("/api/upload")
def upload(image: UploadFile = File(...)):
    image_data = _read_image(image)
    detections = extract_pose_detections(image_data)
    if not detections:
        raise HTTPException(status_code=400, detail="No pose detected in the image.")

    vectors = _validate_vectors([vector for vector, _ in detections])
    image_url = _save_image(image_data)
    records = [
        _pose_record(pose_bounds=_pixel_bounds(bounds, image_data.width, image_data.height))
        for _, bounds in detections
    ]
    _save_index(vectors, image_url, records)
    return JSONResponse(
        {"message": f"Image indexed with {len(vectors)} pose(s).", "pose_count": len(vectors)}
    )


@app.post("/api/segment-page")
def segment_page(image: UploadFile = File(...)):
    image_data = _read_image(image)
    panels = segment_panels(image_data)
    return JSONResponse(
        {
            "width": image_data.width,
            "height": image_data.height,
            "panel_count": len(panels),
            "max_panels": MAX_PANELS,
            "panels": [
                {"panel_index": index, "bounds": list(bounds)}
                for index, bounds in enumerate(panels)
            ],
        }
    )


@app.post("/api/full-page/search")
def search_full_page(image: UploadFile = File(...)):
    _ensure_index()
    image_data = _read_image(image)
    vectors, query_records, panels = _page_detections(image_data)
    if not vectors:
        raise HTTPException(
            status_code=400,
            detail=f"No poses detected across {len(panels)} segmented panel(s).",
        )

    vector_array = _validate_vectors(vectors)
    uploaded_url = _save_image(image_data, prefix="search_page_")
    matches = []
    for vector, record in zip(vector_array, query_records):
        match_index, distance = _match_pose(vector)
        matches.append(
            {
                "query_panel_index": record["panel_index"],
                "query_panel_bounds": record["panel_bounds"],
                "query_pose_bounds": record["pose_bounds"],
                "match": file_paths[match_index],
                "distance": distance,
                "match_panel_bounds": pose_records[match_index].get("panel_bounds"),
                "match_pose_bounds": pose_records[match_index].get("pose_bounds"),
            }
        )
    matches.sort(key=lambda result: result["distance"])
    return JSONResponse(
        {
            "uploaded_image": uploaded_url,
            "panel_count": len(panels),
            "pose_count": len(vectors),
            "matches": matches[:30],
        }
    )


@app.post("/api/full-page/upload")
def upload_full_page(image: UploadFile = File(...)):
    image_data = _read_image(image)
    vectors, records, panels = _page_detections(image_data)
    if not vectors:
        raise HTTPException(
            status_code=400,
            detail=f"No poses detected across {len(panels)} segmented panel(s).",
        )

    vector_array = _validate_vectors(vectors)
    image_url = _save_image(image_data)
    _save_index(vector_array, image_url, records)
    return JSONResponse(
        {
            "message": f"Indexed {len(vectors)} pose(s) from {len(panels)} panel(s).",
            "panel_count": len(panels),
            "pose_count": len(vectors),
            "image": image_url,
            "panels": [
                {"panel_index": index, "bounds": list(bounds)}
                for index, bounds in enumerate(panels)
            ],
        }
    )


@app.post("/api/clear-index")
def clear_index():
    global pose_index, file_paths, pose_records, index_schema_version
    image_names = {
        Path(image_url).name
        for image_url in file_paths
        if image_url.startswith("/images/") and not Path(image_url).name.startswith("search_")
    }
    for image_name in image_names:
        image_path = (IMAGES_DIR / image_name).resolve()
        if image_path.parent != IMAGES_DIR.resolve():
            raise HTTPException(status_code=400, detail="Invalid image path in pose index.")
        if image_path.exists():
            image_path.unlink()

    for path in (INDEX_PATH, DB_PATH, POSE_RECORDS_PATH):
        if path.exists():
            path.unlink()
    pose_index = None
    file_paths = []
    pose_records = []
    index_schema_version = POSE_SCHEMA_VERSION
    return JSONResponse({"message": "Pose index and indexed images cleared."})
