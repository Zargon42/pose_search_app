import io
import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import UploadFile as FormUploadFile

from .catalog import Catalog, GENDERS, uid
from .importing import ImportManager, SUPPORTED_EXTENSIONS, natural_key, open_image, panel_candidates
from .page_segmentation import MAX_PANELS, segment_panels
from .pose_detector import _manga_pose_detector, _pose_detector, extract_pose_detections, extract_pose_details
from .pose_detector import normalize_pose as _normalize_pose

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
MAX_BATCH_BYTES = 4 * MAX_UPLOAD_BYTES


class DatasetInput(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    source_url: str = Field(default="", max_length=2000)
    license: str = Field(default="", max_length=2000)
    attribution: str = Field(default="", max_length=2000)
    notes: str = Field(default="", max_length=10_000)


class PanelInput(BaseModel):
    bounds: list[int] | None = Field(default=None, min_length=4, max_length=4)
    participant_count: int | None = Field(default=None, ge=0, le=64)
    genders: list[str] | None = Field(default=None, max_length=64)
    status: Literal["pending", "accepted", "skipped"] | None = None
    notes: str | None = Field(default=None, max_length=10_000)
    poses: list[list[list[float]]] | None = Field(default=None, max_length=64)
    pose_scores: list[list[float]] | None = Field(default=None, max_length=64)


class BoundsInput(BaseModel):
    bounds: list[int] = Field(min_length=4, max_length=4)


async def import_uploads(request: Request):
    async with request.form(max_files=20_000) as form:
        files = form.getlist("files")
        if any(not isinstance(file, FormUploadFile) for file in files):
            raise HTTPException(422, "The files field must contain uploaded files.")
        yield files


def normalize_pose(coords):
    return _normalize_pose(coords)


def extract_pose_vector(image):
    detections = extract_pose_detections(image)
    return detections[0][0] if detections else None


def _pixel_bounds(bounds, width, height):
    return [round(bounds[0] * width), round(bounds[1] * height), round(bounds[2] * width), round(bounds[3] * height)]


def _page_detections(image):
    """Compatibility helper for callers of the original page pipeline."""
    panels, vectors, records = segment_panels(image), [], []
    for index, bounds in enumerate(panels):
        left, top, right, bottom = bounds
        for vector, local_bounds in extract_pose_detections(image.crop(bounds), manga=True):
            x0, y0, x1, y1 = _pixel_bounds(local_bounds, right - left, bottom - top)
            vectors.append(vector)
            records.append({"panel_bounds": list(bounds), "pose_bounds": [left + x0, top + y0, left + x1, top + y1], "panel_index": index, "schema_version": 1})
    return vectors, records, panels


def validate_bounds(bounds, width, height):
    left, top, right, bottom = bounds
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise HTTPException(422, "Panel bounds must describe a positive rectangle inside the source page.")
    return bounds


def corrected_poses(joint_lists, bounds, width, height, score_lists=None):
    left, top, right, bottom = bounds
    poses = []
    if score_lists is not None and len(score_lists) != len(joint_lists):
        raise HTTPException(422, "Supply confidence values for each skeleton.")
    for index, joints in enumerate(joint_lists):
        coords = np.asarray(joints, dtype=np.float32)
        if coords.shape != (33, 2) or not np.isfinite(coords).all():
            raise HTTPException(422, "Each skeleton must contain 33 finite x/y landmark pairs.")
        if (coords < 0).any() or (coords[:, 0] > width).any() or (coords[:, 1] > height).any():
            raise HTTPException(422, "Skeleton joints must lie inside the source page.")
        normalized = (coords - [left, top]) / [right - left, bottom - top]
        confidence = np.asarray(score_lists[index] if score_lists is not None else [1.] * 33, np.float32)
        if confidence.shape != (33,) or not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
            raise HTTPException(422, "Each skeleton needs 33 confidence values between zero and one.")
        poses.append({"joints": coords.tolist(), "vector": normalize_pose(normalized),
                      "joint_scores": confidence.tolist(), "backend": "manual",
                      "bounds": [float(coords[:, 0].min()), float(coords[:, 1].min()), float(coords[:, 0].max()), float(coords[:, 1].max())]})
    return poses


def create_app(data_dir=None, warm_up=True):
    catalog = Catalog(Path(data_dir or os.environ.get("POSE_DATA_DIR", BASE_DIR / "data")))

    @asynccontextmanager
    async def lifespan(application):
        catalog.initialize()
        manager = ImportManager(catalog)
        application.state.imports = manager
        try:
            if warm_up:
                _pose_detector.warm_up()
            yield
        finally:
            manager.close()
            if warm_up:
                _pose_detector.close()
                _manga_pose_detector.close()

    application = FastAPI(title="Pose Library", lifespan=lifespan)
    application.state.catalog = catalog
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @application.middleware("http")
    async def local_requests(request, call_next):
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin and urlsplit(origin).netloc != request.url.netloc:
            return JSONResponse({"detail": "Cross-origin writes to this local library are disabled."}, status_code=403)
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.exception_handler(KeyError)
    async def missing(_, exc):
        return JSONResponse({"detail": str(exc.args[0])}, status_code=404)

    @application.exception_handler(ValueError)
    async def invalid(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @application.exception_handler(RuntimeError)
    async def model_unavailable(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    application.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=BASE_DIR / "templates")

    @application.get("/")
    def home(request: Request):
        return templates.TemplateResponse(request=request, name="index.html")

    @application.get("/upload")
    @application.get("/library")
    def library(request: Request):
        return templates.TemplateResponse(request=request, name="upload.html")

    @application.get("/full-page")
    def full_page(request: Request):
        return templates.TemplateResponse(request=request, name="full_page.html")

    @application.get("/api/datasets")
    def datasets():
        return {"datasets": catalog.datasets()}

    @application.post("/api/datasets", status_code=201)
    def create_dataset(body: DatasetInput):
        values = body.model_dump()
        values["name"] = values["name"].strip()
        if not values["name"]:
            raise HTTPException(422, "Dataset name cannot be blank.")
        return catalog.dataset(catalog.create_dataset(**values))

    @application.patch("/api/datasets/{dataset_id}")
    def edit_dataset(dataset_id: str, body: DatasetInput):
        values = body.model_dump()
        values["name"] = values["name"].strip()
        if not values["name"]:
            raise HTTPException(422, "Dataset name cannot be blank.")
        catalog.update_dataset(dataset_id, values)
        return catalog.dataset(dataset_id)

    @application.delete("/api/datasets/{dataset_id}")
    def delete_dataset(dataset_id: str):
        catalog.delete_dataset(dataset_id)
        return {"message": "Dataset and its rendered pages deleted."}

    @application.post("/api/datasets/{dataset_id}/imports", status_code=202, openapi_extra={
        "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
            "type": "object", "required": ["files"], "properties": {
                "files": {"type": "array", "items": {"type": "string", "format": "binary"}}
            }
        }}}}
    })
    async def import_files(dataset_id: str, request: Request, files: list[UploadFile] = Depends(import_uploads)):
        catalog.dataset(dataset_id)
        if not files or len(files) > 20_000:
            raise HTTPException(422, "Choose between 1 and 20,000 files.")
        for file in files:
            if Path(file.filename or "").suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise HTTPException(422, f"Unsupported file type: {file.filename}")
        manager = request.app.state.imports
        directory = manager.staging / uid()
        directory.mkdir()
        staged, batch_size = [], 0
        try:
            for file in sorted(files, key=lambda item: natural_key(item.filename or "")):
                path = directory / uid()
                size = 0
                with path.open("wb") as target:
                    while chunk := await file.read(1024 * 1024):
                        size += len(chunk)
                        batch_size += len(chunk)
                        if size > MAX_UPLOAD_BYTES or batch_size > MAX_BATCH_BYTES:
                            raise HTTPException(413, "Imports allow 1 GiB per file and 4 GiB per batch.")
                        target.write(chunk)
                staged.append((path, file.filename or "Untitled image"))
            job_id = catalog.create_job(dataset_id, len(files))
            manager.start(job_id, staged, directory)
            return catalog.job(job_id)
        except Exception:
            shutil.rmtree(directory)
            raise
        finally:
            for file in files:
                await file.close()

    @application.get("/api/imports")
    def imports(dataset_id: str | None = None):
        return {"jobs": catalog.jobs(dataset_id)}

    @application.post("/api/datasets/{dataset_id}/redetect", status_code=202)
    def redetect_dataset(dataset_id: str, request: Request):
        catalog.dataset(dataset_id)
        if any(job["status"] in {"queued", "running", "cancelling"} for job in catalog.jobs(dataset_id)):
            raise HTTPException(409, "Wait for this collection's active job or cancel it first.")
        # Fail before creating a job if model installation is incomplete.
        with _manga_pose_detector._lock:
            _manga_pose_detector._load()
        panel_ids = catalog.redetection_panels(dataset_id)
        if not panel_ids:
            raise HTTPException(422, "No unreviewed automatic panels need re-detection.")
        job_id = catalog.create_job(dataset_id, len(panel_ids), kind="redetect")
        try:
            catalog.backup_before_redetect(job_id)
            with catalog.connect() as db:
                pages = db.execute("SELECT COUNT(DISTINCT pg.id) FROM pages pg JOIN panels p ON p.page_id=pg.id WHERE pg.dataset_id=? AND p.status='pending'", (dataset_id,)).fetchone()[0]
            catalog.update_job(job_id, total_pages=pages)
            request.app.state.imports.start_redetection(job_id, panel_ids)
        except Exception as exc:
            catalog.update_job(job_id, status="failed", error=str(exc))
            raise
        return catalog.job(job_id)

    @application.get("/api/imports/{job_id}")
    def import_status(job_id: str):
        return catalog.job(job_id)

    @application.post("/api/imports/{job_id}/cancel")
    def cancel_import(job_id: str, request: Request):
        catalog.job(job_id)
        return request.app.state.imports.cancel(job_id)

    @application.get("/api/panels")
    def panels(dataset_id: str | None = None, status: Literal["pending", "accepted", "skipped"] | None = "accepted",
               participant_count: int | None = Query(None, ge=0, le=64), gender: str | None = None,
               limit: int = Query(40, ge=1, le=100), offset: int = Query(0, ge=0)):
        if gender is not None and gender not in GENDERS:
            raise HTTPException(422, "Unknown gender tag.")
        return catalog.list_panels(dataset_id, status, participant_count, gender, limit, offset)

    @application.get("/api/panels/{panel_id}")
    def panel(panel_id: str):
        return catalog.panel(panel_id)

    @application.patch("/api/panels/{panel_id}")
    def edit_panel(panel_id: str, body: PanelInput):
        original = catalog.panel(panel_id)
        values = body.model_dump(exclude_none=True, exclude={"poses", "pose_scores"})
        bounds = validate_bounds(values.get("bounds", original["bounds"]), original["width"], original["height"])
        poses = None
        if body.poses is not None:
            poses = corrected_poses(body.poses, bounds, original["width"], original["height"], body.pose_scores)
        elif body.pose_scores is not None:
            raise HTTPException(422, "Confidence changes require skeleton coordinates.")
        elif bounds != original["bounds"]:
            if any(pose["joints"] is None for pose in original["poses"]):
                raise HTTPException(422, "Re-detect this migrated panel before changing its crop.")
            poses = corrected_poses([pose["joints"] for pose in original["poses"]], bounds, original["width"], original["height"],
                                    [pose["joint_scores"] for pose in original["poses"]])
        count = values.get("participant_count", original["participant_count"])
        pose_count = len(poses if poses is not None else original["poses"])
        if count < pose_count:
            raise HTTPException(422, "Participant count cannot be smaller than the number of skeletons. Remove false detections first.")
        genders = values.get("genders", original["genders"][:count] + ["unknown"] * max(0, count - len(original["genders"])))
        if len(genders) != count or any(gender not in GENDERS for gender in genders):
            raise HTTPException(422, "Provide one supported gender tag (or unknown) per participant.")
        if values.get("status", original["status"]) == "accepted" and not pose_count:
            raise HTTPException(422, "Add or detect at least one skeleton before accepting this panel.")
        values["genders"] = genders
        catalog.update_panel(panel_id, values, poses)
        return catalog.panel(panel_id)

    @application.post("/api/panels/{panel_id}/redetect")
    def redetect_panel(panel_id: str, body: BoundsInput):
        original = catalog.panel(panel_id)
        bounds = validate_bounds(body.bounds, original["width"], original["height"])
        page = catalog.page(original["page_id"])
        with Image.open(catalog.assets / page["image_name"]) as image:
            candidate = panel_candidates(image, [bounds])[0]
        count = len(candidate["poses"])
        catalog.update_panel(panel_id, {"bounds": bounds, "participant_count": count, "genders": ["unknown"] * count, "status": "pending"}, candidate["poses"])
        return catalog.panel(panel_id)

    @application.post("/api/pages/{page_id}/panels", status_code=201)
    def add_panel(page_id: str, body: BoundsInput):
        page = catalog.page(page_id)
        bounds = validate_bounds(body.bounds, page["width"], page["height"])
        with Image.open(catalog.assets / page["image_name"]) as image:
            candidate = panel_candidates(image, [bounds])[0]
        return catalog.panel(catalog.add_panel(page_id, candidate))

    @application.get("/api/pages/{page_id}/image")
    def page_image(page_id: str):
        page = catalog.page(page_id)
        return FileResponse(catalog.assets / page["image_name"], media_type="image/png")

    @application.get("/api/panels/{panel_id}/image")
    def panel_image(panel_id: str):
        panel = catalog.panel(panel_id)
        page = catalog.page(panel["page_id"])
        with Image.open(catalog.assets / page["image_name"]) as image:
            output = io.BytesIO()
            image.crop(panel["bounds"]).save(output, "PNG")
        return Response(output.getvalue(), media_type="image/png")

    def read_upload(upload):
        try:
            return open_image(upload.file)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise HTTPException(400, f"Invalid image: {exc}") from exc

    @application.post("/api/search")
    def search(image: UploadFile = File(...), dataset_id: str | None = Form(None),
               participant_count: int | None = Form(None, ge=0, le=64), gender: str | None = Form(None),
               limit: int = Form(12, ge=1, le=50)):
        if gender is not None and gender not in GENDERS:
            raise HTTPException(422, "Unknown gender tag.")
        with read_upload(image) as query:
            detections = extract_pose_details(query)
            if not detections:
                raise HTTPException(400, "No pose detected. Try a photo showing more of the person's body.")
            matches = catalog.search([pose["vector"] for pose in detections], dataset_id, participant_count, gender, limit,
                                      query_scores=[pose.get("joint_scores", [1.] * 33) for pose in detections],
                                      query_dimensions=(query.width, query.height))
            for match in matches:
                match["match"] = match["image_url"]
                match["match_panel_bounds"] = match["bounds"]
                match["query_pose_bounds"] = _pixel_bounds(detections[match["query_pose_index"]]["bounds"], query.width, query.height)
            result = {"matches": matches, "query_pose_count": len(detections)}
            if matches:
                result.update({key: matches[0][key] for key in ("match", "distance", "query_pose_index", "query_pose_bounds", "match_panel_bounds", "match_pose_bounds")})
            return result

    def default_dataset():
        with catalog.connect() as db:
            db.execute("INSERT OR IGNORE INTO datasets(id,name) VALUES('uploads','Image uploads')")
        return "uploads"

    @application.post("/api/upload")
    def upload(image: UploadFile = File(...)):
        with read_upload(image) as source:
            candidates = panel_candidates(source, [[0, 0, source.width, source.height]])
            count = len(candidates[0]["poses"])
            if not count:
                raise HTTPException(400, "No pose detected. Use Library import to add skeletons manually.")
            panel_ids = catalog.add_page(default_dataset(), image.filename or "Image upload", 1, source, candidates, status="accepted")
        return {"message": f"Image indexed with {count} pose(s).", "pose_count": count, "panel_ids": panel_ids}

    @application.post("/api/segment-page")
    def segment_page(image: UploadFile = File(...)):
        with read_upload(image) as source:
            bounds = segment_panels(source)
            return {"width": source.width, "height": source.height, "panel_count": len(bounds), "max_panels": MAX_PANELS,
                    "panels": [{"panel_index": index, "bounds": list(box)} for index, box in enumerate(bounds)]}

    @application.post("/api/full-page/upload")
    def upload_full_page(image: UploadFile = File(...)):
        with read_upload(image) as source:
            candidates = panel_candidates(source)
            count = sum(len(candidate["poses"]) for candidate in candidates)
            ids = catalog.add_page(default_dataset(), image.filename or "Page upload", 1, source, candidates)
            for panel_id, candidate in zip(ids, candidates):
                if candidate["poses"]:
                    catalog.update_panel(panel_id, {"status": "accepted"})
            return {"message": f"Indexed {count} pose(s). Panels without detected poses await review in Library.",
                    "panel_count": len(candidates), "pose_count": count, "panel_ids": ids,
                    "panels": [{"panel_index": index, "bounds": candidate["bounds"]} for index, candidate in enumerate(candidates)]}

    @application.post("/api/full-page/search")
    def search_full_page(image: UploadFile = File(...)):
        with read_upload(image) as source:
            vectors, records, bounds = _page_detections(source)
            if not vectors:
                raise HTTPException(400, "No poses detected in the segmented page.")
            matches = []
            for vector, record in zip(vectors, records):
                box = record["panel_bounds"]
                for match in catalog.search([vector], limit=1, query_dimensions=(box[2]-box[0], box[3]-box[1])):
                    matches.append({**match, "match": match["image_url"], "query_panel_index": record["panel_index"],
                                    "query_panel_bounds": record["panel_bounds"], "query_pose_bounds": record["pose_bounds"]})
            return {"matches": sorted(matches, key=lambda match: match["distance"])[:30], "panel_count": len(bounds), "pose_count": len(vectors)}

    @application.post("/api/clear-index")
    def clear_index():
        if any(job["status"] in {"queued", "running", "cancelling"} for job in catalog.jobs()):
            raise HTTPException(409, "Cancel active imports before clearing the library.")
        for dataset in catalog.datasets():
            catalog.delete_dataset(dataset["id"])
        return {"message": "Local library cleared. Legacy backup files, if present, were preserved."}

    return application


app = create_app()
