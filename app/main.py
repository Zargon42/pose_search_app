import json #interpret json files
import numpy as np #vector math
import uuid  # generate unique filenames
import shutil  # for removing directories

from fastapi import FastAPI, File, UploadFile, Request, HTTPException # handle file uploads and requests
from fastapi.responses import HTMLResponse, JSONResponse # return html and json responses
from fastapi.staticfiles import StaticFiles # serve static files stored on the server
from fastapi.templating import Jinja2Templates # render html templates
from pathlib import Path # handle file paths
from PIL import Image # handle image files
# from mediapipe.tasks import vision
# from mediapipe.tasks.python import vision

import mediapipe as mp # import mediapipe to use the pose detection model

BASE_DIR = Path(__file__).resolve().parent # get the base directory of the project

INDEX_PATH = BASE_DIR / "data" / "pose_index.npz"
DB_PATH = BASE_DIR / "data" / "db_files.json"
IMAGES_DIR = BASE_DIR / "data" / "images"

app = FastAPI(title="Pose Search App") # create a FastAPI app instance
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static") # mount the static files directory to serve static files

# Create images directory if it doesn't exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images") # mount the images directory to serve uploaded images

templates = Jinja2Templates(directory=BASE_DIR / "templates") # create a Jinja2Templates instance to render html templates

# Model URL (lite version – fast and good enough for most cases)
model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
model_path = str(BASE_DIR / "pose_landmarker.task")

pose_index = None
file_paths = []


def load_index():
    global pose_index, file_paths
    if not INDEX_PATH.exists() or not DB_PATH.exists():
        pose_index = None
        file_paths = []
        return
    data = np.load(INDEX_PATH)
    pose_index = data["vectors"]
    with open(DB_PATH, "r", encoding="utf-8") as f:
        file_paths = json.load(f)


load_index()


def normalize_pose(coords: np.ndarray) -> np.ndarray: # normalize the pose coordinates to have zero mean and unit variance
    if coords.size == 0:
        return coords
    coords = coords.reshape(-1, 2)
    center = coords.mean(axis=0)
    coords = coords - center
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale > 0:
        coords = coords / scale
    return coords.flatten()


def extract_pose_vector(image: Image.Image) -> np.ndarray | None:
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    # Convert PIL image to RGB and then to MediaPipe Image
    image_rgb = image.convert("RGB")
    image_array = np.array(image_rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    
    detection_result = detector.detect(mp_image)
    
    if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
        return None
    
    # Extract coordinates from the first detected pose
    landmarks = detection_result.pose_landmarks[0]
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y])
    
    return normalize_pose(np.array(coords, dtype=np.float32))


# def extract_pose_vector(image: Image.Image) -> np.ndarray | None:
#     image = image.convert("RGB")
#     image_np = np.array(image)
#     results = mp_pose.process(image_np)
#     if not results.pose_landmarks:
#         return None
#     coords = []
#     for lm in results.pose_landmarks.landmark:
#         coords.extend([lm.x, lm.y])
#     return normalize_pose(np.array(coords, dtype=np.float32))


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/upload")
def upload_page(request: Request):
    return templates.TemplateResponse(request=request, name="upload.html")


@app.post("/api/search")
def search(image: UploadFile = File(...)):
    global pose_index, file_paths
    
    if pose_index is None or len(file_paths) == 0:
        raise HTTPException(status_code=400, detail="Index is empty. Please upload images first using the Upload page.")
    
    try:
        image_data = Image.open(image.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")
    
    vector = extract_pose_vector(image_data)
    if vector is None:
        raise HTTPException(status_code=400, detail="No pose detected in the image.")
    
    if vector.shape[0] != pose_index.shape[1]:
        raise HTTPException(status_code=500, detail="Pose vector size mismatch.")
    distances = np.linalg.norm(pose_index - vector, axis=1)
    best_idx = int(np.argmin(distances))
    
    # Save the uploaded image temporarily for display
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    file_ext = Path(image.filename).suffix
    uploaded_filename = f"search_{uuid.uuid4()}{file_ext}"
    uploaded_path = IMAGES_DIR / uploaded_filename
    image_data.save(uploaded_path)
    uploaded_url = f"/images/{uploaded_filename}"
    
    return JSONResponse({
        "uploaded_image": uploaded_url,
        "match": file_paths[best_idx],
        "distance": float(distances[best_idx]),
    })


@app.post("/api/upload")
def upload(image: UploadFile = File(...)):
    global pose_index, file_paths
    
    try:
        image_data = Image.open(image.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")
    
    vector = extract_pose_vector(image_data)
    if vector is None:
        raise HTTPException(status_code=400, detail="No pose detected in the image.")
    
    # Create data and images directories if they don't exist
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename and save image
    file_ext = Path(image.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    image_path = IMAGES_DIR / unique_filename
    image_data.save(image_path)
    
    # Store the image path relative to the images directory for serving
    image_url_path = f"/images/{unique_filename}"
    
    if pose_index is None:
        new_index = np.array([vector])
        new_file_paths = [image_url_path]
    else:
        new_index = np.vstack([pose_index, vector])
        new_file_paths = file_paths + [image_url_path]
    
    np.savez(INDEX_PATH, vectors=new_index)
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(new_file_paths, f)
    load_index()
    return JSONResponse({"message": "Image uploaded and index updated."})


@app.post("/api/clear-index")
def clear_index():
    global pose_index, file_paths
    
    # Delete images directory and all uploaded images
    if IMAGES_DIR.exists():
        shutil.rmtree(IMAGES_DIR)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Delete index files
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
    if DB_PATH.exists():
        DB_PATH.unlink()
    
    # Reset global variables
    pose_index = None
    file_paths = []
    
    return JSONResponse({"message": "Index cleared. All images and index files deleted."})