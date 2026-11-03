**Pose Search App**

A FastAPI app that indexes and searches images by human pose. The MediaPipe Pose Landmarker is initialized lazily and reused across requests, and detects up to eight people per image. A separate full-page workflow segments manga pages at near-white gutters, then searches or indexes detected poses from the panels.

**Quick Start**
- **Prerequisites:** : Python 3.8+ and a working virtual environment.
- **Install:** :

```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- **Run (development):** :

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- **Open:** : Visit http://localhost:8000/ in your browser.

**Usage**
- **Web UI:** : The home page lets you perform a pose search; the Upload page lets you add images to the index; `/full-page` provides manga-page segmentation, search, and indexing.
- **API endpoints:** :
  - `POST /api/upload` — Upload an image to add to the index.
  - `POST /api/search` — Upload a query image to find the best pose match.
  - `POST /api/segment-page` — Return panel bounds without running pose inference.
  - `POST /api/full-page/search` — Segment a page and return the closest indexed match for each detected pose.
  - `POST /api/full-page/upload` — Segment a page and index its detected poses.
  - `POST /api/clear-index` — Clear the index and uploaded images.

**Project Structure**
- **app:** : Core FastAPI application and pose extraction logic. See [app/main.py](app/main.py).
- **data:** : Stores `pose_index.npz` and `db_files.json` and uploaded images. See [data/pose_index.npz](data/pose_index.npz) and [data/db_files.json](data/db_files.json).
- **static:** : Client-side JS and CSS used by the web UI. See [static/app.js](static/app.js) and [static/styles.css](static/styles.css).
- **templates:** : Jinja2 HTML templates. See [templates/index.html](templates/index.html) and [templates/upload.html](templates/upload.html).
- **scripts:** : Utility scripts for batch upload and other tasks. See [scripts/batch_upload.py](scripts/batch_upload.py) and [scripts/pinterest_upload.py](scripts/pinterest_upload.py).

**Notes & Tips**
- **Model file:** : The app expects the MediaPipe task file at `app/pose_landmarker.task`. If it's not present, download the lite model from the MediaPipe model zoo or provide the file at that path.
- **Pose vectors:** : Indexed vectors remain 66 normalized MediaPipe landmark coordinates. New page-index entries also record panel and pose bounds in `app/data/pose_records.json`; existing indexes without this metadata continue to load.
- **Manga limitations:** : Panel segmentation is a CPU-friendly gutter heuristic, not a learned panel detector. It can miss borderless or overlapping layouts and may split at large blank areas. MediaPipe is still a photo-oriented pose model; manga accuracy and the 500 ms per-panel target must be measured on representative pages before claiming either is met.
- **Dependencies:** : See `requirements.txt`; MediaPipe handles pose extraction and Pillow/NumPy handle panel segmentation.
- **Images served:** : Uploaded images are stored under `data/images` and served at the `/images` route.
- **Playwright:** : The repository lists `playwright` in `requirements.txt` — if you use any automation that relies on Playwright, run `playwright install` after installing Python deps.

**Development**
- **Run tests:** : `python -m unittest discover -s tests`
- **Run/manual checks:** : Use the development server command above and test single-image and full-page flows via the UI.
- **Reset index:** : Use the `POST /api/clear-index` endpoint to remove all uploaded images and reset indexing.

**License**
- **License:** : This project is provided as-is. Add a license file if you plan to distribute or publish the code.
