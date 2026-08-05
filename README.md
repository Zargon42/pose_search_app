**Pose Search App**

A small FastAPI web app that indexes and searches images by human pose using MediaPipe. Upload images to build a pose index, then search by uploading a query image to find the closest pose match.

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
- **Web UI:** : The home page lets you perform a pose search; the Upload page lets you add images to the index.
- **API endpoints:** :
  - `POST /api/upload` — Upload an image to add to the index.
  - `POST /api/search` — Upload a query image to find the best pose match.
  - `POST /api/clear-index` — Clear the index and uploaded images.

**Project Structure**
- **app:** : Core FastAPI application and pose extraction logic. See [app/main.py](app/main.py).
- **data:** : Stores `pose_index.npz` and `db_files.json` and uploaded images. See [data/pose_index.npz](data/pose_index.npz) and [data/db_files.json](data/db_files.json).
- **static:** : Client-side JS and CSS used by the web UI. See [static/app.js](static/app.js) and [static/styles.css](static/styles.css).
- **templates:** : Jinja2 HTML templates. See [templates/index.html](templates/index.html) and [templates/upload.html](templates/upload.html).
- **scripts:** : Utility scripts for batch upload and other tasks. See [scripts/batch_upload.py](scripts/batch_upload.py) and [scripts/pinterest_upload.py](scripts/pinterest_upload.py).

**Notes & Tips**
- **Model file:** : The app expects the MediaPipe task file at `app/pose_landmarker.task`. If it's not present, download the lite model from the MediaPipe model zoo or provide the file at that path.
- **Dependencies:** : See `requirements.txt` for libraries; `mediapipe` is required for pose extraction and may need platform-specific wheel support.
- **Images served:** : Uploaded images are stored under `data/images` and served at the `/images` route.
- **Playwright:** : The repository lists `playwright` in `requirements.txt` — if you use any automation that relies on Playwright, run `playwright install` after installing Python deps.

**Development**
- **Run tests/manual checks:** : Use the development server command above and test the upload/search flows via the UI.
- **Reset index:** : Use the `POST /api/clear-index` endpoint to remove all uploaded images and reset indexing.

**License**
- **License:** : This project is provided as-is. Add a license file if you plan to distribute or publish the code.
