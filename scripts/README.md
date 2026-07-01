# Batch Upload Script

This folder contains utility scripts for the Pose Search App.

## batch_upload.py

Batch upload images from a folder to the Pose Search App with automatic duplicate detection.

### Features
- Uploads multiple images at once
- Detects duplicate poses using vector similarity
- Skips images that are too similar to existing poses
- Provides detailed feedback on each image processed
- Configurable similarity threshold

### Usage

Make sure the app is running first:
```powershell
cd c:\Users\bensa\pose_search_app
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

Then in another terminal:
```powershell
cd c:\Users\bensa\pose_search_app
.\venv\Scripts\Activate.ps1
python scripts\batch_upload.py <folder_path>
```

### Examples

Upload images from a folder with default settings:
```powershell
python scripts\batch_upload.py C:\path\to\images
```

Upload with custom similarity threshold (lower = stricter duplicate detection):
```powershell
python scripts\batch_upload.py C:\path\to\images --similarity-threshold 0.03
```

Upload to a different API server:
```powershell
python scripts\batch_upload.py C:\path\to\images --api-url http://192.168.1.100:8000
```

### Output

The script will display:
- Progress for each image (UPLOADED, SKIPPED, FAILED)
- Distance metric for duplicate matches
- Summary statistics at the end

Example output:
```
Found 10 image(s) to process.
Similarity threshold: 0.05
API URL: http://localhost:8000

Loading current index...
Current index has 5 image(s).

[1/10] Processing: photo1.jpg... UPLOADED
[2/10] Processing: photo2.jpg... SKIPPED - Duplicate (distance: 0.0234, similar to: /images/abc123.jpg)
[3/10] Processing: photo3.jpg... UPLOADED
...
==================================================
BATCH UPLOAD SUMMARY
==================================================
Uploaded:   8
Duplicates: 1
Failed:     1
Total:      10
==================================================
```

### Similarity Threshold

The similarity threshold controls how strict the duplicate detection is:
- **Lower values (e.g., 0.01)**: Stricter, only very similar poses are detected as duplicates
- **Higher values (e.g., 0.10)**: Looser, more poses considered as duplicates
- **Default: 0.05**: Good balance for most use cases

The distance is measured using Euclidean norm of the normalized pose vectors.
