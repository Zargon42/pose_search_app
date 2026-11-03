# Local batch imports

Start the app, then run:

```powershell
python scripts/batch_upload.py "C:\path\to\volume.pdf" --name "My references"
python scripts/batch_upload.py "C:\path\to\page-folder" --dataset-id EXISTING_ID
python scripts/batch_upload.py "C:\path\to\collection.cbz" --license "Your source's terms" --attribution "Creator"
```

The script recursively finds supported local files, streams them to the local app, reports background import progress, and prints a review link. It never reads the old NumPy index. Press Ctrl+C after the upload completes to request cancellation; completed pages remain available for review. Near-duplicate pose removal is no longer automatic: different panels can contain similar poses that you still want to keep.

Use `--api-url http://127.0.0.1:8000` for a different local port. The same batch limits as the UI apply: 20,000 files, 1 GiB per file, 4 GiB per batch. ZIP/CBZ files should contain page images, not nested archives.

`pinterest_upload.py` is an older optional scraper, separate from the import workflow. The app does not fetch online images or supply an external dataset. Check reuse permissions for any material you obtain elsewhere.
