# Pose Library

A local FastAPI web app for finding manga poses that resemble a human photo. Import your own sources, review panels and skeletons, then search individual figures or browse pose suggestions. No manga collection is supplied with the app.

## Run locally

Use Python 3.12 and a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/setup_pose_models.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Run one server worker per library. The app has no account system and accepts localhost hostnames; it is intended to run on your computer. MediaPipe is used for photos. Manga uses a HumanArt-trained YOLOX person detector followed by RTMPose body estimation. The setup script downloads approximately 150 MiB of official model files into ignored `app/data/models/`; inference uses those local files and never downloads models or sends images. Set `POSE_MODEL_DIR` to store models elsewhere. Upstream URLs and hashes are recorded in `manifest.json`. PDF rendering uses pypdfium2; a separate Poppler installation is not required.

## Build a collection

1. In **My library**, create a collection and optionally record its source, license or permission, attribution, and notes.
2. Import local PDFs, page images, image folders, or CBZ/ZIP archives containing page images. Folder imports retain relative filenames. Archives use natural filename order; their paths are never extracted onto disk.
3. Watch import progress. Pages are rendered and processed sequentially in the background. You can cancel between panels; already completed pages remain available. Invalid pages generate warnings while valid pages continue.
4. Review the candidate panels. Adjust the crop using page-pixel fields or **Draw crop**, then **Re-detect** if needed. **Save crop as another panel** lets you split or add a missed panel from the same source page.
5. Correct joints by dragging them on the source page or editing their coordinates. Only confident joints are shown by default; **Show uncertain joints** reveals the remaining estimates. Setting or dragging a joint marks it as trusted. Add a skeleton for a missed figure, or remove false detections. Set the total participant count, which may include figures without complete skeletons. Gender tags describe characters and default to unknown.
6. **Accept & index** makes a panel searchable. **Save changes** keeps its current status. **Skip panel** removes it from search; skipped panels remain editable.

Imports accept up to 20,000 files, 1 GiB per file, and 4 GiB per batch. Archive images are limited to 100 MiB each, source images to 40 megapixels, PDFs to 10,000 pages, and image archives to 20,000 pages. PDF pages render at up to 2× scale with a 2,000-pixel longest side. Animated/multipage image files use their first frame. No source PDF or archive is retained after import; rendered page images and provenance are retained.

Re-detection replaces skeletons and resets character tags to unknown; the UI asks before replacing those edits.

To update an existing collection after upgrading the detector, choose **Re-detect unreviewed panels**. This cancellable job reuses saved crops, without re-uploading or re-segmenting the source. It skips accepted/skipped panels, tagged or annotated panels, corrected participant counts, and skeletons manually edited after this upgrade. Older joint edits had no provenance marker: pending panels edited before the upgrade may be replaced if they have no tags or notes. The confirmation explains this, and a SQLite backup is saved under `app/data/backups/before-redetect-<job-id>.sqlite3` before processing. To restore a backup, stop the app and replace `catalog.sqlite3` with that file; page images are unchanged by re-detection.

For terminal imports, see [scripts/README.md](scripts/README.md).

## Match and suggest poses

**Match my pose** compares detected individuals in your photo with individual figures in accepted panels. It returns the closest panels, deduplicated so a group panel appears only once. The matching figure is outlined. Distance is a relative pose distance, not an accuracy percentage.

Matching compares the 12 shared shoulder/elbow/wrist/hip/knee/ankle landmarks, corrects for image aspect ratio, and aligns scale and translation using jointly visible landmarks. Low-confidence joints, extra face points, and unsupported fingers do not affect the comparison. Partial-body matches need at least four shared body landmarks and carry a coverage penalty. This lets an upper-body drawing match a full-body photo without treating invented legs as reliable evidence.

The optional participant-count filter applies to the **whole panel**. A photo of one person can therefore match a figure within a two-person panel. The gender filter includes panels with at least one character explicitly tagged with that value. Unknown tags match only when you select Unknown; they do not match a chosen known gender.

**Suggest poses** browses accepted panels without a query image using the same filters. Query photos are processed locally, are not added to the catalog, and are not retained by the app. The preview uses a browser object URL. Imported pages are served only through catalog-backed image routes, with caching disabled; the old unrestricted `/images` mount is removed.

## Data, migration, and backups

The local data directory contains `catalog.sqlite3`, rendered `pages/`, and temporary `staging/` uploads. Set `POSE_DATA_DIR` to use a different directory; the default is `app/data/`, excluded from Git. Stop the app before copying the entire data directory for a backup.

On first launch, existing `pose_index.npz`, `db_files.json`, and optional `pose_records.json` are imported into **Migrated collection**. Figures sharing an image and panel bounds stay grouped. Existing vectors and image content are preserved; gender labels start unknown. Original files remain as private backups, and a migration marker prevents deleted datasets from being reimported. Migration failures leave the originals intact. Migrated vectors have no raw joints: re-detect a panel or replace its skeleton to edit joints.

Deleting a collection removes its catalog records and rendered pages, preserving original source files and legacy backups. Active imports must be stopped first. An interrupted import is marked Interrupted on restart; completed pages remain available, and you can import the remaining source files again. Importing the same source twice creates separate records; automatic deduplication/resume is not implemented.

## API

Interactive documentation is available at `/docs`.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/datasets`, `POST /api/datasets` | List or create collections with provenance |
| `PATCH /api/datasets/{id}`, `DELETE /api/datasets/{id}` | Edit details or delete a collection |
| `POST /api/datasets/{id}/imports` | Multipart `files` uploads; returns a background import job |
| `POST /api/datasets/{id}/redetect` | Re-detect eligible unreviewed panels; returns a cancellable job |
| `GET /api/imports`, `GET /api/imports/{id}` | Import progress and warnings |
| `POST /api/imports/{id}/cancel` | Request cancellation |
| `GET /api/panels` | Paginated panels; filters: `dataset_id`, `status`, `participant_count`, `gender` |
| `GET /api/panels/{id}`, `PATCH /api/panels/{id}` | Review metadata and skeletons |
| `POST /api/panels/{id}/redetect` | Re-run pose detection for supplied `bounds` |
| `POST /api/pages/{id}/panels` | Add a candidate panel with supplied `bounds` |
| `GET /api/panels/{id}/image`, `GET /api/pages/{id}/image` | Panel crop or full source page |
| `POST /api/search` | Multipart `image`, optional `dataset_id`, `participant_count`, `gender`, `limit`; ranked `matches` |

Panel bounds are `[left, top, right, bottom]` in source-page pixels. PATCH `poses`, when provided, replaces the figure skeletons with arrays of 33 MediaPipe `[x,y]` landmark pairs in source-page pixels. Participant count must cover all skeletons; `genders` has one tag per participant. Tags are `unknown`, `male`, `female`, `nonbinary`, or `other`. An accepted panel requires at least one skeleton.

The 33-landmark storage/editor layout is retained for compatibility. Pose records also contain `joint_scores` (33 numbers in `[0,1]`) and `backend`. RTMPose's named joints are mapped to this layout; unsupported landmarks have zero confidence. PATCH `pose_scores`, when supplied alongside `poses`, preserves confidence for unedited joints. If scores are omitted, manually supplied joints are trusted. Older databases gain these columns automatically; old vectors remain intact and their backend is marked `legacy` until re-detected or edited.

The original `/api/upload`, `/api/segment-page`, `/api/full-page/search`, `/api/full-page/upload`, and `/api/clear-index` endpoints remain available. Explicit legacy upload/index actions add detected poses immediately; use the new import route for review before indexing. `/api/search` keeps first-match fields for existing callers, but no longer returns `uploaded_image`; use your local file for a query preview. **Page tools** retains the single-page workflow.

## Verification and limitations

```powershell
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Tests use temporary collections and synthetic images/PDFs. They cover PDF/image/archive import, review gating, cancellation, source provenance, tag filters, ranking, migration, deletion, and query-photo retention. Inference is stubbed in pipeline tests; they verify behavior rather than retrieval accuracy.

Panel segmentation uses a gutter heuristic and can miss overlapping or borderless panels. The drawing pipeline rejects detections without enough confident body joints and never creates a full-image skeleton when person detection finds no figure. It can still miss foreshortened or highly stylized bodies. At most eight pose candidates per panel are retained; a face alone is not a body pose. Automatic participant count is therefore a pose-candidate estimate, not a complete census of all characters. Manual corrections remain useful. Detection quality has been visually compared on a small local panel sample; retrieval accuracy still needs a larger labeled evaluation. Matching does not compare group arrangements or guarantee that the highest-ranked result looks correct.

## Source rights

Public distribution of this software is separate from a user's private image collection. The app does not supply or fetch a JoJo corpus. Online availability is not a reuse license, and running locally does not by itself determine whether copying or indexing particular source files is permitted. Record the source terms and obtain appropriate advice before hosting uploads or distributing a manga dataset.
