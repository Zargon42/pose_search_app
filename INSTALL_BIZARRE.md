# Bizarre Pose Estimator (legacy)

The app no longer invokes the external Bizarre Pose Estimator. Its previous integration started a separate Python process for each image and could take several minutes; its keypoint layout was also not guaranteed to match the MediaPipe vectors stored in the search index.

Photos use MediaPipe. Manga uses local HumanArt YOLOX detection and RTMPose body estimation through ONNX Runtime, with per-joint confidence. Install models with `python scripts/setup_pose_models.py`; see README.md for setup and existing-collection re-detection. Models are loaded once and reused. The external `bizarre-pose-estimator` checkout and `BIZARRE_MODEL` setting are not used by the app.

The older Bizarre project assumes a single full-body character, which limits its usefulness on cropped or multi-person manga panels. Replacement models must map named joints consistently, preserve missing/uncertain landmark confidence, and be compared on representative pages before being enabled.
