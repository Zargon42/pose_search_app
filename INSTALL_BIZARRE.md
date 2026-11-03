# Bizarre Pose Estimator (legacy)

The app no longer invokes the external Bizarre Pose Estimator. Its previous integration started a separate Python process for each image and could take several minutes; its keypoint layout was also not guaranteed to match the MediaPipe vectors stored in the search index.

The active pose backend is the in-process MediaPipe Pose Landmarker. It is initialized lazily and reused across image and panel inference. The external `bizarre-pose-estimator` checkout and `BIZARRE_MODEL` setting are not used by the app.

To compare another pose model in the future, adapt its named joints to the index's versioned pose schema and benchmark inference and retrieval quality before enabling it.
