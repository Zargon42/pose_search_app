Setup instructions for integrating the bizarre-pose-estimator

This project includes an optional integration with the WACV2022 "bizarre-pose-estimator" repository. The upstream project requires a non-trivial environment (GPU, specific PyTorch setup, and model checkpoints). The integration in `app/pose_detector.py` will run the upstream script as a subprocess and parse printed keypoints.

Steps to enable:

1. Clone the upstream repo into the workspace root (project root `pose_search_app`):

```bash
cd /path/to/pose_search_app
git clone https://github.com/ShuhongChen/bizarre-pose-estimator.git
```

2. Download the model checkpoints as described in the upstream README (see "download" section). Place or reference a checkpoint path for `BIZARRE_MODEL`.

3. Install the upstream environment. The upstream repo recommends using Docker for GPU support. If you choose to run locally in your Python environment, install the required packages inside the bizarre repo. Typical requirements include `torch` and other scientific/python libs — consult `bizarre-pose-estimator` for details.

4. Set the `BIZARRE_MODEL` environment variable to point to the checkpoint file you want to use. This can be absolute or relative to the `bizarre-pose-estimator` folder. Example (PowerShell):

```powershell
$env:BIZARRE_MODEL = "./_train/character_pose_estim/runs/feat_concat+data.ckpt"
# Or absolute path
$env:BIZARRE_MODEL = "C:\path\to\bizarre-pose-estimator\_train\...ckpt"
```

5. Start this app as usual. When `BIZARRE_MODEL` is set and the cloned repo exists, the server will attempt to run the upstream `pose_estimator` script for each image search/upload. If that script fails, the app will automatically fall back to the bundled MediaPipe pose extractor.

Notes and troubleshooting:
- The upstream code often expects GPU and a matching PyTorch/CUDA setup; running successfully outside Docker may require careful environment setup.
- If you prefer not to use the upstream estimator, unset `BIZARRE_MODEL` and the app will use MediaPipe (default).
- Running the upstream estimator can be slower and may require increasing the subprocess timeout inside `app/pose_detector.py`.

