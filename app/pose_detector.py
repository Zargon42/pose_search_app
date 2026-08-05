import os
import re
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent / "bizarre-pose-estimator"


def normalize_pose(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return coords
    coords = coords.reshape(-1, 2)
    center = coords.mean(axis=0)
    coords = coords - center
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale > 0:
        coords = coords / scale
    return coords.flatten()


def extract_pose_vector_bizarre(image: Image.Image) -> Optional[np.ndarray]:
    """Run the external bizarre-pose-estimator script to extract keypoints.

    Requirements / expectations:
    - The upstream repo must be cloned to the workspace root at `bizarre-pose-estimator`.
    - An environment variable `BIZARRE_MODEL` must point to the model checkpoint path
      (relative to the bizarre repo or absolute).

    This function saves the image to a temporary file, runs the estimator as a subprocess,
    parses stdout for printed keypoints, and returns the normalized vector.
    """
    model_path = os.environ.get("BIZARRE_MODEL")
    if not model_path:
        raise RuntimeError("BIZARRE_MODEL env var not set; cannot use bizarre estimator")

    if not REPO_DIR.exists():
        raise RuntimeError(f"Expected bizarre repo at {REPO_DIR} but not found")

    # Save image to temporary PNG file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp_path = Path(tf.name)
    try:
        image.save(tmp_path)

        env = os.environ.copy()
        env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

        cmd = [sys.executable, "-m", "_scripts.pose_estimator", str(tmp_path), model_path]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_DIR),
                capture_output=True,
                text=True,
                check=True,
                timeout=240,
                env=env,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"bizarre estimator failed: {e.stderr}") from e

        out = proc.stdout
        # The script prints a block starting with 'keypoints' then lines with '(x, y) <label>'
        coords = []
        for m in re.finditer(r"\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)", out):
            x = float(m.group(1))
            y = float(m.group(2))
            coords.extend([x, y])

        if len(coords) == 0:
            return None

        vec = np.array(coords, dtype=np.float32)
        return normalize_pose(vec)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass
