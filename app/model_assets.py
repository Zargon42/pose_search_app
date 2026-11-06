"""Explicit model installation; inference never downloads weights or sends images."""
from pathlib import Path
import os

MANGA_MODEL_DIR = Path(os.environ.get("POSE_MODEL_DIR", Path(__file__).resolve().parent / "data" / "models"))
MANGA_MODELS = {
    "humanart_detector.onnx": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
    "rtmpose_body26.onnx": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
}
