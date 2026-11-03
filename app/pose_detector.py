import threading
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image


MODEL_PATH = Path(__file__).resolve().parent / "pose_landmarker.task"
MAX_POSES = 8
MAX_IMAGE_SIDE = 1280
PHOTO_POSE_CONFIDENCE = 0.3
MANGA_POSE_CONFIDENCE = 0.1


def normalize_pose(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return coords.astype(np.float32)
    coords = coords.reshape(-1, 2).astype(np.float32)
    coords -= coords.mean(axis=0)
    scale = float(np.max(np.linalg.norm(coords, axis=1)))
    if scale > 0:
        coords /= scale
    return coords.flatten()


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


class PoseDetector:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        pose_confidence: float = PHOTO_POSE_CONFIDENCE,
    ) -> None:
        self._model_path = model_path
        self._pose_confidence = pose_confidence
        self._detector = None
        self._lock = threading.Lock()

    def _get_detector(self):
        if self._detector is None:
            if not self._model_path.is_file():
                raise FileNotFoundError(f"MediaPipe pose model not found: {self._model_path}")
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self._model_path)),
                num_poses=MAX_POSES,
                min_pose_detection_confidence=self._pose_confidence,
                min_pose_presence_confidence=self._pose_confidence,
                output_segmentation_masks=False,
            )
            self._detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        return self._detector

    def warm_up(self) -> None:
        self.detect(Image.new("RGB", (256, 256), "white"))

    def detect_details(self, image: Image.Image) -> list[dict]:
        rgb = image.convert("RGB")
        if max(rgb.size) > MAX_IMAGE_SIDE:
            scale = MAX_IMAGE_SIDE / max(rgb.size)
            rgb = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.Resampling.LANCZOS)

        image_array = np.asarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)

        with self._lock:
            result = self._get_detector().detect(mp_image)

        candidates = []
        for landmarks in result.pose_landmarks or []:
            coords = np.array([(landmark.x, landmark.y) for landmark in landmarks], dtype=np.float32)
            bounds = (
                float(np.clip(coords[:, 0].min(), 0, 1)),
                float(np.clip(coords[:, 1].min(), 0, 1)),
                float(np.clip(coords[:, 0].max(), 0, 1)),
                float(np.clip(coords[:, 1].max(), 0, 1)),
            )
            visibility = float(np.mean([landmark.visibility for landmark in landmarks]))
            candidates.append((visibility, normalize_pose(coords.copy()), bounds, coords))

        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        detections = []
        kept_bounds = []
        for _, vector, bounds, coords in candidates:
            if any(_box_iou(bounds, kept) > 0.55 for kept in kept_bounds):
                continue
            detections.append({"vector": vector, "bounds": bounds, "joints": coords.tolist()})
            kept_bounds.append(bounds)
        return detections

    def detect(self, image: Image.Image) -> list[tuple[np.ndarray, tuple[float, float, float, float]]]:
        return [(pose["vector"], pose["bounds"]) for pose in self.detect_details(image)]

    def close(self) -> None:
        with self._lock:
            if self._detector is not None:
                self._detector.close()
                self._detector = None


_pose_detector = PoseDetector()
_manga_pose_detector = PoseDetector(pose_confidence=MANGA_POSE_CONFIDENCE)


def extract_pose_detections(
    image: Image.Image,
    manga: bool = False,
) -> list[tuple[np.ndarray, tuple[float, float, float, float]]]:
    detector = _manga_pose_detector if manga else _pose_detector
    return detector.detect(image)


def extract_pose_details(image: Image.Image, manga: bool = False) -> list[dict]:
    detector = _manga_pose_detector if manga else _pose_detector
    return detector.detect_details(image)


def extract_pose_vectors(image: Image.Image) -> list[np.ndarray]:
    return [vector for vector, _ in extract_pose_detections(image)]


def extract_pose_vector(image: Image.Image) -> np.ndarray | None:
    vectors = extract_pose_vectors(image)
    return vectors[0] if vectors else None
