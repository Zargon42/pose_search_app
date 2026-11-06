import threading
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

from .model_assets import MANGA_MODEL_DIR


MODEL_PATH = Path(__file__).resolve().parent / "pose_landmarker.task"
MAX_POSES = 8
MAX_IMAGE_SIDE = 1280
PHOTO_POSE_CONFIDENCE = 0.3
JOINT_CONFIDENCE = 0.35
BODY_JOINTS = np.array([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])
BODY_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


def halpe26_to_landmarks(keypoints, scores, width, height):
    """Map named joints; unsupported MediaPipe fingers/face points remain untrusted."""
    mapping = [0, 1, 1, 1, 2, 2, 2, 3, 4, 0, 0, 5, 6, 7, 8, 9, 10,
               9, 10, 9, 10, 9, 10, 11, 12, 13, 14, 15, 16, 24, 25, 20, 21]
    coords = np.asarray(keypoints, np.float32)[mapping] / [width, height]
    confidence = np.asarray(scores, np.float32)[mapping].copy()
    confidence[~np.isfinite(confidence)] = 0
    confidence[[1, 3, 4, 6, 9, 10, 17, 18, 19, 20, 21, 22]] = 0
    confidence[~np.isfinite(coords).all(axis=1) | (coords < 0).any(axis=1) | (coords > 1).any(axis=1)] = 0
    return np.clip(np.nan_to_num(coords), 0, 1), np.clip(np.nan_to_num(confidence), 0, 1)


def usable_body(coords, scores, image_size=(1, 1)):
    visible = np.asarray(scores) >= JOINT_CONFIDENCE
    points = np.asarray(coords)[BODY_JOINTS[visible[BODY_JOINTS]]]
    if len(points) < 4:
        return False
    spread = np.ptp(points * image_size, axis=0)
    return (float(spread.min() / max(spread.max(), 1e-8)) >= .07
            and int(visible[[11, 12, 23, 24]].sum()) >= 2
            and sum(bool(visible[a] and visible[b]) for a, b in BODY_EDGES) >= 2
            and float(np.max(np.ptp(points, axis=0))) > .04)


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
            joint_scores = [float(landmark.visibility * landmark.presence) for landmark in landmarks]
            candidates.append((visibility, normalize_pose(coords.copy()), bounds, coords, joint_scores))

        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        detections = []
        kept_bounds = []
        for _, vector, bounds, coords, joint_scores in candidates:
            if any(_box_iou(bounds, kept) > 0.55 for kept in kept_bounds):
                continue
            detections.append({"vector": vector, "bounds": bounds, "joints": coords.tolist(),
                               "joint_scores": joint_scores, "backend": "mediapipe-photo"})
            kept_bounds.append(bounds)
        return detections

    def detect(self, image: Image.Image) -> list[tuple[np.ndarray, tuple[float, float, float, float]]]:
        return [(pose["vector"], pose["bounds"]) for pose in self.detect_details(image)]

    def close(self) -> None:
        with self._lock:
            if self._detector is not None:
                self._detector.close()
                self._detector = None


class MangaPoseDetector:
    """Detect illustrated people before estimating poses; no full-image fallback."""
    def __init__(self):
        self._lock = threading.Lock()
        self._detector = self._pose_model = None

    def _load(self):
        if self._detector is not None:
            return
        detector_path = MANGA_MODEL_DIR / "humanart_detector.onnx"
        pose_path = MANGA_MODEL_DIR / "rtmpose_body26.onnx"
        if not detector_path.exists() or not pose_path.exists():
            raise RuntimeError("Manga pose models are missing. Run: python scripts/setup_pose_models.py")
        try:
            from rtmlib import YOLOX, RTMPose
        except ImportError as exc:
            raise RuntimeError("Manga pose dependencies are missing. Run: python -m pip install -r requirements.txt") from exc
        detector = YOLOX(str(detector_path), model_input_size=(640, 640), score_thr=.45,
                         nms_thr=.55, backend="onnxruntime", device="cpu")
        pose_model = RTMPose(str(pose_path), model_input_size=(192, 256),
                            backend="onnxruntime", device="cpu", to_openpose=False)
        # RTMLib 0.0.16 has no SessionOptions hook. Replace its default sessions once
        # so two models do not each create a thread pool spanning every CPU core.
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        for tool in (detector, pose_model):
            tool.session = ort.InferenceSession(tool.onnx_model, sess_options=options, providers=["CPUExecutionProvider"])
        self._detector, self._pose_model = detector, pose_model

    def detect_details(self, image):
        rgb = image.convert("RGB")
        if max(rgb.size) > MAX_IMAGE_SIDE:
            factor = MAX_IMAGE_SIDE / max(rgb.size)
            rgb = rgb.resize((max(1, round(rgb.width * factor)), max(1, round(rgb.height * factor))), Image.Resampling.LANCZOS)
        pixels = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])
        with self._lock:
            self._load()
            boxes = self._detector(pixels)
            # RTMLib otherwise invents a full-image person when boxes is empty.
            if not len(boxes):
                return []
            keypoints, scores = self._pose_model(pixels, bboxes=boxes)
        candidates = []
        for points, confidence in zip(keypoints, scores):
            coords, confidence = halpe26_to_landmarks(points, confidence, rgb.width, rgb.height)
            if not usable_body(coords, confidence, rgb.size):
                continue
            visible_coords = coords[confidence >= JOINT_CONFIDENCE]
            bounds = (float(visible_coords[:, 0].min()), float(visible_coords[:, 1].min()),
                      float(visible_coords[:, 0].max()), float(visible_coords[:, 1].max()))
            candidates.append({"vector": normalize_pose(coords.copy()), "bounds": bounds, "joints": coords.tolist(),
                               "joint_scores": confidence.tolist(), "backend": "rtmpose-humanart"})
        return sorted(candidates, key=lambda item: -sum(item["joint_scores"][i] for i in BODY_JOINTS))[:MAX_POSES]

    def detect(self, image):
        return [(pose["vector"], pose["bounds"]) for pose in self.detect_details(image)]

    def close(self):
        with self._lock:
            self._detector = self._pose_model = None


_pose_detector = PoseDetector()
_manga_pose_detector = MangaPoseDetector()


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
