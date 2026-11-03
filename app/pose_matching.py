"""Compare shared, trustworthy body joints across photo and drawing models."""
import numpy as np

from .pose_detector import BODY_JOINTS, JOINT_CONFIDENCE


def body_distances(matrix, query, scores, query_scores, dimensions=None, query_dimensions=None):
    bodies = matrix.reshape(-1, 33, 2)[:, BODY_JOINTS].astype(np.float32)
    target = np.asarray(query, np.float32).reshape(33, 2)[BODY_JOINTS]
    if dimensions is not None and query_dimensions is not None:
        bodies *= np.asarray(dimensions, np.float32)[:, None, :]
        target = target * np.asarray(query_dimensions, np.float32)
    confidence = np.minimum(scores[:, BODY_JOINTS], np.asarray(query_scores, np.float32)[BODY_JOINTS])
    visible = confidence >= JOINT_CONFIDENCE
    weights = np.where(visible, confidence ** 2, 0)
    total = np.maximum(weights.sum(axis=1), 1e-8)
    centers = (bodies * weights[..., None]).sum(axis=1) / total[:, None]
    target_centers = (target[None] * weights[..., None]).sum(axis=1) / total[:, None]
    bodies -= centers[:, None]
    targets = target[None] - target_centers[:, None]
    scales = np.sqrt(((bodies ** 2).sum(axis=2) * weights).sum(axis=1) / total)
    target_scales = np.sqrt(((targets ** 2).sum(axis=2) * weights).sum(axis=1) / total)
    differences = bodies / np.maximum(scales[:, None, None], 1e-8) - targets / np.maximum(target_scales[:, None, None], 1e-8)
    distances = np.sqrt(((differences ** 2).sum(axis=2) * weights).sum(axis=1) / total)
    # A match based on only one arm must not outrank a comparable full-body match for free.
    distances += .2 * (1 - visible.sum(axis=1) / len(BODY_JOINTS))
    valid = (visible.sum(axis=1) >= 4) & (visible[:, [0, 1, 6, 7]].sum(axis=1) >= 2) & (scales > 1e-6) & (target_scales > 1e-6)
    return np.where(valid, distances, np.inf)
