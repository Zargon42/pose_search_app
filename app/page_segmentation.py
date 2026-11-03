import numpy as np
from PIL import Image


MAX_SEGMENTATION_SIDE = 1600
MAX_PANELS = 40
INK_THRESHOLD = 245
MAX_GUTTER_INK_RATIO = 0.005


def _candidate_cut(
    ink: np.ndarray,
    horizontal: bool,
) -> tuple[int, int] | None:
    height, width = ink.shape
    axis_length, orthogonal_length = (height, width) if horizontal else (width, height)
    if axis_length < 80 or orthogonal_length < 80:
        return None

    occupancy = ink.mean(axis=1 if horizontal else 0)
    is_gutter = occupancy <= MAX_GUTTER_INK_RATIO
    minimum_length = max(4, round(min(height, width) * 0.012))

    candidates = []
    start = None
    for position, empty in enumerate(is_gutter):
        if empty and start is None:
            start = position
        elif not empty and start is not None:
            if position - start >= minimum_length:
                center = (start + position) // 2
                if axis_length * 0.04 < center < axis_length * 0.96:
                    candidates.append((position - start, center))
            start = None
    if start is not None and axis_length - start >= minimum_length:
        center = (start + axis_length) // 2
        if axis_length * 0.04 < center < axis_length * 0.96:
            candidates.append((axis_length - start, center))

    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[0])


def _split_region(
    ink: np.ndarray,
    bounds: tuple[int, int, int, int],
    budget: int,
) -> list[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bounds
    if budget <= 1 or min(x1 - x0, y1 - y0) < 80:
        return [bounds]

    region = ink[y0:y1, x0:x1]
    cuts = []
    horizontal_cut = _candidate_cut(region, True)
    vertical_cut = _candidate_cut(region, False)
    if horizontal_cut is not None:
        cuts.append((horizontal_cut, True))
    if vertical_cut is not None:
        cuts.append((vertical_cut, False))
    cuts.sort(
        key=lambda item: item[0][0] / (region.shape[1] if item[1] else region.shape[0]),
        reverse=True,
    )
    split = None
    for (_, cut_position), horizontal in cuts:
        if horizontal:
            if cut_position < 80 or region.shape[0] - cut_position < 80:
                continue
            first = (x0, y0, x1, y0 + cut_position)
            second = (x0, y0 + cut_position, x1, y1)
        else:
            if cut_position < 80 or region.shape[1] - cut_position < 80:
                continue
            first = (x0, y0, x0 + cut_position, y1)
            second = (x0 + cut_position, y0, x1, y1)
        if not ink[first[1]:first[3], first[0]:first[2]].any():
            continue
        if not ink[second[1]:second[3], second[0]:second[2]].any():
            continue
        split = (first, second)
        break
    if split is None:
        return [bounds]
    first, second = split

    first_panels = _split_region(ink, first, budget - 1)
    second_budget = budget - len(first_panels)
    second_panels = _split_region(ink, second, second_budget)
    return first_panels + second_panels


def segment_panels(
    image: Image.Image,
    max_panels: int = MAX_PANELS,
) -> list[tuple[int, int, int, int]]:
    """Split a page at near-white gutters and return panel boxes as (left, top, right, bottom)."""
    if max_panels < 1:
        raise ValueError("max_panels must be at least one")

    rgb = image.convert("RGB")
    original_width, original_height = rgb.size
    if original_width == 0 or original_height == 0:
        raise ValueError("Page image must have non-zero dimensions")

    scale = min(1.0, MAX_SEGMENTATION_SIDE / max(original_width, original_height))
    if scale < 1.0:
        rgb = rgb.resize(
            (round(original_width * scale), round(original_height * scale)),
            Image.Resampling.LANCZOS,
        )

    gray = np.asarray(rgb.convert("L"))
    ink = gray < INK_THRESHOLD
    if not ink.any():
        return [(0, 0, original_width, original_height)]
    boxes = _split_region(ink, (0, 0, gray.shape[1], gray.shape[0]), max_panels)

    scale_x = original_width / gray.shape[1]
    scale_y = original_height / gray.shape[0]
    converted = []
    for x0, y0, x1, y1 in boxes:
        converted.append(
            (
                max(0, round(x0 * scale_x)),
                max(0, round(y0 * scale_y)),
                min(original_width, round(x1 * scale_x)),
                min(original_height, round(y1 * scale_y)),
            )
        )
    return sorted(converted, key=lambda box: (box[1], -box[2]))
