import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from app.page_segmentation import segment_panels
from app.pose_detector import normalize_pose


def make_grid_page(rows: int = 2, columns: int = 2) -> Image.Image:
    width, height, gutter = 1000, 800, 20
    panel_width = (width - (columns - 1) * gutter) // columns
    panel_height = (height - (rows - 1) * gutter) // rows
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for row in range(rows):
        for column in range(columns):
            left = column * (panel_width + gutter)
            top = row * (panel_height + gutter)
            draw.rectangle(
                (left + 5, top + 5, left + panel_width - 5, top + panel_height - 5),
                fill="black",
            )
    return image


def make_page_with_narrow_gray_gutter() -> Image.Image:
    image = Image.new("RGB", (659, 1024), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 390, 420), fill=(200, 200, 200))
    draw.rectangle((408, 20, 650, 420), fill=(200, 200, 200))
    draw.rectangle((20, 455, 650, 1000), fill=(200, 200, 200))
    draw.rectangle((391, 20, 407, 420), fill=(235, 235, 235))
    for y in range(433, 450):
        for x in range(0, 659, 50):
            draw.point((x, y), fill="black")
    return image


class PanelSegmentationTests(unittest.TestCase):
    def test_splits_regular_page_gutters(self):
        panels = segment_panels(make_grid_page())

        self.assertEqual(len(panels), 4)
        for left, top, right, bottom in panels:
            self.assertGreater(right, left)
            self.assertGreater(bottom, top)

    def test_respects_panel_limit(self):
        panels = segment_panels(make_grid_page(rows=3, columns=3), max_panels=4)

        self.assertEqual(len(panels), 4)

    def test_splits_narrow_gray_vertical_gutter_and_noisy_horizontal_gutter(self):
        panels = segment_panels(make_page_with_narrow_gray_gutter())

        self.assertEqual(len(panels), 3)
        upper_panels = sorted(
            (panel for panel in panels if panel[1] == 0),
            key=lambda panel: panel[0],
        )
        self.assertEqual(len(upper_panels), 2)
        self.assertLessEqual(upper_panels[0][2], upper_panels[1][0])

    def test_blank_page_is_one_panel(self):
        image = Image.new("RGB", (120, 200), "white")

        self.assertEqual(segment_panels(image), [(0, 0, 120, 200)])

    def test_outer_white_margin_is_not_split_into_empty_panels(self):
        image = Image.new("RGB", (400, 400), "white")
        ImageDraw.Draw(image).rectangle((50, 50, 350, 350), fill="black")

        self.assertEqual(segment_panels(image), [(0, 0, 400, 400)])

    def test_rejects_non_positive_panel_limit(self):
        with self.assertRaises(ValueError):
            segment_panels(Image.new("RGB", (100, 100), "white"), max_panels=0)


class PoseVectorTests(unittest.TestCase):
    def test_normalization_preserves_media_pipe_vector_shape(self):
        coords = np.arange(66, dtype=np.float32)

        vector = normalize_pose(coords)

        self.assertEqual(vector.shape, (66,))
        self.assertAlmostEqual(
            float(np.linalg.norm(vector.reshape(-1, 2), axis=1).max()),
            1.0,
            delta=1e-6,
        )


class PagePosePipelineTests(unittest.TestCase):
    @patch("app.main.extract_pose_detections")
    def test_maps_detected_pose_bounds_to_page_coordinates(self, detect):
        from app.main import _page_detections

        detect.return_value = [
            (np.zeros(66, dtype=np.float32), (0.25, 0.25, 0.75, 0.75))
        ]
        vectors, records, panels = _page_detections(make_grid_page())

        self.assertEqual(len(panels), 4)
        self.assertEqual(len(vectors), 4)
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0]["panel_index"], 0)
        left, top, right, bottom = panels[0]
        self.assertEqual(
            records[0]["pose_bounds"],
            [
                left + round((right - left) * 0.25),
                top + round((bottom - top) * 0.25),
                left + round((right - left) * 0.75),
                top + round((bottom - top) * 0.75),
            ],
        )


if __name__ == "__main__":
    unittest.main()
