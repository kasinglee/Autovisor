import unittest
from PIL import Image, ImageDraw

from modules.vision_candidates import extract_candidates, selected_point


class CandidateTests(unittest.TestCase):
    def test_detects_colors_and_points_inside_objects_without_labels(self):
        image = Image.new("RGB", (320,160), (231,231,231))
        draw = ImageDraw.Draw(image)
        draw.rectangle((25,30,75,85), fill=(50,130,240))
        draw.rectangle((200,90,260,145), fill=(90,180,50))
        candidates = extract_candidates(image)
        self.assertEqual(len(candidates), 2)
        self.assertEqual([c["color"] for c in candidates], ["blue", "green"])
        for candidate in candidates:
            x,y = candidate["center_px"]
            self.assertNotEqual(image.getpixel((x,y)), (231,231,231))

    def test_empty_image_is_not_clickable(self):
        with self.assertRaises(ValueError):
            extract_candidates(Image.new("RGB",(320,160),(231,231,231)))

    def test_uncertain_and_nonexistent_ids_are_not_clicked(self):
        for answer in ({"selected_id": 0, "uncertain": True}, {"selected_id": 1, "uncertain": False}, {"selected_id": True, "uncertain": False}):
            with self.assertRaises(ValueError):
                selected_point(answer, [{"center_px":[50,50]}])
