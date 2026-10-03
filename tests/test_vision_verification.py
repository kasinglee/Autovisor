import unittest

from modules.vision_verification import parse_point, to_page_point


class VisionPointTests(unittest.TestCase):
    def test_point_converted_for_scaled_offset_dialog(self):
        point = parse_point('```json\n{"center_px":[100,50],"uncertain":false}\n```', 400, 200)
        self.assertEqual(to_page_point(point, (400, 200), {
            "x": 600, "y": 300, "width": 800, "height": 400,
        }), (800, 400))

    def test_rejects_uncertain_outside_and_invalid_coordinates(self):
        for content in (
            '{"center_px":[100,50],"uncertain":true}',
            '{"center_px":[400,50],"uncertain":false}',
            '{"center_px":[100,-1],"uncertain":false}',
            '{"center_px":[true,50],"uncertain":false}',
            '{"center_px":[NaN,50],"uncertain":false}',
            '{"center_px":null,"uncertain":false}',
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_point(content, 400, 200)

    def test_model_explanation_does_not_break_json_parsing(self):
        self.assertEqual(parse_point('说明文字\n```json\n{"center_px":[500,500],"uncertain":false}\n```', 1000, 1000), (500, 500))
