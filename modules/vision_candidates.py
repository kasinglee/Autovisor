"""Image-derived candidates: no task answers or fixture labels are consulted."""
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def extract_candidates(image):
    rgb = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Flat pale-gray backgrounds are excluded; retain colorful and gray glyphs.
    mask = (((hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 60)) | (hsv[:, :, 2] < 185)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for label in range(1, count):
        x,y,w,h,area = map(int, stats[label])
        if area < 50 or w < 8 or h < 15 or w > image.width * .5 or h > image.height * .9:
            continue
        if x > image.width * .88 and y < image.height * .25:
            continue  # The real widget's upper-right refresh control is not a target.
        pixels = hsv[labels == label]
        colored = pixels[pixels[:, 1] > 55]
        if len(colored) < len(pixels) * .2:
            color = "gray"
        else:
            hue = float(np.median(colored[:, 0]))
            color = ("red" if hue < 10 or hue > 170 else "orange" if hue < 23 else
                     "yellow" if hue < 35 else "green" if hue < 85 else
                     "cyan" if hue < 100 else "blue" if hue < 130 else "purple" if hue < 160 else "pink")
        # Distance-transform maximum is inside the glyph, unlike a box center
        # that may fall in the hole of a digit or between strokes.
        component = (labels[y:y+h,x:x+w] == label).astype(np.uint8)
        distance = cv2.distanceTransform(np.pad(component, 1), cv2.DIST_L2, 5)
        py,px = np.unravel_index(distance.argmax(), distance.shape)
        candidates.append({"bbox": [x,y,x+w,y+h], "center_px": [x+int(px)-1,y+int(py)-1], "color": color})
    candidates.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
    for index, candidate in enumerate(candidates):
        candidate["id"] = index
    if not candidates or len(candidates) > 12:
        raise ValueError("Ambiguous image segmentation")
    return candidates


def candidate_image(image, candidates, large_labels=False):
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default(size=max(12, image.width // 30)) if large_labels else None
    for candidate in candidates:
        x0,y0,x1,y1 = candidate["bbox"]
        draw.rectangle((x0,y0,x1,y1), outline="black", width=1)
        label = f"[{candidate['id']}]" if large_labels else str(candidate['id'])
        if large_labels:
            bounds = draw.textbbox((0,0), label, font=font)
            width,height = bounds[2]+6, bounds[3]+4
        else:
            width,height = 14,11
        y = max(0,y0-height-1)
        draw.rectangle((x0,y,x0+width,y+height), fill="white", outline="black")
        draw.text((x0+2,y), label, fill="black", font=font)
    return annotated


def selection_format(explicit=False, short=False):
    result = {"type": "json_schema", "json_schema": {"name": "candidate_selection", "strict": True, "schema": {
        "type": "object", "properties": {
            "reference_attribute": {"type": "string"},
            "selected_id": {"type": "integer", "minimum": 0, "maximum": 11},
            "uncertain": {"type": "boolean"},
        }, "required": ["reference_attribute", "selected_id", "uncertain"], "additionalProperties": False,
    }}}
    if explicit:
        schema = result["json_schema"]["schema"]
        schema["properties"] = {
            "target_character": {"type": "string"},
            "reference_id": {"type": "integer", "minimum": -1, "maximum": 11},
            **schema["properties"],
        }
        schema["required"] = list(schema["properties"])
    if short:
        schema = result["json_schema"]["schema"]
        schema["properties"].pop("reference_attribute")
        schema["required"].remove("reference_attribute")
    return result


def grounded_prompt(instruction, candidates):
    return (
        f"Task: {instruction}. Objects are boxed and numbered. Select ONE box ID matching the instruction. "
        "The numbered white tags are annotations, not objects. Candidate colors are measured, not guesses. "
        "For same-color tasks identify the reference character, then choose the TARGET character with matching measured color. "
        "For side-on letters choose the thin/narrow letter with a prominent visible side, not the broad frontal letter. "
        "Do not select the reference object. Return reference_attribute, selected_id, uncertain as JSON. "
        "Candidates: " + json.dumps([{k:v for k,v in c.items() if k != 'center_px'} for c in candidates])
    )


def selected_point(answer, candidates):
    index = answer.get("selected_id")
    if answer.get("uncertain") is not False or type(index) is not int or not (0 <= index < len(candidates)):
        raise ValueError("Uncertain or invalid candidate selection")
    return candidates[index]["center_px"]
