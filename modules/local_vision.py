"""llama.cpp launch arguments shared by CPU, Metal, and CUDA experiments."""

import json
import re


def server_command(binary, model, projector, backend, port=8091, devices="", threads=4):
    command = [str(binary), "-m", str(model), "--mmproj", str(projector),
               "--host", "127.0.0.1", "--port", str(port), "-c", "4096",
               "--parallel", "1", "-t", str(threads), "--alias", "local-vision", "--verbosity", "4"]
    if backend == "cpu":
        command += ["--device", "none", "-ngl", "0", "--no-mmproj-offload", "--no-op-offload"]
    elif backend in ("metal", "cuda"):
        prefix = "MTL" if backend == "metal" else "CUDA"
        match = re.search(r"^\s*(" + prefix + r"\w*)\s*:", devices, re.MULTILINE)
        if not match:
            raise ValueError(f"No {backend} device: install a llama.cpp build with that backend")
        command += ["--device", match.group(1), "-ngl", "99"]
    else:
        raise ValueError("backend must be cpu, metal, or cuda")
    return command


def decode_json(text):
    return json.JSONDecoder().raw_decode(text[text.index("{"):])[0]


def point_response_format(observations=False, bbox=False, candidates=False):
    result = {"type": "json_schema", "json_schema": {
        "name": "vision_point", "strict": True, "schema": {
            "type": "object", "properties": {
                "target": {"type": "string"},
                "center_px": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000}, "minItems": 2, "maxItems": 2},
                "uncertain": {"type": "boolean"},
            }, "required": ["target", "center_px", "uncertain"], "additionalProperties": False,
        },
    }}
    if observations:
        schema = result["json_schema"]["schema"]
        schema["properties"] = {"observations": {"type": "string"}, **schema["properties"]}
        schema["required"] = ["observations", *schema["required"]]
    if bbox:
        schema = result["json_schema"]["schema"]
        schema["properties"].pop("center_px")
        schema["properties"]["bbox_2d"] = {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 2000}, "minItems": 4, "maxItems": 4}
        schema["required"] = ["bbox_2d" if field == "center_px" else field for field in schema["required"]]
    if candidates:
        schema = result["json_schema"]["schema"]
        schema["properties"] = {
            "reference_attribute": {"type": "string"},
            "candidates": {"type": "array", "minItems": 1, "maxItems": 6, "items": {
                "type": "object", "properties": {
                    "character": {"type": "string"}, "color": {"type": "string"}, "facing": {"type": "string"},
                    "center_px": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 2000}, "minItems": 2, "maxItems": 2},
                }, "required": ["character", "color", "facing", "center_px"], "additionalProperties": False,
            }},
            "selected_index": {"type": "integer", "minimum": 0, "maximum": 5}, "uncertain": {"type": "boolean"},
        }
        schema["required"] = list(schema["properties"])
    return result


def vision_prompt(instruction, width, height, variant="baseline"):
    if variant == "candidates":
        return (
            f"Task: {instruction}. Image size {width}x{height} pixels. "
            "Before selecting, detect EVERY instance of the requested target letter, digit, or shape. "
            "If the instruction uses a reference character, first identify its requested attribute, such as its front-face color. "
            "Return reference_attribute, then candidates array ONLY of matching target characters/shapes (not the reference). "
            "For each candidate record character, actual visible front-face color, front-facing or side-facing, and center_px. "
            "Do not stop at the first matching letter. Compare each candidate's attribute to the reference or instruction. "
            "Finally set selected_index to the zero-based index of the ONE matching candidate, and uncertain. "
            f"All center_px are absolute image pixels, x in 0..{width-1}, y in 0..{height-1}. "
            "Use interior centers, not text baselines. Output JSON only."
        )
    if variant == "baseline":
        return (
            f"Task: {instruction}. Find the ONE matching object in this image. Return JSON only: "
            '{"target":"object name", "center_px":[x,y],"uncertain":false}. '
            f"Image width {width} pixels, height {height} pixels. "
            f"x and y are absolute image pixels, origin top left, bottom right [{width-1},{height-1}]. "
            "Point inside the target face. If unsure return uncertain true."
        )
    if variant == "bbox":
        return (
            f"Instruction: {instruction}. Identify ALL instances of the requested target letter, number, or shape. "
            "Select the instance satisfying the instruction, not just the first matching character. "
            "For a color comparison, match the target front-face color to the reference front-face color. "
            "For side-on characters, compare the width of the front face with the visible side face. "
            f"Provide the 2D bounding box of only the selected target in this {width}x{height} pixel image. "
            "Use absolute pixel coordinates [x_min,y_min,x_max,y_max], origin top-left. "
            "Write one concise observation of the selected object, then target, bbox_2d, uncertain as JSON. "
            "Do not output a click point or normalized coordinates. If the target cannot be distinguished, uncertain=true."
        )
    if variant != "compare":
        raise ValueError("unknown prompt variant")
    return (
        f"Selection instruction: {instruction}. This image contains competing objects. "
        "Select exactly ONE target, not the reference object. Uppercase and lowercase differ. "
        "For 'same color as X', identify X's front-face color, then compare ONLY objects of the requested target character. "
        "For 'same facing direction', compare which side face is visible and its direction. "
        "For 'sideways', select the character with a narrow front face and a prominent side face, not merely a tilted character. "
        "Ignore shadow color. Do not select the first instance just because its letter matches. "
        "In observations, give one short sentence describing visible reference/target attributes. "
        "Then return target, center_px, uncertain. center_px must be INSIDE the selected object's front face. "
        f"The image is {width}x{height} pixels; x is 0..{width-1}, y is 0..{height-1}, origin upper-left. "
        "Use absolute pixels, not percentages or 0-1000 normalization. Return JSON only."
    )
