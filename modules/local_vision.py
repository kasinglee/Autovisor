"""llama.cpp launch arguments shared by CPU, Metal, and CUDA experiments."""

import json
import re


def server_command(binary, model, projector, backend, port=8091, devices=""):
    command = [str(binary), "-m", str(model), "--mmproj", str(projector),
               "--host", "127.0.0.1", "--port", str(port), "-c", "4096",
               "--parallel", "1", "-t", "4", "--alias", "local-vision", "--verbosity", "4"]
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


def point_response_format():
    return {"type": "json_schema", "json_schema": {
        "name": "vision_point", "strict": True, "schema": {
            "type": "object", "properties": {
                "target": {"type": "string"},
                "center_px": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000}, "minItems": 2, "maxItems": 2},
                "uncertain": {"type": "boolean"},
            }, "required": ["target", "center_px", "uncertain"], "additionalProperties": False,
        },
    }}
