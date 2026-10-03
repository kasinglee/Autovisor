"""Run identical, annotated image cases through a temporary localhost server."""

import argparse
import base64
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.local_vision import decode_json, point_response_format, server_command


def rss_mb(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "rss="], capture_output=True, text=True)
    return float(result.stdout.strip() or 0) / 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--backends", nargs="+", default=["cpu", "metal"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-limit-gb", type=float, default=8)
    args = parser.parse_args()
    binary = shutil.which("llama-server")
    if not binary:
        parser.error("llama-server not installed")
    devices = subprocess.run([binary, "--list-devices"], capture_output=True, text=True)
    device_text = devices.stdout + devices.stderr
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = json.loads((ROOT / "tests/fixtures/vision/cases.json").read_text())
    report = {"model": args.name, "devices": device_text, "memory_limit_gb": args.memory_limit_gb,
              "method": "Annotated real and synthetic screenshots; supplied English instruction; JSON constrained decoding; absolute image-pixel output checked against normalized target boxes. Not live platform pass rate. RSS excludes some Metal allocations.",
              "runs": []}
    for backend in args.backends:
        run = {"backend": backend, "cases": []}
        report["runs"].append(run)
        try:
            command = server_command(binary, args.model_dir / "model.gguf", args.model_dir / "mmproj.gguf", backend, devices=device_text)
        except ValueError as exc:
            run["status"] = "unavailable"
            run["error"] = str(exc)
            continue
        log_path = args.output.parent / f"{args.name}-{backend}-server.log"
        run["command"] = command
        start = time.monotonic()
        with log_path.open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            peak = [0.0]
            try:
                with httpx.Client(base_url="http://127.0.0.1:8091", timeout=120, trust_env=False) as client:
                    while True:
                        peak[0] = max(peak[0], rss_mb(process.pid))
                        if peak[0] > args.memory_limit_gb * 1024:
                            raise RuntimeError("Server RSS exceeded memory budget")
                        if process.poll() is not None:
                            raise RuntimeError("llama-server exited; inspect server log")
                        try:
                            if client.get("/health", timeout=1).status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        if time.monotonic() - start > 120:
                            raise TimeoutError("model load timeout")
                        time.sleep(0.5)
                    run["load_seconds"] = round(time.monotonic() - start, 2)
                    print(args.name, backend, "ready", flush=True)
                    # Sample throughout inference, not only after it finishes.
                    import threading
                    stop_sample = threading.Event()
                    def sample():
                        while not stop_sample.wait(0.1):
                            peak[0] = max(peak[0], rss_mb(process.pid))
                            if peak[0] > args.memory_limit_gb * 1024:
                                process.terminate()
                                break
                    sampler = threading.Thread(target=sample, daemon=True)
                    sampler.start()
                    try:
                        for case in cases:
                            with Image.open(ROOT / "tests/fixtures/vision" / case["file"]) as image:
                                crop = image.crop(tuple(case["crop"]))
                                crop = crop.resize((crop.width * 3, crop.height * 3))
                                buffer = io.BytesIO()
                                crop.save(buffer, format="PNG")
                            prompt = (
                                f"Task: {case['instruction_en']}. Chinese original: {case['instruction_zh']}. "
                                "Find the ONE matching object in this image. Return JSON only: "
                                '{"target":"object name", "center_px":[x,y],"uncertain":false}. '
                                f"Image width {crop.width} pixels, height {crop.height} pixels. "
                                f"x and y are absolute image pixels, origin top left, bottom right [{crop.width-1},{crop.height-1}]. "
                                "Point inside the target face. If unsure return uncertain true."
                            )
                            started = time.monotonic()
                            response = client.post("/v1/chat/completions", json={
                                "model": "local-vision", "temperature": 0, "max_tokens": 160,
                                "response_format": point_response_format(),
                                "messages": [{"role": "user", "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}},
                                ]}],
                            })
                            response.raise_for_status()
                            data = response.json()
                            result = {"id": case["id"], "seconds": round(time.monotonic()-started, 2),
                                      "raw": data["choices"][0]["message"]["content"], "usage": data.get("usage"),
                                      "timings": data.get("timings"), "point_inside_expected_box": False}
                            try:
                                answer = decode_json(result["raw"])
                                x, y = answer["center_px"]
                                a,b,c,d = case["expected_box_1000"]
                                result["point_inside_expected_box"] = (answer.get("uncertain") is False and
                                    type(x) in (int,float) and type(y) in (int,float) and
                                    a <= x * 1000 / crop.width <= c and b <= y * 1000 / crop.height <= d)
                            except (ValueError, TypeError, KeyError):
                                result["parse_error"] = True
                            run["cases"].append(result)
                            print(args.name, backend, case["id"], result["seconds"], result["point_inside_expected_box"], flush=True)
                    finally:
                        stop_sample.set()
                        sampler.join(timeout=2)
                run["status"] = "completed"
            except Exception as exc:
                run["status"] = "failed"
                run["error"] = str(exc)
                print(args.name, backend, "FAILED", str(exc), flush=True)
            finally:
                run["peak_rss_mb"] = round(peak[0], 1)
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                run["memory_breakdown_lines"] = [
                    line for line in log_path.read_text(errors="replace").splitlines()
                    if "common_memory_breakdown_print:" in line
                ][-8:]
                report_text = json.dumps(report, ensure_ascii=False, indent=2)
                args.output.write_text(report_text)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
