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
import threading

import httpx
import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.local_vision import decode_json, point_response_format, server_command, vision_prompt


def rss_mb(pid):
    try:
        return psutil.Process(pid).memory_info().rss / (1024 * 1024)
    except psutil.NoSuchProcess:
        return 0.0


def nvidia_sample(pid=None):
    """WDDM may hide process memory; retain whole-card data with its scope."""
    if not shutil.which("nvidia-smi"):
        return {"error": "nvidia-smi unavailable"}
    options = {"capture_output": True, "text": True,
               "creationflags": subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], timeout=5, **options)
        processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                    "--format=csv,noheader,nounits"], timeout=5, **options)
        server_vram = next((line.split(",", 1)[1].strip() for line in processes.stdout.splitlines()
                            if line.split(",", 1)[0].strip() == str(pid)), None) if pid else None
        return {"gpu_csv": gpu.stdout.strip(), "server_vram_mib": server_vram,
                "server_pid": pid, "error": (gpu.stderr + processes.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--backends", nargs="+", default=["cpu", "metal"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-limit-gb", type=float, default=8)
    parser.add_argument("--prompt", choices=["baseline", "compare", "bbox", "candidates", "grounded", "grounded-explicit", "grounded-short"], default="baseline")
    parser.add_argument("--image-scale", type=int, choices=[1, 2, 3, 4], default=3)
    parser.add_argument("--split", choices=["all", "calibration", "holdout"], default="all")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--large-labels", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    binary = shutil.which("llama-server")
    if not binary:
        parser.error("llama-server not installed")
    devices = subprocess.run([binary, "--list-devices"], capture_output=True, text=True)
    device_text = devices.stdout + devices.stderr
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = json.loads((ROOT / "tests/fixtures/vision/cases.json").read_text(encoding="utf-8"))
    cases = [case for case in cases if args.split == "all" or case.get("split") == args.split]
    if not cases or args.repeats < 1:
        parser.error("Empty case set or invalid repeat count")
    report = {"model": args.name, "devices": device_text, "memory_limit_gb": args.memory_limit_gb,
              "prompt": args.prompt, "image_scale": args.image_scale, "split": args.split, "repeats": args.repeats, "large_labels": args.large_labels, "threads": args.threads,
              "method": "Annotated real and synthetic screenshots; supplied English instruction; JSON constrained decoding; absolute image-pixel output checked against normalized target boxes. Not live platform pass rate. RSS excludes some Metal allocations.",
              "runs": []}
    for backend in args.backends:
        run = {"backend": backend, "cases": []}
        report["runs"].append(run)
        try:
            command = server_command(binary, args.model_dir / "model.gguf", args.model_dir / "mmproj.gguf", backend, devices=device_text, threads=args.threads)
        except ValueError as exc:
            run["status"] = "unavailable"
            run["error"] = str(exc)
            continue
        log_path = args.output.parent / f"{args.name}-{backend}-server.log"
        run["command"] = command
        run["nvidia_baseline"] = nvidia_sample()
        run["resource_samples"] = []
        start = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            peak = [0.0]
            stop_monitor = threading.Event()
            phase = ["load"]
            def monitor():
                while not stop_monitor.is_set():
                    sampled_rss = rss_mb(process.pid)
                    peak[0] = max(peak[0], sampled_rss)
                    run["resource_samples"].append({"elapsed_seconds": round(time.monotonic()-start, 3),
                        "phase": phase[0], "rss_mb": round(sampled_rss, 1), **nvidia_sample(process.pid)})
                    stop_monitor.wait(1.0)
            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
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
                        for trial in range(args.repeats * len(cases)):
                            case = cases[trial % len(cases)]
                            phase[0] = case["id"]
                            with Image.open(ROOT / "tests/fixtures/vision" / case["file"]) as image:
                                crop = image.crop(tuple(case["crop"]))
                                crop = crop.resize((crop.width * args.image_scale, crop.height * args.image_scale))
                                if args.prompt.startswith("grounded"):
                                    from modules.vision_candidates import extract_candidates, candidate_image, selection_format
                                    candidates = extract_candidates(crop)
                                    crop = candidate_image(crop, candidates, args.large_labels)
                                buffer = io.BytesIO()
                                crop.save(buffer, format="PNG")
                            if args.prompt.startswith("grounded"):
                                prompt = (
                                    f"Task: {case['instruction_en']}. Chinese: {case['instruction_zh']}. "
                                    "Objects are boxed and numbered. Select ONE box ID matching the instruction. "
                                    "The numbered white tags are annotations, not objects. Candidate colors are measured, not guesses. "
                                    "For same-color tasks identify the reference character, then choose the TARGET character with matching measured color. "
                                    "For side-on letters choose the thin/narrow letter with a prominent visible side, not the broad frontal letter. "
                                    "Do not select the reference object. Return reference_attribute, selected_id, uncertain as JSON. "
                                    "Candidates: " + json.dumps([{k:v for k,v in c.items() if k != 'center_px'} for c in candidates])
                                )
                                if args.prompt == "grounded-explicit":
                                    prompt += (
                                        " First output target_character as the exact requested target letter, digit, or shape. "
                                        "Then reference_id as the reference object's ID, or -1 when there is no reference. "
                                        "reference_attribute must state the actual color or facing direction, not the word 'color'. "
                                        "selected_id must refer to the requested target_character and must NOT equal reference_id. "
                                        "For same-color questions use the measured color of reference_id to filter target candidates."
                                    )
                                if args.prompt == "grounded-short":
                                    prompt = (
                                        f"Choose ONE numbered object: {case['instruction_en']}. "
                                        "White number tags and black boxes are annotations, not objects. "
                                        "Measured candidate colors: " + json.dumps([{k:v for k,v in c.items() if k not in {'bbox','center_px'}} for c in candidates]) +
                                        ". Select the exact requested character, case, color and shape. "
                                        "When comparing colors, find the reference character's color and select the TARGET character with that color; never select the reference. "
                                        "Sideways means a narrow front face and a prominent side face. "
                                        "Return selected_id and uncertain as JSON only."
                                    )
                                response_format = selection_format(args.prompt == "grounded-explicit", args.prompt == "grounded-short")
                            else:
                                prompt = vision_prompt(case['instruction_en'] + f" Chinese original: {case['instruction_zh']}", crop.width, crop.height, args.prompt)
                                response_format = point_response_format(args.prompt != "baseline", args.prompt == "bbox", args.prompt == "candidates")
                            started = time.monotonic()
                            response = client.post("/v1/chat/completions", json={
                                "model": "local-vision", "temperature": 0, "max_tokens": 400 if args.prompt == "candidates" else (256 if args.prompt != "baseline" else 160),
                                "response_format": response_format,
                                "messages": [{"role": "user", "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}},
                                ]}],
                            })
                            response.raise_for_status()
                            data = response.json()
                            result = {"id": case["id"], "split": case.get("split"), "synthetic": case.get("synthetic", False), "repeat": trial // len(cases), "image_size": list(crop.size), "seconds": round(time.monotonic()-started, 2),
                                      "raw": data["choices"][0]["message"]["content"], "usage": data.get("usage"),
                                      "timings": data.get("timings"), "point_inside_expected_box": False}
                            try:
                                answer = decode_json(result["raw"])
                                if args.prompt.startswith("grounded"):
                                    index = answer["selected_id"]
                                    if type(index) is not int or not (0 <= index < len(candidates)):
                                        raise ValueError("Unknown candidate ID")
                                    if args.prompt == "grounded-explicit" and "same" in case["instruction_en"].lower() and index == answer["reference_id"]:
                                        raise ValueError("Selected reference instead of target")
                                    x,y = candidates[index]["center_px"]
                                    result["image_candidates"] = candidates
                                    result["derived_center"] = [x,y]
                                elif args.prompt == "candidates":
                                    index = answer["selected_index"]
                                    if type(index) is not int or not (0 <= index < len(answer["candidates"])):
                                        raise ValueError("Invalid candidate index")
                                    x, y = answer["candidates"][index]["center_px"]
                                    result["derived_center"] = [x,y]
                                elif args.prompt == "bbox":
                                    x0,y0,x1,y1 = answer["bbox_2d"]
                                    if not (0 <= x0 < x1 <= crop.width and 0 <= y0 < y1 <= crop.height):
                                        raise ValueError("Invalid bounding box")
                                    x, y = (x0+x1)/2, (y0+y1)/2
                                    result["derived_center"] = [x,y]
                                else:
                                    x, y = answer["center_px"]
                                a,b,c,d = case["expected_box_1000"]
                                result["point_inside_expected_box"] = (answer.get("uncertain") is False and
                                    type(x) in (int,float) and type(y) in (int,float) and
                                    a <= x * 1000 / crop.width <= c and b <= y * 1000 / crop.height <= d)
                            except (ValueError, TypeError, KeyError):
                                result["parse_error"] = True
                            run["cases"].append(result)
                            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
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
                stop_monitor.set()
                monitor_thread.join(timeout=12)
                run["peak_rss_mb"] = round(peak[0], 1)
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                run["memory_breakdown_lines"] = [
                    line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    if "common_memory_breakdown_print:" in line
                ][-8:]
                report_text = json.dumps(report, ensure_ascii=False, indent=2)
                args.output.write_text(report_text, encoding="utf-8")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
