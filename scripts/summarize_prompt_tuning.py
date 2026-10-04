import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
directory = ROOT / "reports/prompt-tuning"
rows = []
for path in sorted(directory.glob("*.json")):
    report = json.loads(path.read_text())
    for run in report["runs"]:
        cases = run["cases"]
        if not cases:
            continue
        hit = sum(case["point_inside_expected_box"] for case in cases)
        real = [c for c in cases if not c.get("synthetic",False)]
        synthetic = [c for c in cases if c.get("synthetic",False)]
        real_score = f"{sum(c['point_inside_expected_box'] for c in real)}/{len(real)}"
        synthetic_score = f"{sum(c['point_inside_expected_box'] for c in synthetic)}/{len(synthetic)}"
        seconds = sum(case["seconds"] for case in cases) / len(cases)
        rows.append(f"| {report['model']} | {run['backend']} | {hit}/{len(cases)} | {real_score} | {synthetic_score} | {seconds:.2f} | {run.get('peak_rss_mb', 'pending')} |")
lines = ["# 3B prompt and speed experiment", "", "All measurements are local on the M4 16GB Mac. Scores are points inside manually annotated target rectangles, not live-platform CAPTCHA pass rates.", "", "The fixture set has four real screenshots and six synthetic diagrams. Five cases were initially calibration and five initially holdout. After inspecting holdout failures and refining prompts, the ten cases are a development set, not a blind estimate of generalization. Model temperatures were zero; duplicate runs are not independent samples.", "", "The readable-marker combination reached 8/10 overall on Metal, but only 2/4 real screenshots and 6/6 synthetic diagrams. It has NOT demonstrated 80% success on real platform challenges. Accuracy improved primarily on the easier synthetic cases.", "", "| Experiment | Backend | All hits | Real screenshots | Synthetic diagrams | Mean seconds | Peak sampled RSS MiB |", "| --- | --- | --- | --- | --- | --- | --- |", *rows, "", "- baseline / compare / bbox / candidates are prompt-only or output-format variants.", "- grounded variants additionally use OpenCV connected components and image-measured colors. The VLM selects a candidate ID; code chooses an interior pixel, without consulting ground-truth labels.", "- short / explicit variants alter the output contract; readable changes marker sizes. Changing the segmentation/marker layout can change model decisions, so earlier scores are not guarantees for the final code.", "- scale1 is native image size, scale2 doubles it, scale3 triples it. Native inputs are faster but may lose grounding accuracy.", "- Q3_K_M text weights came from DhruvalLabs/Qwen2.5-VL-3B-Instruct-GGUF, revision 40859f27e17513a6a04328b6ff663eb14206c076. The 1.48GB file is smaller than the existing ~1.9GB Q4_K_M, but these tests showed severe degradation. This is one specific quantized artifact, not a universal Q3 accuracy claim.", "- Q8_0 vision projector is the official ggml-org artifact at revision 5037fcf163dd95d1e41d1974465f0898ed108ca2 (~806MiB vs the ~1.2GiB f16 projector). It preserves the text-model Q4 weights. File size reduction does not imply proportional CPU acceleration.", "- RSS does not include all Metal allocations. Production accounts still use their previous DeepSeek configuration.", "", "To reproduce, run `uv sync --locked --extra captcha --extra benchmark`, then `scripts/benchmark_local_vision.py --prompt grounded --image-scale 3 --large-labels` with the chosen model directory. Use `--backends cpu metal`; `--split calibration` reproduces the original five-case tuning tests. See raw JSON for all outputs and per-case timing."]
(directory / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
