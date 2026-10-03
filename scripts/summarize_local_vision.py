import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report_dir = ROOT / "reports/local-vision"
lines = ["# Local vision measurements", "", "M4 Mac mini, 16GB; llama.cpp 0.4.1 / build 10964. Five offline cases, including two synthetic diagrams.", "", "| Model | Backend | Target-box hits | Mean request seconds | Peak sampled RSS MiB |", "| --- | --- | --- | --- | --- |"]
for path in sorted(report_dir.glob("*.json")):
    report = json.loads(path.read_text())
    for run in report["runs"]:
        log = report_dir / f"{report['model']}-{run['backend']}-server.log"
        if log.exists():
            run["memory_breakdown_lines"] = [line for line in log.read_text(errors="replace").splitlines() if "common_memory_breakdown_print:" in line][-8:]
        cases = run["cases"]
        if cases:
            hits = sum(case["point_inside_expected_box"] for case in cases)
            seconds = sum(case["seconds"] for case in cases) / len(cases)
            lines.append(f"| {report['model']} | {run['backend']} | {hits}/{len(cases)} | {seconds:.2f} | {run['peak_rss_mb']:.1f} |")
        else:
            lines.append(f"| {report['model']} | {run['backend']} | {run['status']} | — | — |")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
lines += ["", "Hits are manual offline target rectangles, not live website verification outcomes. English instructions are provided; Chinese OCR was not benchmarked. RSS excludes some Metal allocations. Read the JSON memory breakdown and LOCAL_VISION.md before drawing 8GB hardware conclusions.", "", "Small sample sizes do not establish production reliability. Runtime and prompt tuning may change results. Live scheduled account tasks were not switched to the local provider."]
(report_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
