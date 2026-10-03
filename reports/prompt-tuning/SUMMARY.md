# 3B prompt and speed experiment

All measurements are local on the M4 16GB Mac. Scores are points inside manually annotated target rectangles, not live-platform CAPTCHA pass rates.

The fixture set has four real screenshots and six synthetic diagrams. Five cases were initially calibration and five initially holdout. After inspecting holdout failures and refining prompts, the ten cases are a development set, not a blind estimate of generalization. Model temperatures were zero; duplicate runs are not independent samples.

The readable-marker combination reached 8/10 overall on Metal, but only 2/4 real screenshots and 6/6 synthetic diagrams. It has NOT demonstrated 80% success on real platform challenges. Accuracy improved primarily on the easier synthetic cases.

| Experiment | Backend | All hits | Real screenshots | Synthetic diagrams | Mean seconds | Peak sampled RSS MiB |
| --- | --- | --- | --- | --- | --- | --- |
| q3-grounded-scale3 | metal | 0/5 | 0/3 | 0/2 | 4.69 | 1974.6 |
| q3-readable-scale3 | metal | 0/5 | 0/3 | 0/2 | 4.63 | 1961.9 |
| q4-baseline-scale3 | metal | 2/5 | 1/3 | 1/2 | 3.76 | 1689.3 |
| q4-baseline-ten | metal | 6/10 | 2/4 | 4/6 | 3.66 | 2058.7 |
| q4-bbox-scale3 | metal | 1/5 | 1/3 | 0/2 | 4.42 | 1859.7 |
| q4-bbox-scale4 | metal | 2/5 | 1/3 | 1/2 | 6.69 | 2109.3 |
| q4-candidates-scale3 | metal | 1/5 | 0/3 | 1/2 | 5.75 | 1882.8 |
| q4-compare-scale1 | metal | 0/5 | 0/3 | 0/2 | 2.21 | 1684.1 |
| q4-compare-scale3 | metal | 2/5 | 1/3 | 1/2 | 4.41 | 1799.6 |
| q4-explicit-scale3-ten | metal | 4/10 | 1/4 | 3/6 | 4.32 | 2208.6 |
| q4-grounded-scale1 | metal | 3/5 | 1/3 | 2/2 | 1.51 | 1630.3 |
| q4-grounded-scale2 | metal | 3/5 | 1/3 | 2/2 | 2.37 | 1659.3 |
| q4-grounded-scale3-ten | metal | 7/10 | 2/4 | 5/6 | 3.90 | 2111.8 |
| q4-grounded-scale3 | metal | 4/5 | 2/3 | 2/2 | 3.86 | 1826.2 |
| q4-short-scale2-ten | metal | 3/10 | 1/4 | 2/6 | 1.90 | 1816.9 |
| q4-short-scale3-ten | metal | 3/10 | 1/4 | 2/6 | 3.44 | 2103.6 |
| q4-visionq8-candidates-scale3 | metal | 1/5 | 0/3 | 1/2 | 5.76 | 1498.6 |
| q4-visionq8-explicit-scale3-ten | metal | 4/10 | 1/4 | 3/6 | 4.35 | 1886.4 |
| q4-visionq8-grounded-native-ten | metal | 5/10 | 1/4 | 4/6 | 1.50 | 1279.3 |
| q4-visionq8-grounded-native-ten | cpu | 6/10 | 2/4 | 4/6 | 8.91 | 4476.6 |
| q4-visionq8-grounded-ten | metal | 6/10 | 1/4 | 5/6 | 3.90 | 1667.4 |
| q4-visionq8-grounded-ten | cpu | 6/10 | 1/4 | 5/6 | 38.05 | 4715.5 |
| q4-visionq8-readable-native-ten | metal | 5/10 | 1/4 | 4/6 | 1.50 | 1199.4 |
| q4-visionq8-readable-scale2-ten | metal | 7/10 | 1/4 | 6/6 | 2.32 | 1357.7 |
| q4-visionq8-readable-ten-cpu | cpu | 7/10 | 1/4 | 6/6 | 38.02 | 4413.1 |
| q4-visionq8-readable-ten | metal | 8/10 | 2/4 | 6/6 | 3.90 | 1668.8 |
| q4-visionq8-short-scale3-ten | metal | 3/10 | 1/4 | 2/6 | 3.48 | 1732.1 |

- baseline / compare / bbox / candidates are prompt-only or output-format variants.
- grounded variants additionally use OpenCV connected components and image-measured colors. The VLM selects a candidate ID; code chooses an interior pixel, without consulting ground-truth labels.
- short / explicit variants alter the output contract; readable changes marker sizes. Changing the segmentation/marker layout can change model decisions, so earlier scores are not guarantees for the final code.
- scale1 is native image size, scale2 doubles it, scale3 triples it. Native inputs are faster but may lose grounding accuracy.
- Q3_K_M text weights came from DhruvalLabs/Qwen2.5-VL-3B-Instruct-GGUF, revision 40859f27e17513a6a04328b6ff663eb14206c076. The 1.48GB file is smaller than the existing ~1.9GB Q4_K_M, but these tests showed severe degradation. This is one specific quantized artifact, not a universal Q3 accuracy claim.
- Q8_0 vision projector is the official ggml-org artifact at revision 5037fcf163dd95d1e41d1974465f0898ed108ca2 (~806MiB vs the ~1.2GiB f16 projector). It preserves the text-model Q4 weights. File size reduction does not imply proportional CPU acceleration.
- RSS does not include all Metal allocations. Production accounts still use their previous DeepSeek configuration.

To reproduce, run `uv sync --locked --extra captcha`, then `scripts/benchmark_local_vision.py --prompt grounded --image-scale 3 --large-labels` with the chosen model directory. Use `--backends cpu metal`; `--split calibration` reproduces the original five-case tuning tests. See raw JSON for all outputs and per-case timing.
