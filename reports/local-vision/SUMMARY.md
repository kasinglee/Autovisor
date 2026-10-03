# Local vision measurements

M4 Mac mini, 16GB; llama.cpp 0.4.1 / build 10964. Five offline cases, including two synthetic diagrams.

| Model | Backend | Target-box hits | Mean request seconds | Peak sampled RSS MiB |
| --- | --- | --- | --- | --- |
| qwen25vl-3b | metal | 3/5 | 4.40 | 1636.5 |
| qwen25vl-3b | cpu | 3/5 | 46.51 | 4350.0 |
| qwen25vl-3b | cuda | unavailable | — | — |
| qwen25vl-7b | metal | 4/5 | 7.61 | 1716.0 |
| qwen25vl-7b | cpu | 4/5 | 71.66 | 6190.3 |
| qwen25vl-7b | cuda | unavailable | — | — |
| smolvlm-256m | cpu | 0/5 | 7.46 | 740.4 |
| smolvlm-256m | metal | 0/5 | 0.53 | 522.5 |
| smolvlm-256m | cuda | unavailable | — | — |
| smolvlm-500m | cpu | 0/5 | 8.13 | 1275.9 |
| smolvlm-500m | metal | 0/5 | 0.63 | 653.1 |
| smolvlm-500m | cuda | unavailable | — | — |

Hits are manual offline target rectangles, not live website verification outcomes. English instructions are provided; Chinese OCR was not benchmarked. RSS excludes some Metal allocations. Read the JSON memory breakdown and LOCAL_VISION.md before drawing 8GB hardware conclusions.

Small sample sizes do not establish production reliability. Runtime and prompt tuning may change results. Live scheduled account tasks were not switched to the local provider.
