# Local vision experiment

Branch: `codex/local-vision-cpu-metal`. This worktree starts from the latest local
account-b popup and DeepSeek fixes. Production account directories and LaunchAgents
are not switched to this branch.

## Reproduce

```bash
brew install llama.cpp
uv sync --locked
uv run python scripts/generate_vision_fixtures.py
uv run python scripts/benchmark_local_vision.py \
  --model-dir /Users/jasonlei/.cache/autovisor-models/smolvlm-256m \
  --name smolvlm-256m --backends cpu metal cuda \
  --memory-limit-gb 8 --output reports/local-vision/smolvlm-256m.json
```

Each model directory contains `model.gguf` and its matching `mmproj.gguf`.
Do not pair a model with another model's vision projector. The tested model revisions:

| Model | Hugging Face repository | Revision | Text / projector |
| --- | --- | --- | --- |
| SmolVLM 256M | ggml-org/SmolVLM-256M-Instruct-GGUF | b9e4379657e1450d04d02eec8e345667265b0a00 | Q8_0 / f16 |
| SmolVLM 500M | ggml-org/SmolVLM-500M-Instruct-GGUF | 72e986006ef53e37cdd3f6d4241c90b0f01df376 | Q8_0 / f16 |
| Qwen2.5-VL 3B | ggml-org/Qwen2.5-VL-3B-Instruct-GGUF | 5037fcf163dd95d1e41d1974465f0898ed108ca2 | Q4_K_M / f16 |
| Qwen2.5-VL 7B | ggml-org/Qwen2.5-VL-7B-Instruct-GGUF | 508edd0afaa66bb9e9f40587acc2184f02daf1f6 | Q4_K_M / f16 |

## Backend selection

- CPU uses `--device none -ngl 0 --no-mmproj-offload --no-op-offload`.
- Metal uses a discovered `MTL` device and offloads model and projector.
- CUDA uses a discovered `CUDA` device. No CUDA inference was run on this Mac.
  On an NVIDIA machine install the CUDA toolkit and build llama.cpp with:

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release --target llama-server
```

Put that binary on PATH, then select `--backend cuda`. Missing requested devices
produce an explicit error instead of silently using CPU.

## Optional experimental course integration

```bash
uv run python scripts/serve_local_vision.py \
  --model-dir /Users/jasonlei/.cache/autovisor-models/qwen25vl-3b --backend metal
```

In another terminal, copy `config.llama.ini.example` to a private local configuration,
fill in the account and course URL, then run Autovisor with `--config`. The relevant
options are `enableVisionCaptcha=True` and `[vision-option] provider=llama`.
The localhost provider does not retrieve or send the DeepSeek key. The default
provider remains DeepSeek. No local model service is installed as a recurring task.

## What the benchmark establishes

Five fixtures contain three real screenshots and two locally generated diagrams.
The test supplies an English translation of the instruction, so it primarily tests
visual reasoning, coordinate output, and runtime rather than Chinese instruction OCR.
Manual target rectangles are stored in `tests/fixtures/vision/cases.json` and can be
reviewed against the PNGs. A case passes only when the returned point falls within
the annotated target and the model reports it is not uncertain.

JSON schema decoding forces the output fields, separating format failures from
wrong target positions. Local models are asked for absolute pixels of the enlarged
image, then mapped to the browser image rectangle; DeepSeek retains its normalized
0–1000 contract. The initial normalized-coordinate trial failed even for otherwise
correct Qwen targets, and the final test uses the absolute-pixel contract.
The raw outputs, latency, token usage, and peak sampled RSS
are saved in `reports/local-vision/*.json`. These five cases do not establish a
general success rate or a successful live website verification.

The 8GB guard watches server RSS every 0.1 seconds. On Apple Silicon some Metal
allocations may not appear in RSS; RSS is not the total unified-memory requirement,
and a result on a 16GB M4 does not certify operation on an 8GB Mac. Server startup,
warm inference, and backend details are retained separately. Temporary servers
are stopped when each run completes.

Sources: [llama.cpp multimodal](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md),
[CPU/Metal/CUDA builds](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
