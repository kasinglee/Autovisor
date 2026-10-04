# Windows CPU / NVIDIA CUDA 本机基准结果

测试日期：2026-10-04（Asia/Singapore）。分支：`codex/windows-cpu-cuda-test`，测试基础分支提交：`f6f2bcc4c6c5`。

## 结论

- CPU 与 CUDA 均实际完成三种规定配置各 10 张图；每项 CPU 与 CUDA 顺序运行，温度 0、单次推理。另对原尺寸 CPU 测了 16 线程。70 次回答全部解析成功。图集为 4 张真实截图和 6 张合成图，曾用于调参，结果不是独立泛化概率，也不是线上平台通过率。
- GPU 确认：llama.cpp `--list-devices` 列出 `CUDA0`；CUDA 日志显示 37/37 模型层卸载至 RTX 4080 SUPER，并明确 `CLIP using CUDA0 backend`。运行采样最高全卡利用率 94%，显存从约 2.2 GiB 测前基线升至约 6.1 GiB；无 CPU 回退。
- 原尺寸 Grounded 的最快 CPU 实测为 16 线程，平均推理 2.63/张，命中 5/10；比同配置 4 线程快 38%。但该配置命中率只有 5/10。测试集中得分最高的是 Grounded ×3 + 大标记：CPU 9/10（真实 3/4，合成 6/6），平均 13.86 秒/图；CUDA 7/10（真实 1/4，合成 6/6），平均 0.30 秒/图，约快 46 倍。CUDA 相对四线程 CPU 的平均推理加速倍数：原始坐标约 38 倍、Grounded ×3 约 46 倍、原尺寸 Grounded 约 19 倍。

## 主机、模型与软件

- Windows：Microsoft Windows 11 专业版 10.0.26200（build 26200）。
- CPU：AMD Ryzen 7 9700X 8-Core Processor，8 核 / 16 线程。
- RAM：31.1 GiB。
- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB VRAM；驱动 591.86，驱动报告 CUDA 13.1。
- 模型：`ggml-org/Qwen2.5-VL-3B-Instruct-GGUF`，revision `5037fcf163dd95d1e41d1974465f0898ed108ca2`；Q4_K_M 文本模型 + Q8_0 视觉投影器。实际文件大小分别 1.8 GiB / 806 MiB。
- SHA256：模型 `D02FE9B69AD8CADBBD228E387667AF66612C44BED29FFC8EB1E7CAF9AC486C12`；视觉投影器 `980C9B2F78C04E6CFF93D277ADA09E768394F112D75DB3B4E9DEA8A69F9FB904`。
- llama.cpp：`0.4.1-dev (build 10964, commit b29c606e2)`，Windows x64 Clang 20.1.8；Python 3.12.15，uv 0.12.23。
- 仓库测试：105 项通过，3 项跳过。CPU 固定 `--device none -ngl 0 --no-mmproj-offload --no-op-offload`；CUDA 用 `CUDA0 -ngl 99`。

## 配置对照

“命中”表示模型被解析的选择点位于手工标注框内且 `uncertain=false`。延迟是每次 HTTP 推理请求时间；不含图片裁剪、缩放及 OpenCV 候选框预处理。RAM 是服务器进程峰值 Windows working set。VRAM 为 NVIDIA 全卡基线 → 运行峰值（MiB）；WDDM 对进程级显存均报 N/A，因此不能归因到单进程。token/s 为生成速度均值。

| 配置 | 后端 | 命中 总/真实/合成 | 平均秒/图 | P50 | P95 | 模型加载秒 | 峰值 RAM MiB | 全卡 VRAM 基线→峰值 MiB | 增量 MiB | 生成 token/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始坐标 | CPU | 5/10 · 1/4 · 4/6 | 12.61 | 12.02 | 15.69 | 2.66 | 5160 | 2190 → 2474 | 284 | 25.30 |
| 原始坐标 | CUDA | 6/10 · 2/4 · 4/6 | 0.33 | 0.31 | 0.42 | 5.12 | 2683 | 2191 → 6111 | 3920 | 217.29 |
| Grounded ×3 + 大标记 | CPU | 9/10 · 3/4 · 6/6 | 13.86 | 13.34 | 16.94 | 2.34 | 5209 | 2257 → 2578 | 321 | 23.92 |
| Grounded ×3 + 大标记 | CUDA | 7/10 · 1/4 · 6/6 | 0.30 | 0.30 | 0.34 | 2.56 | 2732 | 2229 → 6182 | 3953 | 214.99 |
| Grounded 原尺寸 | CPU | 5/10 · 1/4 · 4/6 | 3.64 | 3.55 | 4.18 | 2.44 | 4616 | 2243 → 2514 | 271 | 25.93 |
| Grounded 原尺寸 | CUDA | 5/10 · 1/4 · 4/6 | 0.19 | 0.19 | 0.23 | 2.09 | 2472 | 2237 → 6141 | 3904 | 221.48 |

### 逐图结果及 CPU/CUDA 答案对照

“答案”列对 Grounded 表示选出的候选 ID；原始点坐标方案显示模型返回中心点。相同候选 ID 表示选中了同一候选框。逐字原始回答及 token 用量保存在对应 JSON。

**原始坐标**

| 图片 | 类别 | CPU 命中 / 答案 | CPU 秒 | CUDA 命中 / 答案 | CUDA 秒 | 判定 |
|---|---|---|---:|---|---:|---|
| blue-cube | 真实截图 | ✓ / blue cube @ [800, 270] | 12.02 | ✓ / blue cube @ [800, 280] | 0.45 | 不同 |
| number-orientation | 真实截图 | ✗ / object name @ [412, 198] | 12.02 | ✗ / object name @ [412, 192] | 0.31 | 不同 |
| sideways-S | 真实截图 | ✗ / S @ [402, 300] | 12.30 | ✗ / S @ [409, 300] | 0.31 | 不同 |
| synthetic-green-F | 合成 | ✗ / object name @ [105, 199] | 12.11 | ✗ / object name @ [105, 218] | 0.31 | 不同 |
| synthetic-blue-cube | 合成 | ✓ / blue cube @ [728, 300] | 11.86 | ✓ / blue cube @ [728, 300] | 0.33 | 相同 |
| synthetic-red-P | 合成 | ✗ / object name @ [384, 199] | 11.70 | ✗ / object name @ [412, 219] | 0.27 | 不同 |
| synthetic-lowercase-k | 合成 | ✓ / k @ [430, 328] | 11.78 | ✓ / k @ [432, 330] | 0.30 | 不同 |
| synthetic-yellow-square | 合成 | ✓ / yellow square @ [150, 250] | 12.09 | ✓ / yellow square @ [150, 250] | 0.31 | 相同 |
| synthetic-purple-8 | 合成 | ✓ / object name @ [779, 252] | 11.75 | ✓ / object name @ [784, 252] | 0.31 | 不同 |
| real-green-F | 真实截图 | ✗ / object name @ [200, 100] | 18.47 | ✓ / object name @ [192, 380] | 0.38 | 不同 |

**Grounded ×3 + 大标记**

| 图片 | 类别 | CPU 命中 / 答案 | CPU 秒 | CUDA 命中 / 答案 | CUDA 秒 | 判定 |
|---|---|---|---:|---|---:|---|
| blue-cube | 真实截图 | ✓ / ID 4 | 14.09 | ✓ / ID 4 | 0.34 | 相同 |
| number-orientation | 真实截图 | ✓ / ID 4 | 13.53 | ✗ / ID 0 | 0.31 | 不同 |
| sideways-S | 真实截图 | ✓ / ID 3 | 13.91 | ✗ / ID 5 | 0.30 | 不同 |
| synthetic-green-F | 合成 | ✓ / ID 3 | 13.02 | ✓ / ID 3 | 0.28 | 相同 |
| synthetic-blue-cube | 合成 | ✓ / ID 2 | 13.66 | ✓ / ID 2 | 0.28 | 相同 |
| synthetic-red-P | 合成 | ✓ / ID 2 | 13.14 | ✓ / ID 2 | 0.30 | 相同 |
| synthetic-lowercase-k | 合成 | ✓ / ID 2 | 12.67 | ✓ / ID 2 | 0.31 | 相同 |
| synthetic-yellow-square | 合成 | ✓ / ID 2 | 12.66 | ✓ / ID 2 | 0.28 | 相同 |
| synthetic-purple-8 | 合成 | ✓ / ID 1 | 12.64 | ✓ / ID 1 | 0.30 | 相同 |
| real-green-F | 真实截图 | ✗ / ID 2 | 19.27 | ✗ / ID 2 | 0.34 | 相同 |

**Grounded 原尺寸**

| 图片 | 类别 | CPU 命中 / 答案 | CPU 秒 | CUDA 命中 / 答案 | CUDA 秒 | 判定 |
|---|---|---|---:|---|---:|---|
| blue-cube | 真实截图 | ✓ / ID 3 | 4.03 | ✓ / ID 3 | 0.25 | 相同 |
| number-orientation | 真实截图 | ✗ / ID 3 | 3.77 | ✗ / ID 1 | 0.19 | 不同 |
| sideways-S | 真实截图 | ✗ / ID 0 | 4.11 | ✗ / ID 1 | 0.17 | 不同 |
| synthetic-green-F | 合成 | ✓ / ID 2 | 3.55 | ✓ / ID 2 | 0.17 | 相同 |
| synthetic-blue-cube | 合成 | ✓ / ID 2 | 2.88 | ✓ / ID 2 | 0.17 | 相同 |
| synthetic-red-P | 合成 | ✓ / ID 2 | 3.56 | ✓ / ID 2 | 0.20 | 相同 |
| synthetic-lowercase-k | 合成 | ✗ / ID 0 | 3.41 | ✗ / ID 0 | 0.19 | 相同 |
| synthetic-yellow-square | 合成 | ✗ / ID 0 | 3.36 | ✗ / ID 0 | 0.17 | 相同 |
| synthetic-purple-8 | 合成 | ✓ / ID 1 | 3.48 | ✓ / ID 1 | 0.19 | 相同 |
| real-green-F | 真实截图 | ✗ / ID 0 | 4.23 | ✗ / ID 2 | 0.20 | 不同 |

### CPU 线程调优

原尺寸 Grounded CPU 4 线程：平均 3.64 秒/图、命中 5/10；16 线程：平均 2.63 秒/图、命中 5/10（真实 1/4，合成 4/6），峰值工作集 4618 MiB。16 线程快约 38%；一次十图的调优观测，不代表独立准确率。原始数据：`q4-fast-cpu16.json`。

## 资源预算与边界

- 32 GiB 主机 RAM；CPU 峰值 working set 约 5209 MiB（约 5.1 GiB），低于 8 GiB 进程 RSS 监控预算。CUDA 进程峰值 RAM 约 2732 MiB。
- 16 GiB 显存；测试中全卡峰值 6182 MiB，低于 8 GiB 单项预算。非测试时桌面基线约 2.4 GiB；实际进程专属 VRAM 因 WDDM 返回 N/A，显存增量只能用全卡基线估算。
- 所有必跑组合均在同一固定十图开发集各跑一次，未做重复稳定性、16 GiB 放宽或其他图像缩放精度扫描。视觉权重和 CUDA 模型日志属于本机实测环境；仓库测试仅需 Python 环境。
- 该图集已有调参历史；4 张真实截图样本很少。即使合成图命中较高，也不足以证明真实网站验证码可靠，或支持替换现有线上服务。适合继续评估 CPU/GPU 本地加速，但上线前需在新的、代表实际场景且未参与调参的数据上验证。

## 原始记录

- `q4-baseline.json`、`q4-readable.json`、`q4-fast.json`：逐图延迟、命中、原始模型回答、token/timing、加载时间、working set、显存基线和采样、设备配置。
- 对应 `*-cpu-server.log` / `*-cuda-server.log`：llama.cpp 后端与卸载日志；`*-run.txt`：逐图运行进度。
- `environment.json`、`model-sha256.json`、`llama-version.txt`、`llama-devices.txt`、`unit-tests.txt`：机器、模型、版本、设备及仓库测试记录。
- 模型权重被仓库忽略规则排除，不随提交。
