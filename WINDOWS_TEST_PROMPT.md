# 给 Windows 主机 AI 的测试任务

请在这台 Windows 主机上执行并完成 Autovisor 本地视觉模型的 CPU 与 NVIDIA CUDA GPU 对照测试。CPU 不是 CPI。先检查硬件和环境，随后运行实际测试、保存原始结果并提交报告；不要只给安装建议。

## 仓库与背景

- 仓库：https://github.com/kasinglee/Autovisor
- 分支：`codex/local-vision-cpu-metal`
- 克隆后切换并拉取该分支，保留本机已有改动，使用独立实验目录。
- 阅读 `LOCAL_VISION.md`、`reports/prompt-tuning/SUMMARY.md` 和基准脚本。
- 使用已提交的十张测试图，无需登录课程账号或调用 DeepSeek API。
- 十张图由四张真实截图和六张合成图组成，已用于调参，不能把命中数称为独立泛化概率或平台验证通过率。
- Mac 上 Qwen2.5-VL 3B Q4 + Q8 视觉组件 + OpenCV 编号 + 大号编号，在 Metal 上混合集 8/10，但真实截图只有 2/4。Windows 的目标是复现并检验 CPU/CUDA 的性能和准确率，而不是预设必须得到 80%。

## 1. 环境

记录 Windows 版本、CPU 型号/物理与逻辑核心数、内存、NVIDIA GPU 型号、显存、驱动和 CUDA 版本。
使用 Python 3.12，并运行：

```powershell
uv sync --locked --extra captcha --extra benchmark
uv run --extra captcha --extra benchmark python -m unittest discover -s tests -q
```

安装支持 CUDA 的官方 llama.cpp Windows 二进制（或构建 CUDA 版本），记录版本、构建信息。
Mac 参考版本为 0.4.1/build 10964/commit b29c606e2。参考官方安装/构建文档匹配现有 NVIDIA 驱动。

```powershell
llama-server --version
llama-server --list-devices
nvidia-smi
```

需要看到真实的 `CUDA0` 等设备。若没有 NVIDIA GPU 或 CUDA 加载失败，明确报告 CUDA 未测试，不要把 CPU 回退当成 GPU 成功。

## 2. 下载相同模型

在实验目录创建 `models\qwen25vl-3b-vision-q8`，其中两个文件必须分别命名为 `model.gguf`、`mmproj.gguf`：

```powershell
$modelDir = Join-Path (Get-Location) 'models\qwen25vl-3b-vision-q8'
New-Item -ItemType Directory -Force $modelDir | Out-Null
curl.exe -fL --retry 3 -o "$modelDir\model.gguf" 'https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/5037fcf163dd95d1e41d1974465f0898ed108ca2/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf'
curl.exe -fL --retry 3 -o "$modelDir\mmproj.gguf" 'https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/5037fcf163dd95d1e41d1974465f0898ed108ca2/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf'
```

确认下载成功且文件大小合理，记录 SHA256。不要错配其他模型的视觉组件。

## 3. 必跑测试矩阵

CPU 与 CUDA 分别顺序运行相同十张不同图片，温度 0、`--repeats 1`；两个后端不要同时跑，以免竞争硬件。
先保持四线程与 Mac 可比，再单独选择适合本机 CPU 的线程数做调优，并在报告中分开标注。

```powershell
# A：原始点坐标方案
uv run --extra captcha --extra benchmark python scripts/benchmark_local_vision.py --model-dir "$modelDir" --name windows-q4-baseline --backends cpu cuda --threads 4 --output reports/windows/q4-baseline.json

# B：较高精度组合，三倍图片 + OpenCV + 大号编号
uv run --extra captcha --extra benchmark python scripts/benchmark_local_vision.py --model-dir "$modelDir" --name windows-q4-readable --backends cpu cuda --threads 4 --prompt grounded --image-scale 3 --large-labels --output reports/windows/q4-readable.json

# C：CPU 提速组合，原尺寸图片 + OpenCV + 小编号
uv run --extra captcha --extra benchmark python scripts/benchmark_local_vision.py --model-dir "$modelDir" --name windows-q4-fast --backends cpu cuda --threads 4 --prompt grounded --image-scale 1 --output reports/windows/q4-fast.json
```

脚本使用 psutil 采集跨平台进程 RSS/Windows working set；此数值不是 CUDA 显存。
测试 CPU 时确认语言模型和视觉组件均未卸载到 GPU（`--device none -ngl 0 --no-mmproj-offload --no-op-offload`）。
CUDA 时检查服务器日志中的语言模型和视觉组件 GPU 卸载情况，配合 NVIDIA 监控确认确实使用 GPU。

若某项 Windows 兼容性问题阻碍测试，先定位、做最小修复、测试并继续。不要修改标注答案来提高得分。

## 4. 显存、速度与稳健性

- 对每次运行采样 NVIDIA 显存：优先按 llama-server PID 记录使用量；Windows WDDM 若返回 N/A，则记录 GPU 总使用量及测试前基线，并明确是全卡指标，不能冒充进程显存。
- 保存加载时间、每图延迟、平均/P50/P95、峰值 working set、显存指标、生成 token 速度、原始回答、解析/坐标错误、每图是否命中目标框。
- 把真实截图和合成图分别统计；显示 CPU/CUDA 是否选出了不同答案。
- 同图重复十次可以单独用作稳定性测试，但不能等同十张独立样本。
- 先按 8GB 预算尝试，不够再按实际硬件能力考虑 16GB；RAM 与 VRAM 单独核算，脚本 `--memory-limit-gb` 仅监视主机进程 RSS，不控制显存。
- 基线跑通后，可比较不同 CPU 线程数、二倍图片、F16/Q8 视觉组件。保持其他条件相同，给出准确率与速度取舍。
- 更小的 Q3_K_M 已在 Mac 上出现 0/5，若另测请记录模型发布者/版本；不要假定文件小就一定更快或仍准确。

## 5. 交付

生成 `reports/windows/SUMMARY.md` 和原始 JSON/必要日志，清晰回答：

1. CPU 与 CUDA 是否实际跑通？使用哪种硬件、模型与后端？
2. 三种配置各自的整体、真实图、合成图命中数及耗时是多少？
3. CPU 最快且仍有可用准确率的配置是什么？CUDA 比 CPU 快多少？
4. RAM/VRAM 是否符合 8GB/16GB 预算？哪些指标无法取得？
5. 本地方案是否值得继续优化？真实截图结果是否足以支持替换线上方案？

在新的 `codex/windows-cpu-cuda-test` 分支提交 Windows 兼容性修复和报告，完成后告诉我提交号。将推送视为这项测试任务的一部分；若当前主机没有仓库认证，保留本地提交并说明推送未完成。不要提交模型权重、账号配置、Cookie、API Key 或用户私人文件。
