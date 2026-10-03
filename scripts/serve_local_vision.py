import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.local_vision import server_command

parser = argparse.ArgumentParser(description="Start local vision on CPU, Metal, or CUDA")
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--backend", choices=["cpu", "metal", "cuda"], required=True)
parser.add_argument("--port", type=int, default=8091)
args = parser.parse_args()
binary = shutil.which("llama-server")
if not binary:
    parser.error("Install llama.cpp with your desired backend first")
devices = subprocess.run([binary, "--list-devices"], capture_output=True, text=True)
try:
    command = server_command(binary, args.model_dir / "model.gguf", args.model_dir / "mmproj.gguf", args.backend, args.port, devices.stdout + devices.stderr)
except ValueError as exc:
    parser.error(str(exc))
os.execv(binary, command)
