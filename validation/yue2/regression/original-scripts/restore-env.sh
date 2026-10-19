#!/bin/bash
set -euo pipefail
root=/root/autodl-tmp/yue2-repatch-20261019
old=/root/autodl-tmp/yue2-wsb-4090d-20261016-01
mkdir -p "$root/evidence" "$root/tmp"
exec > "$root/evidence/restore-env.log" 2>&1
trap 'rc=$?; echo "$rc" > "$root/evidence/restore-env.exit"; date -Is; exit "$rc"' EXIT
export UV_CACHE_DIR="$root/uv-cache" TMPDIR="$root/tmp"
export UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4 UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
uv="$old/bin/uv"
date -Is
"$uv" venv --python "$old/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12" "$root/venv"
"$uv" pip install --python "$root/venv/bin/python" --no-deps -r "$old/requirements.txt"
"$uv" pip check --python "$root/venv/bin/python"
"$uv" pip freeze --python "$root/venv/bin/python" > "$root/evidence/installed.txt"
CUDA_VISIBLE_DEVICES='' "$root/venv/bin/python" -c 'import torch,importlib.metadata as m; print(torch.__version__,m.version("vllm")); print(torch.ones(2)+1)'
