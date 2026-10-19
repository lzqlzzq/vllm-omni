cd /root/autodl-tmp/yue2-wsb-4090d-20261016-01
export PATH="$PWD/venv/bin:$PWD/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="$PWD/source:$PWD/scripts"
export HF_HOME="$PWD/hf-cache" HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$PWD/tmp" UV_CACHE_DIR="$PWD/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/python"
export UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4 UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
