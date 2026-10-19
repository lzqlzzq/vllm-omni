export YUE2_ROOT=/root/autodl-tmp/yue2-4090-e2e-20261012-01
export PATH="$YUE2_ROOT/venv/bin:/root/miniconda3/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-13.0"
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HOME="$YUE2_ROOT/hf-cache" HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export YUE2_ARM=${YUE2_ARM:?arm required}
export YUE2_SOURCE=${YUE2_SOURCE:?arm source required}
unset LD_LIBRARY_PATH
export PYTHONPATH="$YUE2_ROOT/$YUE2_SOURCE:$YUE2_ROOT/scripts"
cd "$YUE2_ROOT"
export YUE2_MPS=off
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
