# YuE2 validation evidence

Companion evidence for [`feat/yue2-feature-enhancements`](https://github.com/lzqlzzq/vllm-omni/tree/feat/yue2-feature-enhancements), implementation commit `d413a22364d1acd297778fea7edfc1a9de7ee12b` on main `2c72dea3aab6a6781ea650798cfdf85748abf0ef`. This branch contains validation material only; its repository root does not include the implementation patch.

## Results and scope

| Evidence | Source revision | Result |
| --- | --- | --- |
| [CPU regressions](regression/remote) | `2c72dea3` + PR | 246 YuE2 tests passed; 277 shared tests passed, 2 CUDA-only skips |
| [Pre-commit](regression/precommit-final.log) | `2c72dea3` + PR | All applicable checks passed |
| [Native ABC/CFG/HTTP](performance/artifacts) | `f9143976` + PR | AR1/AR2, 402 paired sampling steps, five HTTP successes/two parameter rejections |
| [Performance](performance/artifacts/comparison.json) | `f9143976`, main + common loading fix vs PR | Throughput +6.22/+6.38%; E2EL P95 −13.94/−14.10%; RTF P95 −15.75/−15.86% |
| [WSB PER/SongBench](wsb/scoring/nar1-full-scores-01/summary.json) | `f9143976` + PR | 384 outputs scored; 192 selected by PER |
| [WSB MuLan/AllMusicCaps](wsb/scoring/nar1-additional-01/summary.json) | Same generated outputs | Same 192 selected outputs scored |

Performance: one RTX4090 24 GiB, driver 595.71.05, Python 3.12.11, PyTorch 2.13.0+cu130, vLLM 0.31.0. Order main → PR → PR → main; each process used two full warmups and 32 measured songs. Both used AR1/KV3 GiB/client2, full CoT/CFG1; PR used NAR1/VAE1, no batching wait/budget. Peak device memory: main 22.6877 GiB, PR 21.3733 GiB. PR output duration was 2.454% shorter. Thermal throttling occurred in both versions. No general concurrency-scaling claim is made.

Selected WSB scores: Musicality **5.903877**, Global Avg **6.729727**, MuLan **0.511322**, AllMusicCaps **0.403524**, PER micro **0.090754**. Generation used RTX4090D 24 GiB, 32 midpoint steps and FP32 tiled legacy VAE. Four metrics completed; Q3O and batched NAR acceptance remain pending. These are not paired quality comparisons against current main.

PER execution removed one redundant parent-process CUDA synchronization after blocking ASR results. [Original/executed function and hashes](wsb/scoring/nar1-full-scores-01/per-worker-execution.json) are retained. Scoring parameters and algorithms were unchanged. ASR reached its output cap for 11/384 outputs (38 repeats), including 3/192 selected outputs; none were excluded. Top-level scorer input rows retain `pending`; each metric's `status` and the completed summary are authoritative.

GPU/WSB evidence predates the rebase. No new-head GPU run or full WSB rerun is claimed. See [submission checks](PRECHECK.md).

## Verify without a GPU

From this branch's repository root:

```bash
python3 validation/yue2/verify.py
```

The standard-library verifier checks published file hashes, request/seed coverage, latency/RTF percentiles, TPS, sampled device peaks, WSB selection identity and metric aggregation. `provenance.json` distinguishes byte-identical original records from the numeric WSB extracts. The extracts retain every row and all fields needed for reported score aggregation; lyrics, ASR transcripts and repeated raw scorer payloads are omitted. Audio, latents, model weights, caches and duplicate source tarballs are not included.

Archived filenames and internal labels remain byte-identical: `upstream` means **main**, `candidate` means **PR**. These are historical evidence labels, not model or deployment entry points.

## Reproduce the measured performance

Use a Linux NVIDIA GPU 0 with sufficient memory, a CUDA 13-capable driver/toolkit, and the [recorded dependencies](performance/evidence/environment.json). Activate that Python environment first. Populate the Hugging Face cache with the pinned model/tokenizer snapshots from [assets.json](performance/evidence/assets.json), including the deployment's `gpt2` tokenizer. Choose an idle GPU; disable MPS and leave allocator settings, clocks and power unchanged.

```bash
export HF_HOME=/absolute/path/to/populated/hf-cache
export CUDA_HOME=/usr/local/cuda-13.0
export YUE2_MODEL=/absolute/path/to/YuE2-3B/snapshot
export YUE2_VAE=/absolute/path/to/YuE2-Vae-legacy/snapshot
export YUE2_RUN_DIR=/absolute/path/to/a-new-run-directory
bash validation/yue2/reproduce-performance.sh
```

The wrapper reconstructs the two historical sources from `f9143976` and the included patches, verifies all 1875 main / 1879 PR production file hashes and model weight hashes, and runs the original measured harness/configurations. Only local asset paths change. Results go to the new directory. It fails if the directory already exists. Original run scripts, protocols, per-request measurements and environment records remain under `performance/`.

The wrapper was syntax-checked and source reconstruction was verified locally; it has not itself been used for another GPU measurement. For testing current code, use the implementation branch and the commands below rather than treating the historical harness as current-head acceptance.

## Repeat the current CPU regressions

In an implementation checkout at `d413a22364d1acd297778fea7edfc1a9de7ee12b`, with the [recorded dependencies](regression/remote/installed.txt):

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/worker/test_yue2_*.py tests/entrypoints/openai_api/test_tts_adapter.py tests/core/sched/test_omni_cfg_ar_scheduler.py -m 'core_model and cpu' --run-level core_model -q
CUDA_VISIBLE_DEVICES='' python -m pytest tests/worker/test_gpu_ar_model_runner.py tests/core/test_prefix_cache.py tests/engine/test_output_processor.py tests/core/sched/test_omni_ar_scheduler_logprobs.py -m 'core_model and cpu' --run-level core_model -q
pre-commit run --from-ref 2c72dea3aab6a6781ea650798cfdf85748abf0ef --to-ref HEAD --show-diff-on-failure
```

## WSB protocol and scripts

WSB is pinned to [`e361b85a`](https://huggingface.co/datasets/m-a-p/WildSongBench/tree/e361b85a8d7365d3079fa9647641a87ce3c8c5b5). Models: YuE2-3B `1a96eca688d6ae5d7f0feb88573fec89920fcd19`; legacy VAE `b54118f0fc462f08999d1ec07e88817f4ee3f770`. The [generation scripts/configs](wsb/generation), [scoring scripts](wsb/scoring/scripts), evaluator asset lock and environment records preserve the executed protocol.

These archived WSB shell scripts retain the original machine paths and require the matching frozen source, official evaluator package and generated artifacts. They are execution records, not a portable installer. Obtain the official evaluator from the pinned WSB repository and apply the recorded PER override if reproducing that execution. No WSB generation or scoring is triggered by `verify.py`.
