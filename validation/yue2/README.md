# YuE2 validation evidence

Run the published main/PR performance comparison or CPU regressions against an implementation checkout.

## Results and scope

| Check | Tested source | Result / evidence |
| --- | --- | --- |
| Performance | `f9143976` + recorded main/PR patches | Throughput +6.22/+6.38%; E2EL P95 −13.94/−14.10%; RTF P95 −15.75/−15.86%. [Results](performance/artifacts/comparison.json) |
| NAR retirement / capture preparation removal | Old PR `0b982dc8` → new PR `298d5258` | Performance effectively unchanged; throughput +0.65/+0.07%, allocated peak −1.53 GiB, device peak +1.58 GiB; zero OOM/retries. [Four-run comparison](nar-retirement/README.md) |
| ABC / paired CFG / HTTP | `f9143976` + PR | AR1/AR2, 402 paired sampling steps, five HTTP successes/two rejections. [Records](performance/artifacts) |
| CPU fixes | Exception-chain snapshot; logging at `9b62169b` | 256 YuE2 + 277 shared passes/2 CUDA-only skips; later 40 acoustic passes. [Exact scope and logs](review-regression/README.md) |
| WildSongBench | `f9143976` + PR, NAR1 | 384 outputs; same 192 PER-selected outputs for all four metrics. [Scores](wsb/scoring/nar1-full-scores-01/summary.json) |

The main/PR performance comparison used one RTX4090 24 GiB, AR1/KV3 GiB/client2, full CoT/CFG1, NAR1/VAE1. PR audio was 2.454% shorter; both versions thermally throttled.
The later NAR retirement comparison used a 4090 D with matching frame counts and token hashes across all four runs and no sampled thermal-throttling flags.
Batched NAR acceptance, Q3O and long-form CFG quality remain pending. [Per-change analysis](performance-attribution/README.md) · [Historical submission check](PRECHECK.md).

## Verify without a GPU

```bash
python3 validation/yue2/verify.py
python3 validation/yue2/nar-retirement/scripts/analyze.py
```

Checks file hashes and recomputes performance/WSB aggregates using the Python standard library. `provenance.json` and `performance-attribution/sources.json` identify the source records.

## Reproduce the measured performance

Use Linux, Bash, Git, tar, an idle NVIDIA GPU 0 with MPS off, a CUDA 13 toolkit/compatible driver, and the [recorded environment](performance/evidence/environment.json): Python 3.12.11, PyTorch 2.13.0+cu130, vLLM 0.31.0. Activate that environment; keep allocator settings, clocks and power unchanged.
Populate the offline HF cache with the [pinned assets](performance/evidence/assets.json), including the `gpt2` tokenizer. Use a full Git clone containing `f9143976`.

```bash
export HF_HOME=/absolute/path/to/populated/hf-cache
export CUDA_HOME=/usr/local/cuda-13.0
export YUE2_MODEL=/absolute/path/to/pinned/YuE2-3B
export YUE2_VAE=/absolute/path/to/pinned/YuE2-Vae-legacy
export YUE2_RUN_DIR=/absolute/path/to/new-run-directory
bash validation/yue2/reproduce-performance.sh
```

The script reconstructs and checks main/PR sources and weights, then runs main → PR → PR → main, each with two full warmups and 32 measured songs. Logs and results go under `$YUE2_RUN_DIR/artifacts`; the directory must be new.

## Repeat the current CPU regressions

Activate the [recorded CPU environment](regression/remote/installed.txt), with the selected implementation and its test dependencies installed. Select a clean implementation checkout to identify the tested commit; weights and GPU are unnecessary.

```bash
bash validation/yue2/reproduce-cpu.sh /absolute/path/to/implementation /absolute/path/to/new-cpu-results
```

Runs the YuE2/adapter/CFG suite, then shared runner/cache/output/scheduler contracts; stops on failure. Uses the active Python and a new output directory, recording source identity, environment, logs and exit codes.

## WSB protocol and scripts

[Generation](wsb/generation) · [Scoring](wsb/scoring/scripts) · [Additional scores](wsb/scoring/nar1-additional-01/summary.json). These archived launchers retain their original machine paths.
Selected 192: PER micro **0.090754**, SongBench Global Avg **6.729727**, Musicality **5.903877**, MuLan **0.511322**, AllMusicCaps **0.403524**. Scores cover the NAR1 PR outputs.
PER removed one redundant parent CUDA synchronization; [original/executed function](wsb/scoring/nar1-full-scores-01/per-worker-execution.json) is retained. ASR hit its cap for 11/384 outputs, including 3/192 selected; none were excluded.

Archived internal labels `upstream` / `candidate` mean main / PR. Source hashes cover the original logs, configs, patches and numeric data. Model weights and generated media are stored separately.
