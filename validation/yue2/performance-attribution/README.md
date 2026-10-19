# YuE2 performance: per-change comparisons

Incremental experiments behind PR #9098. Each row uses the implementation and recipe recorded in its linked report.

## 1. Bottleneck and architectural motivation

Cosmos3’s same-process conditioning/generation design motivated the architecture experiments.

YuE already cached conditioning KV. The first separate-worker split removed HOLD work but reduced concurrent throughput. Fixed-input replay showed NAR taking 3.432 s alone versus 5.776 s alongside AR; MPS reduced the latter to 3.804 s. This led us toward same-worker execution. [Cosmos source review](cosmos/report.txt), [NAR isolation](nar-isolation/evidence/comparison.json).

Running the full acoustic loop synchronously still blocked other AR requests. We kept native AR batching and added an independent acoustic queue, releasing completed semantic requests' AR slots/KV while synthesis continued.

The October 9 profile, after reverting phase-separated AR queues and before the packed-KV/FlashInfer changes, found 105.471 s between acoustic jobs with no ready request, versus 1.64 ms with ready work awaiting dispatch. NAR velocity occupied 437.600 s of CUDA activity intervals versus 23.772 s for VAE. We then focused on NAR batching, copies and kernels. [Profile](profile/report.txt), [recorded configuration](profile/evidence/plan.json).

## 2. Individual changes retained or made configurable

Each row uses the same hardware within its comparison. Recipe changes are listed in the row.

| Retained change | Comparison and result |
| --- | --- |
| [Independent acoustic queue / early AR release](queue/report.txt) | Same base, NAR/VAE unchanged, AR4/KV4/NAR1, three rounds/load: throughput **+3.12% / +6.28% / +6.44%** at client concurrency 4 / 8 / continuous 16 capped at 8. |
| [Packed NAR](nar-batch/report.txt) | Same source, NAR1→NAR2 with VAE1, four alternating-order pairs: equal-length throughput **+6.07–12.59%**; P50 latency increased. Mixed-length results varied. NAR1 default. |
| [Direct packed-KV writes](kv-and-depth/report.txt) | Intermediate storage/copy removed: allocated peak **37.627→36.948 GiB**; throughput **+0.244% / +0.837%** against bracketing original runs. |
| [FlashInfer BF16 SiLU×gate](flashinfer/report.txt) | Fixed-input complete NAR solves, five B1/B2/B4 shapes: **NAR throughput +2.38–3.58%**; tested velocity/latent outputs exact. |
| [Single-element concatenation removal](single-cat/report.txt) | **15/15 FP32 audio pairs exact**, covering B1/B3/B4, cross-chunk and cancellation/reuse cases. |
| [Pre-capture synchronization/cache reclamation](capture-prep/report.txt) | Same load: failure at song 18 became **32/32 completed, zero retries/OOM**. Reserved memory at that capture fell **19.633→13.502 GiB**. |

Automatic ABC, paired CFG and HF loading are functional additions. The automatic-ABC/two-call diagnostic measured throughput −1.38% and handler CPU wall time at 0.415% of the run. Performance comparisons used CFG1. [ABC diagnostic](abc/report.txt).

## 3. Regressions and changes without a demonstrated E2E benefit

The last column gives our explanation/hypothesis.

| Experiment | Result | Explanation / hypothesis |
| --- | --- | --- |
| [Separate-worker KV-transfer split](separate-workers/report.txt) | Concurrency-4 throughput **−6.29%** | IPC stalls and longer co-running NAR outweighed removed HOLD work. |
| [One AR GPU + one acoustic GPU vs two replicas](two-gpu/report.txt) | Continuous-16 throughput **−30.16%** | Stage imbalance: about 45% AR / 94% acoustic utilization; CPU-staged transfer and 4 vs 8 total AR slots. |
| [Blocking complete-generation loop](direct-generation/report.txt) | Concurrent throughput **−6.29% to −7.84%** | Other AR requests cannot progress during synthesis. |
| [MPS on the same-worker phase-queue version](mps/report.txt) | Off/on/on/off: **−0.69% / −1.51%** throughput | AR/NAR already shared a context; compute contention remained. |
| [Phase-specific logits projection](phase-logits/report.txt) | Mixed projection **54.51–61.43% slower** | Extra projection and gather/scatter outweighed saved work. Reverted. |
| [Separate ABC/semantic queues](phase-queue/report.txt) | Throughput **−5.09%**; mean AR batch **22.185→16.260** | Smaller batches required more steps. Reverted. |
| [Equal acoustic/AR stream priority](priority/report.txt) | **+1.12%** throughput in one profiled pair; AR waits remained | Priority did not resolve shared-resource contention. |
| [Submission depth 2→1](kv-and-depth/report.txt) | **+0.170% / +0.763%**, RTF P95 slightly worse | Less queued work changed ordering without reducing NAR computation. |
| [VAE batch 1→2](vae-batch/report.txt) | Throughput within **±0.4%**; VAE1 peak **2.84 GiB lower** | Too little critical-path work to repay additional batch storage. |
| [Wait window 0/200/1000 ms](wait-window/report.txt) | Throughput **16.075/16.060/16.067**; B2 coverage **6/6/10** of 32 | Additional batching did not repay the wait. |
| [Max-B2→max-B4 with 32k/48k token budgets](token-budget/report.txt) | Throughput **+0.225% / −1.080%**; neither run formed B4 | A larger cap did not produce enough simultaneously ready requests. |

NAR operator profiling found attention and MLP dominant. FlashInfer attention/GEMM alternatives gave mixed results across shapes; the SiLU replacement was retained. [Profile](nar-hotspots/report.txt), [kernel comparison](flashinfer/report.txt).

## 4. Whole-path results

The October 10 packed-KV/FlashInfer snapshot measured throughput **+0.739% / −0.886%** against main with AR32/KV24/client64/NAR≤4/token32768/window200ms/VAE1. Main→PR→PR→main, each with 2 full warmups + 64 measured songs. [Plan](h800-e2e/evidence/plan.json), [results](h800-e2e/artifacts/pairs.json), [report](h800-e2e/report.txt).

Changes made after that run:

| Change | Record |
| --- | --- |
| Single-element concatenation removal | [Fixed-input audio and allocator profiling](single-cat/report.txt) |
| Unified default entry and CFG scheduling | [Ordinary-request admission and CFG-capacity handling](unified-entry/report.txt) |
| Graph pre-capture synchronization/cache reclamation | [Intervention](capture-prep/report.txt), [adopted patch](../performance/evidence/candidate.patch) |
| Upstream migration to `f9143976`, including delayed-output token handling | [Recorded patch](../performance/evidence/candidate.patch) |

The October 14 main/PR ABBA used base `f9143976` + the common loading fix versus the recorded PR patch. Both used AR1/KV3/client2, context24576, full-CoT/CFG1, ABC4096/semantic9000; PR NAR1/VAE1/window0/token-budget0. Main→PR→PR→main, fresh processes, each with 2 full warmups + 32 measured songs:

| Metric | PR relative to main, two pairs |
| --- | --- |
| Audio throughput | **+6.22% / +6.38%** |
| Content TPS | **+5.80% / +5.95%** |
| E2EL P95 | **−13.94% / −14.10%** |
| RTF P95 | **−15.75% / −15.86%** |
| Device peak | **22.688→21.373 GiB** |

PR audio was 2.454% shorter; both arms thermally throttled. [Sources, recipes and per-request results](../README.md#reproduce-the-measured-performance).

## Evidence provenance

`sources.json` records original workspace paths and SHA256. Archived reports and numeric records preserve their original bytes and labels; references inside the reports refer to their original experiment directories. The parent `SHA256.json` covers the published files.
