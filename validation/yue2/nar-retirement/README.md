# NAR retirement and capture preparation

Commit `298d5258296710b30ec69aafd901e9cabe8dd799` retires NAR state after a CUDA event completes and before VAE decoding. It removes the device synchronization and cache clearing previously performed before NAR graph capture. Model arithmetic, sampling parameters and VAE computation are unchanged.

## Protocol

The comparison used old PR `0b982dc825fbf27b9d31bf4c7b7db08f5d1b997a` and new PR `298d5258296710b30ec69aafd901e9cabe8dd799`, in old → new → new → old order. Each run used a separate process, two complete warmup songs and eight measured long songs.

Hardware and configuration: RTX 4090 D 24 GiB, driver 570.124.04, Python 3.12.11, PyTorch 2.13.0+cu129, vLLM 0.31.0+cu129, FlashInfer 0.7.0.post1; AR1, KV3 GiB, client concurrency2, NAR1/VAE1, context24576, full CoT, CFG1, ABC4096, semantic9000, 32 midpoint steps, Graph1, MPS off, OMP/MKL1.

The fixed sample used indices `[0,4,8,12,17,20,24,28]` from the previous 32-song sequence with their original seeds. Source and configuration hashes were checked before and after each run. The two runtime snapshots differed only in `yue2.py` and `batched_synthesis.py`.

Timing excluded initialization, warmup and shutdown. Stage, capture and allocator-history probes were disabled. Device memory, clocks, temperature and power were sampled through NVML every 100 ms. E2EL starts when a request acquires a client slot and ends when its final result arrives. Content TPS counts returned ABC and semantic tokens over the measured makespan.

## Results

All four runs completed on October 20, 2026, with controller exit0: 32 measured requests and eight warmups. Performance was effectively unchanged.

| Run | Audio seconds/second | Content TPS | E2EL P50/P95 (s) | RTF P50/P95 |
| --- | ---: | ---: | ---: | ---: |
| Old PR 1 | 3.5721 | 120.522 | 131.821 / 175.859 | 0.5646 / 0.6390 |
| New PR 1 | 3.5952 | 121.301 | 130.906 / 174.588 | 0.5614 / 0.6360 |
| New PR 2 | 3.5777 | 120.711 | 131.414 / 175.466 | 0.5636 / 0.6407 |
| Old PR 2 | 3.5754 | 120.632 | 131.630 / 175.499 | 0.5646 / 0.6386 |

The two new/old pairs changed throughput and TPS by +0.647% / +0.065%, E2EL P95 by −0.723% / −0.019%, and RTF P95 by −0.462% / +0.323%. Each run produced 1980.549 seconds of audio. Frame counts and ABC/semantic token hashes matched across all four runs. OOMs, allocator retries and budget truncations were zero; final request and queue state was empty.

## Memory

Each version produced the same peaks in both repetitions.

| Measured peak (GiB) | Old PR | New PR | Change |
| --- | ---: | ---: | ---: |
| Allocated | 14.897 | 13.365 | −1.531 |
| Reserved | 19.924 | 21.506 | +1.582 |
| Device memory | 20.842 | 22.424 | +1.582 |

NAR retirement lowered the live-tensor peak. With explicit cache clearing removed, the allocator retained more cached memory and the device-memory peak increased. Both versions completed within 24 GiB.

Median SM clock was 2730 MHz in all four runs. Temperature ranges were 53–71, 54–70, 53–71 and 55–72°C; sampled software/hardware thermal-throttling flags were zero. Mean power was 308.39, 308.33, 308.11 and 309.70 W.

## Evidence and scope

[Analysis](evidence/analysis.json), [protocol](evidence/plan.json), [controller log](remote/evidence/controller.log), [source identities](remote/evidence/runtime-sources.json), and [raw requests and device samples](remote/artifacts) record the comparison. The [recovery manifest](remote/recovery.json) covers 48 files totaling 4,528,254 bytes; [collection verification](evidence/collection-result.json) checked every size and SHA256. Archived scripts retain their original machine paths and bytes.

The logs retain initialization cache warnings, base-class no-audio cleanup warnings and process-manager shutdown warnings. Output shape and finite-value checks passed. After shutdown, GPU usage was 1 MiB/0% with no compute process.

This run covers NAR1 at client concurrency2 in the recorded CUDA 12.9 environment. Each P95 describes eight requests. Waveform parity, higher concurrency, packed NAR and multi-GPU execution were not tested. WSB was not rerun for this lifetime/synchronization change; the existing WSB results remain tied to their recorded implementation snapshot.
