# YuE2 test consolidation

Implementation `15d1c014eab7f4d13a815cf6fd914a8fa993b7b0` follows `d413a22364d1acd297778fea7edfc1a9de7ee12b` with tests only. Production files are unchanged.

Six added test modules became four, reducing them from 1,094 to 957 lines (137 fewer). Existing modified test modules are untouched.

- Merge automatic ABC and paired CFG into `test_yue2_request.py`, sharing scheduler setup and parameterizing HTTP sampling isolation.
- Merge duplicate completion/cancellation coverage into the real parent scheduler path; retain AR slot release, unfinished-request accounting, delayed tokens, errors and late-output suppression.
- Parameterize independent NAR reference comparisons across fallback/packed boundaries; fold kernel and Graph capture regressions into acoustic tests.
- Remove tests of unchanged VAE checkpoint loading and trivial config serialization; retain HF revision/cache/auth forwarding and VAE selection precedence.

The new combinations produce 247 passed tests, 18 warnings in 13.19 seconds. All applicable pre-commit checks passed. Ruff fixed formatting/imports on its first pass; both logs are retained. Remote/local changed-test and log SHA256 checks passed; 2,036 production files matched the previous remote source.

Reproduce in the recorded Python 3.12.11 / PyTorch 2.13.0+cu130 / vLLM 0.31.0 environment, from the implementation checkout; no weights or GPU needed:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/worker/test_yue2_*.py tests/entrypoints/openai_api/test_tts_adapter.py tests/core/sched/test_omni_cfg_ar_scheduler.py -m 'core_model and cpu' --run-level core_model -q
pre-commit run --files tests/worker/test_yue2_request.py tests/worker/test_yue2_runtime.py tests/worker/test_yue2_acoustic_batch.py tests/worker/test_yue2_loading.py --show-diff-on-failure
```

The earlier 277 shared-contract passes remain attached to `d413a223`; those tests and production code did not change and were not rerun. No GPU, performance or WSB run was added. The original raw evidence is preserved.
