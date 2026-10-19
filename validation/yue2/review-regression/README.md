# CPU review regressions

Each row identifies the source snapshot used for that run.

| Change | Tested identity | Result |
| --- | --- | --- |
| Release chained exception tracebacks | `602bae5c` + [patch](chain/final.patch); [4,237-file manifest](chain/source-final.json) | [YuE2: 256 passed, 18 warnings](chain/yue2-cpu.log); [shared: 277 passed, 2 CUDA-only skipped, 16 warnings](chain/shared-cpu.log) |
| Preserve exception causes in ERROR logs | `8c592e3d` + [patch](logging/final.patch), committed as `9b62169b864d78b4de3d021df888553d6c813fe1` | [Acoustic module: 40 passed, 16 warnings](logging/after.log) |

Both controllers exited 0: [chain](chain/regress.exit), [logging](logging/regress.exit). The chain snapshot was checked before/after execution; its [final source check](chain/source-verification.json) is retained. The logging run's [SHA list](logging/SHA256SUMS) covers its remote source and records; the package `SHA256.json` verifies the files published here.

Python 3.12.11, PyTorch 2.13.0+cu130, vLLM 0.31.0, CUDA disabled. [Dependency record](../regression/remote/installed.txt). To rerun both CPU suites against a selected implementation, use [reproduce-cpu.sh](../reproduce-cpu.sh); its output directory records the actual revision and result.
