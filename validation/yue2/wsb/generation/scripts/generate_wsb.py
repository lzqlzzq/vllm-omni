"""Generate WSB through native AsyncOmni; retain the official evaluator artifacts."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from vllm_omni import AsyncOmni
from vllm_omni.model_executor.models.yue2.queued import Yue2QueueWorker
from vllm_omni.model_executor.models.yue2.request import prepare_song
from vllm_omni.model_executor.models.yue2.yue2 import (
    ABC_SAMPLING,
    SEMANTIC_SAMPLING,
    CODEC_OFFSET,
    CODEC_SIZE,
)
from vllm_omni.tokenizers.yue2_prompt import semantic_prefix_ids, song_text
from vllm_omni.tokenizers.yue2_tokenizer import YuE2TextTokenizer

ROOT = Path(__file__).resolve().parents[1]


class ArtifactWorker(Yue2QueueWorker):
    def capture_latents(self):
        model = self.model_runner.model.sample.__self__
        from vllm_omni.model_executor.models.yue2.yue2 import _SynthesisJob

        queue = model._synthesis
        original = _SynthesisJob.release
        captures = Path(os.environ["WSB_OUTPUT"]) / "captures"
        captures.mkdir()

        def release(job):
            if job._released:
                return original(job)
            # Retain references across native synchronization/release; B1 pump
            # can release a job before finish() receives its output.
            latents = [x for x in job.keep if isinstance(x, torch.Tensor)
                       and x.ndim == 2 and x.shape[1] == 64]
            members = (list(queue.members) if hasattr(queue, "members")
                       else [job])
            original(job)
            if job.cancelled or job.error is not None or job.finite is None or not bool(job.finite):
                return
            assert len(latents) == len(members), (len(latents), len(members))
            records = []
            for state, latent in zip(members, latents, strict=True):
                rid = external_id(state.request_id)
                path = captures / f"{rid}.npy"
                assert not path.exists(), path
                value = latent.detach().float().cpu().numpy()
                assert np.isfinite(value).all()
                np.save(path, value)
                records.append(dict(id=rid, internal_id=state.request_id,
                                    frames=latent.shape[0]))
            with (captures / "batches.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(members=records, size=len(records),
                                            start=job.t_start, done=job.t_done)) + "\n")

        _SynthesisJob.release = release
        from vllm_omni.model_executor.models.yue2 import batched_synthesis
        original_nar = batched_synthesis.BatchedCachedNAR

        def packed_nar(*args, **kwargs):
            engine = original_nar(*args, **kwargs)
            with (captures / "packed-nar.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(
                    ids=[external_id(state.request_id) for state in queue.members],
                    frames=engine.lengths, query_tokens=engine.nar_length,
                    kv_tokens=engine.key_offsets[-1])) + "\n")
            return engine

        batched_synthesis.BatchedCachedNAR = packed_nar
        return True

    def final_state(self):
        model = self.model_runner.model.sample.__self__
        return dict(states=len(model._states), active=bool(model._synthesis.active),
                    waiting=len(model._synthesis.waiting),
                    allocated=torch.cuda.memory_allocated(),
                    reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(),
                    peak_reserved=torch.cuda.max_memory_reserved(),
                    allocator={k: torch.cuda.memory_stats().get(k, 0)
                               for k in ("num_alloc_retries", "num_ooms")})


def external_id(request_id):
    match = re.fullmatch(r"(wsb_\d{3}_r[01]_s\d+)(?:-[0-9a-f]{8})?", request_id)
    if match is None:
        raise ValueError(f"Unexpected WSB request ID: {request_id}")
    return match[1]



def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--prompt-index", type=int, action="append")
    parser.add_argument("--nar2", action="store_true")
    args = parser.parse_args()
    deploy = ROOT / "configs" / ("nar2.yaml" if args.nar2 else "nar1.yaml")
    concurrency = 4 if args.nar2 else 2
    stage = yaml.safe_load(deploy.read_text())["stages"][0]
    assert stage["max_num_seqs"] == (2 if args.nar2 else 1)
    assert stage["hf_overrides"]["yue2_nar_batch_size"] == (2 if args.nar2 else 1)
    assert stage["hf_overrides"]["yue2_vae_batch_size"] == 1
    out = ROOT / "artifacts" / args.label
    out.mkdir(parents=True, exist_ok=False)
    os.environ["WSB_OUTPUT"] = str(out)
    rows = [
        json.loads(line)
        for line in (ROOT / "upstream/reproduction_manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [r["prompt_index"] for r in rows] == list(range(192))
    if args.prompt_index is not None:
        assert len(set(args.prompt_index)) == len(args.prompt_index)
        assert all(0 <= i < 192 for i in args.prompt_index)
        rows = [r for r in rows if r["prompt_index"] in args.prompt_index]
    assets = json.loads((ROOT / "evidence/assets.json").read_text())
    weights = {}
    for name, asset in zip(["mot", "vae"], assets, strict=True):
        p = Path(asset["path"])
        assert digest(p / "model.safetensors") == asset["sha256"]
        weights[name] = {
            "files": {
                "model.safetensors": {
                    "sha256": asset["sha256"],
                    "bytes": (p / "model.safetensors").stat().st_size,
                }
            },
            "config_sha256": digest(p / "config.json"),
        }
    sys.path.insert(0, str(ROOT / "wsb_metrics"))
    from artifact_storage import collect_hashes, identity, verify_result
    from wsb_eval import GENERATION, validate_generation

    for phase, preset in [("abc", ABC_SAMPLING), ("semantic", SEMANTIC_SAMPLING)]:
        assert all(preset[k] == v for k, v in GENERATION[phase].items())
    expected_source = json.loads((ROOT / "evidence/candidate-source.json").read_text())
    assert all(digest(ROOT / "source" / name) == sha
               for name, sha in expected_source.items())
    production = {
        str(p.relative_to(ROOT / "source")): digest(p)
        for p in sorted((ROOT / "source/vllm_omni").rglob("*"))
        if p.suffix in {".py", ".yaml"}
    }
    write(out / "source-hashes.json", production)
    runtime = identity(
        {
            "production": production,
            "harness": digest(__file__),
            "deploy": digest(deploy),
            "torch": torch.__version__,
            "vllm": importlib.metadata.version("vllm"),
        }
    )
    config = dict(
        generation=GENERATION,
        overrides={},
        cot="full",
        cfg_scale=1.0,
        cfg_negative="same_instruction_and_exact_abc",
        backend="vllm",
        implementation="vllm-omni/yue2",
        quantization="none",
        model_dtype="bfloat16",
        vae_dtype="float32",
        vae_decode="halo_crop",
        vae_core_frames=1024,
        vae_halo_frames=16,
        memory_budget_gib=24,
        offload_ar=False,
        runtime_sha256=runtime,
    )
    tokenizer = YuE2TextTokenizer.from_pretrained(
        assets[0]["path"], revision=assets[0]["revision"]
    )
    for row in rows:
        assert (
            song_text(row["style"], row["lyrics"], "full") == row["generation_prompt"]
        )
        assert row["candidate_ar_seeds"] == row["candidate_nar_seeds"]
        prepare_song(
            tokenizer.encode, row["style"], row["lyrics"], cot="full", max_frames=9000
        )
    engine = AsyncOmni(
        model=assets[0]["path"],
        revision=assets[0]["revision"],
        deploy_config=str(deploy),
        stage_overrides={
            "0": {
                "worker_cls": "generate_wsb.ArtifactWorker",
                "hf_overrides": {
                    "yue2_vae_path": assets[1]["path"],
                    "yue2_vae_revision": assets[1]["revision"],
                },
            }
        },
    )
    entries, results = [], []
    semaphore = asyncio.Semaphore(concurrency)
    try:
        reply = await engine.collective_rpc("capture_latents")
        while isinstance(reply, list):
            assert len(reply) == 1
            reply = reply[0]
        assert reply is True, reply
        async def generate(row, j, seed):
            async with semaphore:
                await candidate(row, j, seed)

        async def candidate(row, j, seed):
            i = row["prompt_index"]
            assert seed == [831001, 831019][j] + i
            rid = f"wsb_{i:03d}_r{j}_s{seed}"
            folder = out / "songs" / rid
            if shutil.disk_usage(out).free < 512 * 2**20:
                raise RuntimeError(
                    "Less than 512 MiB disk space; preserving completed candidates"
                )
            folder.mkdir(parents=True)
            request = dict(
                id=rid,
                style=row["style"],
                lyrics=row["lyrics"],
                cot="full",
                cfg_scale=1.0,
                seed=seed,
                abc=None,
            )
            prompt, params = prepare_song(
                tokenizer.encode,
                row["style"],
                row["lyrics"],
                cot="full",
                seed=seed,
                max_frames=9000,
            )
            start = time.perf_counter()
            result = None
            async for result in engine.generate(
                prompt, request_id=rid, sampling_params_list=[params]
            ):
                pass
            elapsed = time.perf_counter() - start
            assert result is not None
            def save():
                mm = result.multimodal_output
                meta = mm.get("meta") or {
                    k[5:]: v for k, v in mm.items() if k.startswith("meta.")
                }
                assert not torch.as_tensor(meta["error"]).item(), meta
                wave = mm.get("audio", mm.get("model_outputs")).float()
                semantic = [
                    t - CODEC_OFFSET
                    for t in result.outputs[0].token_ids
                    if CODEC_OFFSET <= t < CODEC_OFFSET + CODEC_SIZE
                ]
                abc = mm["abc_token_ids"].tolist()
                assert (
                    wave.shape == (2, len(semantic) * 1920 - 64)
                    and torch.isfinite(wave).all()
                )
                (out / "captures" / f"{rid}.npy").rename(folder / "latent.npy")
                latent = np.load(folder / "latent.npy")
                assert latent.shape == (len(semantic), 64) and np.isfinite(latent).all()
                sf.write(folder / "audio.flac", wave.numpy().T, 48000, subtype="PCM_24")
                np.save(folder / "semantic.npy", np.asarray(semantic, dtype=np.int32))
                np.save(
                    folder / "prefix.npy",
                    np.asarray(
                        semantic_prefix_ids(
                            tokenizer.encode, row["style"], row["lyrics"], "full", abc
                        ),
                        dtype=np.int32,
                    ),
                )
                np.save(folder / "abc.npy", np.asarray(abc, dtype=np.int32))
                write(folder / "request.json", request)
                write(folder / "config.json", config)
                receipt = dict(
                    status="complete",
                    identity=identity(
                        {"request": request, "config": config, "weights": weights}
                    ),
                    truncated={
                        "abc": bool(
                            torch.as_tensor(meta.get("abc_truncated", False)).item()
                        ),
                        "semantic": bool(torch.as_tensor(meta["truncated"]).item()),
                    },
                    sample_rate=48000,
                    audio_seconds=wave.shape[1] / 48000,
                    weights=weights,
                    timing={"e2el_with_latent_capture_s": elapsed},
                    artifacts=collect_hashes(folder),
                )
                validate_generation(request, config, receipt)
                write(folder / "result.json", receipt)
                verify_result(folder)
                return abc, semantic, receipt

            abc, semantic, receipt = await asyncio.to_thread(save)
            entries.append(
                dict(
                    prompt_index=i,
                    candidate_index=j,
                    seed=seed,
                    path=f"songs/{rid}",
                )
            )
            (out / "evaluation_inputs.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in sorted(entries, key=lambda e: (e["prompt_index"], e["candidate_index"])))
            )
            results.append(
                dict(
                    id=rid,
                    abc_tokens=len(abc),
                    frames=len(semantic),
                    **receipt["timing"],
                    truncated=receipt["truncated"],
                )
            )
            write(
                out / "generation_report.json",
                dict(
                    complete=False,
                    expected_candidates=len(rows) * 2,
                    results=results,
                ),
            )
            print(json.dumps(results[-1]), flush=True)

        tasks = [asyncio.create_task(generate(row, j, seed))
                 for row in rows for j, seed in enumerate(row["candidate_ar_seeds"][:2])]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        batches = [json.loads(line) for line in (out / "captures/batches.jsonl").read_text().splitlines()]
        ids = [member["id"] for batch in batches for member in batch["members"]]
        assert sorted(ids) == sorted(result["id"] for result in results)
        packed_path = out / "captures/packed-nar.jsonl"
        packed = [json.loads(line) for line in packed_path.read_text().splitlines()] if packed_path.exists() else []
        if not args.nar2:
            assert all(batch["size"] == 1 for batch in batches) and not packed
        write(out / "batch_summary.json", dict(client_concurrency=concurrency, ar_slots=2 if args.nar2 else 1,
              nar_cap=2 if args.nar2 else 1, vae_cap=1, jobs=len(batches),
              packed_nar_calls=len(packed), packed_request_ids=sorted({rid for group in packed for rid in group["ids"]}),
              batch_sizes={str(size): sum(b["size"] == size for b in batches) for size in (1, 2)}))
        final = await engine.collective_rpc("final_state")
        while isinstance(final, list):
            assert len(final) == 1
            final = final[0]
        write(out / "final_state.json", final)
        assert final["states"] == 0 and not final["active"] and final["waiting"] == 0, final
        assert final["allocator"]["num_ooms"] == 0, final
        assert all(digest(ROOT / "source" / name) == sha for name, sha in production.items())
    finally:
        engine.close()
    write(
        out / "generation_report.json",
        dict(complete=True, expected_candidates=len(rows) * 2, results=results),
    )


if __name__ == "__main__":
    asyncio.run(main())
