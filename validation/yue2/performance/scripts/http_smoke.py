"""Actual unified speech endpoint: automatic/external ABC and default off."""

import argparse
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assets = json.loads((root / "evidence/assets.json").read_text())
    out = root / "artifacts" / args.label
    out.mkdir(exist_ok=False)
    config = yaml.safe_load((root / "source/vllm_omni/deploy/yue2.yaml").read_text())
    config["stages"][0].update(max_num_seqs=2, max_model_len=4096, max_num_batched_tokens=8192, kv_cache_memory_bytes=3*1024**3)
    config["stages"][0]["hf_overrides"].update(
        yue2_vae_path=assets[1]["path"],
        yue2_vae_revision=assets[1]["revision"],
    )
    deploy = out / "serve.yaml"
    deploy.write_text(yaml.safe_dump(config))
    command = [
        str(Path(sys.executable).with_name("vllm")),
        "serve",
        assets[0]["path"],
        "--omni",
        "--served-model-name",
        "m-a-p/YuE2-3B",
        "--revision",
        assets[0]["revision"],
        "--deploy-config",
        str(deploy),
        "--host",
        "127.0.0.1",
        "--port",
        "19092",
        "--api-key",
        "yue2-validation",
    ]
    item = {"style":"Warm acoustic folk, gentle guitar", "lyrics":"[Verse]\nMorning light across the bay\nCarry all my dreams away\n[Outro]", "seed":831001}
    results = {}
    with (out / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(
                base_url="http://127.0.0.1:19092",
                timeout=180,
                headers={"Authorization": "Bearer yue2-validation"},
            ) as client:
                deadline = time.monotonic() + 180
                while True:
                    assert server.poll() is None, "Server exited before readiness"
                    try:
                        if client.get("/health", timeout=2).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    assert time.monotonic() < deadline, "Readiness timeout"
                    time.sleep(0.5)
                for label, extra in (
                    ("off", {}),
                    ("guided-off", {"cfg_scale": 1.5}),
                    (
                        "melody-auto",
                        {"cot": "melody", "abc_max_tokens": 32, "cfg_scale": 1.5},
                    ),
                    (
                        "full-auto",
                        {"cot": "full", "abc_max_tokens": 32, "cfg_scale": 1.5},
                    ),
                    (
                        "external",
                        {
                            "cot": "full",
                            "cfg_scale": 1.5,
                            "abc": "X:1\nT:Test\nM:4/4\nL:1/8\nK:C\nCDEF GABc|",
                        },
                    ),
                    ("cfg-rejected", {"cfg_scale": 21}),
                    ("budget-rejected", {"cot": "full", "abc_max_tokens": 0}),
                ):
                    response = client.post(
                        "/v1/audio/speech",
                        json={
                            "model": "m-a-p/YuE2-3B",
                            "input": item["lyrics"],
                            "instructions": item["style"],
                            "seed": item["seed"],
                            "max_new_tokens": 200,
                            "response_format": "wav",
                            "extra_params": extra,
                        },
                    )
                    row = {"status": response.status_code}
                    if label.endswith("rejected"):
                        assert response.status_code == 400, response.text
                        row["body"] = response.json()
                    else:
                        assert response.status_code == 200, response.text[:1000]
                        wave, sr = sf.read(
                            io.BytesIO(response.content), dtype="float32"
                        )
                        assert (
                            sr == 48000
                            and wave.shape == (200 * 1920 - 64, 2)
                            and np.isfinite(wave).all()
                        )
                        row.update(
                            shape=list(wave.shape),
                            sr=sr,
                            headers={
                                k: v
                                for k, v in response.headers.items()
                                if "audio" in k
                            },
                        )
                    results[label] = row
                    print(label, json.dumps(row), flush=True)
                (out / "result.json").write_text(json.dumps(results, indent=2))
        finally:
            server.terminate()
            try:
                server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
                raise
        results["server_exit_code"] = server.returncode
        (out / "result.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
