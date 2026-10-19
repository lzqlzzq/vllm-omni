"""Fetch pinned WSB generation assets into the existing content-addressed cache."""

import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
assets = [
    (
        "m-a-p/YuE2-3B",
        "1a96eca688d6ae5d7f0feb88573fec89920fcd19",
        "1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9",
    ),
    (
        "m-a-p/YuE2-Vae-legacy",
        "b54118f0fc462f08999d1ec07e88817f4ee3f770",
        "b6d283628913bb41145ba99e2314eef613905ee95f690eb70e8212d5f4965044",
    ),
]
report = []
for repo, revision, expected in assets:
    path = Path(
        snapshot_download(
            repo,
            revision=revision,
            allow_patterns=["*.json", "*.tiktoken", "*.safetensors"],
            max_workers=2,
        )
    )
    with (path / "model.safetensors").open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    assert digest == expected, (repo, digest)
    report.append(dict(repo=repo, revision=revision, path=str(path), sha256=digest))
    print(json.dumps(report[-1]), flush=True)
(root / "evidence/assets.json").write_text(json.dumps(report, indent=2) + "\n")
