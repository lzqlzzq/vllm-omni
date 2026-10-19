"""Add E2E content-token throughput without changing frozen measurements."""
import json
import math
import sys
from pathlib import Path


def metrics(rows, elapsed):
    assert rows and elapsed > 0
    semantic = sum(row["frames"] for row in rows)
    abc = sum(row["abc_tokens"] for row in rows)
    assert semantic > 0 and abc >= 0
    return dict(semantic_tokens=semantic, abc_content_tokens=abc,
                semantic_tps=semantic / elapsed, abc_content_tps=abc / elapsed,
                total_content_tps=(semantic + abc) / elapsed)


def main():
    root = Path(__file__).resolve().parents[1]
    labels = sys.argv[1:]
    assert labels
    # Aggregate counts over one shared interval, not per-request rate averages.
    sample = metrics([dict(frames=20, abc_tokens=4), dict(frames=10, abc_tokens=2)], 2)
    assert sample["semantic_tps"] == 15 and sample["total_content_tps"] == 18
    results = []
    for label in labels:
        folder = root / "artifacts" / label
        summary = json.loads((folder / "summary.json").read_text())
        rows = [json.loads(line) for line in (folder / "requests.jsonl").read_text().splitlines()]
        assert summary["complete"] and len(rows) == summary["count"]
        elapsed = max(row["end"] for row in rows) - min(row["start"] for row in rows)
        assert math.isclose(elapsed, summary["elapsed_s"], abs_tol=1e-6)
        item = dict(label=label, elapsed_s=elapsed, **metrics(rows, elapsed))
        # Audio has 1920 samples per codec frame, minus 64 samples per song.
        audio_rate = (item["semantic_tokens"] * 1920 - len(rows) * 64) / 48000 / elapsed
        assert math.isclose(audio_rate, summary["throughput_audio_s_s"], abs_tol=1e-9)
        results.append(item)
    result = dict(complete=True, scope="Aggregate content tokens divided by full E2E makespan, warmup excluded. Semantic counts codec frames; ABC counts retained score tokens. Excludes prompts/control tokens, CFG negative-branch work and NAR ODE evaluations. Not phase-local decode TPS.", results=results)
    (root / "artifacts/tps.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
