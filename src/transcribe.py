"""Whisper ASR pipeline.

Run ONCE over the eval set and persist to results/transcripts.jsonl
(transcription is slow). All downstream NLP analysis loads from disk.

Each JSONL line:
    {"utt_id", "text", "avg_logprob", "no_speech_prob", "compression_ratio"}
``avg_logprob`` is the ASR-confidence proxy used in nlp_features.py.

whisper is imported lazily so the rest of the framework imports without it.
"""
from __future__ import annotations

import json
from pathlib import Path


def _segment_stats(result: dict) -> dict:
    """Aggregate Whisper segment-level fields into utterance-level confidence."""
    segs = result.get("segments", [])
    if not segs:
        return {"avg_logprob": float("nan"), "no_speech_prob": float("nan"),
                "compression_ratio": float("nan")}
    n = len(segs)
    return {
        "avg_logprob": sum(s["avg_logprob"] for s in segs) / n,
        "no_speech_prob": sum(s["no_speech_prob"] for s in segs) / n,
        "compression_ratio": sum(s["compression_ratio"] for s in segs) / n,
    }


def transcribe_dataset(trials, *, model_size: str = "base",
                       out: str | Path = "results/transcripts.jsonl",
                       device: str | None = None) -> Path:
    """Transcribe every trial with Whisper, appending to a JSONL cache.

    Resumable: utt_ids already present in ``out`` are skipped, so an interrupted
    run continues where it left off.
    """
    import whisper

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if out.exists():
        with open(out) as f:
            done = {json.loads(line)["utt_id"] for line in f if line.strip()}

    model = whisper.load_model(model_size, device=device)
    todo = [r for r in trials.itertuples(index=False) if r.utt_id not in done]
    with open(out, "a") as f:
        for row in todo:
            result = model.transcribe(row.path, fp16=False)
            rec = {"utt_id": row.utt_id, "text": result["text"].strip(),
                   **_segment_stats(result)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
    return out


def load_transcripts(path: str | Path = "results/transcripts.jsonl") -> dict:
    """Load the transcript cache keyed by utt_id."""
    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["utt_id"]] = rec
    return out
