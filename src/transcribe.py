"""Whisper ASR pipeline.

Run ONCE over the eval set and persist to results/transcripts.jsonl
(transcription is slow). All downstream NLP analysis loads from disk.

Each JSONL line: {"utt_id", "text", "avg_logprob", "no_speech_prob", "segments"}
where avg_logprob is the ASR-confidence proxy used in nlp_features.py.
"""
from __future__ import annotations

from pathlib import Path


def transcribe_dataset(trials, model_size: str = "base",
                       out: str | Path = "results/transcripts.jsonl") -> None:
    """Transcribe every trial with Whisper, appending to a JSONL cache.

    TODO: load whisper, iterate trials, skip already-transcribed utt_ids so the
    run is resumable, persist text + confidence fields.
    """
    raise NotImplementedError


def load_transcripts(path: str | Path = "results/transcripts.jsonl") -> dict:
    """Load the transcript cache keyed by utt_id."""
    raise NotImplementedError
