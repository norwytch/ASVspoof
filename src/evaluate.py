"""Main evaluation loop: sweep all degradation conditions and write results.

Outputs:
    results/results.csv          one metric row per (condition, param)
    results/per_attack_eer.csv   EER per attack id x condition (failure analysis)
    results/scores/*.npz         cached per-utterance scores (resume-friendly)

Usage:
    python -m src.evaluate --protocol <key> --flac-dir <dir> [--subset 5000 | --full]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from . import degradations as deg
from .dataset import load_trials
from .metrics import compute_eer, per_attack_eer, summarize
from .model import DEFAULT_MODEL_ID, SpoofDetector

SCORES_DIR = Path("results/scores")

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **_):  # type: ignore
        return it


def _slug(family: str, params: dict) -> str:
    """Stable filename/label slug for a condition, e.g. 'mp3_bitrate_kbps=32'."""
    if not params:
        return family
    body = ",".join(f"{k}={v}" for k, v in params.items())
    return re.sub(r"[^A-Za-z0-9=,_.-]", "", f"{family}_{body}")


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load a (flac/wav) file as mono float32 + sample rate."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _degrade(audio: np.ndarray, sr: int, family: str, params: dict) -> np.ndarray:
    """Apply a non-streaming degradation. Streaming is handled in run_condition."""
    if family == "clean":
        return audio
    if family == "mp3":
        return deg.apply_mp3_compression(audio, sr, **params)
    if family == "noise":
        return deg.apply_noise(audio, sr, **params)
    if family == "telephony":
        mode = params.get("mode")
        if mode == "bandpass":
            return deg.apply_bandpass(audio, sr)
        if mode == "g711":
            return deg.apply_telephony(audio, sr)
        raise ValueError(f"unknown telephony mode {mode!r}")
    raise ValueError(f"unknown degradation family {family!r}")


def _build_loader(paths, family, params, batch_size, num_workers):
    """DataLoader yielding (B, MAX_SAMPLES) fp32 waveform batches.

    The per-file load + degrade (incl. the MP3 ffmpeg round-trip) runs inside
    worker processes, so it overlaps GPU compute instead of blocking it. Order is
    preserved (shuffle=False), so scores stay aligned with ``paths``/``trials``.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    from .model import _fix_length, _to_mono_16k

    class _DegradeDS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            audio, sr = load_audio(paths[i])
            audio = _degrade(audio, sr, family, params)
            return _fix_length(_to_mono_16k(audio, sr)).float()

    return DataLoader(_DegradeDS(), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=(num_workers > 0))


def run_condition(detector: SpoofDetector, trials: pd.DataFrame,
                  family: str, params: dict, *, cache: bool = True,
                  batch_size: int = 32, num_workers: int = 8, amp: bool = True):
    """Score every trial under one condition. Returns (labels, scores, attack_ids).

    Non-streaming families are scored with a parallel DataLoader + batched
    (optionally bf16-autocast) forward pass, which keeps the GPU busy and overlaps
    the ffmpeg/decoding cost. Streaming keeps the per-utterance path because each
    utterance produces a variable number of chunks (already batched internally by
    ``predict_streaming``).

    Results are cached to results/scores/<slug>.npz; a cached file is reused if
    its utt_ids match the current trial set (so re-running metrics is free and
    an interrupted sweep resumes cleanly).
    """
    slug = _slug(family, params)
    cache_path = SCORES_DIR / f"{slug}.npz"
    utt_ids = trials.utt_id.to_numpy()

    if cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        if np.array_equal(z["utt_id"], utt_ids):
            return z["label"], z["score"], z["attack_id"]

    if family == "streaming":
        scores = np.empty(len(trials), dtype=np.float32)
        for i, row in enumerate(tqdm(trials.itertuples(index=False),
                                     total=len(trials), desc=slug)):
            audio, sr = load_audio(row.path)
            _, agg = detector.predict_streaming(
                audio, sr, chunk_ms=params.get("chunk_ms"),
                overlap_ms=params.get("overlap_ms", 0))
            scores[i] = agg
    else:
        import torch
        detector.load()
        use_amp = amp and str(detector.device).startswith("cuda")
        loader = _build_loader(trials.path.tolist(), family, params, batch_size, num_workers)
        out = []
        for batch in tqdm(loader, total=len(loader), desc=slug):
            with torch.inference_mode():
                if use_amp:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        s = detector._forward(batch)
                else:
                    s = detector._forward(batch)
            out.append(s.float())
        scores = torch.cat(out).numpy().astype(np.float32)

    labels = trials.label.to_numpy()
    attack_ids = trials.attack_id.to_numpy()
    if cache:
        SCORES_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, utt_id=utt_ids, label=labels,
                 score=scores, attack_id=attack_ids)
    return labels, scores, attack_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True, help="ASVspoof CM key/protocol file")
    p.add_argument("--flac-dir", required=True, help="Directory of eval .flac files")
    p.add_argument("--model-id", default=None)
    p.add_argument("--subset", type=int, default=5000, help="Stratified subset size")
    p.add_argument("--full", action="store_true", help="Use the entire eval set")
    p.add_argument("--out", default="results/results.csv")
    p.add_argument("--per-attack-out", default="results/per_attack_eer.csv")
    p.add_argument("--codec-out", default="results/codec_eer.csv")
    p.add_argument("--batch-size", type=int, default=32, help="batch size for non-streaming scoring")
    p.add_argument("--num-workers", type=int, default=8, help="DataLoader workers for parallel load+degrade")
    p.add_argument("--no-amp", action="store_true", help="disable bf16 autocast (use fp32)")
    args = p.parse_args()
    score_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, amp=not args.no_amp)

    trials = load_trials(args.protocol, args.flac_dir, n=args.subset, full=args.full)
    print(f"Evaluating {len(trials)} trials "
          f"({trials.attrs['n_bonafide']} bona fide / {trials.attrs['n_spoof']} spoof)")

    detector = SpoofDetector(args.model_id or DEFAULT_MODEL_ID)

    # Clean baseline first — every Δ EER is measured against it.
    base_labels, base_scores, base_attacks = run_condition(detector, trials, "clean", {}, **score_kw)
    baseline_eer, _ = compute_eer(base_labels, base_scores)
    print(f"Clean baseline EER = {baseline_eer * 100:.2f}%")

    # Native-codec stratified EER on the clean condition. ASVspoof 2021 LA bakes in
    # real telephony/codec transmission (the `codec` column; label-correlated), so
    # this is the principled channel-effect analysis — re-applying our own synthetic
    # `telephony` degradation would double-count it. 'none' = uncompressed reference.
    codecs = trials["codec"].to_numpy()
    codec_rows = []
    for c in sorted(set(map(str, codecs))):
        m = codecs.astype(str) == c
        if m.sum() < 2 or len(set(base_labels[m].tolist())) < 2:
            continue
        eer_c, _ = compute_eer(base_labels[m], base_scores[m])
        codec_rows.append({"codec": c, "n": int(m.sum()),
                           "n_bonafide": int((base_labels[m] == 1).sum()),
                           "eer_pct": round(eer_c * 100.0, 3)})
    Path(args.codec_out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(codec_rows).to_csv(args.codec_out, index=False)
    print(f"Native-codec EER (clean) -> {args.codec_out}:")
    print(pd.DataFrame(codec_rows).to_string(index=False))

    rows, per_attack_rows = [], []
    for family, configs in deg.DEGRADATIONS.items():
        # Synthetic telephony is redundant with LA's native telephony codec layer
        # (see --codec-out breakdown); functions are kept in degradations.py and can
        # be re-enabled by removing this skip.
        if family == "telephony":
            continue
        for params in configs:
            labels, scores, attacks = run_condition(detector, trials, family, params, **score_kw)
            row = {"condition": family, "param": _slug(family, params).replace(f"{family}_", "") if params else "—"}
            row.update(summarize(labels, scores, baseline_eer=baseline_eer))
            rows.append(row)

            for attack, eer in per_attack_eer(labels, scores, attacks).items():
                per_attack_rows.append({"condition": family,
                                        "param": row["param"],
                                        "attack_id": attack,
                                        "eer_pct": eer * 100.0})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(per_attack_rows).to_csv(args.per_attack_out, index=False)
    print(f"Wrote {args.out} ({len(rows)} conditions) and {args.per_attack_out}")


if __name__ == "__main__":
    main()
