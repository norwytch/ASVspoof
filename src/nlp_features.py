"""Extension 1: Transcript-conditioned anomaly signals.

Hypothesis: synthetic speech yields atypical ASR behavior — higher LM
perplexity on the recovered transcript, more degenerate repetition, and lower
Whisper confidence — and these correlate with the audio model's errors.

Features per utterance:
    - gpt2_perplexity : GPT-2 perplexity of the Whisper transcript (batched)
    - repetition_rate : fraction of repeated n-grams in the transcript
    - asr_confidence  : Whisper avg_logprob (+ no_speech_prob, compression_ratio)

Output: results/nlp_scores.csv, then correlate each signal with the audio
model's per-utterance score. transformers/torch are imported lazily.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Text-only features (no heavy deps)
# --------------------------------------------------------------------------- #
def repetition_rate(text: str, n: int = 2) -> float:
    """Fraction of n-grams that are repeats (degenerate-decoding proxy).

    0.0 == every n-gram unique; → 1.0 == highly repetitive. Empty/short text → 0.
    """
    tokens = text.lower().split()
    if len(tokens) < n + 1:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


# --------------------------------------------------------------------------- #
# GPT-2 perplexity (batched)
# --------------------------------------------------------------------------- #
def gpt2_perplexity(texts: list[str], *, batch_size: int = 16,
                    model_name: str = "gpt2", device: str | None = None) -> list[float]:
    """Batched GPT-2 perplexity per text. Empty texts → nan.

    Perplexity is exp(mean token NLL); computed with right-padding and a loss
    mask so padding tokens don't contribute.
    """
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = GPT2TokenizerFast.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()

    out: list[float] = []
    for i in range(0, len(texts), batch_size):
        batch = [t if t.strip() else tok.eos_token for t in texts[i:i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(device)
        ids, mask = enc.input_ids, enc.attention_mask
        with torch.inference_mode():
            logits = model(ids, attention_mask=mask).logits
        # shift for next-token prediction
        sl = logits[:, :-1, :]
        st = ids[:, 1:]
        sm = mask[:, 1:].float()
        nll = torch.nn.functional.cross_entropy(
            sl.reshape(-1, sl.size(-1)), st.reshape(-1), reduction="none"
        ).view(st.size())
        tok_nll = (nll * sm).sum(1) / sm.sum(1).clamp(min=1)
        out.extend(torch.exp(tok_nll).cpu().tolist())
    # blank originals → nan
    return [float("nan") if not t.strip() else p for t, p in zip(texts, out)]


# --------------------------------------------------------------------------- #
# Assembly + correlation
# --------------------------------------------------------------------------- #
def build_nlp_table(transcripts: dict, audio_scores: dict | None = None,
                    labels: dict | None = None, *,
                    out: str | Path = "results/nlp_scores.csv",
                    skip_perplexity: bool = False) -> pd.DataFrame:
    """Per-utterance NLP features + (optional) audio score & label.

    ``transcripts`` is the dict from transcribe.load_transcripts. ``audio_scores``
    / ``labels`` map utt_id -> value (from evaluate's cached clean scores).
    """
    utt_ids = list(transcripts)
    texts = [transcripts[u]["text"] for u in utt_ids]
    ppl = ([float("nan")] * len(texts) if skip_perplexity
           else gpt2_perplexity(texts))

    rows = []
    for u, text, p in zip(utt_ids, texts, ppl):
        t = transcripts[u]
        row = {
            "utt_id": u,
            "gpt2_perplexity": p,
            "repetition_rate": repetition_rate(text),
            "asr_avg_logprob": t.get("avg_logprob", float("nan")),
            "asr_no_speech_prob": t.get("no_speech_prob", float("nan")),
            "asr_compression_ratio": t.get("compression_ratio", float("nan")),
        }
        if audio_scores is not None:
            row["audio_score"] = audio_scores.get(u, float("nan"))
        if labels is not None:
            row["label"] = labels.get(u)
        rows.append(row)

    df = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def correlate_with_audio(df: pd.DataFrame, target: str = "audio_score") -> dict:
    """Spearman correlation of each NLP signal with the audio score.

    Spearman (rank) is used because the relationships are expected monotonic but
    not linear. NaNs are dropped pairwise.
    """
    from scipy.stats import spearmanr

    signals = ["gpt2_perplexity", "repetition_rate", "asr_avg_logprob",
               "asr_no_speech_prob", "asr_compression_ratio"]
    res = {}
    for s in signals:
        if s not in df or target not in df:
            continue
        pair = df[[s, target]].dropna()
        if len(pair) < 3 or pair[s].std() == 0:
            res[s] = (float("nan"), float("nan"))
            continue
        rho, pval = spearmanr(pair[s], pair[target])
        res[s] = (float(rho), float(pval))
    return res
