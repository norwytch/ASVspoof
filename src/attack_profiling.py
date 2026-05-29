"""Extension 2: TTS attack profiling.

Central hypothesis: text-conditioned (neural TTS) and speaker-conditioned
(voice conversion) attacks degrade differently under real-world conditions
because their artifacts live in different parts of the signal.

Pipeline:
    1. Load data/attack_taxonomy.json (attack_id -> category, system).
    2. Cluster Whisper transcripts via sentence embeddings; cross-reference
       clusters with attack ids (does a system's text cluster together?).
    3. Re-report degradation EER broken down by attack CATEGORY.
    4. Visualize: UMAP of embeddings (colored by category), EER heatmap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_taxonomy(path: str | Path = "data/attack_taxonomy.json") -> dict:
    with open(path) as f:
        return json.load(f)


def cluster_by_transcript(transcripts: list[str], n_clusters: int = 5,
                          model_name: str = "all-MiniLM-L6-v2"):
    """Embed transcripts and KMeans-cluster. Returns (labels, embeddings).

    TODO: SentenceTransformer(model_name).encode + KMeans(random_state=42).
    """
    raise NotImplementedError


def eer_by_category(labels: np.ndarray, scores: np.ndarray,
                    attack_ids: np.ndarray, taxonomy: dict) -> dict[str, float]:
    """Map attack_ids -> category via taxonomy, then per-category EER vs bona fide."""
    raise NotImplementedError
