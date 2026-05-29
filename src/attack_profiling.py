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

sentence-transformers / umap are imported lazily.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_taxonomy(path: str | Path = "data/attack_taxonomy.json") -> dict:
    """Load the attack_id -> {category, system} map. Returns the 'attacks' dict."""
    with open(path) as f:
        data = json.load(f)
    return data.get("attacks", data)


def category_of(attack_id: str, taxonomy: dict) -> str:
    """Map an attack id to its category; bona fide ('-') -> 'bonafide'."""
    if attack_id in ("-", "", None):
        return "bonafide"
    entry = taxonomy.get(attack_id, {})
    return entry.get("category", "unknown")


# --------------------------------------------------------------------------- #
# Transcript clustering
# --------------------------------------------------------------------------- #
def embed_transcripts(transcripts: list[str], model_name: str = "all-MiniLM-L6-v2"):
    """Sentence embeddings for a list of transcripts. Returns (N, D) array."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name).encode(transcripts,
                                                   show_progress_bar=False)


def cluster_by_transcript(transcripts: list[str], n_clusters: int = 5,
                          model_name: str = "all-MiniLM-L6-v2", seed: int = 42):
    """Embed transcripts and KMeans-cluster. Returns (cluster_labels, embeddings)."""
    from sklearn.cluster import KMeans

    embeddings = embed_transcripts(transcripts, model_name)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return km.fit_predict(embeddings), embeddings


def cluster_attack_agreement(cluster_labels: np.ndarray,
                             attack_ids: np.ndarray) -> dict:
    """Do transcript clusters line up with attack systems?

    Returns adjusted Rand index + adjusted mutual information between the
    transcript clustering and the attack-id partition. High values => the
    transcript carries system-specific information.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    return {
        "adjusted_rand": float(adjusted_rand_score(attack_ids, cluster_labels)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(attack_ids, cluster_labels)),
    }


# --------------------------------------------------------------------------- #
# Degradation EER by attack category
# --------------------------------------------------------------------------- #
def eer_by_category(labels: np.ndarray, scores: np.ndarray, attack_ids: np.ndarray,
                    taxonomy: dict) -> dict[str, float]:
    """Per-category EER (each spoof category vs the shared bona-fide pool)."""
    from .metrics import compute_eer

    labels = np.asarray(labels)
    scores = np.asarray(scores)
    cats = np.array([category_of(a, taxonomy) for a in attack_ids])
    bona = labels == 1
    out: dict[str, float] = {}
    for cat in sorted(set(cats[labels == 0])):
        mask = bona | (cats == cat)
        eer, _ = compute_eer(labels[mask], scores[mask])
        out[cat] = eer
    return out


def category_degradation_table(scores_dir: str | Path, taxonomy: dict,
                               conditions: list[str]) -> pd.DataFrame:
    """Build the condition x attack-category EER table from cached score files.

    ``conditions`` are score-file slugs (e.g. ['clean','mp3_bitrate_kbps=32']).
    Mirrors the main results table but rows are attack categories.
    """
    scores_dir = Path(scores_dir)
    rows = []
    for slug in conditions:
        z = np.load(scores_dir / f"{slug}.npz", allow_pickle=True)
        cat_eer = eer_by_category(z["label"], z["score"], z["attack_id"], taxonomy)
        rows.append({"condition": slug, **{k: v * 100 for k, v in cat_eer.items()}})
    return pd.DataFrame(rows).set_index("condition")
