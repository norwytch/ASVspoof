"""Part 2 — linear probing with control-task selectivity (research-design.md §3.4).

Operates on cached, utterance-level SSL embeddings (see embeddings.py). All
probes and detector heads are linear, so this runs on CPU.

Key methodological point (Hewitt & Liang 2019; Belinkov 2022): raw probe
accuracy is not evidence. We report **selectivity** = accuracy(real labels) −
accuracy(control task with shuffled labels). A high-capacity probe can fit
random labels; only the *gap* indicates the property is linearly encoded.
"""
from __future__ import annotations

import numpy as np


def _pipeline(C: float, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=C, random_state=seed,
                           multi_class="auto"),
    )


def probe_accuracy(X: np.ndarray, y: np.ndarray, *, C: float = 1.0,
                   cv: int = 5, seed: int = 0) -> float:
    """Cross-validated linear-probe accuracy for predicting ``y`` from ``X``."""
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(_pipeline(C, seed), X, y, cv=cv, scoring="accuracy")
    return float(scores.mean())


def probe_selectivity(X: np.ndarray, y: np.ndarray, *, C: float = 1.0,
                      cv: int = 5, seed: int = 0) -> dict[str, float]:
    """Selectivity = real-label accuracy − control-task (shuffled-label) accuracy.

    The control preserves the label distribution but destroys the X↔y mapping, so
    its accuracy is the capacity floor for this probe on this data.
    """
    real = probe_accuracy(X, y, C=C, cv=cv, seed=seed)
    rng = np.random.default_rng(seed)
    y_ctrl = rng.permutation(y)
    ctrl = probe_accuracy(X, y_ctrl, C=C, cv=cv, seed=seed)
    return {"accuracy": real, "control": ctrl, "selectivity": real - ctrl}


def layerwise_selectivity(layer_X: dict[int, np.ndarray], y: np.ndarray, *,
                          C: float = 1.0, cv: int = 5, seed: int = 0) -> dict[int, dict]:
    """Run probe_selectivity per layer. ``layer_X`` maps layer index -> (N, D).

    Returns {layer: {accuracy, control, selectivity}} — the layer-wise profile
    (§3.4); generator artifacts may peak in early layers while the head reads a
    weighted sum.
    """
    return {L: probe_selectivity(X, y, C=C, cv=cv, seed=seed)
            for L, X in sorted(layer_X.items())}


def capacity_sweep(X: np.ndarray, y: np.ndarray, *, Cs=(0.1, 1.0, 10.0),
                   cv: int = 5, seed: int = 0) -> dict[float, dict]:
    """Selectivity at ≥2 probe strengths so a null isn't a weak-probe artifact (§3.4)."""
    return {C: probe_selectivity(X, y, C=C, cv=cv, seed=seed) for C in Cs}


def detector_scores(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, *,
                    C: float = 1.0, seed: int = 0) -> np.ndarray:
    """Fit a linear bona-fide/spoof head and return P(bona fide) on the test set.

    Output follows the repo convention (higher == more bona fide; label 1 = bona
    fide), so it plugs straight into metrics.compute_eer for the LOAO grid.
    """
    clf = _pipeline(C, seed).fit(X_train, y_train)
    classes = list(clf.named_steps["logisticregression"].classes_)
    proba = clf.predict_proba(X_test)
    return proba[:, classes.index(1)]
