import numpy as np
import pandas as pd

from experiments.loao import run_loao
from src import probes


def test_probe_selectivity_signal_vs_noise():
    rng = np.random.default_rng(0)
    y = np.array([0] * 80 + [1] * 80)
    X_sep = np.vstack([rng.normal(0, 1, (80, 16)), rng.normal(1.5, 1, (80, 16))])
    X_noise = rng.normal(0, 1, (160, 16))
    assert probes.probe_selectivity(X_sep, y, seed=0)["selectivity"] > 0.3
    assert abs(probes.probe_selectivity(X_noise, y, seed=0)["selectivity"]) < 0.15


def test_detector_scores_higher_for_bonafide():
    rng = np.random.default_rng(0)
    X_tr = np.vstack([rng.normal(0, 1, (80, 16)), rng.normal(2, 1, (80, 16))])  # spoof, bona
    y_tr = np.array([0] * 80 + [1] * 80)
    X_te = np.vstack([rng.normal(2, 1, (20, 16)), rng.normal(0, 1, (20, 16))])  # bona-like, spoof-like
    sc = probes.detector_scores(X_tr, y_tr, X_te, seed=0)
    assert sc[:20].mean() > sc[20:].mean()


def test_run_loao_recovers_planted_geometry_effect():
    """Embeddings where more-distinct families are both more separable and harder to
    generalize to. run_loao should rank them so the most-distinct family has the
    biggest non-transfer gap and selectivity↔gap correlates positively."""
    rng = np.random.default_rng(0)
    D = 32
    spoofiness = np.zeros(D)
    spoofiness[0] = 2.0
    fams = {"voice_conversion": 0.5, "neural_tts": 1.5, "hybrid_tts_vc": 3.0, "concatenative_tts": 5.0}
    aid = {"voice_conversion": "A17", "neural_tts": "A07", "hybrid_tts_vc": "A13", "concatenative_tts": "A16"}
    rows, embs = [], []
    for i in range(160):
        embs.append(rng.normal(0, 1, D))
        rows.append(dict(utt_id=f"b{i}", label=1, attack_id="-", family="bonafide"))
    for j, (fam, m) in enumerate(fams.items()):
        d = np.zeros(D)
        d[j + 1] = m                                    # deterministic family direction
        for i in range(80):
            embs.append(rng.normal(0, 1, D) + spoofiness + d)
            rows.append(dict(utt_id=f"{fam}{i}", label=0, attack_id=aid[fam], family=fam))
    meta = pd.DataFrame(rows)
    X = np.vstack(embs)

    res = run_loao(meta, X, seed=0)
    pf = res["per_family"].set_index("family")
    assert pf.loc["concatenative_tts", "gap_pct"] > pf.loc["voice_conversion", "gap_pct"]
    assert res["correlation"]["rho"] > 0
