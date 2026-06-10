import numpy as np

from src.dataset import leave_one_attack_out, parse_protocol, select_eval_subset

TAX = {a: {"category": c} for a, c in [
    ("A07", "neural_tts"), ("A10", "neural_tts"), ("A13", "hybrid_tts_vc"),
    ("A16", "concatenative_tts"), ("A19", "voice_conversion")]}
ATTACKS = ["A07", "A10", "A13", "A16", "A19"]


def _write_2021(path, n_spoof=120, n_bona=40):
    lines = []
    for i in range(n_spoof):
        lines.append(f"LA_00{i % 9} LA_E_{1000 + i} alaw ita_tx {ATTACKS[i % 5]} spoof notrim eval")
    for i in range(n_bona):
        lines.append(f"LA_00{i % 9} LA_E_{5000 + i} none none - bonafide notrim eval")
    lines.append("LA_001 LA_E_9999 opus dummy A07 spoof notrim progress")  # should be filtered
    path.write_text("\n".join(lines) + "\n")


def test_parse_2021_layout_and_phase_filter(tmp_path):
    p = tmp_path / "meta.txt"
    _write_2021(p)
    df = parse_protocol(str(p), "/flac", phase="eval")
    assert len(df) == 160                                   # progress row filtered
    assert (df.phase == "progress").sum() == 0
    assert set(df[df.label == 0].attack_id) == set(ATTACKS)
    assert (df.label == 1).sum() == 40
    assert df.path.iloc[0].endswith(".flac")


def test_parse_2019_layout(tmp_path):
    p = tmp_path / "p19.txt"
    p.write_text("LA_0079 LA_T_1 - - bonafide\nLA_0079 LA_T_2 - A01 spoof\n")
    df = parse_protocol(str(p), "/x", phase="eval")         # no phase tokens -> filter skipped
    assert list(df.label) == [1, 0]
    assert list(df.attack_id) == ["-", "A01"]


def test_subset_preserves_groups(tmp_path):
    p = tmp_path / "meta.txt"
    _write_2021(p, n_spoof=300, n_bona=100)
    df = parse_protocol(str(p), "/f")
    sub = select_eval_subset(df, n=80, min_per_group=5)
    assert sub.groupby(["label", "attack_id"]).ngroups == df.groupby(["label", "attack_id"]).ngroups
    assert len(sub) < len(df)


def test_loao_splits_isolate_held_family(tmp_path):
    p = tmp_path / "meta.txt"
    _write_2021(p)
    df = parse_protocol(str(p), "/f")
    splits = list(leave_one_attack_out(df, TAX, seed=0))
    fams = [f for f, _, _ in splits]
    assert set(fams) == {"neural_tts", "concatenative_tts", "voice_conversion", "hybrid_tts_vc"}
    for f, tr, te in splits:
        assert f not in set(tr[tr.label == 0].family)        # held family excluded from train
        assert set(te[te.label == 0].family) == {f}          # test spoof is only the held family
        bona_tr = set(tr[tr.label == 1].utt_id)
        bona_te = set(te[te.label == 1].utt_id)
        assert bona_tr.isdisjoint(bona_te)                   # no bona-fide leakage
