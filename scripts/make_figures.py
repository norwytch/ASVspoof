"""Phase 2 figure driver: build the degradation-sweep plots from cached scores.

Reads results/scores/*.npz + results/per_attack_eer.csv and writes PNGs to
results/figures/. Re-runnable; depends only on the cached sweep outputs.
"""
import pandas as pd

from src import visualize as viz
from src.metrics import compute_eer

FIG = viz.FIGDIR
saved = []


def eer_of(slug):
    labels, scores = viz.load_scores(slug)
    return compute_eer(labels, scores)[0]


clean_eer = eer_of("clean")

# 1. ROC + 2. DET overlay: clean vs one representative of each degradation.
rep = {
    "clean": viz.load_scores("clean"),
    "mp3 8kbps": viz.load_scores("mp3_bitrate_kbps=8"),
    "noise 0dB": viz.load_scores("noise_snr_db=0"),
    "stream 0.5s": viz.load_scores("streaming_chunk_ms=500,overlap_ms=0"),
}
saved.append(viz.plot_roc(rep, "ROC — clean vs degradations", FIG / "roc_overview.png"))
saved.append(viz.plot_det(rep, "DET — clean vs degradations", FIG / "det_overview.png"))

# 3. EER vs MP3 bitrate (ascending bitrate; left = most compressed).
br = [8, 16, 32, 64, 128]
saved.append(viz.plot_eer_sweep(
    br, [eer_of(f"mp3_bitrate_kbps={b}") for b in br],
    "MP3 bitrate (kbps)", FIG / "eer_vs_mp3.png", baseline_eer=clean_eer))

# 4. EER vs noise SNR (ascending SNR; left = noisiest).
snr = [0, 5, 10, 20, 30]
saved.append(viz.plot_eer_sweep(
    snr, [eer_of(f"noise_snr_db={s}") for s in snr],
    "additive noise SNR (dB)", FIG / "eer_vs_noise.png", baseline_eer=clean_eer))

# 5. EER vs streaming chunk size (no-overlap variants).
chunks = [500, 2000, 4000]
saved.append(viz.plot_eer_sweep(
    chunks, [eer_of(f"streaming_chunk_ms={c},overlap_ms=0") for c in chunks],
    "streaming chunk size (ms)", FIG / "eer_vs_streaming.png", baseline_eer=clean_eer))

# 6. Per-attack EER heatmap over all conditions.
pa = pd.read_csv("results/per_attack_eer.csv")
saved.append(viz.plot_attack_heatmap(pa, FIG / "per_attack_heatmap.png"))

# 7. Clean score-separation histogram (bona fide vs spoof).
lbl, sc = viz.load_scores("clean")
saved.append(viz.plot_score_hist(lbl, sc, "Clean score separation", FIG / "score_hist_clean.png"))

print(f"clean baseline EER = {clean_eer*100:.2f}%")
for p in saved:
    print("wrote", p)
