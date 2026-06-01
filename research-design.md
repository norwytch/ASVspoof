# Fingerprints vs. Process Artifacts: A Representational Study of Why Audio Anti-Spoofing Detectors Fail to Generalize

_Research-design note for the ASVspoof robustness project. The rigorous, falsification-driven framing — grounded against the literature, with every cited claim verified (see §9). This is the "research-grade" reframing of the engineering project in [audio_deepfake_robustness_proposal.md](audio_deepfake_robustness_proposal.md)._

---

## 1. Central hypothesis (one falsifiable claim)

**H1 (causal-predictive).** *Across held-out spoofing families, the degree to which a frozen SSL embedding linearly encodes the held-out family's generator identity predicts how badly a detector built on that embedding fails to generalize to that family. If a targeted high-frequency vocoder-artifact band-mask is applied at train time, leave-one-attack-out (LOAO) EER on families whose discriminative artifact lives in that band improves relative to a bandwidth-matched control mask.*

Falsifiable two independent ways:
- (a) the across-family rank correlation between *probe-recovers-generator* (selectivity) and *LOAO generalization failure* is null or negative;
- (b) the targeted band-mask fails to beat the bandwidth-matched control.

**Why the original framing was dropped.** The initial "learns synthesis-*process* artifacts **vs.** memorizes generator-specific *fingerprints*" dichotomy is a false dichotomy — a vocoder fingerprint *is* a process artifact. The literature's honest open question (Müller et al. 2024, *Harder or Different?*) is **transferable vs. non-transferable cues**: do detectors key on attack-agnostic forensic cues (the upsampling/vocoder-artifact class shared across generators, which transfers) or on family-specific cues (which don't)? "Generator identity recoverable by a linear probe" is the operational marker of the *non-transferable* end — not proof of an identity fingerprint. Stated accordingly: probe-recoverable family identity is a **predictor of non-transfer**, not evidence of memorization.

Part 1 of the motivating premise — detectors collapse on unseen attacks — is **settled consensus**, stated without hedging. Part 2 — *why* — is **genuinely open** (Müller et al. 2024 state plainly that a comprehensive analysis of the underlying challenges is lacking). That gap is the target.

---

## 2. Claims restated defensibly

Each row was adversarially verified; where the verification weakened the premise, the claim is restated or dropped. Full verdicts in §9.

| Original premise | Verdict | Restated for the design |
|---|---|---|
| Silence alone is **near-SOTA** | Overstated | Silence-duration alone reaches **~15% EER** on 2019/2021 LA — far from sub-1% SOTA, but a real shortcut that inflates strong models (RawNet2 **3.6% → 15.5% EER when trimmed**) [Müller 2021]. Silence also carries *genuine* TTS/VC cues [Zhang 2023], so trimming is itself confounded — report **trimmed and untrimmed**. |
| Vocoder = "checkerboard," CV intuition transfers | Supported, scoped | Shared *mechanism* is the upsampling operator [Pons 2021 ↔ Odena 2016]. But the *signature differs*: images show 2D spatial grids, audio shows **tonal/horizontal-line artifacts + HF spectral replicas — not a literal 2D checkerboard**. Detection transfer is real [Sun 2023; Frank & Schönherr 2021]. **Scope caveat:** anti-aliased vocoders (BigVGAN) make the HF footprint less pronounced though still detectable [Gasenzer & Wolter 2024]; the diffusion-vocoder case is *not* covered by that reference — do not claim it without a diffusion-specific source. Phase/bispectral cues [AlBadawy 2019] are a **separate channel** — do not conflate with magnitude/upsampling. |
| Linear probe recovers *which* generator ⇒ fingerprinting | Partially supported | On a binary-fine-tuned-then-frozen SSL CM embedding, a light head recovers **vocoder identity 73.7%** and **acoustic-model identity 91.4%** (two-stage regime); an end-to-end SSL probe reaches 84.6% / 99.4% [Klein 2024, *Source Tracing of Audio Deepfake Systems*]. Well above chance, **uneven across attribute type**, degrades on unseen generators. Design consequences: (i) test **both** frozen regimes — off-the-shelf XLS-R *and* binary-fine-tuned-then-frozen; (ii) recovery is *consistent with* keying on per-generator artifacts, **not** proof a fingerprint drives detection. |
| EER noisy across seeds ⇒ multi-seed needed | Supported | RawGAT-ST/AASIST spanned **1.19–2.06% EER across seeds** [Jung 2022]; multi-round EER reporting is the documented norm [Wang & Yamagishi 2021]. Seed variance and eval-set sampling CI are **distinct** sources — report both. |

---

## 3. Experimental protocol

### 3.1 Backbone (frozen, cache-once)
wav2vec2-XLS-R front end [Tak 2022]; cache per-layer embeddings once to disk. **Two frozen regimes, stated explicitly:**
- **(A)** off-the-shelf pretrained XLS-R (truly frozen — "does the SSL prior already encode this?");
- **(B)** XLS-R fine-tuned on the binary spoof task, then frozen (the Klein-2024 regime — "did the detector learn it?").

Probe and detector heads are **light** (linear / 1-layer). All conditions share the identical front end and feature cache — **vary one factor at a time** (front-end choice alone moved EER ~37% relative in prior work, the most-criticized confound).

### 3.2 LOAO matrix
ASVspoof 2021 LA reuses 2019 attacks **A07–A19**. Group by *waveform-generation attribute* — the most attributable axis [Klein 2024] — not by system name. Use the verified mapping in [data/attack_taxonomy.json](data/attack_taxonomy.json):

- **neural_tts** — A07, A08, A09, A10, A11, A12
- **concatenative_tts** — A16 (single system — interpret with care)
- **voice_conversion** — A17, A18, A19
- **hybrid_tts_vc** — A13, A14, A15

For each family *f*: train on bona fide + all spoof families **except** *f*, evaluate on held-out *f*. Report the **full per-family grid + pooled** — pooled-only hides the generalization signal and is gameable by easy attacks [Liu 2023]. A large pooled-vs-averaged gap flags incompatible per-attack score distributions.

### 3.3 Shortcut ablations (run before any "artifact" claim)
1. **Silence-only ceiling** — classifier on leading/trailing-silence duration alone; reproduce ~15% EER to show the shortcut exists in *this* pipeline, then show it is removed [Müller 2021].
2. **VAD trim** — identical endpointing at train and test [Chettri 2020, on ASVspoof 2017]; report trimmed and untrimmed; add mask-silence vs. mask-non-silence to dissociate duration-proportion from silence-content cues [Zhang 2023].
3. **Duration / energy / peak-amplitude equalization** — the confounds ASVspoof 5 organizers suppress [Wang et al. 2025, §4.2].
4. **Channel/codec** — 2021 LA mixes telephony codecs (label-correlated). Balance/randomize codec across classes or augment with an acoustic simulator [Chen 2021]; **verify the HF artifact survives the codec** — band-limiting can erase the very signal under study.
5. **Residual bias check** — bona fide stays "easier" even after trimming [Shim 2024]; report loss asymmetry so no single ablation is claimed to "close" shortcuts.

### 3.4 Generator-identity probing (proper methodology)
Probe the frozen embedding for *family identity* among the **training** families (held-out *f* is the generalization target, never a probe class).
- **Linear probes only**, with **control tasks + selectivity** = (target acc − random-label-control acc) [Hewitt & Liang 2019; Belinkov 2022]. Raw accuracy alone is not evidence.
- **Layer-wise** profile, not a single layer — artifacts may sit in early CNN / low transformer layers while the head reads a weighted sum [Pasad 2021].
- **Baselines:** chance/majority, random-feature, untrained/shuffled-model.
- **Confound controls:** run the *same* probe on silence-trimmed, energy/duration-equalized, channel-randomized inputs and within fixed speaker/content — the generator signal must **survive** or it was codec/speaker/silence, not synthesis [Pîrlogeanu 2026 warns attribution collapses under vocoder/prompt mismatch].
- **Probe-capacity sanity:** ≥2 probe strengths so a null isn't a weak-probe artifact.
- **Cheap CV-analog baseline:** standardized average-residual fingerprint (signal minus low-pass/EnCodec-filtered) [Pizarro 2024] to compare against the SSL probe.

### 3.5 The H1 correlation (core test)
Per held-out family *f*: x = probe selectivity for *f*'s identity (from a probe where *f* is in-distribution — how separable it is), y = detector LOAO failure on *f* (EER_LOAO − in-domain EER). Test **Spearman rank** correlation — n ≈ 4 families (or ≈ 13 systems), so rank stats, CIs, a **pre-registered directional prediction**, and no point-estimate over-reading. Report at both family (n≈4) and system (n≈13) granularity.

### 3.6 Band-masking intervention (causal handle, pre-registered)
- Adapt Frequency Feature Masking [Kwak 2022] from *random* bands to a **targeted HF vocoder-artifact band** (localize via integrated gradients / per-family average log-spectra; expect >4 kHz for GAN vocoders [Gasenzer & Wolter 2024; Frank 2020]).
- **Predicted outcome:** training with the targeted HF mask **improves LOAO EER** on families whose artifact lives in that band (forces the detector off the band-localized shortcut).
- **Mandatory control:** a **bandwidth-matched low/mid-band mask**. Targeted-HF > control ⇒ artifact-band-specific. Both equal ⇒ generic regularization — report honestly.
- **Pre-registered failure mode:** on anti-aliased / diffusion vocoders (or ASVspoof 5) the HF story may collapse; test the *hardest* held-out family explicitly.

---

## 4. Novelty delta — the contribution is the integration

| Prior thread | Already established | Did **not** do |
|---|---|---|
| Klein 2024; Yan 2022; Pizarro 2024; Sun 2023 | Generator/vocoder identity **is** recoverable from (SSL) embeddings | Never **correlated** recoverability with **detector LOAO failure**; used it as a training signal or standalone attribution |
| Müller 2024 (*Harder or Different?*) | The gap is "difference," not "hardness," at score/dataset level | No **representational** localization; no frozen-embedding probe; no frequency-band attribution |
| Müller 2021/2022; Chettri 2020; Wang 2025 (ASVspoof 5) | Silence/duration/energy shortcuts; OOD collapse | Do **not** treat the HF vocoder artifact as a shortcut; no train-time **targeted band-mask** with a directional LOAO prediction |
| Kwak 2022 (FFM) | Random-band masking improves robustness | Not **targeted** HF bands; never measures **LOAO** delta vs. a matched-bandwidth control |
| Frank 2020; Pons 2021; Wang 2020 (vision) | Upsampling → spectral fingerprints; single-source + aug generalizes | The **CV→audio fingerprint framing on a frozen SSL backbone for ASVspoof LOAO** is undrawn in the audio literature |

**The synthesis contribution (and only this):** (1) selectivity-controlled probing of the *frozen detection embedding* + (2) the cross-family **correlation** "family-identity recoverability ↔ LOAO non-transfer" as a falsifiable test + (3) a **pre-registered, control-matched targeted band-mask** intervention + (4) the scope-limited CV-fingerprint framing. Honest caveat: every *ingredient* exists; novelty is the causal/predictive integration, not any single component. Re-check against 2025–26 ASVspoof 5 source-tracing work before claiming full novelty.

---

## 5. Statistical rigor

- **Two variance sources, both reported and labeled:** (i) **seed variance** — retrain the light head ≥5 seeds (cheap, front end frozen), report mean ± std [Jung 2022; Wang & Yamagishi 2021]; (ii) **eval-set sampling CI** — utterance-level bootstrap (~1000 reps) [Bisani & Ney 2004], **paired** bootstrap for A/B (targeted vs. control mask), not two independent CIs.
- **Metrics:** pooled **and** per-family EER; min t-DCF for 2021 LA comparability [Liu 2023]; state operating point/priors (EER is calibration-blind [Brümmer 2021]); for any ASVspoof 5 extension use min a-DCF [Shim/Jung et al. 2024, *a-DCF*].
- **Correlation:** Spearman + CI, small-n caveat foregrounded, pre-registered direction.
- **No best-of-N:** report the full distribution, never the best run.

---

## 6. Constraints respected

Frozen XLS-R, embeddings cached once to disk, all probes/detectors linear or 1-layer → **CPU or a single cheap GPU**. The one-time cost is feature extraction; LOAO, probes, ablations, and the band-mask are all cheap on cached features — rigor and the compute budget point the same way. CV background is leveraged *exactly as far as the verification supports*: the upsampling-operator mechanism transfers [Pons ↔ Odena], the *visual* checkerboard does **not** (audio = tonal lines + HF replicas), phase is a separate channel, and the HF-artifact spine is scoped to GAN-class vocoders with a pre-registered risk on diffusion/BigVGAN/ASVspoof 5.

---

## 7. What makes it a finding vs. a demo — and the biggest way it's wrong

**Three results that elevate it:**
1. A **significant per-family rank correlation** (selectivity ↔ LOAO EER inflation) that survives the silence/energy/codec controls — the novel empirical claim.
2. **Targeted HF mask beats the bandwidth-matched control** on LOAO EER for the predicted families, multi-seed, paired-bootstrap significant — correlation → causal handle.
3. **Layer- and frequency-localization that agree** — the embedding layers where family identity is most selective coincide with the band whose masking most helps LOAO: one coherent story, not two disconnected effects.

**The single most likely way the thesis is wrong:** the probe's "generator identity" is a **confound** — speaker, content, or (most dangerously in 2021 LA) **codec/channel** — *and* the band-mask "improvement" is generic regularization. **The design reveals this rather than hides it:** (a) the probe must survive channel-randomized / fixed-speaker controls — if selectivity evaporates, H1 is *rejected*, not rescued; (b) the bandwidth-matched control mask directly tests regularization-vs-artifact; (c) the two frozen regimes separate "SSL prior already encodes it" from "the detector learned it." A clean negative is itself interview-grade: it would show generalization failure is *not* explained by linearly-recoverable family identity, redirecting Müller's "difference" term away from the fingerprint story.

---

## 8. Mapping onto this codebase

What the rigorous design needs, against what [src/](src/) already has. Roughly 60% reuses existing code; three genuinely new pieces.

| Protocol element (§3) | Status | Where / what to add |
|---|---|---|
| **Frozen SSL embedding cache** (XLS-R, per-layer) | adapt | [reconstruction.py](src/reconstruction.py) already has `load_encoder` (frozen HuBERT) and `_features` (extracts `last_hidden_state`). New `src/embeddings.py`: swap to `Wav2Vec2Model` XLS-R, call with `output_hidden_states=True`, add `cache_embeddings(trials, layers, out_dir)` writing one `.npy` per utt. The freeze + lazy-import pattern is already there. |
| **LOAO splits** | new (small) | `dataset.leave_one_attack_out(trials, taxonomy)` yielding `(held_family, train_df, test_df)`. Reuses `attack_profiling.category_of` for grouping and the stratification logic already in `select_eval_subset`. |
| **Light detector + identity probe** | new | `src/probes.py`: sklearn `LogisticRegression` on cached embeddings — same `StandardScaler` + CV pattern as `prosody.prosody_eer`. Add `probe_selectivity(emb, labels, seed)` = target_acc − shuffled-label control_acc [Hewitt & Liang]; loop over layers. |
| **Silence-only ceiling + VAD trim** | new (degradations) | [degradations.py](src/degradations.py): `trim_silence(audio, sr)` (energy/`librosa.effects.trim`), `silence_duration_features(audio, sr)` for the ceiling classifier. Sits alongside the existing degradation callables + `DEGRADATIONS` registry. |
| **Duration/energy equalization** | new (degradations) | `degradations.equalize_energy`, `fix_duration` — small, same signature `(audio, sr) -> audio`. |
| **Codec randomization** | reuse | `apply_mp3_compression` / `apply_telephony` already exist; wrap with random param choice. |
| **Band-mask intervention** | new (key) | `degradations.apply_band_mask(audio, sr, low_hz, high_hz)` via STFT zeroing (`scipy.signal.stft`/`istft`). Two configs: targeted-HF and bandwidth-matched control. This is the one genuinely new *signal-processing* primitive. |
| **Per-family EER / correlation** | reuse | `metrics.per_attack_eer` and `attack_profiling.eer_by_category` already produce per-family EER. Add `metrics.spearman_with_ci` (scipy `spearmanr` is already imported in `nlp_features`). |
| **Multi-seed + bootstrap CI** | new (small) | `metrics.bootstrap_eer_ci(labels, scores, n=1000, paired=False)`; extend `summarize` to take a seed list and emit mean±std. |
| **LOAO experiment runner** | adapt | [evaluate.py](src/evaluate.py)'s `run_condition` + `.npz` score-cache pattern is the template; new `experiments/loao.py` (or extend `evaluate`) loops families × seeds, caches scores, writes the grid. |

**Net new files:** `src/embeddings.py`, `src/probes.py`, `experiments/loao.py`. **New functions in existing files:** `dataset.leave_one_attack_out`; `degradations.{trim_silence, silence_duration_features, equalize_energy, fix_duration, apply_band_mask, matched_control_band}`; `metrics.{spearman_with_ci, bootstrap_eer_ci, seed_variance}`. Everything else composes from what's built.

> **Implementation status (built & unit-tested).** All of the above now exist and pass synthetic-data tests; heavy deps (torch/transformers) are lazily imported so the logic tests on CPU without them. `experiments.loao.run_loao` was validated against a *planted* H1 effect — embeddings where more-distinct families are both more probe-separable and harder to generalize to — and the pipeline recovered ρ≈0.8 (selectivity↔gap), with the expected wide CI at n=4 families. Still pending real execution: the XLS-R embedding cache over downloaded ASVspoof audio (`python -m src.embeddings`), then `python -m experiments.loao`.

> Note: this reframing **complements** the existing degradation-robustness project rather than replacing it. The degradation pipeline (MP3/telephony/noise/streaming) becomes one axis of the shortcut/robustness analysis (§3.3, channel/codec), and the existing reconstruction/prosody extensions remain as independent detection baselines.

---

## 9. References (verified)

All 30 references below were checked against arXiv / the publishing venue — **all resolve to real papers; none fabricated.** Flags mark where the original attribution needed a fix; these corrections are already applied in the text above.

**Generalization & "why detectors fail"**
- Müller, Evans, Tak, Sperl, Böttinger (2024). *Harder or Different? Understanding Generalization of Audio Deepfake Detection.* Interspeech 2024. arXiv:2406.03512. ⚠️ *Distinct from* the 2022 paper *Does Audio Deepfake Detection Generalize?* (arXiv:2203.16263) — don't merge the two.
- Liu, Wang, Sahidullah, et al. (2023). *ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection in the Wild.* IEEE/ACM TASLP. arXiv:2210.02437.
- Wang et al. (incl. Yamagishi) (2025). *ASVspoof 5: Design, Collection and Validation…* arXiv:2502.08857. ⚠️ Cite as **Wang et al.**, not Yamagishi et al. (he is senior co-author).

**Shortcut / spurious-cue learning**
- Müller, Dieckmann, Czempin, et al. (2021). *Speech is Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn?* ASVspoof 2021 Workshop. arXiv:2106.12914. ✓ numbers confirmed (~15.1% EER; RawNet2 3.6%→15.5%).
- Zhang, Li, Lu, Hua, Wang, Zhang (2023). *The Impact of Silence on Speech Anti-Spoofing.* IEEE/ACM TASLP. arXiv:2309.11827.
- Shim, Sahidullah, Jung, Watanabe, Kinnunen (2024). *Beyond Silence: Bias Analysis through Loss and Asymmetric Approach in Audio Anti-Spoofing.* arXiv:2406.17246. ⚠️ supports loss-asymmetry/bona-fide-bias; verify exact "residual-after-trimming" wording against the body.
- Chettri, Benetos, Sturm (2020). *Dataset artefacts in anti-spoofing systems: a case study on the ASVspoof 2017 benchmark.* IEEE/ACM TASLP. arXiv:2010.07913. ⚠️ on ASVspoof **2017**; endpointing-as-confound is the takeaway, paraphrased.

**Vocoder / upsampling artifacts & source attribution**
- Pons, Pascual, Cengarle, Serrà (2021). *Upsampling artifacts in neural audio synthesis.* ICASSP 2021. arXiv:2010.14356.
- Odena, Dumoulin, Olah (2016). *Deconvolution and Checkerboard Artifacts.* Distill. DOI 10.23915/distill.00003.
- Sun, Jia, Hou, AlBadawy, Lyu (2023). *AI-Synthesized Voice Detection Using Neural Vocoder Artifacts.* CVPRW 2023. arXiv:2302.09198.
- Frank & Schönherr (2021). *WaveFake: A Data Set to Facilitate Audio Deepfake Detection.* NeurIPS D&B. arXiv:2111.02813.
- Gasenzer & Wolter (2024). *Towards generalizing deep-audio fake detection networks.* TMLR. arXiv:2305.13033. ⚠️ covers GAN/Avocodo/**BigVGAN** (HF less pronounced, still detectable); does **not** cover diffusion (DiffWave/WaveGrad) — don't attribute the diffusion claim here.
- Yan, Yi, Tao, et al. (2022). *An Initial Investigation for Detecting Vocoder Fingerprints of Fake Audio.* DDAM '22 (ACM-MM). arXiv:2208.09646.
- AlBadawy, Lyu, Farid (2019). *Detecting AI-Synthesized Speech Using Bispectral Analysis.* CVPRW 2019. (phase/bispectral = separate channel.)
- Pizarro, Laszkiewicz, Kolossa, Fischer (2024). *Lightweight Model Attribution and Detection of Synthetic Speech via Audio Residual Fingerprints.* arXiv:2411.14013.
- Frank, Eisenhofer, Schönherr, Fischer, Kolossa, Holz (2020). *Leveraging Frequency Analysis for Deep Fake Image Recognition.* ICML 2020. arXiv:2003.08685.
- Wang, Wang, Zhang, Owens, Efros (2020). *CNN-generated images are surprisingly easy to spot… for now.* CVPR 2020. arXiv:1912.11035.

**Probing / representations**
- Klein, Chen, Tak, Casal, Khoury (2024). *Source Tracing of Audio Deepfake Systems.* Interspeech 2024. arXiv:2407.08016. ⚠️ exact numbers: two-stage frozen = **73.7% vocoder / 91.4% AM**; end-to-end SSL = 84.6% / 99.4%. Don't merge the ranges.
- Hewitt & Liang (2019). *Designing and Interpreting Probes with Control Tasks.* EMNLP-IJCNLP. arXiv:1909.03368.
- Belinkov (2022). *Probing Classifiers: Promises, Shortcomings, and Advances.* Computational Linguistics 48(1). arXiv:2102.12452.
- Pasad, Chou, Livescu (2021). *Layer-wise Analysis of a Self-supervised Speech Representation Model.* ASRU 2021. arXiv:2107.04734.
- Pîrlogeanu, Stan, Cucu (2026). *Understanding the strengths and weaknesses of SSL models for audio deepfake model attribution.* ICASSP 2026. arXiv:2603.13488. (Real despite the 2026 date.)

**Models / augmentation / metrics**
- Tak, Todisco, Wang, Jung, Yamagishi, Evans (2022). *ASV Spoofing and Deepfake Detection Using wav2vec 2.0 and Data Augmentation.* Speaker Odyssey 2022. arXiv:2202.12233.
- Jung, Heo, Tak, et al. (2022). *AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks.* ICASSP 2022. arXiv:2110.01200.
- Kwak, Kwag, Lee, et al. (2022). *Low-quality Fake Audio Detection through Frequency Feature Masking.* DDAM '22. DOI 10.1145/3552466.3556533.
- Chen, Zhang, Zhu, Duan (2021). *UR Channel-Robust Synthetic Speech Detection System for ASVspoof 2021.* arXiv:2107.12018.
- Wang & Yamagishi (2021). *A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection.* Interspeech 2021. arXiv:2103.11326. ⚠️ general CM/front-end/loss study (not "SSL front ends").
- Brümmer, Ferrer, Swart (2021). *Out of a hundred trials, how many errors does your speaker verifier make?* Interspeech 2021. arXiv:2104.00732.
- Shim, Jung, Kinnunen, Evans, Bonastre, Lapidot (2024). *a-DCF: an architecture-agnostic metric…* Odyssey 2024. arXiv:2403.01355. ⚠️ **not** "Lee et al." — fix author attribution.
- Bisani & Ney (2004). *Bootstrap estimates for confidence intervals in ASR performance evaluation.* ICASSP 2004. DOI 10.1109/ICASSP.2004.1326009.
