# Why Audio Anti-Spoofing Detectors Fail to Generalize: A Representational Study

_This is the **design** document: the falsification-driven framing, the pre-registered hypotheses, the protocol and controls, and the literature behind them (every cited claim verified, see §7). It states what the study set out to test and why that is rigorous._

_For what was actually run and the measured results, read [report.md](report.md). This document is the plan and its grounding; report.md is the outcome. Where the two diverge, report.md is authoritative._

---

## 1. Main idea

**H1 (predictive).** Take a frozen SSL embedding and ask, for a held-out spoofing family, how strongly it linearly encodes that family's generator identity. The claim is that this degree of encoding predicts how badly a detector built on the embedding fails to generalize to that family.

H1 is falsified if the across-family rank correlation between *probe-recovers-generator* (selectivity) and *leave-one-attack-out (LOAO) generalization failure* is null or negative.

The original framing was dropped for the following reason. The initial framing pitted "learns synthesis-*process* artifacts" against "memorizes generator-specific *fingerprints*," which is a false dichotomy, because a vocoder fingerprint is itself a process artifact. The open question in the literature is instead about transferable vs. non-transferable cues (Müller et al. 2024, *Harder or Different?*). Do detectors key on attack-agnostic forensic cues, the upsampling/vocoder-artifact class that is shared across generators and therefore transfers? Or on family-specific cues that don't transfer? "Generator identity recoverable by a linear probe" is the operational marker of the non-transferable end. It is not proof of an identity fingerprint. Stated accordingly, probe-recoverable family identity is a predictor of non-transfer, not evidence of memorization.

The motivating premise has two parts. Part 1, that detectors collapse on unseen attacks, is settled consensus and is stated without hedging. Part 2, why they collapse, is open: Müller et al. 2024 state plainly that a comprehensive analysis of the underlying challenges is lacking. That gap is the target.

---

## 2. Adversarial verification

Each premise below was checked against the literature. Where verification weakened it, the claim is restated or dropped.

| Original premise | Verdict | Restated for the design |
|---|---|---|
| Silence alone is near-SOTA | Overstated | Silence-duration alone reaches ~15% EER on 2019/2021 LA, far from sub-1% SOTA, but a real shortcut that inflates strong models (RawNet2 3.6% to 15.5% EER when trimmed) [Müller 2021]. Silence also carries genuine TTS/VC cues [Zhang 2023], so trimming is itself confounded; report trimmed and untrimmed. |
| Vocoder = "checkerboard," CV intuition transfers | Supported, scoped | The shared mechanism is the upsampling operator [Pons 2021 ↔ Odena 2016], but the signature differs: images show 2D spatial grids, audio shows tonal/horizontal-line artifacts plus HF spectral replicas rather than a literal 2D checkerboard. Detection transfer is real [Sun 2023; Frank & Schönherr 2021]. Scope caveat: anti-aliased vocoders (BigVGAN) make the HF footprint less pronounced though still detectable [Gasenzer & Wolter 2024]; the diffusion-vocoder case is not covered by that reference, so do not claim it without a diffusion-specific source. Phase/bispectral cues [AlBadawy 2019] are a separate channel; do not conflate with magnitude/upsampling. |
| Linear probe recovers *which* generator ⇒ fingerprinting | Partially supported | On a binary-fine-tuned-then-frozen SSL CM embedding, a light head recovers vocoder identity 73.7% and acoustic-model identity 91.4% (two-stage regime); an end-to-end SSL probe reaches 84.6% / 99.4% [Klein 2024, *Source Tracing of Audio Deepfake Systems*]. Well above chance, uneven across attribute type, degrades on unseen generators. Design consequences: (i) test both frozen regimes, off-the-shelf XLS-R and binary-fine-tuned-then-frozen; (ii) recovery is consistent with keying on per-generator artifacts, but is not proof a fingerprint drives detection. |
| EER noisy across seeds ⇒ multi-seed needed | Supported | RawGAT-ST/AASIST spanned 1.19–2.06% EER across seeds [Jung 2022]; multi-round EER reporting is the documented norm [Wang & Yamagishi 2021]. Seed variance and eval-set sampling CI are distinct sources; report both. |

---

## 3. Experimental protocol

### 3.1 Backbone (frozen, cache-once)

The front end is wav2vec2-XLS-R [Tak 2022]. Cache per-layer embeddings once to disk. Use two frozen regimes, stated explicitly:

- (A) off-the-shelf pretrained XLS-R, truly frozen. This asks: does the SSL prior already encode this?
- (B) XLS-R fine-tuned on the binary spoof task, then frozen, the Klein-2024 regime. This asks: did the detector learn it?

Probe and detector heads are light (linear / 1-layer). All conditions share the identical front end and feature cache, so the design varies one factor at a time. Front-end choice alone moved EER ~37% relative in prior work, the most-criticized confound.

### 3.2 LOAO matrix

ASVspoof 2021 LA reuses 2019 attacks A07–A19. Group them by waveform-generation attribute, the most attributable axis [Klein 2024], rather than by system name. Use the verified mapping in [data/attack_taxonomy.json](data/attack_taxonomy.json):

- neural_tts: A07, A08, A09, A10, A11, A12
- concatenative_tts: A16 (single system, interpret with care)
- voice_conversion: A17, A18, A19
- hybrid_tts_vc: A13, A14, A15

For each family *f*: train on bona fide plus all spoof families except *f*, then evaluate on held-out *f*.

Report the full per-family grid alongside the pooled number. Pooled-only hides the generalization signal and is gameable by easy attacks [Liu 2023]. A large pooled-vs-averaged gap flags incompatible per-attack score distributions.

### 3.3 Shortcut ablations

1. Silence-only ceiling: train a classifier on leading/trailing-silence duration alone. Reproduce ~15% EER to show the shortcut exists in this pipeline, then show it is removed [Müller 2021].
2. VAD trim: use identical endpointing at train and test [Chettri 2020, on ASVspoof 2017]; report trimmed and untrimmed. Add mask-silence vs. mask-non-silence to dissociate duration-proportion from silence-content cues [Zhang 2023].
3. Duration / energy / peak-amplitude equalization: the confounds the ASVspoof 5 organizers suppress [Wang et al. 2025, §4.2].
4. Channel/codec: 2021 LA mixes telephony codecs, which are label-correlated. Balance or randomize codec across classes, or augment with an acoustic simulator [Chen 2021]. Then verify the HF artifact survives the codec, since band-limiting can erase the very signal under study.
5. Residual bias check: bona fide stays "easier" even after trimming [Shim 2024]. Report loss asymmetry, so that no single ablation is claimed to "close" shortcuts.

### 3.4 Generator-identity probing

Probe the frozen embedding for *family identity* among the training families. The held-out family *f* is the generalization target and is never a probe class.

- Linear probes only, with control tasks and selectivity, where selectivity = (target acc − random-label-control acc) [Hewitt & Liang 2019; Belinkov 2022]. Raw accuracy alone is not evidence.
- Layer-wise profile, not a single layer. Artifacts may sit in early CNN or low transformer layers while the head reads a weighted sum [Pasad 2021].
- Baselines: chance/majority, random-feature, untrained/shuffled-model.
- Confound controls: run the same probe on silence-trimmed, energy/duration-equalized, and channel-randomized inputs, and within fixed speaker/content. The generator signal must survive, or it was codec/speaker/silence rather than synthesis. Pîrlogeanu 2026 warns that attribution collapses under vocoder/prompt mismatch.
- Probe-capacity sanity: use ≥2 probe strengths, so a null isn't a weak-probe artifact.
- Cheap CV-analog baseline: a standardized average-residual fingerprint (signal minus its low-pass/EnCodec-filtered version) to compare against the SSL probe [Pizarro 2024].

### 3.5 The H1 correlation (core test)

For each held-out family *f*, define two quantities:

- x = probe selectivity for *f*'s identity, taken from a probe where *f* is in-distribution, i.e. how separable it is.
- y = detector LOAO failure on *f*, measured as EER_LOAO − in-domain EER.

Test the Spearman rank correlation between x and y. With n ≈ 4 families (or ≈ 13 systems), this calls for rank statistics, CIs, a pre-registered directional prediction, and no over-reading of point estimates. Report at both family granularity (n≈4) and system granularity (n≈13).

---

## 4. Contribution to literature

| Prior thread | Already established | Did not do |
|---|---|---|
| Klein 2024; Yan 2022; Pizarro 2024; Sun 2023 | Generator/vocoder identity is recoverable from (SSL) embeddings | Never correlated recoverability with detector LOAO failure; used it as a training signal or standalone attribution |
| Müller 2024 (*Harder or Different?*) | The gap is "difference," not "hardness," at score/dataset level | No representational localization; no frozen-embedding probe of the detection representation |
| Frank 2020; Pons 2021; Wang 2020 (vision) | Upsampling → spectral fingerprints; single-source + aug generalizes | The CV→audio fingerprint framing on a frozen SSL backbone for ASVspoof LOAO is undrawn in the audio literature |

The synthesis contribution is the combination of three pieces:

1. selectivity-controlled probing of the *frozen detection embedding*;
2. the cross-family correlation "family-identity recoverability ↔ LOAO non-transfer" as a falsifiable test;
3. the scope-limited CV-fingerprint framing.

Every ingredient already exists; the novelty is the predictive integration, not any single component.

---

## 5. Statistical rigor

- Two variance sources, both reported and labeled:
  - seed variance: retrain the light head ≥5 seeds (cheap, since the front end is frozen), report mean ± std [Jung 2022; Wang & Yamagishi 2021];
  - eval-set sampling CI: utterance-level bootstrap (~1000 reps) [Bisani & Ney 2004], using a paired bootstrap when comparing two conditions on the same trials, not two independent CIs.
- Metrics: report pooled and per-family EER. Add min t-DCF for 2021 LA comparability [Liu 2023]. State the operating point and priors, since EER is calibration-blind [Brümmer 2021]. For any ASVspoof 5 extension, use min a-DCF [Shim/Jung et al. 2024, *a-DCF*].
- Correlation: Spearman plus CI, with the small-n caveat foregrounded and the direction pre-registered.
- No best-of-N: report the full distribution, never the best run.

---

## 6. Implementation status

This design is implemented in the repository. The leave-one-attack-out transfer study, the generator-identity probing (the H1 selectivity test), and the boundary-geometry analysis (H2) were all run on real ASVspoof embeddings; those results are in [report.md](report.md). The degradation pipeline of §3.3 also serves as Part 1's robustness study.

The one part of §3.4 not yet run on real data is the confound controls (codec/speaker), which await the protocol metadata key.

report.md also goes beyond this design. Its boundary-geometry and conformal-coverage analyses became the causal and predictive handle the study delivers: they localize non-transfer to bona-fide proximity, and show where a calibrated coverage guarantee breaks under attack shift.

---

## 7. References

All references were verified against arXiv or the publishing venue, including authors, year, venue, and that each paper supports the claim it is cited for.

**Generalization & "why detectors fail"**
- Müller, Evans, Tak, Sperl, Böttinger (2024). *Harder or Different? Understanding Generalization of Audio Deepfake Detection.* Interspeech 2024. arXiv:2406.03512. (Distinct from the 2022 paper *Does Audio Deepfake Detection Generalize?*, arXiv:2203.16263.)
- Liu, Wang, Sahidullah, et al. (2023). *ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection in the Wild.* IEEE/ACM TASLP. arXiv:2210.02437.
- Wang et al. (incl. Yamagishi) (2025). *ASVspoof 5: Design, Collection and Validation…* arXiv:2502.08857.

**Shortcut / spurious-cue learning**
- Müller, Dieckmann, Czempin, et al. (2021). *Speech is Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn?* ASVspoof 2021 Workshop. arXiv:2106.12914.
- Zhang, Li, Lu, Hua, Wang, Zhang (2023). *The Impact of Silence on Speech Anti-Spoofing.* IEEE/ACM TASLP. arXiv:2309.11827.
- Shim, Sahidullah, Jung, Watanabe, Kinnunen (2024). *Beyond Silence: Bias Analysis through Loss and Asymmetric Approach in Audio Anti-Spoofing.* arXiv:2406.17246.
- Chettri, Benetos, Sturm (2020). *Dataset artefacts in anti-spoofing systems: a case study on the ASVspoof 2017 benchmark.* IEEE/ACM TASLP. arXiv:2010.07913.

**Vocoder / upsampling artifacts & source attribution**
- Pons, Pascual, Cengarle, Serrà (2021). *Upsampling artifacts in neural audio synthesis.* ICASSP 2021. arXiv:2010.14356.
- Odena, Dumoulin, Olah (2016). *Deconvolution and Checkerboard Artifacts.* Distill. DOI 10.23915/distill.00003.
- Sun, Jia, Hou, AlBadawy, Lyu (2023). *AI-Synthesized Voice Detection Using Neural Vocoder Artifacts.* CVPRW 2023. arXiv:2302.09198.
- Frank & Schönherr (2021). *WaveFake: A Data Set to Facilitate Audio Deepfake Detection.* NeurIPS D&B. arXiv:2111.02813.
- Gasenzer & Wolter (2024). *Towards generalizing deep-audio fake detection networks.* TMLR. arXiv:2305.13033. (Covers GAN vocoders, Avocodo, and BigVGAN, not diffusion vocoders.)
- Yan, Yi, Tao, et al. (2022). *An Initial Investigation for Detecting Vocoder Fingerprints of Fake Audio.* DDAM '22 (ACM-MM). arXiv:2208.09646.
- AlBadawy, Lyu, Farid (2019). *Detecting AI-Synthesized Speech Using Bispectral Analysis.* CVPRW 2019. (phase/bispectral = separate channel.)
- Pizarro, Laszkiewicz, Kolossa, Fischer (2024). *Lightweight Model Attribution and Detection of Synthetic Speech via Audio Residual Fingerprints.* arXiv:2411.14013.
- Frank, Eisenhofer, Schönherr, Fischer, Kolossa, Holz (2020). *Leveraging Frequency Analysis for Deep Fake Image Recognition.* ICML 2020. arXiv:2003.08685.
- Wang, Wang, Zhang, Owens, Efros (2020). *CNN-generated images are surprisingly easy to spot… for now.* CVPR 2020. arXiv:1912.11035.

**Probing / representations**
- Klein, Chen, Tak, Casal, Khoury (2024). *Source Tracing of Audio Deepfake Systems.* Interspeech 2024. arXiv:2407.08016.
- Hewitt & Liang (2019). *Designing and Interpreting Probes with Control Tasks.* EMNLP-IJCNLP. arXiv:1909.03368.
- Belinkov (2022). *Probing Classifiers: Promises, Shortcomings, and Advances.* Computational Linguistics 48(1). arXiv:2102.12452.
- Pasad, Chou, Livescu (2021). *Layer-wise Analysis of a Self-supervised Speech Representation Model.* ASRU 2021. arXiv:2107.04734.
- Pîrlogeanu, Stan, Cucu (2026). *Understanding the strengths and weaknesses of SSL models for audio deepfake model attribution.* ICASSP 2026. arXiv:2603.13488.

**Models / augmentation / metrics**
- Tak, Todisco, Wang, Jung, Yamagishi, Evans (2022). *ASV Spoofing and Deepfake Detection Using wav2vec 2.0 and Data Augmentation.* Speaker Odyssey 2022. arXiv:2202.12233.
- Jung, Heo, Tak, et al. (2022). *AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks.* ICASSP 2022. arXiv:2110.01200.
- Chen, Zhang, Zhu, Duan (2021). *UR Channel-Robust Synthetic Speech Detection System for ASVspoof 2021.* arXiv:2107.12018.
- Wang & Yamagishi (2021). *A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection.* Interspeech 2021. arXiv:2103.11326.
- Brümmer, Ferrer, Swart (2021). *Out of a hundred trials, how many errors does your speaker verifier make?* Interspeech 2021. arXiv:2104.00732.
- Shim, Jung, Kinnunen, Evans, Bonastre, Lapidot (2024). *a-DCF: an architecture-agnostic metric…* Odyssey 2024. arXiv:2403.01355.
- Bisani & Ney (2004). *Bootstrap estimates for confidence intervals in ASR performance evaluation.* ICASSP 2004. DOI 10.1109/ICASSP.2004.1326009.
