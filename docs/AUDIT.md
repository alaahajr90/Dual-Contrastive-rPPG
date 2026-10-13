# Manuscript-to-Code Traceability and Author Verification Report

**Status: requires author verification before representing this repository as a reproduction of the published results.** This report compares the uploaded 41-page manuscript with the uploaded UBFC-only source archive. The PURE and COHFACE training scripts and historical model checkpoint provenance were not provided.

| Component | Manuscript / provided reference | Canonical repository behavior | Reproduction risk |
|---|---|---|---|
| Dataset protocol | PURE 10 subjects, COHFACE 40, UBFC 42; subject-exclusive 60/40 dev/test, 5-fold development CV | Enforced grouping and five subject folds | PURE held-out IDs and UBFC archived IDs confirmed; COHFACE held-out IDs unknown |
| UBFC test | 17 subjects, 3,739 clips | Exactly the 17 subject IDs from the provided revised Stage-2 text file | Original memberships retained; clip count requires actual input data |
| UBFC train/val | 20 train / 5 val subjects, 4,611 / 1,179 clips | Frozen membership in `ubfc_archived_final.json` | Stage-1 historical archive contains **different** train/val lists; do not mix them |
| Model input | `[B,3,128,64,64]` | Identical tensor convention | ROI geometry unspecified |
| Encoder | Lightweight 3D CNN | 64-output-channel encoder ported from supplied UBFC Stage 1 | No PURE/COHFACE encoder source supplied |
| Branches | P5/P6/P7/Pout called hierarchical | Four contiguous 16-channel chunks of one final 64-channel encoder output | This is what the supplied historical UBFC code actually does; **not** four distinct encoder layers |
| Projection | Two-layer SSL head | 64 → 512 → 128 with nonlinear activation/dropout | Based on supplied `grid_config_stage1.json`; paper does not state all dimensions |
| NT-Xent | SimCLR positive/negative clip views; temperature τ | Correct **clip-level symmetric NT-Xent** with τ=0.07 | Supplied historical code instead used per-frame embeddings, margin=0.3, a malformed denominator and unreachable code; **loss changed** |
| RoPE | Rotary relative temporal encoding | Applied correctly to Q/K immediately before scaled-dot-product attention | Historical code rotated features before QKV projection; incompatible state dict |
| Memory attention | Four blocks for each P5/P6/P7/Pout, 8 heads | Four per pathway | Supplied historical UBFC script had **six** blocks per pathway |
| Long-range attention | Four extra blocks on Pout | Four extra blocks on Pout | Historical UBFC script had **six**, not four |
| Regression head | 128 → 64 → 32 → 1, GELU, dropout 0.25 | Same linear sizes/activations/dropout | Historical code used dropout 0.1, final Hardtanh ±1.5 and normalized predictions twice |
| Parameter count | **546,532** in manuscript | **588,772** in this refactor; historical provided UBFC model **642,692** | Parameter mismatch **NOT RESOLVED**; architecture/parameter source needed |
| SmoothL1 | α=0.5; β PURE/UBFC=0.5, COHFACE=1.0 | Identical numeric settings | Ground-truth scaling and model normalization need historical verification |
| FFT loss | PURE normalized magnitude; UBFC log magnitude; COHFACE 0.7–4 Hz normalized log spectrum | Three differentiable dataset-specific implementations | Exact historical preprocessing and FFT normalization details not fully disclosed |
| Loss weights | PURE FFT=1.0; UBFC FFT=1.5/stat=0.5; COHFACE FFT=2.5/smooth=0.1 | Dataset YAML values match manuscript Table 3 | COHFACE smoothing formula and PURE exact FFT source unavailable |
| UBFC signed-skew filter | Threshold 1.15 in manuscript; training-only | 1.15, training-only; val/test untouched | Uploaded UBFC `main_stage2.py` also calls a threshold of **1.25**; author must identify experimental version |
| HR estimation | 128 Welch segment, nfft=256, Hann, 50% overlap, bandpass, parabolic peak | Same signal estimation settings | Historical UBFC code had alternate `extract_hr` with `nfft=1024`; paper and older code differ |
| Final Stage-2 optimizer | Adam with batch=8 | Adam at 1e-4, batch=8 | Learning rate comes from UBFC source; cannot assert identical PURE/COHFACE configuration |
| Checkpoint selection | Validation-loss-only early stop | Same | Historical best weights intentionally omitted |
| Bootstrap | 10,000 subject-level resamples | Same, retaining all subject sessions together | Must use archived held-out session predictions to match Table 6 |
| Multiple seeds | 42/123/2026 | Same, fixed Stage-1 encoder | Actual original seed-specific weights are not provided |
| PURE motion | Six conditions, held-out IDs 02/05/09/10, 24 sessions | Six-condition report via session predictions | Manuscript motion paragraph describes overlap-add waveform reconstruction, not main clip-HR averaging; exact overlap-add code unavailable |
| PURE occlusion | Mask 0,25,50,75,100% ROI and mean-fill at inference | Same five levels and per-channel time/space mean fill | Shape of exact paper mask needs author confirmation |
| Zero-shot cross-dataset | Frozen PURE model on held-out UBFC and COHFACE | No target adaptation or fit | Reference scores not regenerated without weights/data |
| Stage-1 embedding figures | t-SNE, KMeans k=6, cosine distributions | Code generates both from actual saved Stage-1 encoder | Historical figure embeddings / epoch 1 snapshots were not supplied |
| Computation | Manuscript 0.0247 GMAC, 0.0494 GFLOP, 20.06 ms | Profiles current model forward and reports measured parameters/time | No claim of matching hardware/runtime, and parameter count differs |

## Manuscript values vs current experimental outputs

`reproducibility/reference_results.json` stores Table 5, Table 6, PURE motion/occlusion, cross-dataset and ablation values as **static, clearly attributed references**. A run outputs its own `metrics.json` and `session_estimates.csv`. The code never writes fabricated metrics or treats reference values as a measured run.

The paper's original Table 5 MAE values (PURE 0.18, COHFACE 2.36, UBFC 0.55 bpm) are **different experiments/summaries** from Table 6's subject-disjoint fixed-model bootstrap point estimates (PURE 2.518, COHFACE 3.556, UBFC 0.809 bpm). They must not be declared equivalent or merged into a single performance claim without experiment provenance and identifiers.

The paper's reported ablation study has **missing** configurations on some datasets; those missing scores are preserved as absent rather than invented. `rppg ablate` provides code paths for running the ablations, not invented reproductions of their Table 10 values.

## Blocking tasks before exact reproduction claim

1. Supply the final trained and tested **PURE + COHFACE** source implementations or original Git revision history, including exact preprocessing ROI geometry, the source of the 546,532-parameter model, and the final wave normalization and FFT/smoothness definitions.
2. Supply the COHFACE **archived subject IDs**, especially the 16 held-out participants. A new random 24/16 split is reproducible but does not establish the original reported experimental result.
3. Confirm whether Stage 1 used per-frame contrastive logits with negative margin, or clip-level standard NT-Xent, and whether the Stage-2 original used four or six attention blocks. Choose the **actual reported run**, then align code and manuscript.
4. Confirm the PURE/COHFACE learning rates, scheduler, early-stopping details, Stage-1 and Stage-2 random seeds, and checkpoint selection metadata. Exact supplied Stage-2 epochs differ from best epoch due to early stopping.
5. Obtain/verify all raw dataset access rights and clinical/biometric handling permissions; do not distribute third-party face footage in the code repository.
6. Rerun five-fold development cross-validation and the frozen held-out evaluations, record exact package/CUDA versions and hardware, compare result files to Tables 4–10, and add resulting command logs and commit hash to a release report.

This checklist is deliberately conservative: matching settings is necessary but does **not** establish that the supplied source exactly generated the manuscript tables.
