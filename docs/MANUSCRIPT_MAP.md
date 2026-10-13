# Manuscript-to-Implementation Component Map

| Manuscript section | Canonical code / artifact | Notes |
|---|---|---|
| 3.1 Benchmark datasets | `preprocessing/readers.py`, `configs/*.yaml` | PURE (PNG+JSON), COHFACE (AVI+HDF5), UBFC-rPPG (AVI+TXT) |
| 3.2 Dlib/ROI | `preprocessing/roi.py` | Cheeks + forehead, exclude eyes/nose/mouth, 64×64 |
| 3.2.1 Temporal clip slicing | `preprocessing/manifest.py` | 128 frames, stride 8, frame-aligned reference |
| 3.2.2 Dataset handling | `preprocessing/readers.py` | Separate converters but shared output contract |
| 3.2.3 NumPy storage | `preprocessing/manifest.py`, `preprocessing/dataset.py` | Identical file and manifest schema for all datasets |
| 3.3 Spatial/temporal perturbations | `preprocessing/augmentation.py` | Rotation, translation, crop/zoom, flip, color jitter, blur, JPEG, temporal reordering |
| 3.4 Encoder/projection | `stage1/model.py` | 3D CNN and two-layer projection |
| 3.5 Contrastive NT-Xent | `stage1/losses.py` | Correct symmetric clip-level implementation; mismatch audited |
| 3.6 Memory attention / RoPE / long-range | `stage2/attention.py`, `stage2/model.py` | Four branches, four standard + four Pout long-range blocks |
| 3.7 SmoothL1/FFT/auxiliary | `stage2/losses.py`, `configs/*.yaml` | β, α, FFT weight and optional stat/smooth terms |
| 3.7.4 Session HR metrics | `evaluation/hr.py` | Welch PSD, parabolic interpolation, MAE/RMSE/r |
| 3.8 Implementation | `training.py`, `config.py` | Adam, early stopping, data quality training-only |
| 3.9 Complexity/runtime | `experiments/profiling.py`, `rppg profile` | Current network only; parameters mismatch paper |
| 4.1.1 60/40 split and five-fold CV | `splits.py`, `rppg split`, `rppg cv` | Subject-exclusive folds, archived UBFC membership |
| 4.1.2–4.1.4 Embedding/similarity figures | `experiments/ssl_figures.py`, `rppg ssl-plots` | Recomputed training/test projections using current encoder |
| 4.2.1 HR results | `rppg evaluate`, `evaluation/plots.py` | Actual evaluation outputs vs static reference values |
| 4.2.2 Subject bootstrap and multiple seeds | `evaluation/statistics.py`, `rppg seeds`, `rppg bootstrap` | 10,000 subject-cluster draws; 42,123,2026 seeds |
| 4.2.3 Motion/occlusion | `experiments/motion.py`, `experiments/robustness.py` | Inference-time occlusion, six PURE conditions |
| 4.2.4 Zero-shot transfer | `rppg cross-dataset` | Frozen source model, target held-out partition only |
| 4.2.5 Ablation/FFT sensitivity | `experiments/ablations.py`, `rppg ablate`, `rppg fft-sweep` | No SSL/RoPE/memory/long-range/FFT variants |
| Figures 8–9 Prediction/BA/correlation | `evaluation/plots.py`, `evaluation/waveforms.py`, `rppg waveforms` | Session-level BA/correlation plus held-out waveform/FFT paired figures |
| Table 11 published comparator methods | `reproducibility/reference_results.json` (study-only context) | Comparator implementations were not supplied and are not falsely reimplemented |

No historical figure, benchmark score, or cited external method is automatically reproduced solely by storing the manuscript's reported numbers. The experiment commands must be executed on the actual datasets for new values to be reported.
