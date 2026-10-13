# Dual Contrastive Learning with Attention for Contactless rPPG Heart-Rate Estimation

**Research-code companion to:** *Dual Contrastive Learning with Attention Mechanism for Contactless rPPG-based Heart Rate Estimation via Facial Videos* (manuscript under journal submission).

[![Research tests](https://img.shields.io/badge/tests-pytest-informational)](#verification) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)

This repository implements a unified, **two-stage**, video-based remote photoplethysmography (rPPG) workflow for PURE, COHFACE, and UBFC-rPPG. It contains preprocessing adapters, subject-exclusive partitioning, Stage-1 contrastive pretraining, Stage-2 supervised regression, waveform-to-HR evaluation, uncertainty analysis, and controlled experiments. Code and technical comments are in English.

> **Reproduction status: architecture/protocol-aligned implementation, not a verified bit-for-bit reproduction of all published numbers.** Only the UBFC source scripts, the manuscript, and the UBFC partition lists were supplied for this refactor. The complete PURE/COHFACE source implementations, raw data, historical final model weights, and training provenance were not supplied. There are documented inconsistencies between the source archive and manuscript; see [`docs/AUDIT.md`](docs/AUDIT.md) before citing experimental reproduction. Manuscript results stored in `reproducibility/reference_results.json` are explicitly **reference values**, not claimed rerun outputs.

## Repository layout

```text
dual_contrastive_rppg_github/
├── configs/
│   ├── pure.yaml                  # PURE acquisition + waveform/spectral settings
│   ├── cohface.yaml               # COHFACE acquisition + waveform/spectral settings
│   └── ubfc.yaml                  # UBFC-rPPG acquisition + waveform/spectral settings
├── src/rppg/
│   ├── config.py                  # Shared hyperparameters and deterministic seeding
│   ├── splits.py                  # 60/40 subject splits, frozen split, 5-fold CV
│   ├── preprocessing/
│   │   ├── readers.py             # Dataset-specific video and BVP acquisition
│   │   ├── roi.py                 # Dlib 68-point cheek/forehead ROI extraction
│   │   ├── manifest.py            # Frame/BVP synchronization + 128/8 clip generation
│   │   ├── augmentation.py        # Clip-consistent contrastive views
│   │   └── dataset.py             # Single paired NumPy dataloader for all experiments
│   ├── stage1/
│   │   ├── model.py               # Shared 3D-CNN encoder and two-layer projection head
│   │   └── losses.py              # Symmetric NT-Xent
│   ├── stage2/
│   │   ├── attention.py           # RoPE query/key memory attention
│   │   ├── model.py               # P5/P6/P7/Pout waveform heads
│   │   └── losses.py              # Dataset-specific spectral + multiscale losses
│   ├── evaluation/
│   │   ├── hr.py                  # Welch PSD -> clip HR -> session HR -> MAE/RMSE/r
│   │   ├── statistics.py          # Cluster bootstrap and multiple-seed mean ± SD
│   │   ├── plots.py               # Correlation and Bland–Altman (generated PDF)
│   │   └── waveforms.py           # Paired waveform and spectral visualizations
│   ├── experiments/
│   │   ├── ablations.py           # no SSL, no RoPE, no memory, no FFT, etc.
│   │   ├── ssl_analysis.py        # Embedding, KMeans and similarity diagnostics
│   │   ├── ssl_figures.py         # Stage-1 t-SNE and cosine distribution plots
│   │   ├── motion.py              # PURE condition-wise metrics
│   │   ├── robustness.py          # Synthetic central occlusion at inference
│   │   └── profiling.py           # Model-only profiling
│   ├── training.py                # Central training/validation/evaluation loops
│   └── cli.py                     # Public command-line entry point
├── reproducibility/
│   ├── splits/ubfc_archived_final.json  # Original provided UBFC train/val/test subjects
│   └── reference_results.json          # Manuscript reference data (not rerun results)
├── tests/                         # Deterministic unit, model and split tests
├── docs/
│   ├── DATA_LAYOUT.md             # Input naming, synchronization and ROI assumptions
│   ├── AUDIT.md                   # Differences requiring author verification
│   └── JOURNAL_CHECKLIST.md       # Source release and data-statement checklist
└── .github/workflows/tests.yml    # GitHub Actions without private biometric files
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e '.[preprocess,test]'
python -m pytest -q
rppg --help
```

Use a CUDA-compatible PyTorch installation for GPU training. Dlib is an optional preprocessing requirement (download its predictor yourself; **do not commit model data files**). Commands can also be run as `python -m rppg` after installation.

## Raw datasets and preprocessing

Refer to [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md) for exact PURE JSON/PNG, COHFACE HDF5/AVI, and UBFC text/AVI arrangements. The processed clip representation is identical across the three datasets:

- `session_clipN.npy`: RGB ROI video with shape `(128, 64, 64, 3)` and dtype `uint8`.
- `session_clipN_bvp.npy`: frame-aligned reference waveform `(128,)`, dtype `float32`.
- `manifest.csv`: dataset, subject, session, clip ID, ROI file, BVP file, FPS, original start frame.

**Important:** The manuscript specifies forehead+cheek ROIs and Dlib 68 landmarks but does not supply exact polygon vertices. `preprocessing/roi.py` is a documented explicit approximation; verify it against the original extraction pipeline before claiming numerical reproduction.

```bash
rppg preprocess --config configs/ubfc.yaml --raw-root data/raw \
  --predictor models/shape_predictor_68_face_landmarks.dat

rppg preprocess --config configs/pure.yaml --raw-root data/raw \
  --predictor models/shape_predictor_68_face_landmarks.dat

rppg preprocess --config configs/cohface.yaml --raw-root data/raw \
  --predictor models/shape_predictor_68_face_landmarks.dat
```

### Frozen subject partitioning (critical)

**All partitions are subject-exclusive, with 60% development and 40% held-out test.** Five-fold cross-validation operates **only on development subjects**, with no overlap of subject clips across folds. Even though adjacent clips overlap by 120/128 frames, split grouping prevents train/test leakage.

**UBFC archive:** The supplied source archive fixes 20 train + 5 validation + 17 held-out subjects (4,611 / 1,179 / 3,739 ROI clips). Its 25 development subjects plus 17 held-out subjects correspond to 25/42 = 59.52% and 17/42 = 40.48% due to integer rounding. Use the archived membership, not an independently generated random split:

```bash
rppg split --config configs/ubfc.yaml \
  --manifest data/processed/ubfc/manifest.csv \
  --frozen reproducibility/splits/ubfc_archived_final.json \
  --write data/processed/ubfc/split.json
```

**PURE:** The manuscript explicitly names the four unseen subjects as `02, 05, 09, 10`; six other subjects form the development partition.

```bash
rppg split --config configs/pure.yaml \
  --manifest data/processed/pure/manifest.csv \
  --write data/processed/pure/split.json
```

**COHFACE:** The manuscript specifies 24 development and 16 test subjects but does not identify which ones; the default command produces a **reproducible new partition**, not a purported recovery of the archived original. To reproduce published numbers, provide the **actual archived subject identities** using `--frozen`.

```bash
rppg split --config configs/cohface.yaml \
  --manifest data/processed/cohface/manifest.csv \
  --seed 42 --write data/processed/cohface/split.json
```

## Automatic Stage 1 -> Stage 2 execution

Run with one dataset and its corresponding processed manifest and frozen split. The `pipeline` command trains Stage 1 first, passes its encoder weights to Stage 2, selects the best Stage-2 checkpoint **on validation loss only**, and optionally evaluates the held-out test **after training**. The checkpoint files are created under the ignored `outputs/` directory, never distributed through GitHub.

```bash
rppg pipeline --config configs/ubfc.yaml \
  --manifest data/processed/ubfc/manifest.csv \
  --split data/processed/ubfc/split.json \
  --output outputs --test
```

Use equivalent `pure.yaml`/`cohface.yaml` and paths for the other datasets. For a development smoke run only, use `--epochs 1` (not for reporting paper metrics).

### Independent Stage commands

```bash
rppg stage1 --config configs/pure.yaml --manifest data/processed/pure/manifest.csv \
  --split data/processed/pure/split.json

rppg stage2 --config configs/pure.yaml --manifest data/processed/pure/manifest.csv \
  --split data/processed/pure/split.json \
  --encoder outputs/pure/stage1/encoder_best.pth

rppg evaluate --config configs/pure.yaml --manifest data/processed/pure/manifest.csv \
  --split data/processed/pure/split.json --partition test \
  --checkpoint outputs/pure/stage2/seed_42/best_model.pth
```

## Research analyses

All commands accept the corresponding dataset config, manifest, subject split, and output path except `bootstrap` (which consumes session predictions). Examples below show PURE; for UBFC/COHFACE replace the config and file paths.

```bash
# Five development folds, redoing Stage 1 independently for each fold:
rppg cv --config configs/pure.yaml --manifest data/processed/pure/manifest.csv \
  --split data/processed/pure/split.json --stage 2

# Stage-2 variability, keeping the Stage-1 model fixed:
rppg seeds --config configs/pure.yaml --manifest data/processed/pure/manifest.csv \
  --split data/processed/pure/split.json --encoder outputs/pure/stage1/encoder_best.pth \
  --values 42 123 2026

# Bootstrap subject clusters (10,000 independent subject draws):
rppg bootstrap --session-csv outputs/pure/evaluation/test/session_estimates.csv \
  --repetitions 10000 --seed 42 --output outputs/pure/bootstrap.json

# No-SSL, no-RoPE, long-range attention, memory and spectral variants:
rppg ablate --config configs/pure.yaml --manifest data/processed/pure/manifest.csv \
  --split data/processed/pure/split.json --encoder outputs/pure/stage1/encoder_best.pth \
  --variants no_ssl no_rope no_long_range no_memory no_attention no_fft

# Vary independent FFT loss weights:
rppg fft-sweep --config configs/pure.yaml --manifest data/processed/pure/manifest.csv \
  --split data/processed/pure/split.json --encoder outputs/pure/stage1/encoder_best.pth \
  --weights 0 0.5 1.0 2.5 3.0

# PURE motion conditions and mean-filled inference-only ROI occlusion:
rppg motion --config configs/pure.yaml --manifest data/processed/pure/manifest.csv \
  --split data/processed/pure/split.json --checkpoint outputs/pure/stage2/seed_42/best_model.pth
rppg occlusion --config configs/pure.yaml --manifest data/processed/pure/manifest.csv \
  --split data/processed/pure/split.json --checkpoint outputs/pure/stage2/seed_42/best_model.pth \
  --fractions 0 0.25 0.5 0.75 1

# Stage-1 embedding and contrastive-pair figures:
rppg ssl-plots --config configs/pure.yaml --manifest data/processed/pure/manifest.csv \
  --split data/processed/pure/split.json --encoder outputs/pure/stage1/encoder_best.pth

# Figure 8-style predicted-vs-reference waveform and PSD pairs:
rppg waveforms --config configs/pure.yaml --manifest data/processed/pure/manifest.csv \
  --split data/processed/pure/split.json \
  --checkpoint outputs/pure/stage2/seed_42/best_model.pth --examples 2

# Zero-shot transfer: evaluate a frozen PURE-trained final Stage-2 model on UBFC test:
rppg cross-dataset --config configs/ubfc.yaml --manifest data/processed/ubfc/manifest.csv \
  --split data/processed/ubfc/split.json --source-dataset pure \
  --checkpoint outputs/pure/stage2/seed_42/best_model.pth

# Actual model-only runtime/parameter profile for the current architecture:
rppg profile --config configs/pure.yaml --iterations 300
```

**Evaluation aggregation:** each 128-frame waveform produces one Welch-based HR estimate (128-sample segment, 256-point FFT, default 50% overlap, in-band peak with interpolation). Clip HRs are averaged into recording/session HR, and only then are MAE, RMSE and Pearson r computed. This avoids treating heavily overlapping clips as independent samples. No ground-truth quality filtering is applied to validation or test. For the manuscript-specific PURE *motion* experiment, its phrase “reconstructed waveform from overlapping clips” differs from this main evaluator; see [`docs/AUDIT.md`](docs/AUDIT.md).

## Verification

```bash
python -m pytest -q
python -m compileall -q src
```

Tests check archived UBFC membership, PURE and COHFACE subject counts, no subject intersection, five-fold coverage, differentiable FFT losses, RoPE, waveform heads, Welch HR, bootstrap, video augmentations, and mean-filled occlusion. This does **not** replace the full GPU/raw-dataset reproduction necessary to validate manuscript results.

## Data, artifacts and responsible release

- Raw facial recordings, frame landmarks, paired BVP, private paths, generated arrays, trained weights and run outputs are omitted from version control via `.gitignore`.
- Original source archive was used **only as an audit input**; historical duplicate/defective scripts are not published alongside the canonical modules.
- A separate benchmark is needed to verify the manuscript's **546,532 parameters** and runtime claim; this implementation presently has **588,772 parameters** with manuscript's described 4+4 attention settings. The existing UBFC historical Stage-2 Python code used 6 blocks and had **642,692 parameters**. These are **not identical models**. Do not silently present new-model performance as historical published numbers.
- Reference benchmark numbers are kept in a separate data file and must not be substituted for measured outputs.
- Public dataset licenses, author code ownership, and the final public software license should be confirmed before distribution.

## Citation

Cite the accompanying manuscript once bibliographic metadata/DOI is finalized. The repository includes a draft `CITATION.cff` without an invented publication DOI.
