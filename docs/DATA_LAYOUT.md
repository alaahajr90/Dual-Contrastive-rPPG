# Data Layout and Synchronization Contract

Provide public benchmark data locally. The repository does **not** redistribute raw biometric videos, reference BVP, Dlib weights, or checkpoints.

## PURE

```text
data/raw/pure/
├── 01-01/
│   ├── 1234567890.000.png
│   ├── 1234567890.033.png
│   └── ...
├── 01-01.json
├── 01-02/
├── 01-02.json
└── ...
```

The PURE adapter sorts PNGs by numeric frame timestamp and reads JSON `FullPackage` records with `Timestamp` and `Value.waveform`. It normalizes timestamps to seconds, aligns the ground-truth waveform by interpolation, and creates overlapping ROI clips. Session names are assumed to be `subject-condition` with zero-padded two-digit subjects. The paper's six conditions map condition IDs `01..06` to steady, talking, slow translation, fast translation, small rotation and medium rotation.

**Verify** your distribution's timestamp naming and JSON schema. File names that are not numeric require adapting the reader; do not silently replace the absolute timestamp logic with uniform-frame assumptions.

## COHFACE

```text
data/raw/cohface/
├── 1/
│   ├── 0/data.avi
│   ├── 0/data.hdf5
│   ├── 1/data.avi
│   ├── 1/data.hdf5
│   └── ...
├── 2/
└── ...
```

The loader finds `*.avi` recursively and expects a paired `*.hdf5`/`*.h5` with a `pulse` (or `bvp` / `BVP`) dataset. It uses the manuscript's 256-Hz reference acquisition rate and derives frame timestamps using capture FPS. Both paths must correspond to the same subject/session; unexpected FPS or mismatched durations raise errors.

## UBFC-rPPG

```text
data/raw/ubfc/
├── subject1/
│   ├── vid.avi
│   └── ground_truth.txt
├── subject2/
│   ├── vid.avi
│   └── ground_truth.txt
└── ...
```

For UBFC, the first text row is the BVP waveform, second is nominal HR, third (when present and valid) supplies BVP timestamps. The loader uses the video's **measured FPS** and resamples reference BVP to individual video frames, preserving the `subjectN` identity required for the archived split.

## Shared 68-landmark ROI

`preprocessing/roi.py` detects a face using Dlib, builds polygons from eyebrow/forehead and cheek landmarks, masks eyes/nose/mouth, and resizes to `64 × 64`. Its ROI polygon placement is a documented approximation because no exact coordinates are provided in the manuscript. The proprietary/third-party 68-landmark predictor is not included; download it from its authorized provider and verify applicable terms.

The ROI detector carries forward the previous face bounding box when a detector intermittently fails, but fails explicitly when the initial face is not found. It does not drop individual frames silently, which would corrupt waveform synchronization. The acquisition metadata includes FPS, alignment choice and frame count for audits.

**Time alignment:** the loader never creates a clip until the BVP reference timestamp range covers the frame timestamps within a 100-ms tolerance. Timestamp overlap problems cause an error rather than an unnoticed extrapolation. Any changes to acquisition time reference or signal sample rate should be documented in `acquisition_metadata.json`.

## Processed clip format

```text
data/processed/pure/manifest.csv
              /01-01_clip0.npy
              /01-01_clip0_bvp.npy
              /01-01_clip1.npy
              /01-01_clip1_bvp.npy
              /acquisition_metadata.json
```

The same format applies to `cohface/` and `ubfc/`. The manifest stores **absolute local filenames** and is intentionally not committed, because paths differ by installation and may expose private data. Use `rppg split` after preprocessing and keep resulting subject membership fixed for both stages.

Clips advance eight frames and include 128 frames each (93.75% overlap with the preceding clip). Preprocessing takes place **before** applying subject partitions, but all Stage-1 and Stage-2 training loaders select clips using subject identity. Overlapping clips from one participant never cross partition boundaries.
