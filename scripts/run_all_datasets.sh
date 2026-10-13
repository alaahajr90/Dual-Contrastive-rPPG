#!/usr/bin/env bash
# Execute both stages independently on each *preprocessed* dataset.
# A valid split JSON must already exist for each dataset. In particular, the
# COHFACE test membership MUST be verified before presenting published scores.
set -euo pipefail
for DATASET in pure cohface ubfc; do
  CONFIG="configs/${DATASET}.yaml"
  MANIFEST="data/processed/${DATASET}/manifest.csv"
  SPLIT="data/processed/${DATASET}/split.json"
  if [[ ! -s "${MANIFEST}" || ! -s "${SPLIT}" ]]; then
    printf 'ERROR: Missing %s or %s; preprocess and freeze splits first.\n' "${MANIFEST}" "${SPLIT}" >&2
    exit 1
  fi
  echo "Running Stage 1 then Stage 2 on ${DATASET} (no automatic held-out test use)."
  rppg pipeline --config "${CONFIG}" --manifest "${MANIFEST}" \
    --split "${SPLIT}" --output outputs
done
