#!/usr/bin/env bash
# Download checkpoint from GCS and run demo_test.py evaluation.
# Designed to run inside the Vertex AI container.
set -e

GCS_CKPT="${GCS_CKPT:-gs://kishmakov-trans-count-outputs/456p_s43/checkpoints/best.pt}"
OUT_JSON="${OUT_JSON:-/app/results/demo_test_results.json}"
DATA_DIR="${DATA_DIR:-/app/results/data}"
GCS_OUTPUT="${GCS_OUTPUT:-gs://kishmakov-trans-count-outputs/demo_test}"

echo "=== Downloading checkpoint from ${GCS_CKPT} ==="
mkdir -p /tmp/ckpt
gsutil cp "${GCS_CKPT}" /tmp/ckpt/best.pt

echo "=== Running demo_test.py ==="
python demo_test.py \
  --ckpt /tmp/ckpt/best.pt \
  --device cpu \
  --data-dir "${DATA_DIR}" \
  --out "${OUT_JSON}"

echo "=== Uploading results to ${GCS_OUTPUT} ==="
gsutil cp "${OUT_JSON}" "${GCS_OUTPUT}/demo_test_results.json"

echo "=== Done. Results at ${GCS_OUTPUT}/demo_test_results.json ==="
