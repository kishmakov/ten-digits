#!/usr/bin/env bash
# build_docker.sh – build and push Docker image for training.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

REGION="us-central1"
IMAGE_NAME="trans-count"

IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_URI="${REGION}-docker.pkg.dev/project-39b7a321-7b37-42c3-bdb/kishmakov-dockers/${IMAGE_NAME}:${IMAGE_TAG}"

echo "==> Building image: ${IMAGE_URI}"
docker build -t "${IMAGE_URI}" -f "${REPO_ROOT}/dockers/Dockerfile" "${REPO_ROOT}"

echo "==> Pushing image..."
docker push "${IMAGE_URI}"
