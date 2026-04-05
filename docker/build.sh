#!/usr/bin/env bash
set -euo pipefail

TAG=ocr-v2-cli-py310
DOCKERFILE=docker/Dockerfile

if command -v podman &>/dev/null; then
    RUNTIME=podman
elif command -v docker &>/dev/null; then
    RUNTIME=docker
else
    echo "Error: neither podman nor docker found" >&2
    exit 1
fi

cd "$(dirname "$0")/.."

# Download model weights (skips if already present)
wget -nc https://lab.ndl.go.jp/dataset/ndlocr_v2/text_recognition_lightning/resnet-orient2.ckpt \
    -P ./submodules/text_recognition_lightning/models
wget -nc https://lab.ndl.go.jp/dataset/ndlocr_v2/text_recognition_lightning/rf_author/model.pkl \
    -P ./submodules/text_recognition_lightning/models/rf_author/
wget -nc https://lab.ndl.go.jp/dataset/ndlocr_v2/text_recognition_lightning/rf_title/model.pkl \
    -P ./submodules/text_recognition_lightning/models/rf_title/
wget -nc https://lab.ndl.go.jp/dataset/ndlocr_v2/ndl_layout/ndl_retrainmodel.pth \
    -P ./submodules/ndl_layout/models
wget -nc https://lab.ndl.go.jp/dataset/ndlocr_v2/separate_pages_mmdet/epoch_180.pth \
    -P ./submodules/separate_pages_mmdet/models

echo "Building with $RUNTIME ..."
"$RUNTIME" build -t "$TAG" -f "$DOCKERFILE" .
