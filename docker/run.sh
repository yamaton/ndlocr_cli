#!/usr/bin/env bash
set -euo pipefail

TAG=ocr-v2-cli-py310

podman run --device nvidia.com/gpu=all -d --rm --name ocr_cli_runner --shm-size=256m -i "${TAG}:latest"
