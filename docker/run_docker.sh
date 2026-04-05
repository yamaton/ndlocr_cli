#!/usr/bin/env bash
set -euo pipefail

TAG=ocr-v2-cli-py310

docker run --gpus all -d --rm --name ocr_cli_runner --shm-size=256m -i "${TAG}:latest"
