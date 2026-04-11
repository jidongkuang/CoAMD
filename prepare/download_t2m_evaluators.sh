#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

mkdir -p checkpoints/t2m
cd checkpoints/t2m

echo "Downloading HumanML3D evaluator checkpoints..."
gdown --fuzzy https://drive.google.com/file/d/1ejiz4NvyuoTj3BIdfNrTFFZBZ-zq4oKD/view?usp=sharing
unzip -o evaluators_humanml3d.zip
rm evaluators_humanml3d.zip

echo "Done: ./checkpoints/t2m"
