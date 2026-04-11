#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

rm -rf glove
echo "Downloading GloVe embeddings used by the HumanML3D evaluators..."
gdown --fuzzy https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing
unzip -o glove.zip
rm glove.zip
echo "Done: ./glove"
