#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda/bin/activate
conda create -y -n vieth3 python=3.12
conda activate vieth3

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev,data,hf,cuda]" yt-dlp
