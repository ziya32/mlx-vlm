#!/usr/bin/env bash
set -euo pipefail

uv run python -m mlx_vlm convert \
  --hf-path ~/tmp/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2 \
  --mlx-path ~/tmp/Qwen3.5-27B-Opus46-v2-emee8bit \
  -q --q-bits 8 \
  --quant-config ~/tmp/qwen3.5-27B-quant-config.json
