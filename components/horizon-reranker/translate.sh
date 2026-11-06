#!/usr/bin/env bash
# 최종 선택적 번역: bash translate.sh 입력.md 출력.md
set -euo pipefail
python -m reranker translate --config rerank_config.json --glossary /workspace/data/glossary_curated.json \
  --weights results/final/weights.json --experiment E_all --input "$1" --out "$2"
