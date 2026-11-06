#!/usr/bin/env bash
# usage: bash run_rerank.sh DATA_DIR GLOSSARY CONFIG VALID_LIMIT [GLOSSARY_WEIGHT]
set -euo pipefail
DATA=$1; GLOSS=$2; CFG=$3; LIMIT=$4; GW=${5:-0.0}
RUN=runs/$(date +%Y%m%d_%H%M)_v${LIMIT}_gw${GW}
mkdir -p "$RUN"
python -m reranker generate --config "$CFG" --glossary "$GLOSS" --data "$DATA/valid.jsonl" --limit "$LIMIT" --out "$RUN/valid.cands.jsonl"
python -m reranker score    --config "$CFG" --glossary "$GLOSS" --in "$RUN/valid.cands.jsonl" --out "$RUN/valid.feats.jsonl"
python -m reranker tune     --glossary "$GLOSS" --glossary-weight "$GW" --in "$RUN/valid.feats.jsonl" --out "$RUN/weights.json"
if [ "$LIMIT" -le 20 ]; then echo "smoke test done: $RUN"; exit 0; fi
python -m reranker generate --config "$CFG" --glossary "$GLOSS" --data "$DATA/test.jsonl" --out "$RUN/test.cands.jsonl"
python -m reranker score    --config "$CFG" --glossary "$GLOSS" --in "$RUN/test.cands.jsonl" --out "$RUN/test.feats.jsonl"
python -m reranker decode   --config "$CFG" --glossary "$GLOSS" --in "$RUN/test.feats.jsonl" --weights "$RUN/weights.json" --out-dir "$RUN/test_preds"
for f in "$RUN"/test_preds/*.jsonl; do
  case "$f" in *.sentences.jsonl) continue;; esac
  n=$(basename "$f" .jsonl)
  python -m evaluation run --pred "$f" --testset "$DATA/test.jsonl" --glossary "$GLOSS" --out "$RUN/results/$n" > /dev/null
done
python -m evaluation compare "$RUN"/results/A_greedy "$RUN"/results/B_fwd "$RUN"/results/C_fwd_bwd \
  "$RUN"/results/D_fwd_bwd_lm "$RUN"/results/E_all --out "$RUN/compare.md"
python -m evaluation significance --a "$RUN"/test_preds/A_greedy.jsonl --b "$RUN"/test_preds/E_all.jsonl \
  --glossary "$GLOSS" --n 1000 --out "$RUN/significance_A_vs_E.json"
echo "done: $RUN"
