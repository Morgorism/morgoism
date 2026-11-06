#!/usr/bin/env bash
# 중단된 실행을 이어서 합니다. 이미 끝난 단계(결과 파일 + .meta.json이 있는 단계)는 건너뜁니다.
# usage: bash resume_rerank.sh RUN_DIR DATA_DIR GLOSSARY CONFIG VALID_LIMIT [GLOSSARY_WEIGHT]
set -euo pipefail
RUN=$1; DATA=$2; GLOSS=$3; CFG=$4; LIMIT=$5; GW=${6:-0.0}
mkdir -p "$RUN"
done_step() { [ -s "$1" ] && [ -s "$1.meta.json" ]; }      # meta는 결과를 다 쓴 뒤에 생성됨
skip() { echo "[건너뜀] $1 (이미 완료)"; }

done_step "$RUN/valid.cands.jsonl" && skip "valid generate" || \
  python -m reranker generate --config "$CFG" --glossary "$GLOSS" --data "$DATA/valid.jsonl" --limit "$LIMIT" --out "$RUN/valid.cands.jsonl"
done_step "$RUN/valid.feats.jsonl" && skip "valid score" || \
  python -m reranker score --config "$CFG" --glossary "$GLOSS" --in "$RUN/valid.cands.jsonl" --out "$RUN/valid.feats.jsonl"
[ -s "$RUN/weights.json" ] && skip "tune" || \
  python -m reranker tune --glossary "$GLOSS" --glossary-weight "$GW" --in "$RUN/valid.feats.jsonl" --out "$RUN/weights.json"
if [ "$LIMIT" -le 20 ]; then echo "smoke test done: $RUN"; exit 0; fi
done_step "$RUN/test.cands.jsonl" && skip "test generate" || \
  python -m reranker generate --config "$CFG" --glossary "$GLOSS" --data "$DATA/test.jsonl" --out "$RUN/test.cands.jsonl"
done_step "$RUN/test.feats.jsonl" && skip "test score" || \
  python -m reranker score --config "$CFG" --glossary "$GLOSS" --in "$RUN/test.cands.jsonl" --out "$RUN/test.feats.jsonl"
[ -s "$RUN/test_preds/E_all.jsonl" ] && skip "decode" || \
  python -m reranker decode --config "$CFG" --glossary "$GLOSS" --in "$RUN/test.feats.jsonl" --weights "$RUN/weights.json" --out-dir "$RUN/test_preds"
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
