# reranker — 문서 단위 N-best reranking (Yu et al., TACL 2020 방식)

## 무엇을 하나
1. 초록(문서)을 문장으로 나눕니다. 현재 데이터는 논문당 초록 1단락이라 **같은 초록의 앞 문장들이 문맥**입니다.
2. 문장마다 후보를 만듭니다: greedy 1개 + 샘플링 `n_samples`개 (중복 제거).
3. 후보마다 점수를 계산합니다.

| 특징 | 의미 | 계산 |
|---|---|---|
| fwd | 번역 모델이 이 후보를 얼마나 확신하나 log p(y\|x, 문맥) | 학습 프롬프트 + chat template, 토큰 평균 |
| bwd | 이 한국어에서 원문 영어가 복원되나 log p(x\|y) — 누락·오역 탐지 | `train.py`가 KO→EN 샘플 25%를 함께 학습했으므로 같은 어댑터로 계산 |
| lm | 앞 문장 번역과 이어 읽을 때 자연스러운가 log p(y\|앞 번역) | 어댑터를 끈 베이스 모델로 한국어 원문 이어쓰기 점수 |
| glossary | 팀 용어집 준수율 | `glossary_curated.json` (evaluation과 같은 규칙) |
| len | 길이 비율 log(번역 길이/원문 길이) | 가중치가 음수/양수 모두 가능 |
| structure | 수식·코드·인용·placeholder가 깨진 후보는 선택에서 제외 | evaluation/structure.py |

4. **학습(tune)**: valid에서 가중치를 무작위 탐색으로 정합니다. 목표 = 문서 단위 chrF (+ 선택: 용어집 준수율). fwd 가중치는 1로 고정.
5. **decode**: test에서 문장 순서대로, 실제로 고른 앞 문장 번역을 문맥으로 lm(그리고 문맥 프롬프트를 쓰면 fwd)을 다시 계산하며 고릅니다.
6. 실험 A~E를 한 번에 만듭니다.

| 실험 | 사용 특징 | 확인할 것 |
|---|---|---|
| A_greedy | 없음 (첫 greedy 후보) | 문장 단위 번역 기준선 |
| B_fwd | fwd | 후보를 늘린 효과 |
| C_fwd_bwd | + bwd | 누락·오역 감소 |
| D_fwd_bwd_lm | + lm | 문맥 효과. **과번역률이 오르지 않는지 꼭 확인** |
| E_all | + glossary, len | 최종 후보 |

## 어떤 어댑터와 프롬프트를 쓰나

| 설정 파일 | forward (후보 생성·fwd 점수) | backward (bwd 점수) | 프롬프트 |
|---|---|---|---|
| **`configs/final.json` (최종)** | jjha Stage4 Replay LoRA | Bomin CS LoRA `checkpoint-1520` | jjha `prompts.py`와 동일 (`prompt_style: jjha_stage4`) |
| `configs/terminology_lora.json` | 최윤서 Terminology LoRA | Bomin CS LoRA | 윤서 학습 프롬프트 + `{rules}` |
| `configs/cs_lora.json` | Bomin CS LoRA | 같은 어댑터 | test.py 프롬프트 |

**Stage4 설정이 jjha 님 추론과 같은 점** (tests/test_jjha_parity.py로 원본 코드와 대조)
- 프롬프트: 앞 영어 문장 최대 2개를 `[Background Information]`으로 넣고, 앞 문장이 없으면 일반 프롬프트. test 4,099문장 전부 원본 `prompts.py`와 글자 단위로 일치
- 용어 규칙: auto_enforce이면서 context_sensitive가 아닌 용어만. `strict_glossary.json`과 `glossary_curated.json` 어느 쪽을 넣어도 같은 프롬프트
- 이상 출력(빈 출력, 토큰 상한, 깨진 문자, 6어절 3회 반복)이면 repetition_penalty 1.05, no_repeat_ngram 12로 한 번 재생성 → 이 결과가 기준선 후보(A_greedy)
- max_new_tokens 512 고정, bf16
- 이상 출력 후보는 reranking 선택에서도 제외

**reranker가 더하는 것:** jjha 님 모델은 앞 문장의 **영어 원문**만 봅니다. reranker의 lm 점수는 **이미 고른 앞 문장 한국어 번역**과 이어지는지를 보므로, 번역문 쪽 일관성을 추가로 반영합니다 (Yu et al.의 핵심 아이디어).

**jjha 님 문장 정렬 데이터를 쓰면** (`current_en`, `prev_en`, `target_ko`, `paper_id` 형식) 그대로 `--data`로 넣을 수 있습니다. `prev_en`은 주어진 값을 그대로 쓰고, decode가 문장 단위 예측 파일(`*.sentences.jsonl`)도 만들어 jjha 님 숫자와 같은 조건으로 비교할 수 있습니다.

## RunPod 실행 순서 (자세한 설명은 채팅 안내 참고)

처음 한 번: `bash setup_runpod.sh` → 저장소·어댑터·데이터·모델을 받고 `rerank_config.json`을 자동 생성합니다.

최종 번역: `python -m reranker translate --config rerank_config.json --glossary /workspace/data/glossary_curated.json --weights runs/<실행폴더>/weights.json --input paper.md --out paper.ko.md`

아래는 수동으로 할 때의 순서입니다.
```bash
git lfs install && git lfs pull                     # 어댑터 가중치는 LFS에 있음
pip install -r requirements.txt
python -m pytest -q tests                            # 작은 모델로 코드 경로 검증 (GPU 불필요)
# Terminology LoRA는 horizon 저장소 Releases의 terminology_adapter_backup.tar.gz를 받아 풀어 둡니다
# Stage4 어댑터는 horizon-context Releases의 context_replay_release.tar.gz를 받아 풀어 둡니다
cp configs/final.json rerank_config.json         # 경로 수정
bash run_rerank.sh /workspace/data /workspace/glossary_curated.json rerank_config.json 20   # 스모크 테스트 (20문서)
bash run_rerank.sh /workspace/data /workspace/glossary_curated.json rerank_config.json 200  # valid 200문서로 튜닝
```
스모크 테스트에 걸린 시간으로 전체 시간을 추정한 뒤 `--limit`, `n_samples`를 조정하세요. test는 항상 전체(624)로 돌립니다.

## 개선이 진짜인지 확인 (paired bootstrap)
```bash
python -m evaluation significance --a runs/.../test_preds/A_greedy.jsonl --b runs/.../test_preds/E_all.jsonl \
    --glossary glossary_curated.json --n 1000
```
`delta`의 95% 구간(`ci95`)이 0을 포함하거나 `p_B_not_better`가 0.05보다 크면 "개선"이라고 보고하지 않습니다.

## 꼭 비교할 기준선
- 문단 단위 `test.py` 결과 (chrF 35.11). 모델은 초록 전체로 학습했기 때문에 **문장 단위 번역(A_greedy) 자체가 문단 단위보다 낮을 수 있습니다.** reranker 효과는 A 대비, 시스템 전체 효과는 문단 단위 기준선 대비로 보고하세요.

## 주의
- tune은 앞 문장 문맥을 greedy 초안으로 고정한 근사입니다. 최종 점수는 반드시 decode 결과로 평가하세요.
- test로 가중치를 고르지 마세요. 가중치는 valid로만 정합니다.
- 정답 번역의 용어 준수율이 팀 채점 규칙으로 68%(66건 중 45건)라 chrF만 최대화하면 용어 정책과 반대로 갈 수 있습니다. `--glossary-weight 0.1~0.3`도 시도해서 두 지표를 함께 보고하세요.
