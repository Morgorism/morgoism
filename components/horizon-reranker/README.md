## 최종 모델 구성

```
tencent/Hy-MT2-7B (베이스, Hugging Face)
 + Stage4 LoRA          jjha0/horizon-context Release v1.0.0
 + 용어 규칙 프롬프트     glossary_curated.json (jjha 프롬프트 형식)
 + Reranker (이 저장소)   results/final/weights.json + configs/final.json
     후보: 문장마다 greedy 1 + 샘플 6
     점수: 순방향 확률, 역번역 확률(Bomin CS LoRA), 앞 문장 번역과의 연결(LM), 팀 용어 규칙 준수, 길이
     제외: 수식·코드·인용이 깨진 후보, 반복·빈 출력
```
reranker가 학습한 것은 `results/final/weights.json`의 가중치 5개.

## 결과 (test 624편, 문서 단위)

| 시스템 | chrF | BLEU | 팀 용어 규칙 (66건) | 구조 보존 |
|---|---|---|---|---|
| A: Stage4 그대로 | 35.84 | 27.50 | 96.97% | 99.84% |
| **E: + reranker (최종)** | **36.69** | **28.23** | **98.48%** | **100%** |

- chrF +0.85, paired bootstrap 95% 구간 [+0.64, +1.08] → 유의
- 팀 용어 규칙 +1건, 유의하지 않음 (p=0.38, 66건 표본)
- E는 전문용어 괄호 병기가 늘었음 (괄호 안 영어 단어 1,413 → 2,263). 괄호 밖 영어 단어는 2,916 → 3,104(+6%)로 대부분 약어·고유명사
- 정답은 저자가 쓴 국문 초록이라 영어 원문의 충실한 번역이 아님. chrF는 참고 지표로 볼 것
- 전체 비교표: `results/final/compare.md`, 유의성: `results/final/significance_A_vs_E.json`

## 빠른 시작 (RunPod, GPU 48GB 이상, PyTorch 템플릿)

```bash
cd /workspace && git clone https://github.com/Anjihoo0527/horizon-reranker.git && cd horizon-reranker
bash setup_runpod.sh          # 팀 저장소·어댑터·데이터·베이스 모델 받기, rerank_config.json 생성, 테스트
bash translate.sh 논문.md 논문.ko.md
```

코드 블록, 수식 블록, 표는 그대로 두고 제목·목록·문단을 번역.

## 재현

```bash
bash run_rerank.sh /workspace/data /workspace/data/glossary_curated.json rerank_config.json 200
# 중단되면: bash resume_rerank.sh runs/<폴더> /workspace/data /workspace/data/glossary_curated.json rerank_config.json 200
```
valid 200편으로 가중치를 학습하고 test 624편으로 평가합니다. A100급 GPU에서 약 4시간.

## 평가 코드 (팀 공통)

```bash
python -m evaluation run --pred 예측.jsonl --testset data/test.jsonl --glossary glossary_curated.json --out results/이름
python -m evaluation compare results/A results/B
python -m evaluation significance --a A.jsonl --b B.jsonl --glossary glossary_curated.json
```
- `team_*` 지표는 `evaluate_terminology.py`와 같은 규칙이며, 원본 코드와 대조해 같은 값이 나오는 것을 확인함
- `reranker/prompts.py`의 프롬프트는 jjha `prompts.py`와 test 4,099문장에서 글자 단위로 일치 (`tests/test_jjha_parity.py`)

## 폴더

| 경로 | 내용 |
|---|---|
| `reranker/` | 후보 생성, 점수, 가중치 학습, 선택, Markdown 번역 |
| `evaluation/` | BLEU/chrF, 용어, 구조, 출력 이상, 유의성 검정 |
| `configs/final.json` | 최종 설정 (Stage4 + 역번역용 CS LoRA) |
| `results/final/` | 최종 가중치와 평가 결과 |
| `tests/` | 작은 모델로 전체 과정 검증 + 팀원 코드 대조 테스트 |
| `RERANKER.md` | reranker 상세 설명 |

## 한계와 다음 단계

- 번역이 느림: 문장마다 후보 7개를 만들고 점수를 매김 (test 624편에 약 2.5시간)
- 다음 버전: reranker가 고른 번역을 chosen, 탈락한 번역을 rejected로 선호 학습(DPO/CPO, ALMA-R 방식)하면 reranker 없이도 비슷한 품질을 빠르게 낼 수 있음
- 사람 평가 필요: 자동 지표만으로 평가함
