# 자체 검증 기록 (2026-11-06)

## 사용한 입력
- 사용자가 올린 `PDF_to_Markdown-main.zip`의 `sample_input/mineru_result.zip`, `test.pdf`
- 팀원이 구현한 PDF→MD / Markdown→PDF 파일과 최종 Reranker 구현(그대로 패키지에 포함)

## 실제 실행 확인
- `pytest tests/test_pipeline.py`: **9 passed** (보호 토큰/그림/표/수식/HTML 표/깨진 출력 검출/실제 E_all 명령 인자까지)
- **CPU 데모 끝까지 실행 성공**: MinerU 결과 ZIP → PDF→MD 변환 → 팀 Reranker 원본 Markdown 파서 → 모의 텍스트 출력 → 구조 복원 → 팀 Pandoc+XeLaTeX PDF 출력기
- 입력 PDF: **2페이지**; 원본 Markdown QA에서 **수식 5개, 표 2개, 이미지 대체 1개, 2단 페이지 [1,2]** 검출
- 결과: PDF **5페이지**, 생성 성공, 225,957 bytes (원본 2단을 PDF에서 단일 단으로 다시 조판하므로 분량 차이 발생)
- 그림 참조 누락 **0건**, Markdown 파이프 표 행 수정 **0건**
- 별도 단위 테스트에서 **보호 마커 누락 및 이미지 경로 변형 시 오류로 중단** 확인
- PDF를 PNG로 렌더링하여 그림, 표 이미지, 한국어 데모 글꼴이 보이는지 육안 점검

## 모델 연결 테스트 범위
- 실제 Reranker CLI 명령어 `-m reranker translate`, `--config`, `--glossary`, `--weights`, `--experiment E_all`, `--input`, `--out` 연결 검증
- **실제 모델 없이 가짜 출력을 주입**하는 테스트로 경로·파싱·보호·복원·파일 생성 확인
- 팀의 `results/final/weights.json`에 **E_all** 학습된 스코어 가중치가 존재하고 `configs/final.json`의 Stage4 + Bomin 경로 필드가 사용됨을 확인

## 검증하지 못한 것 — 실제 GPU 검증 필요
- Stage4 Replay + Bomin CS LoRA checkpoint가 실제 로드되는지
- Hugging Face 베이스 모델과 CUDA/VRAM 조건에서 전체 번역이 정상 완료되는지
- 실제 한국어 번역 품질 및 Reranker의 최종 성능
- 임의의 모든 논문에 대한 수식 OCR/표 구조/완전한 원본 2단 레이아웃 재현

**데모 PDF는 실제 번역 결과가 아닙니다.** 파일명에 DEMO를 표시함.
