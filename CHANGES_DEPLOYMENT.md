# 배포 문서/사용성 수정 내역

## 이번 수정에서 선택한 방법

**MinerU를 번역 Docker 이미지에 추가하지 않고, PDF를 처리하는 사용자 PC에서 별도로 실행합니다.** GPU 번역 Docker 이미지의 의존성과 MinerU 파서의 의존성을 섞지 않기 위한 설계입니다. 따라서 서버 API의 `.pdf` 직접 업로드는 기존처럼 HTTP 503입니다.

## 교체/추가 파일

- `README.md` — 사용자 관점에서 Docker 요구사항, API 사용법, MinerU 설치/생성, GPU 조건, 제한사항 일원화
- `README_DEPLOY.md` — RunPod는 선택 배포 방식으로 설명, 기본 이미지 MinerU 미포함 명시
- `README_시작하기.md` — 개발자/데모와 실제 배포를 구분
- `deployment/submit_pdf_local.py` — 원본 PDF → 로컬 MinerU Standard ZIP 생성 → API 제출 → 폴링 → 결과 다운로드
- `deployment/test_submit_pdf_local.py` — 새로운 브리지 스크립트의 오류 처리 및 API 업로드 테스트

## 확인 결과

- `python -m pytest -q deployment/test_api.py tests/test_pipeline.py deployment/test_submit_pdf_local.py`
- 결과: **16 passed** (API + 파이프라인 기존 테스트 13개, 새 브리지 테스트 3개)
- `submit_pdf_local.py` 문법 검사 통과
- **하지 않은 검증:** 실제 Docker 이미지 빌드, MinerU 실제 모델 실행, LoRA GPU 실추론. Docker/GPU 호스트에서 별도로 확인해야 합니다.

실행 코드 중 `Dockerfile`, `compose.yaml`, `deployment/app.py`, `run_pipeline.py`는 **원본 상태 그대로**입니다. 사용자가 필요 없는 Docker MinerU 설치를 했다고 주장하지 않습니다.
