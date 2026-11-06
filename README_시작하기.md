# 개발자용 빠른 시작 — CS Paper Pipeline

> **배포용 문서는 [README.md](README.md)가 기준입니다.** 이 문서는 소스코드 구성과 로컬 단위 테스트를 안내합니다. Docker 이미지에 MinerU는 포함되지 않습니다.

## 구성 요소

```text
run_pipeline.py
  ├── components/pdf_to_markdown/make_paper_md.py  # PDF/ZIP/MD → 구조화 Markdown
  ├── components/horizon-reranker/                 # Stage4 + Bomin + E_all
  └── components/markdown_to_pdf/pdf.py            # Pandoc/XeLaTeX PDF 출력

deployment/app.py                # FastAPI 번역/작업 API
deployment/client.py             # ZIP/Markdown 업로드·결과 다운로드
deployment/submit_pdf_local.py   # 사용자 PC의 MinerU → ZIP → API 연동
```

## 입력별 동작

- **통합 CLI (`run_pipeline.py`)**: 실행 환경에 `mineru-kit`이 있으면 PDF 원본도 분석할 수 있습니다. `--input *.zip` 또는 `--input *.md`로 시작하면 기존 변환 결과를 사용합니다.
- **배포 Docker API (`deployment/app.py`)**: 이미지에 `mineru-kit`을 설치하지 않았습니다. 원본 PDF 직접 업로드는 503입니다. 사용자 PC에서 MinerU ZIP을 만들어 보내세요.
- **Markdown에 있는 이미지**: 이미지 경로를 정확히 유지해야 합니다. 단일 `.md` 업로드만으로 `images/` 폴더가 같이 전송되지는 않습니다. 실제 논문 전체는 MinerU ZIP 입력을 권장합니다.

## 의존성과 테스트

```bash
python -m pip install fastapi 'uvicorn[standard]' python-multipart httpx pytest pymupdf pillow requests
python -m pytest -q deployment/test_api.py tests/test_pipeline.py deployment/test_submit_pdf_local.py
```

GPU 없이 실행하는 테스트는 API/파이프라인 연동을 확인할 뿐 실번역을 검증하지 않습니다. 실제 GPU 모델 가중치와 네트워크 준비가 필요합니다.

## 실행 예시

**데모 전용 (AI 번역 아님)**

```bash
python run_pipeline.py --input examples/mineru_result.zip \
  --original-pdf examples/test.pdf --mode demo \
  --renderer skip --name demo_md_only --no-previews
```

**배포 API 사용 (실제 LoRA와 NVIDIA GPU가 준비되었다는 전제)**

```bash
cp .env.example .env
# .env의 API_KEY 반드시 변경
mkdir -p local-data/stage4 local-data/bomin
# 각 폴더에 실제 LoRA 체크포인트 배치
docker compose build
docker compose run --rm --no-deps --entrypoint python api \
  scripts/prepare_model_config.py \
  --stage4 /workspace/stage4 --bomin /workspace/bomin \
  --out /workspace/rerank_config.json
docker compose up -d
```

**사용자 PDF 분석 → 번역 API 요청:** [README.md](README.md) 방법 A를 참고하세요. 각자 원본 PDF를 변환하려는 경우에만 MinerU Standard 설치가 필요하며, ZIP이나 Markdown이 이미 있다면 필요하지 않습니다.

## 주의

- Reranker 후보와 구조 보호 기능이 있어도 수식·표의 의미적 정확도를 자동 보장하지 않습니다.
- 2단 원본 레이아웃은 최종 PDF에서 동일한 위치/쪽수로 복원되지 않을 수 있습니다.
- 모델 가중치를 `git add`하거나 Docker 이미지에 복사하지 마세요.
