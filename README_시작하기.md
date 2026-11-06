# CS 논문 번역 최종 통합 파이프라인

**한 번 실행하는 흐름:** PDF → MinerU Standard → 학술 Markdown → Stage4 Replay 번역 + CS 역번역 모델 + E_all Reranker → 한국어 Markdown → Pandoc/XeLaTeX PDF.

이 패키지는 **2026-11-06에 받은 팀 ZIP 5개**의 코드를 기반으로 새로 만든 통합 프로젝트입니다. 팀원이 구현한 PDF→Markdown 변환기(`components/pdf_to_markdown/make_paper_md.py`), Reranker(`components/horizon-reranker/`), Markdown→PDF 변환기(`components/markdown_to_pdf/pdf.py`)를 원본 그대로 사용하고, `run_pipeline.py`에서 호출·구조 보호·검사를 담당합니다.

**중요:** 번역 모델은 새로 학습하지 않습니다. 팀의 사전학습된 Stage4 LoRA + Bomin CS LoRA와 Reranker의 **이미 학습된 E_all 가중치**를 사용합니다. 단, 이 ZIP에는 실제 수백 MB~수 GB의 모델 가중치가 **포함되지 않습니다.** GitHub ZIP의 `.safetensors`가 134바이트인 것은 Git LFS 포인터이지 모델이 아닙니다.

## A. 네 노트북에서 먼저 확인 — GPU 필요 없음

압축을 풀고 `CS_Paper_Pipeline_Final` 폴더에서 실행하세요.

```powershell
conda activate mineru-env
python -m pip install pymupdf pillow pytest
python -m pytest -q tests/test_pipeline.py
```

본인의 컴퓨터에 **Pandoc, XeLaTeX, 한글 폰트**가 설치되어 있다면 데모 PDF까지 생성:

```powershell
python run_pipeline.py --input examples/mineru_result.zip --original-pdf examples/test.pdf --mode demo --name demo_test --no-previews
```

생성 위치: `output/demo_test/document/paper.md`, `output/demo_test/document/paper.ko.md`, `output/demo_test/demo_test.DEMO.pdf`, `pipeline_report.json`.

**데모 모드는 절대로 실제 번역이 아닙니다.** `【데모: 실제 AI 번역 아님】` 문구가 남는 것이 정상입니다. 팀원 Reranker의 **실제 Markdown 파서/재구성 코드**를 GPU 없이 점검하는 용도입니다.

PDF 변환 도구가 설치되지 않은 경우 Markdown까지 확인:

```powershell
python run_pipeline.py --input examples/mineru_result.zip --original-pdf examples/test.pdf --mode demo --renderer skip --name demo_md_only --no-previews
```

## B. RunPod에서 실제 모델까지 한 번에 실행

RunPod의 GPU 환경에서 아래 조건을 먼저 준비하세요.

1. `components/horizon-reranker/requirements.txt`에 명시된 Python 패키지와 **PyTorch CUDA** 환경
2. **실제 Stage4 Replay LoRA** 가중치 (`adapter_config.json` + 충분한 크기의 `adapter_model.safetensors`)
3. **Bomin CS LoRA** 역번역 가중치 (Git LFS pull 수행, 일반 GitHub ZIP의 작은 포인터 파일로는 불가)
4. Hugging Face 베이스 모델 `tencent/Hy-MT2-7B` 접근 및 다운로드 가능 환경
5. `mineru-kit`의 Standard 모델을 사용할 수 있는 별도 Python 환경
6. `pandoc`, `xelatex`, 한국어 서체(Noto Serif CJK KR / Noto Sans CJK KR)

팀 Reranker의 `setup_runpod.sh`가 GitHub clone/LFS/모델 다운로드/설정 파일 생성 과정을 안내하며, 그대로 패키지에 포함했습니다. **단, 팀원의 `horizon-context/RELEASE_ASSET.md`에는 `stage4_replay_release.tar.gz`, Reranker 설치 스크립트에는 `context_replay_release.tar.gz`가 명시돼 있습니다.** 실제 GitHub Release 자산명을 확인해야 합니다. 불일치한 이름으로 다운로드하면 실패할 수 있습니다. 이미 설치된 환경이 있다면 재설치하지 마세요.

모델의 실제 경로를 확인한 다음 **통합 실행 설정 생성**:

```bash
python scripts/make_rerank_config.py \
  --stage4 /workspace/stage4/adapter \
  --bomin /workspace/finetuning_Bomin/outputs/hymt2_cs/checkpoint-1520 \
  --out rerank_config.json
```

`/workspace/...`는 **예시 경로**입니다. 실제 가중치가 있는 폴더에 맞게 지정하세요. 필요한 파일이 없거나 Git LFS 포인터이면 설정 생성이 실패하도록 했습니다.

이후 전체 실행 (RunPod Linux 예시):

```bash
python run_pipeline.py \
  --input /workspace/test.pdf \
  --mode real \
  --config rerank_config.json \
  --md-python /workspace/cs-mineru-env/bin/python \
  --reranker-python /opt/venv/bin/python \
  --pdf-python /workspace/cs-mineru-env/bin/python \
  --main-font 'Noto Serif CJK KR' \
  --sans-font 'Noto Sans CJK KR' \
  --name final_paper
```

**주의:** 위 Python 경로들도 예시이며, 자신의 RunPod에서 `which python` 및 환경 위치를 확인한 뒤 지정하세요. 세 Python 실행 파일은 같은 파일을 가리켜도 괜찮습니다. MinerU 환경과 번역 GPU 환경이 분리돼 있다면 반드시 각각 지정하세요.

입력을 `.pdf`로 지정하면 **그 GPU 환경에 MinerU Standard 모델 설치/다운로드가 필요**합니다. 이미 MinerU로 추출한 `.zip`을 입력하면 MinerU를 다시 실행하지 않습니다. 확인용 ZIP 실행:

```bash
python run_pipeline.py --input examples/mineru_result.zip --original-pdf examples/test.pdf \
  --mode real --config rerank_config.json --name final_paper \
  --main-font 'Noto Serif CJK KR' --sans-font 'Noto Sans CJK KR'
```

### 최종 출력

```text
output/final_paper/
├── final_paper.ko.pdf         # 실제 한국어 최종 PDF (real 모드)
├── pipeline_report.json      # 실제/데모 여부 및 검증 결과 기록
└── document/
    ├── paper.md              # 영어 원본 Markdown
    ├── paper.for_translation.md  # 수식/그림/코드 보호된 모델 입력
    ├── paper.translated_guarded.md # 모델 직후의 한국어 Markdown
    ├── paper.ko.md           # 보호 표현 복원 완료
    ├── images/               # 그림, 깨진 표의 이미지
    ├── format_report.md      # MinerU 구조 검사 결과
    ├── math_review.json      # 수식 검토 대상
    └── protection_map.json   # 복원에 사용한 보호 토큰 매핑
```

## C. 구성 요소가 실제로 어떻게 연결되는지

- **PDF → MD**: 팀 `make_paper_md.py`를 별도 파이썬 프로세스로 실행. 수식 LaTeX/표 이미지 대체/2단 읽기 순서 QA 수행.
- **MD 보호**: 이미지 참조/인라인 수식/코드/링크/HTML 표/독립 수식 등을 마커로 보호. Reranker의 Markdown 파서가 무시하는 HTML 주석 형식으로 구조 블록 전달.
- **Stage4 + Reranker**: `python -m reranker translate`에 실제 config, 용어집, **E_all weights.json**을 전달. 모델이 역번역용 Bomin LoRA까지 로드하므로 임의로 제외하지 않음. 현재 프로젝트의 glossary는 horizon 팀 ZIP 원본을 복사한 것.
- **구조 복원**: 누락되거나 변형된 보호 마커를 발견하면 조용히 PDF를 만들지 않고 오류로 중단. 이미지 경로, 표 Markdown 셀도 검사.
- **MD → PDF**: 팀원의 `pdf.py`를 그대로 사용. **Pandoc+XeLaTeX**가 Markdown 및 HTML 표, 한국어 폰트를 처리.

## D. 구현 범위 / 남은 제한

- 실제 Stage4/Bomin GPU 추론은 **이 환경에서 실행하지 못함**. `--mode demo`는 모델 호출을 흉내 낸 테스트이지 번역 품질 테스트가 아님.
- 학술 논문의 복잡한 수식·표는 MinerU의 OCR이 틀릴 수 있음. 현재 PDF→MD 팀 구현은 안전한 표기만 교정하고 불확실한 수식은 검수 보고서로 보냄.
- 팀 Reranker의 원래 `markdown_io.py`는 파이프 표 **셀 안 영어를 번역하지 않음**. 이번 요구와 동일하게 표 구조를 그대로 유지함.
- 최종 PDF는 **단일 컬럼 재조판**. 원본 PDF의 2단 지면 좌표까지 동일하게 재현하지 않음. 2단의 텍스트 읽기 순서는 PDF→MD 단계에서 분석함.
- LaTeX 구문/복잡한 인라인 마크다운에 따라 모델이 보호 토큰을 바꾸면 이 파이프라인은 **실패를 감지하고 중지**함. 무조건 복원을 보장하지 않음.
- 각 환경의 소프트웨어/글꼴 설치, RunPod GPU 메모리/인터넷/체크포인트 권한 등은 별도 사전 준비가 필요함.

검증 세부사항은 `TEST_RESULTS.md` 참고.
