# CS Paper Translation — 사용자 NVIDIA GPU로 실행하는 Docker 배포판

> **현재 제공 방식: PDF는 사용자 PC의 MinerU Standard에서 먼저 변환하고, Docker API는 변환된 MinerU ZIP 또는 Markdown을 번역합니다.** Docker 이미지에는 MinerU가 설치되어 있지 않으므로 원본 `.pdf` 파일을 API에 직접 업로드하면 **HTTP 503**이 발생합니다. 서버에서 PDF까지 자동 처리하는 기능은 이 배포판에 구현되어 있지 않습니다.

본 프로젝트는 논문 번역을 제공하는 공개 웹서비스가 아니라 **사용자가 직접 실행하는 자기 호스팅(self-hosted) API**입니다. GitHub에는 실행 코드와 Docker 설정만 있으며, 실제 번역에는 사용자의 **NVIDIA GPU**, Stage4 정방향 LoRA, Bomin 역방향 LoRA, 베이스 모델이 필요합니다. 모델 가중치는 저장소에 포함되지 않습니다.

## 1. 무엇이 어디서 실행되나요?

```text
[사용자 PC: MinerU Standard]       [사용자의 NVIDIA GPU 컴퓨터: Docker]
원본 영어 PDF                    MinerU ZIP 또는 .md 업로드
   │                                  │
   └─ MinerU 변환 → .zip ──────────────┤
                                      ▼
                           Stage4 번역 + Bomin 역번역
                           + E_all Reranker
                                      │
                           Markdown → Pandoc/XeLaTeX
                                      │
                              한국어 PDF/Markdown
```

| 입력 | 지원 여부 | 필요한 준비 |
| --- | --- | --- |
| 영어 원본 `.pdf` → 바로 API 업로드 | **미지원 (503)** | 서버 Docker에는 `mineru-kit` 없음 |
| 원본 `.pdf` → 사용자 PC MinerU → ZIP → API | **지원 경로** | 사용자 PC의 MinerU Standard 및 API 사용을 위한 GPU 서버 |
| 이미 추출된 MinerU `.zip` → API | **지원** | MinerU 재설치 불필요 |
| `.md` → API | **지원** | 구조 이미지 파일은 별도 보존 필요. 이미지가 많은 논문은 ZIP 권장 |

**중요:** 호스트 PC에 MinerU를 설치하는 것만으로 Docker 컨테이너 내부에 `mineru-kit`이 설치되지는 않습니다. 따라서 원본 PDF를 **먼저 호스트에서 ZIP으로 변환해 업로드**해야 합니다. PDF→Markdown의 수식·표·이미지 검사는 MinerU 추출 후 팀의 변환 모듈이 이어받아 처리합니다. **원본 PDF 좌표를 이용하는 2단 검사는 변환 단계에 원본 PDF도 전달되어야 하므로 ZIP만 서버에 올린 경우 일부 좌표 검사가 생략될 수 있습니다.** Markdown→PDF의 2단 지면 복원이나 수식·표의 완전한 정확도는 보장되지 않습니다.

## 2. Docker API를 실행할 사람: 준비물

- **Linux:** Docker Engine + NVIDIA Driver + NVIDIA Container Toolkit. **Windows:** Docker Desktop + WSL2 backend + NVIDIA GPU 지원 드라이버.
- Docker Compose **2.30 이상** (`gpus: all` 사용).
- **NVIDIA GPU** 권장 VRAM 48GB 이상. 24GB급에서 작은 설정으로 실험할 수 있지만, 실제 모델 메모리 사용량은 검증이 필요합니다. Intel 내장 GPU 등에서는 이 배포판의 실번역을 실행할 수 없습니다.
- Stage4/Bomin **실제 LoRA 체크포인트**: 각각 `adapter_config.json`, **1MB보다 큰** `adapter_model.safetensors`가 있어야 합니다. Git LFS 포인터만으로는 동작하지 않습니다.
- 베이스 모델 `tencent/Hy-MT2-7B` 다운로드를 위한 네트워크와 충분한 저장 공간.

[Docker GPU Compose](https://docs.docker.com/compose/how-tos/gpu-support/) / [Windows GPU Docker](https://docs.docker.com/desktop/features/gpu/)

## 3. Docker API 설치와 시작 (GPU 컴퓨터에서)

아래 명령은 **Bash/WSL2 Ubuntu** 기준입니다. `<REPOSITORY_URL>`과 경로는 자신의 환경에 맞게 바꿔야 합니다.

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>

cp .env.example .env
# .env를 열어서 API_KEY=REPLACE_WITH_LONG_RANDOM_TOKEN 을
# 충분히 길고 임의의 비밀 문자열로 반드시 변경한다.

mkdir -p local-data/stage4 local-data/bomin
# local-data/stage4/ 에 Stage4 LoRA 파일 복사
# local-data/bomin/  에 Bomin LoRA 파일 복사

# GPU 호스트/드라이버와 Docker GPU 전달 확인 후 실행
docker compose build

docker compose run --rm --no-deps --entrypoint python api \
  scripts/prepare_model_config.py \
  --stage4 /workspace/stage4 \
  --bomin /workspace/bomin \
  --out /workspace/rerank_config.json \
  --n-samples 2 --gen-batch-size 1 --score-batch-size 1

docker compose up -d
docker compose ps
curl http://127.0.0.1:8000/health
```

첫 번역 때 베이스 모델 가중치를 받으므로 다운로드 시간이 걸릴 수 있습니다. `local-data/hf_cache/`에 보관됩니다. 모델을 다시 학습할 필요는 없습니다.

API 준비 상태 및 실제 번역 확인 (비밀키는 자신의 `.env`와 동일한 값):

```bash
# YOUR_API_KEY를 실제 값으로 변경
curl -H 'X-API-Key: YOUR_API_KEY' http://127.0.0.1:8000/ready
curl -X POST http://127.0.0.1:8000/translate \
  -H 'X-API-Key: YOUR_API_KEY' -H 'Content-Type: application/json' \
  -d '{"text":"The proposed method improves accuracy."}'
```

`/health`의 `ok: true`는 서버 프로세스가 살아 있다는 뜻일 뿐, **모델 로딩 성공을 의미하지 않습니다.** `/ready`의 `configured`, `cuda_available`, `pdf_renderer_available` 값을 살피고 `/translate`의 실제 한국어 출력을 확인해야 합니다. `/ready`의 `pdf_parser_available: false`는 **현재 Docker 이미지에서는 정상 상태**입니다.

## 4. PDF를 번역하려는 사람: MinerU Standard 설치 (각 사용자 PC)

**원본 PDF를 보내고 싶을 때만** 사용자 PC에 MinerU를 설치하면 됩니다. 이미 MinerU ZIP 또는 Markdown이 있다면 이 단계를 생략하세요. MinerU는 Docker 번역 환경과 **분리된 Python 환경**에 설치하는 것이 안전합니다.

최초 설치 예시 (Python 3.11~3.14 환경, 운영체제 및 하드웨어별 의존성은 MinerU 공식 지침 참고):

```bash
python -m venv .mineru-venv
source .mineru-venv/bin/activate   # Linux/macOS
# Windows PowerShell: .\.mineru-venv\Scripts\Activate.ps1
python -m pip install 'mineru>=4.0,<5'
mineru-kit --help
mineru-kit models download --tier standard
mineru-kit models verify --tier standard
```

이미 설치한 사용자는 `mineru-kit --help`와 모델 준비 상태만 확인하면 됩니다. Windows에서 MinerU Standard 실행에 추가 패키지나 백엔드 설정이 필요한 경우 [MinerU 공식 설치 안내](https://opendatalab.github.io/MinerU/quick_start/)를 따르세요. 최초 다운로드 및 문서 파싱은 시간이 걸릴 수 있습니다.

### 방법 A — PDF 파일 하나를 명령어 한 번으로 번역 (권장)

이 저장소에 포함된 `deployment/submit_pdf_local.py`가 **로컬 MinerU → ZIP 생성 → Docker API 업로드 → 상태 확인 → 한국어 PDF 다운로드**를 자동으로 순차 실행합니다. **API 서버는 먼저 GPU 컴퓨터에서 켜져 있어야 합니다.** 로컬 MinerU 실행과 서버 GPU 번역은 서로 다른 컴퓨터여도 가능하지만, 이 경우 API 접근 가능한 주소/인증 설정이 필요합니다.

```bash
python -m pip install requests
export API_KEY='여기에_.env와_같은_비밀키'    # Bash/WSL2
# Windows PowerShell: $env:API_KEY='여기에_.env와_같은_비밀키'
# 다른 컴퓨터의 API라면 API_URL도 설정 (기본 http://127.0.0.1:8000)

python deployment/submit_pdf_local.py ./paper.pdf --out ./paper.ko.pdf
```

실행 후 `paper.ko.pdf`를 확인합니다. MinerU 원본 ZIP도 남기고 싶으면 `--keep-mineru-zip`을 붙입니다. **`.pdf`는 서버 API에 직접 보내지 않으며, ZIP을 보내는 방식**입니다. 다른 컴퓨터의 API에 연결하려면 VPN·SSH 터널 등 적절한 보안 연결이 필요합니다. 로컬 API를 인터넷에 그대로 개방하지 마세요.

### 방법 B — MinerU ZIP을 직접 만들고 업로드

```bash
mineru-kit parse ./paper.pdf -o ./paper.mineru.zip --format zip --tier standard
python -m pip install requests
export API_KEY='여기에_.env와_같은_비밀키'
python deployment/client.py paper ./paper.mineru.zip --output-format pdf
```

Windows PowerShell은 `export API_KEY=...` 대신 `$env:API_KEY='...'`를 사용하세요. 위 `client.py`는 기본적으로 현재 PC의 `http://127.0.0.1:8000` 서버에 요청합니다. 성공하면 `paper.mineru.pdf`가 생성됩니다(입력 ZIP 파일 이름 기준). ZIP은 **MinerU 자체가 생성한 ZIP**이어야 하며, PDF를 그냥 ZIP으로 압축한 파일은 처리할 수 없습니다.

### 방법 C — 이미 추출된 Markdown이 있다면

```bash
export API_KEY='여기에_.env와_같은_비밀키'
python deployment/client.py paper ./paper.md --output-format markdown
```

Markdown에 그림이 있다면 `.md` 하나만 서버에 보내는 것으로는 `images/` 파일이 자동 업로드되지 않습니다. 구조와 이미지까지 필요하면 원본 **MinerU ZIP**을 사용하는 방법을 권장합니다.

## 5. API 목록

| Method | Endpoint | 의미 |
| --- | --- | --- |
| GET | `/health` | 프로세스 정상 여부, 인증 불필요 |
| GET | `/ready` | GPU·LoRA·렌더러·MinerU 유무 점검 |
| POST | `/translate` | 텍스트/Markdown의 실모델 번역 |
| POST | `/jobs?output_format=pdf` | MinerU ZIP 또는 Markdown 작업 등록 |
| GET | `/jobs/{job_id}` | `queued` / `running` / `completed` / `failed` 확인 |
| GET | `/jobs/{job_id}/download` | 번역 PDF 또는 `.md` 다운로드 |

인증이 필요한 API에는 `X-API-Key` 헤더가 필요합니다. 로컬 API 문서: http://127.0.0.1:8000/docs

## 6. 현재 배포판의 한계와 주의 사항

1. **Docker에는 MinerU를 포함하지 않습니다.** 원본 PDF를 `/jobs`에 업로드하면 HTTP 503입니다. 사용자 PC에서 먼저 MinerU ZIP을 만든 뒤 업로드하세요. 이것은 의도한 배포 방식입니다.
2. 모델 가중치와 API 키는 GitHub/Docker 이미지에 넣지 않습니다. `local-data/`와 `.env`는 Git에서 제외됩니다. 체크포인트 사용 권한과 라이선스는 별도 검토하세요.
3. 기본 Compose는 `127.0.0.1:8000`에만 바인딩됩니다. 외부 접속은 추가 보안 설정이 필요합니다.
4. 번역 후보 생성과 E_all Reranker를 실행하려면 실제 NVIDIA GPU가 필요합니다. Docker 자체는 GPU를 제공하지 않습니다.
5. Pandoc/XeLaTeX는 Docker에 설치됩니다. PDF 출력의 수식·표·2단 편집 완전 재현은 보장되지 않습니다.
6. 작업 진행 중 서버를 재시작하면 작업이 중단될 수 있습니다. 결과와 로그는 `local-data/jobs/`에 남습니다. 저장 공간을 주기적으로 확인하세요.
7. 팀의 모델 코드 통합과 API 단위 테스트는 존재하지만, **이 배포본 자체의 실제 Docker 빌드와 GPU 추론은 아직 최종 검증되지 않았습니다.** `/health` 성공을 전체 파이프라인 성공으로 표현하지 마세요.

## 7. 기본 점검

```bash
python -m pip install fastapi 'uvicorn[standard]' python-multipart httpx pytest pymupdf pillow
python -m pytest -q deployment/test_api.py tests/test_pipeline.py
```

RunPod를 통한 GPU 서버 배포 방식은 [`README_DEPLOY.md`](README_DEPLOY.md)를 참고하세요. 그 문서는 **대체 배포 방식**이며, 현재 Docker 이미지의 MinerU 미설치 제한은 동일하게 적용됩니다. 전체 파이프라인 구조와 개발용 설명은 기존 `README_시작하기.md`를 참고하되, **최종 사용자 배포 절차는 이 README.md를 우선**합니다.
