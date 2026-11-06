# CS Paper Pipeline — GitHub + Docker + 사용자 GPU

이 프로젝트는 **서버가 아닌, 실행하는 사람이 소유한 NVIDIA GPU**에서 LLM 번역을 수행하는 자기 호스팅(self-hosted) 프로젝트입니다.

- **GitHub:** 소스코드·Dockerfile·Compose만 게시
- **Hugging Face:** 베이스 모델 다운로드 또는 체크포인트 보관(선택)
- **사용자:** 자신이 소유한 GPU/전기/스토리지로 추론 실행
- **로컬 API:** `http://127.0.0.1:8000`, 문서 `http://127.0.0.1:8000/docs`

## 필수 요구사항

1. Linux + Docker Engine + NVIDIA Container Toolkit **또는** Windows 10/11 + Docker Desktop + WSL2 backend + NVIDIA GPU 드라이버. GPU 패스스루가 활성화되어야 합니다.
2. Docker Compose 2.30 이상 (`gpus: all` 구문을 사용).
3. 실제 Stage4 정방향 LoRA와 Bomin 역방향 LoRA 체크포인트. 이 저장소에는 모델 가중치가 포함되지 않습니다.
4. GPU 메모리: 모델 로더가 **FP16/BF16 7B를 사용**합니다. 가중치만 대략 14~15GB이므로 6GB VRAM GPU는 현재 코드로는 불가능합니다. **최소 24GB급 GPU부터 테스트**하고, 긴 문서·대규모 후보 평가에는 48GB를 권장합니다. 4-bit 양자화는 아직 구현되지 않았습니다.

Docker 공식 GPU 문서: https://docs.docker.com/compose/how-tos/gpu-support/ , https://docs.docker.com/desktop/features/gpu/

## 빠른 시작

```bash
# 1. GitHub에 업로드된 프로젝트를 복제 (본인 실제 URL 사용)
git clone https://github.com/OWNER/REPOSITORY.git
cd REPOSITORY

# 2. 로컬 비밀키 파일 준비
cp .env.example .env
# .env의 API_KEY=REPLACE_WITH_LONG_RANDOM_TOKEN 을 반드시 변경

# 3. 호스트에서 소유한 LoRA 체크포인트를 로컬 볼륨에 배치
mkdir -p local-data/stage4 local-data/bomin
# local-data/stage4/{adapter_config.json,adapter_model.safetensors,...}
# local-data/bomin/{adapter_config.json,adapter_model.safetensors,...}

# 4. 이미지 빌드 (PDF 렌더러/언어 모델 패키지로 인해 대용량)
docker compose build

# 5. 모델 설정 생성 (GPU 불필요; 두 LoRA 파일 확인)
docker compose run --rm --no-deps --entrypoint python api \
  scripts/prepare_model_config.py \
  --stage4 /workspace/stage4 --bomin /workspace/bomin \
  --out /workspace/rerank_config.json \
  --n-samples 2 --gen-batch-size 1 --score-batch-size 1

# 6. GPU를 할당하여 서버 가동
docker compose up -d

# 7. 로그와 서버 상태 확인
docker compose logs --tail=50 -f api
curl http://127.0.0.1:8000/health
```

Windows PowerShell에서는 `cp` 대신 `Copy-Item .env.example .env`를 사용할 수 있고 `mkdir`도 지원됩니다. 줄 바꿈용 `\`는 Bash 구문이므로 PowerShell에서는 설정 생성 명령을 한 줄로 실행하거나 PowerShell의 백틱(`)을 사용하세요. 가장 단순하게는 WSL2 Ubuntu 터미널에서 위 코드를 실행하면 됩니다.

`/health`가 OK여도 모델이 없으면 번역은 실행되지 않습니다. 실 가중치와 GPU 검증에는 `/ready`를 확인하고 `/translate`를 실제 호출해야 합니다.

```bash
# API_KEY 부분을 본인의 .env 값으로 교체
curl -H 'X-API-Key: YOUR_API_KEY' http://127.0.0.1:8000/ready
curl -X POST http://127.0.0.1:8000/translate \
 -H 'X-API-Key: YOUR_API_KEY' -H 'Content-Type: application/json' \
 -d '{"text":"The model improves translation quality."}'
```

Python 클라이언트:

```bash
pip install requests
python deployment/client.py --api-key YOUR_API_KEY text \
  --text 'The model improves translation quality.'
```

논문 작업:

```bash
curl -X POST 'http://127.0.0.1:8000/jobs?output_format=pdf' \
 -H 'X-API-Key: YOUR_API_KEY' -F 'file=@examples/mineru_result.zip'
# 응답 job_id로 GET /jobs/{job_id} 확인 후 GET /jobs/{job_id}/download
```

**원본 PDF 직접 업로드는 기본 이미지에서 지원하지 않습니다.** 현재 원본 파이프라인은 외부 `mineru-kit` CLI를 요구하므로 기본 설치에는 제외했습니다. MinerU 결과 ZIP 또는 Markdown 파일로 시작하세요. MinerU가 포함된 호환 이미지를 별도 구축하면 원본 PDF도 처리할 수 있습니다.

## 데이터와 모델 경로

```
project/
├── Dockerfile
├── compose.yaml                # 로컬 GPU 패스스루
├── .env.example                # 비밀키 예시
├── .gitignore                  # 가중치 / 비밀키 제외
├── local-data/                 # gitignore 적용, Docker에서 /workspace로 연결
│   ├── stage4/adapter_config.json + adapter_model.safetensors
│   ├── bomin/adapter_config.json + adapter_model.safetensors
│   ├── rerank_config.json      # 자동 생성
│   ├── hf_cache/               # 다운로드된 Hugging Face 베이스 모델
│   └── jobs/                   # 작업 결과와 로그
├── deployment/                # FastAPI 및 클라이언트
├── scripts/prepare_model_config.py
└── components/                # 기존 PDF / Reranker 코드
```

베이스 모델은 **처음 추론 요청 때** Hugging Face에서 다운로드하고 `local-data/hf_cache`에 캐시할 수 있습니다. 비공개 모델이면 `.env`에 `HF_TOKEN`을 추가해야 합니다. 다운로드를 허용하지 않거나 네트워크가 없는 환경이라면 베이스 모델을 별도로 준비하고 LoRA 메타데이터/모델 설정에 맞게 경로를 구성하세요.

## API 목록

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | API 프로세스 상태 (인증 불필요) |
| GET | `/ready` | 체크포인트/GPU/PDF 렌더러 준비 상태 |
| POST | `/translate` | 텍스트·Markdown 번역 |
| POST | `/jobs?output_format=pdf` | 논문 변환 작업 큐 등록 (MinerU ZIP / MD) |
| GET | `/jobs/{job_id}` | 작업 상태 조회 |
| GET | `/jobs/{job_id}/download` | 완성 PDF 또는 MD 다운로드 |

인증이 필요한 API에는 `X-API-Key: <.env의 API_KEY>` 헤더를 전송하세요.

## 보안·제약

- 기본 Compose는 `127.0.0.1:8000`에서만 공개합니다. 외부 사용자에게 포트를 개방하려면 별도의 인증·HTTPS·접근 제어를 구성해야 합니다. 보안 없이 `0.0.0.0:8000` 공개는 금지하세요.
- 작업 큐는 단일 프로세스 메모리 중심이므로 재시작 시 진행 중 작업은 중단됩니다. 완료 결과는 로컬 `local-data/jobs`에 남습니다.
- 모델 및 LoRA 파일은 공개 GitHub에 커밋하면 안 됩니다. 배포 권한·모델 라이선스를 검토하세요.
- Docker는 GPU 자원을 제공하지 않습니다. **GPU를 가진 호스트**와 NVIDIA 드라이버/컨테이너 연동이 필요합니다.
- `.env` 비밀키는 Git에서 제외되며, API는 기본적으로 개인 PC 로컬에서만 접속합니다.
- 배포 패키지는 GPU 없는 환경에서 API 및 기본 단위 테스트로만 검증했습니다. 실 Docker 이미지 빌드/실 GPU 추론 테스트는 하지 못했습니다.

테스트:

```bash
python -m pip install fastapi 'uvicorn[standard]' python-multipart httpx pytest pymupdf pillow
python -m pytest -q deployment/test_api.py tests/test_pipeline.py
```

기존 RunPod 배포 문서는 [README_DEPLOY.md](README_DEPLOY.md)에 별도로 유지했습니다. 로컬 사용자 GPU 배포는 **이 README와 compose.yaml**을 기준으로 하세요.
