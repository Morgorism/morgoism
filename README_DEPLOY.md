# RunPod GPU Pod + Docker: Stage4 + E_all 논문 번역 API

> 이 배포본은 **첨부된 `CS_Paper_Pipeline_Final` 원본 프로젝트에 API와 Docker 설정을 추가**한 것입니다. 기존 `run_pipeline.py`, PDF 변환기, Reranker는 그대로 유지합니다. 모델을 학습하거나 가중치를 패키지에 넣지 않습니다.

## 배포 전 확인

- Docker 이미지를 저장할 Docker Hub 계정과 Docker CLI/Desktop(또는 Linux 빌드 환경)이 필요합니다.
- GPU: **NVIDIA L4 24GB 1장부터 시도**. RAM/VRAM 사용량은 입력 길이, 샘플 수에 따라 달라집니다. OOM이면 `--n-samples 2`로 낮추거나 48GB GPU를 사용하세요.
- **실제 LoRA 체크포인트 두 개**가 필요합니다. `Stage4` 정방향 번역, `Bomin CS` 역번역. `adapter_config.json`과 **실제** `adapter_model.safetensors`가 모두 있어야 합니다. Git LFS 포인터는 사용 불가.
- 베이스 모델 `tencent/Hy-MT2-7B`는 첫 요청 시 Hugging Face에서 내려받습니다. `HF_HOME=/workspace/hf_cache`를 사용하므로 `/workspace` 볼륨의 여유 공간을 확보하세요.
- **원본 PDF 직접 처리:** 현재 Dockerfile은 `mineru-kit`을 설치하지 않습니다. 원본 변환기가 `mineru-kit parse`라는 특정 명령어를 요구하기 때문입니다. 검증 가능한 기본 배포에서는 **MinerU 추출 ZIP 또는 .md**를 업로드하세요. `/ready` 응답의 `pdf_parser_available: false`가 정상입니다. 원본 PDF까지 처리하려면 호환되는 MinerU CLI 환경을 별도로 이미지에 설치해야 합니다.

## 1. Docker 이미지 빌드 및 업로드

이 README가 있는 프로젝트 루트에서 실행하세요.

```bash
# Windows Docker Desktop에서도 Linux 컨테이너 모드로 실행
# 아래 DOCKERHUB_ID를 본인 계정으로 교체

docker build --platform linux/amd64 -t DOCKERHUB_ID/cs-paper-api:v1 .
docker login
docker push DOCKERHUB_ID/cs-paper-api:v1
```

이미지는 CUDA/PyTorch와 XeLaTeX가 포함되어 크기가 큽니다. 로컬 Windows에 Docker가 없다면 Linux 빌드 환경(또는 GitHub Actions)을 이용할 수 있습니다. **이 패키지를 만든 환경에서는 실제 Docker 이미지를 빌드하거나 GPU 추론을 실행하지 않았습니다.**

## 2. RunPod Pod 생성

1. [RunPod Pods](https://www.runpod.io/console/pods)에서 GPU Pod 생성. 우선 **L4 24GB** 또는 필요시 **L40S 48GB**.
2. Image: `DOCKERHUB_ID/cs-paper-api:v1` (Custom Template 설정).
3. HTTP Ports: `8000/http`. **Dockerfile의 EXPOSE만으로는 RunPod 프록시 포트가 열리지 않습니다.**
4. GPU와 모델을 보관할 `/workspace` 볼륨 마운트. 가능하면 Pod와 독립적으로 유지되는 Network Volume을 선택하세요. 모델 두 개와 베이스 모델 캐시를 고려해 충분한 공간을 잡습니다(예: 50GB 이상).
5. 환경 변수(필수): `API_KEY=길고_무작위인_비밀키`, `MODEL_CONFIG=/workspace/rerank_config.json`, `HF_HOME=/workspace/hf_cache`.
6. `workers`는 반드시 **1개** 유지. FastAPI worker마다 모델이 로드되면 VRAM이 낭비됩니다.

RunPod에서 HTTP Proxy는 아래 패턴을 제공합니다.

```
https://POD_ID-8000.proxy.runpod.net
```

공식 문서: https://docs.runpod.io/pods/connect-to-a-pod#http-proxy (문서 URL이 변경될 수 있음)

## 3. LoRA 모델을 `/workspace`에 준비

컨테이너가 실행되는 RunPod Pod의 Terminal 또는 SSH에서 실제 가중치를 준비하세요. 다음 경로는 **예시**이므로, 팀원이 제공한 체크포인트 위치에 맞게 변경해야 합니다.

```bash
ls -lh /workspace/stage4/adapter/adapter_*
ls -lh /workspace/finetuning_Bomin/outputs/hymt2_cs/checkpoint-1520/adapter_*
```

Stage4는 `.tar.gz` Release를 실제로 다운로드·압축 해제해 준비하거나 팀원의 모델 파일을 업로드하세요. Bomin 체크포인트를 GitHub에서 가져왔다면 `git lfs pull`이 필요할 수 있습니다. **기존 `setup_runpod.sh`는 Release 이름이 소스와 맞지 않을 가능성이 있으므로 무작정 실행하지 마세요.**

그 다음 설정 생성:

```bash
cd /app
python scripts/prepare_runpod_config.py \
  --stage4 /workspace/stage4/adapter \
  --bomin /workspace/finetuning_Bomin/outputs/hymt2_cs/checkpoint-1520 \
  --out /workspace/rerank_config.json \
  --n-samples 6 --gen-batch-size 1 --score-batch-size 2
```

위 설정은 원본 `configs/final.json`을 복사하며, **E_all 가중치 및 번역 프롬프트는 바꾸지 않고** 메모리 사용을 줄이기 위해 배치 크기만 줄입니다. 여전히 OOM이면 `--n-samples 2`로 변경할 수 있지만, 후보 감소에 따른 번역 품질 차이가 있을 수 있습니다.

## 4. 작동 확인

첫 요청 전에는 모델이 아직 메모리에 적재되지 않는 **lazy loading** 방식입니다. 그래서 `/health`가 200이라고 해서 실제 번역이 동작하는 것은 아닙니다.

```bash
# 서버 상태 (API 키 없이 확인 가능)
curl https://POD_ID-8000.proxy.runpod.net/health

# 파일/설정 및 GPU 상태 (키 필요)
curl -H 'X-API-Key: YOUR_SECRET' \
  https://POD_ID-8000.proxy.runpod.net/ready

# 실제 번역 (첫 요청에서는 모델 다운로드/로드 때문에 상당히 오래 걸릴 수 있음)
curl -X POST https://POD_ID-8000.proxy.runpod.net/translate \
  -H 'X-API-Key: YOUR_SECRET' \
  -H 'Content-Type: application/json' \
  -d '{"text":"The proposed model achieves better translation accuracy."}'
```

성공 시 `translation`, `units`, `system` 필드를 포함한 JSON을 반환합니다. **키를 소스코드·공개 GitHub 저장소에 넣지 마세요.**

### 논문 변환 작업 (대기 → 상태 확인 → 다운로드)

```bash
# 먼저 기존 MinerU ZIP 업로드. PDF를 바로 넣으면 mineru-kit 미설치로 503 반환
curl -X POST 'https://POD_ID-8000.proxy.runpod.net/jobs?output_format=pdf' \
  -H 'X-API-Key: YOUR_SECRET' \
  -F 'file=@examples/mineru_result.zip'

# 반환된 job_id로 확인
curl -H 'X-API-Key: YOUR_SECRET' \
  'https://POD_ID-8000.proxy.runpod.net/jobs/JOB_ID'

# status=completed이면 출력 PDF 다운로드
curl -L -H 'X-API-Key: YOUR_SECRET' \
  'https://POD_ID-8000.proxy.runpod.net/jobs/JOB_ID/download' \
  -o result.ko.pdf
```

Markdown만 필요하면 `?output_format=markdown`으로 요청하세요.

파이프라인 로그는 `/workspace/jobs/JOB_ID/pipeline.log`에 기록됩니다. 원본 `run_pipeline.py`가 **모델 번역 결과에 한국어가 없거나, 보호된 수식·이미지 경로가 훼손되면 실패**시키는 동작을 그대로 유지합니다.

## 5. 로컬 Python 클라이언트

```bash
pip install requests
export API_URL='https://POD_ID-8000.proxy.runpod.net'
export API_KEY='YOUR_SECRET'
python deployment/client.py text --text 'The model is trained on English and Korean data.'
python deployment/client.py paper examples/mineru_result.zip --output-format pdf
```

Windows PowerShell은 `export` 대신 `$env:API_URL="..."`와 `$env:API_KEY="..."`를 사용하세요. FastAPI API 문서는 `https://POD_ID-8000.proxy.runpod.net/docs`에서 확인할 수 있습니다.

## 6. 운영상의 주의 사항

- **GPU 동시 실행 제한:** API 내부에 단일 GPU 락을 사용합니다. 텍스트 요청 중 다른 요청은 HTTP 429가 될 수 있습니다. 논문 작업 시작 시 텍스트 모델을 GPU에서 해제하고 기존 CLI 프로세스가 모델을 로드합니다. 논문 작업 이후 텍스트 요청에서 다시 모델을 로드합니다.
- **긴 요청은 작업 API 사용:** 논문 전체 번역은 시간이 길어질 수 있으므로 `/jobs`로 등록 후 `/jobs/{id}`를 폴링합니다. 진행 중 Pod가 멈추면 작업은 복구되지 않고 `interrupted` 상태가 되며 **재제출이 필요**합니다.
- **저장 공간:** `/workspace/jobs`에 업로드·생성 파일이 누적됩니다. 필요 없는 완료 작업은 서버 터미널에서 직접 삭제하세요. Pod를 종료해도 Network Volume을 사용하면 저장 공간 비용은 계속 발생합니다.
- **모델은 사용자에게 제공되지 않습니다:** 해당 LoRA 파일과 API_KEY는 Docker 이미지에 포함하지 않고 RunPod 환경에만 둡니다.
- **데모가 아닌 실번역:** `/translate`와 `/jobs`는 항상 원본 `Stage4 + E_all` 실추론을 사용하며 가중치가 없으면 실패합니다. 첨부 패키지에 있는 `--mode demo`는 API에서 노출하지 않습니다.
- **설치 검증 범위:** API 라우팅과 기존 PDF 파이프라인 구조 테스트는 로컬 CPU에서 검증했습니다. Docker 이미지 빌드, 의존성의 실제 설치, RunPod GPU 모델 적재 및 품질/성능은 아직 검증되지 않았습니다.

## 7. 배포본 내부 확인

```bash
python -m pip install fastapi 'uvicorn[standard]' python-multipart httpx pytest pymupdf pillow
python -m pytest -q deployment/test_api.py tests/test_pipeline.py
```

현재 실 모델 체크포인트를 제공받지 못했으므로 실 LLM 추론 테스트는 수행할 수 없습니다.
