# 선택 배포 안내 — RunPod GPU + Docker

> **우선 읽을 문서: [README.md](README.md)**. 이 문서는 개인 NVIDIA GPU가 없는 사용자가 **직접 비용을 지불하는 GPU 서버에 Docker API를 올리고 싶을 때** 참고하는 보조 문서입니다. GPU 서버가 자동 제공되거나 공용 웹서비스가 개설되는 것은 아닙니다.

## 현재 지원 범위

- **지원:** MinerU ZIP 또는 Markdown을 API에 올려 Stage4 + Bomin + E_all로 번역 후 Markdown/PDF 다운로드.
- **미지원:** 원본 PDF를 API에 바로 업로드. 현재 Dockerfile에는 `mineru-kit`이 없으며, `.pdf` 요청은 HTTP 503을 반환합니다.
- **원본 PDF 사용:** 사용자 PC에 MinerU Standard를 별도로 설치하여 PDF를 MinerU ZIP으로 변환한 뒤 서버에 업로드해야 합니다. GPU 서버에 MinerU가 설치되어 있지 않아도 됩니다.
- **체크포인트:** 실제 Stage4/Bomin 어댑터 파일은 Docker 이미지/저장소에 포함되어 있지 않습니다.

## 절차 요약

1. 본인이 관리하는 RunPod GPU Pod를 만들고, NVIDIA CUDA 드라이버와 Docker 사용 가능 여부를 확인합니다. 24GB GPU는 작은 후보 설정의 실험 수준이며, 긴 논문과 여러 후보 계산은 48GB 이상을 권장합니다. 모델 GPU 메모리 소모는 실제 실행으로 확인해야 합니다.
2. 프로젝트를 Pod에 복제하거나 Docker Hub에 직접 빌드한 이미지를 배포합니다. API 컨테이너에 GPU 패스스루가 활성화되어야 합니다.
3. `/workspace/stage4/`, `/workspace/bomin/`에 실제 LoRA 파일을 배치하고 `scripts/prepare_model_config.py`로 `/workspace/rerank_config.json`을 생성합니다. Git LFS의 몇 바이트짜리 포인터 파일은 사용할 수 없습니다.
4. 환경 변수 `API_KEY`, `MODEL_CONFIG=/workspace/rerank_config.json`, `HF_HOME=/workspace/hf_cache`, `JOB_DIR=/workspace/jobs`를 지정합니다. GPU 워커/서버 프로세스는 한 개만 사용합니다.
5. API를 시작하여 `/health`로 서버 프로세스를 확인하고, 인증된 `/ready`에서 모델·GPU·XeLaTeX 상태를 확인합니다. **`pdf_parser_available: false`는 기본 이미지에서 정상입니다.**
6. 사용자 PC에서 `mineru-kit parse paper.pdf -o paper.zip --format zip --tier standard`로 ZIP을 만든 후, 해당 ZIP을 `/jobs?output_format=pdf`로 업로드하여 처리 결과를 다운로드합니다.

## API 접속 및 보안

- 외부 GPU 서버에서 API를 접속하려면 HTTPS와 안전한 인증/접근 제어를 구성하세요. 프로젝트의 `compose.yaml`은 기본적으로 **127.0.0.1:8000**에만 바인딩되어 있습니다. 그대로 두면 다른 컴퓨터에서 바로 연결할 수 없습니다.
- RunPod HTTP 프록시를 사용하는 경우 **실제 Pod의 HTTP 포트와 프록시 주소**를 확인해야 합니다. Dockerfile의 `EXPOSE 8000`만으로 외부 접근 경로가 자동 생성되지는 않습니다.
- 반드시 임의의 `API_KEY`를 설정하고, 해당 값이나 LoRA 파일을 공개 GitHub/이미지에 넣지 마세요.
- 작업이 끝나면 GPU Pod를 중지·삭제하여 비용을 관리하세요. 별도의 Network Volume은 Pod 삭제 후에도 저장 요금이 발생할 수 있습니다.

## 기술적 제한

- MinerU는 배포용 Docker 이미지에 설치하지 않는 것을 의도한 구성입니다. 호스트에 설치한 MinerU CLI가 Docker 내부에 자동으로 나타나지 않습니다.
- 실제 Docker 이미지 빌드와 이 배포 이미지 기반 GPU 추론의 완전한 성공은 현재 자동 테스트로 검증되지 않았습니다. `/translate` 또는 `/jobs`로 실모델 추론을 실행해 검증해야 합니다.
- 2단 편집, 복잡한 표와 LaTeX 수식의 완벽한 원본 레이아웃 복원은 보장하지 않습니다.

자세한 사용 명령과 로컬 PDF 자동 제출 스크립트는 [README.md](README.md)의 **방법 A**를 따르세요.
