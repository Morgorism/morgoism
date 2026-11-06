#!/usr/bin/env bash
# RunPod 첫 준비: 저장소·어댑터·데이터를 /workspace에 받고 rerank_config.json을 자동으로 만듭니다.
# usage: bash setup_runpod.sh
set -euo pipefail
W=/workspace
CODE=$(cd "$(dirname "$0")" && pwd)
export HF_HOME=$W/hf_cache                      # 모델 파일을 볼륨 디스크에 저장 (Pod를 껐다 켜도 유지)
grep -q "HF_HOME=$W/hf_cache" ~/.bashrc 2>/dev/null || echo "export HF_HOME=$W/hf_cache" >> ~/.bashrc

echo "== 1/6 도구 설치"
apt-get update -qq && apt-get install -y -qq git-lfs tmux > /dev/null
git lfs install > /dev/null
pip install -q -r "$CODE/requirements.txt"

echo "== 2/6 팀 저장소 받기"
cd $W
[ -d finetuning_Bomin ] || git clone -q https://github.com/nengrafi/finetuning_Bomin.git
(cd finetuning_Bomin && git lfs pull --include="outputs/hymt2_cs/checkpoint-1520/*")
[ -d horizon ] || git clone -q https://github.com/roooutan/horizon.git
[ -d horizon-context ] || git clone -q https://github.com/jjha0/horizon-context.git

echo "== 3/6 Stage4(jjha) 어댑터 받기"
mkdir -p $W/stage4 && cd $W/stage4
ASSET=context_replay_release.tar.gz
SHA=9e552bf3ae484539893bd00e65fa1aa1b06290180f574780644b71ca704475ad
if ! find . -name adapter_config.json | grep -q .; then
  for f in *.tar.gz *.tgz; do [ -f "$f" ] && tar -xzf "$f" && echo "  압축 해제: $f"; done   # 직접 올린 파일 우선
fi
if ! find . -name adapter_config.json | grep -q .; then
  wget -q --show-progress -O $ASSET https://github.com/jjha0/horizon-context/releases/download/v1.0.0/$ASSET
  echo "$SHA  $ASSET" | sha256sum -c - || { echo "!! 체크섬 불일치: 다운로드가 깨졌습니다. 다시 실행하세요"; rm -f $ASSET; exit 1; }
  tar -xzf $ASSET
fi
STAGE4=$(dirname "$(find $W/stage4 -name adapter_config.json | head -1)")
CS=$W/finetuning_Bomin/outputs/hymt2_cs/checkpoint-1520

echo "== 4/6 어댑터 파일 확인"
for d in "$STAGE4" "$CS"; do
  size=$(stat -c %s "$d/adapter_model.safetensors")
  [ "$size" -gt 1000000 ] || { echo "!! $d 의 가중치가 비어 있습니다(LFS 미다운로드). git lfs pull을 확인하세요"; exit 1; }
  echo "  OK $d ($((size/1000000)) MB)"
done

echo "== 5/6 베이스 모델 미리 받기 (약 15GB, 한 번만)"
python -c "from huggingface_hub import snapshot_download; snapshot_download('tencent/Hy-MT2-7B')" \
  || { echo "!! 모델 다운로드 실패. 접근 권한이 필요하면: huggingface-cli login 후 다시 실행"; exit 1; }

echo "== 6/6 설정 파일 만들기"
cd "$CODE"
python - "$STAGE4" "$CS" <<'PY'
import json, sys
cfg = json.load(open("configs/final.json", encoding="utf-8"))
cfg["adapter"], cfg["backward_adapter"] = sys.argv[1], sys.argv[2]
json.dump(cfg, open("rerank_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(cfg, ensure_ascii=False, indent=2))
PY
mkdir -p $W/data
cp $W/finetuning_Bomin/data/valid.jsonl $W/finetuning_Bomin/data/test.jsonl $W/data/
cp "$W/horizon/glossary/찐막glossary_curated.json" $W/data/glossary_curated.json
python -m pytest -q tests
echo "준비 완료. 다음: bash run_rerank.sh /workspace/data /workspace/data/glossary_curated.json rerank_config.json 20"
