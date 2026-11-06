#!/usr/bin/env python3
"""Create an E_all config pointing at pretrained adapters (no training)."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'components/horizon-reranker/configs/final.json'


def require_adapter(directory: Path, name: str) -> None:
    if not (directory / 'adapter_config.json').is_file():
        raise SystemExit(f'{name}: adapter_config.json 파일이 없습니다: {directory}')
    file = directory / 'adapter_model.safetensors'
    if not file.is_file() or file.stat().st_size < 1_000_000:
        raise SystemExit(f'{name}: 실제 .safetensors 가중치가 없습니다 (LFS 포인터는 사용 불가): {file}')


def main():
    parser = argparse.ArgumentParser(description='Stage4+CS 두 LoRA 연결 설정 생성')
    parser.add_argument('--stage4', type=Path, required=True, help='Stage4 Replay adapter 폴더')
    parser.add_argument('--bomin', type=Path, required=True, help='Bomin CS adapter 폴더')
    parser.add_argument('--out', type=Path, default=ROOT / 'rerank_config.json')
    args = parser.parse_args()
    stage4 = args.stage4.expanduser().resolve()
    bomin = args.bomin.expanduser().resolve()
    require_adapter(stage4, 'Stage4')
    require_adapter(bomin, 'Bomin CS')
    cfg = json.loads(TEMPLATE.read_text(encoding='utf-8-sig'))
    cfg['adapter'] = str(stage4)
    cfg['backward_adapter'] = str(bomin)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    print('완료:', args.out)
    print('Stage4:', stage4)
    print('Bomin CS:', bomin)
    print('후보 생성: greedy 1 + sampling 6, Reranker: E_all')


if __name__ == '__main__':
    main()
