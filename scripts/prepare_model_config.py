#!/usr/bin/env python3
"""Build a GPU-friendly copy of the original Stage4 + E_all config.

Adapters stay on user-mounted /workspace storage; no model files are copied
into the Docker image. E_all scoring is unchanged; batch sizes are smaller.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "components/horizon-reranker/configs/final.json"


def adapter_path(value: str, label: str) -> str:
    path = Path(value).expanduser().resolve()
    if not (path / "adapter_config.json").is_file():
        raise SystemExit(f"{label}: adapter_config.json missing: {path}")
    w = path / "adapter_model.safetensors"
    if not w.is_file() or w.stat().st_size < 1_000_000:
        raise SystemExit(f"{label}: model weights missing or Git LFS pointer: {w}")
    return str(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage4", required=True)
    p.add_argument("--bomin", required=True)
    p.add_argument("--out", default="/workspace/rerank_config.json")
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--score-batch-size", type=int, default=2)
    p.add_argument("--gen-batch-size", type=int, default=1)
    args = p.parse_args()
    if args.n_samples < 1 or args.score_batch_size < 1 or args.gen_batch_size < 1:
        p.error("sample and batch values must be positive")
    cfg = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    cfg["adapter"] = adapter_path(args.stage4, "Stage4")
    cfg["backward_adapter"] = adapter_path(args.bomin, "Bomin")
    cfg["n_samples"] = args.n_samples
    cfg["score_batch_size"] = args.score_batch_size
    cfg["gen_batch_size"] = args.gen_batch_size
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved config: {out}")
    print("Stage4:", cfg["adapter"])
    print("Bomin:", cfg["backward_adapter"])
    print("Samples:", cfg["n_samples"], "score batch:", cfg["score_batch_size"])


if __name__ == "__main__":
    main()
