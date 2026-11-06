"""Usage (run from the folder that contains evaluation/ and reranker/)

  # 1) N-best candidates (GPU)
  python -m reranker generate --config rerank_config.json --glossary glossary_curated.json --data data/valid.jsonl --limit 200 --out runs/valid.cands.jsonl
  python -m reranker generate --config rerank_config.json --glossary glossary_curated.json --data data/test.jsonl --out runs/test.cands.jsonl
  # 2) features (GPU)
  python -m reranker score --config rerank_config.json --glossary glossary_curated.json --in runs/valid.cands.jsonl --out runs/valid.feats.jsonl
  python -m reranker score --config rerank_config.json --glossary glossary_curated.json --in runs/test.cands.jsonl --out runs/test.feats.jsonl
  # 3) weight tuning on valid (CPU is enough)
  python -m reranker tune --glossary glossary_curated.json --in runs/valid.feats.jsonl --out runs/weights.json
  # 4) final decode on test, every experiment A..E (GPU when LM/context features are re-scored)
  python -m reranker decode --config rerank_config.json --in runs/test.feats.jsonl --weights runs/weights.json --out-dir runs/test_preds
  # 6) final system: translate a new paper (Markdown) with the tuned weights
  python -m reranker translate --config rerank_config.json --glossary glossary_curated.json --weights runs/weights.json --input paper.md --out paper.ko.md
  # 5) evaluate
  python -m evaluation run --pred runs/test_preds/E_all.jsonl --testset data/test.jsonl --glossary glossary_curated.json --out results/rerank_E
"""
import argparse
import datetime as dt
import json
from pathlib import Path

from evaluation import glossary as G

from . import config as C
from . import pipeline as P
from .segment import build_docs


def _translator(cfg):
    from .model import Translator
    if not cfg.get("adapter"):
        raise SystemExit("set 'adapter' in the config or pass --adapter")
    return Translator(cfg["adapter"], cfg.get("base_model"), backward_adapter=cfg.get("backward_adapter"))


def _cfg(a):
    return C.load(getattr(a, "config", None), adapter=getattr(a, "adapter", None))


def cmd_generate(a):
    cfg = _cfg(a)
    docs = build_docs(a.data, a.limit)
    P.log(f"{len(docs)} documents, {sum(len(d['sentences']) for d in docs)} sentences")
    gloss = G.load(a.glossary) if a.glossary else None
    tr = _translator(cfg)
    P.generate(docs, tr, cfg, gloss)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    P.write_jsonl(a.out, docs)
    _meta(a.out, cfg, tr.adapter, a)


def cmd_score(a):
    cfg = _cfg(a)
    docs = P.read_jsonl(getattr(a, "in"))
    gloss = G.load(a.glossary) if a.glossary else None
    tr = _translator(cfg)
    P.score(docs, tr, cfg, gloss)
    P.write_jsonl(a.out, docs)
    _meta(a.out, cfg, tr.adapter, a)


def cmd_tune(a):
    docs = P.read_jsonl(getattr(a, "in"))
    gloss = G.load(a.glossary) if a.glossary else None
    report = P.tune(docs, gloss, a.trials, a.glossary_weight, a.seed)
    report["_settings"] = {"trials": a.trials, "glossary_weight": a.glossary_weight, "docs": len(docs),
                           "features_file": getattr(a, "in")}
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {x: v[x] for x in ("chrf", "glossary_compliance")} for k, v in report.items()
                      if not k.startswith("_")}, ensure_ascii=False, indent=2))


def cmd_decode(a):
    cfg = _cfg(a)
    docs = P.read_jsonl(getattr(a, "in"))
    report = json.loads(Path(a.weights).read_text(encoding="utf-8"))
    names = [a.experiment] if a.experiment else [k for k in P.EXPERIMENTS]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr = None
    for name in names:
        w = report[name]["weights"]
        if w is None:
            chosen = P.select_static(docs, None)
            picked = {d["doc_id"]: [c[0] for c in d["cands"]] for d in docs}
        else:
            needs_model = w.get("lm", 0) or (w.get("fwd", 0) and C.forward_uses_target_context(cfg))
            if needs_model and tr is None:
                tr = _translator(cfg)
            if needs_model:
                gloss = G.load(a.glossary) if a.glossary else None
                chosen, picked = P.decode(docs, tr, cfg, w, gloss)
            else:
                chosen = P.select_static(docs, w)
                picked = {d["doc_id"]: [P.pick(c, w) for c in d["cands"]] for d in docs}
        P.write_jsonl(out / f"{name}.jsonl", P.to_predictions(docs, chosen, picked, f"rerank:{name}", cfg.get("adapter")))
        if docs and "sentence_refs" in docs[0]:
            P.write_jsonl(out / f"{name}.sentences.jsonl",
                          P.to_sentence_predictions(docs, chosen, picked, f"rerank:{name}", cfg.get("adapter")))
        P.log(f"wrote {out / (name + '.jsonl')}")


def cmd_translate(a):
    """Final system: Markdown in -> Korean Markdown out (generate -> score -> rerank with tuned weights)."""
    from .markdown_io import parse, render
    cfg = _cfg(a)
    text = Path(a.input).read_text(encoding="utf-8")
    blocks, units = parse(text)
    if not units:
        Path(a.out).write_text(text, encoding="utf-8")
        return
    gloss = G.load(a.glossary) if a.glossary else None
    report = json.loads(Path(a.weights).read_text(encoding="utf-8"))
    w = report[a.experiment]["weights"]
    doc = {"doc_id": Path(a.input).stem, "source_en": " ".join(units), "reference_ko": "", "sentences": units}
    tr = _translator(cfg)
    if w is None:
        cfg = dict(cfg, n_samples=0)                      # baseline: greedy (+ repeat recovery) only
    P.generate([doc], tr, cfg, gloss)
    if w is None:
        chosen = [c[0]["text"] for c in doc["cands"]]
    else:
        P.score([doc], tr, cfg, gloss)
        picked, _ = P.decode([doc], tr, cfg, w, gloss)
        chosen = picked[doc["doc_id"]]
    Path(a.out).write_text(render(blocks, chosen), encoding="utf-8")
    P.log(f"translated {len(units)} units -> {a.out}")


def _meta(path, cfg, adapter, a):
    meta = {"created": dt.datetime.now().isoformat(timespec="seconds"), "adapter": adapter,
            "config": cfg, "args": {k: str(v) for k, v in vars(a).items()}}
    Path(str(path) + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(prog="reranker")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--config"); g.add_argument("--adapter")
    g.add_argument("--data", required=True); g.add_argument("--limit", type=int, default=0)
    g.add_argument("--glossary", help="needed when the forward prompt contains {rules}")
    g.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--config"); s.add_argument("--adapter"); s.add_argument("--glossary")
    s.add_argument("--in", required=True); s.add_argument("--out", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--in", required=True); t.add_argument("--glossary"); t.add_argument("--out", required=True)
    t.add_argument("--trials", type=int, default=300)
    t.add_argument("--glossary-weight", type=float, default=0.0,
                   help="objective = chrF + w*100*glossary_compliance (0 = chrF only)")
    t.add_argument("--seed", type=int, default=42)
    d = sub.add_parser("decode")
    d.add_argument("--config"); d.add_argument("--adapter")
    d.add_argument("--in", required=True); d.add_argument("--weights", required=True)
    d.add_argument("--glossary", help="needed only if the forward prompt uses {context_target} and {rules}")
    d.add_argument("--experiment", choices=list(P.EXPERIMENTS)); d.add_argument("--out-dir", required=True)
    tr_ = sub.add_parser("translate", help="final system: translate a Markdown/text file")
    tr_.add_argument("--config"); tr_.add_argument("--adapter"); tr_.add_argument("--glossary")
    tr_.add_argument("--weights", required=True); tr_.add_argument("--experiment", default="E_all",
                                                                 choices=list(P.EXPERIMENTS))
    tr_.add_argument("--input", required=True); tr_.add_argument("--out", required=True)
    a = p.parse_args(argv)
    {"generate": cmd_generate, "score": cmd_score, "tune": cmd_tune, "decode": cmd_decode,
     "translate": cmd_translate}[a.cmd](a)


if __name__ == "__main__":
    main()
