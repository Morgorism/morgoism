"""generate -> score -> tune -> decode.

Context handling
- generate/score use a *draft* context: the greedy candidates of the previous sentences.
- decode re-scores context-dependent features (LM, and forward if its prompt uses {context_target})
  along the path actually chosen by the reranker, sentence by sentence.
- tune uses the draft-context features, which is the standard approximation for fast weight search.
"""
import json
import math
import random
import sys
import time

from evaluation import glossary as G
from evaluation import structure

from . import config as C
from .prompts import ForwardPrompt, abnormal_reasons

FEATURES = ["fwd", "bwd", "lm", "glossary", "len"]

# Ablations reported by `tune` (weights outside the set are fixed to 0).
EXPERIMENTS = {
    "A_greedy": None,                 # first (greedy) candidate, no reranking
    "B_fwd": ["fwd"],
    "C_fwd_bwd": ["fwd", "bwd"],
    "D_fwd_bwd_lm": ["fwd", "bwd", "lm"],
    "E_all": ["fwd", "bwd", "lm", "glossary", "len"],
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def fmt(template, source="", context_source="", context_target="", target="", rules=""):
    return template.format(source=source, context_source=context_source,
                           context_target=context_target, target=target, rules=rules)



def _ctx(doc, i, chosen, k):
    lo = max(0, i - k)
    return " ".join(doc["sentences"][lo:i]), " ".join(chosen[lo:i])


# ---------------------------------------------------------------- generate
def generate(docs, tr, cfg, gloss=None):
    """Adds doc['cands'][i] = list of {"text","kind","hit_max_tokens","abnormal"} with the baseline
    candidate first, and doc['prompts'][i] = forward prompt used (draft context)."""
    fp = ForwardPrompt(cfg, gloss)
    for d in docs:
        d["cands"] = [None] * len(d["sentences"])
        d["prompts"] = [None] * len(d["sentences"])
    draft = {d["doc_id"]: [] for d in docs}
    max_len = max(len(d["sentences"]) for d in docs)
    for i in range(max_len):                               # position-synchronous so context is available
        active = [d for d in docs if i < len(d["sentences"])]
        log(f"generate position {i + 1}/{max_len}: {len(active)} sentences")
        for b in range(0, len(active), cfg["gen_batch_size"]):
            batch = active[b:b + cfg["gen_batch_size"]]
            users = [fp(d, i, draft[d["doc_id"]]) for d in batch]
            outs = tr.generate(users, cfg["n_samples"], cfg["temperature"], cfg["top_p"], cfg["max_new_tokens"],
                               seed=cfg["seed"] + i * 100003 + b, fixed_budget=cfg["fixed_budget"])
            for c in (c for cand in outs for c in cand):
                c["abnormal"] = abnormal_reasons(c["text"], c["hit_max_tokens"])
            if cfg["repeat_recovery"]:
                redo = [j for j, cand in enumerate(outs) if cand[0]["abnormal"]]
                if redo:
                    again = tr.generate([users[j] for j in redo], 0, 0, 0, cfg["max_new_tokens"],
                                        fixed_budget=cfg["fixed_budget"], retry=True)
                    for j, r in zip(redo, again):
                        r[0]["abnormal"] = abnormal_reasons(r[0]["text"], r[0]["hit_max_tokens"])
                        r[0]["initial_abnormal"] = outs[j][0]["abnormal"]
                        outs[j].insert(0, r[0])            # baseline = retry result, as in infer_stage4
            for d, u, cand in zip(batch, users, outs):
                seen, uniq = set(), []
                for c in cand:
                    if c["text"] and c["text"] not in seen:
                        seen.add(c["text"])
                        uniq.append(c)
                if not uniq:
                    uniq = [cand[0]]
                d["cands"][i] = uniq
                d["prompts"][i] = u
                draft[d["doc_id"]].append(uniq[0]["text"])
    return docs


# ---------------------------------------------------------------- features
def _context_free_features(docs, tr, cfg, gloss):
    bwd_t = cfg["prompts"]["backward"]
    jobs = []
    for d in docs:
        for i, cands in enumerate(d["cands"]):
            for c in cands:
                jobs.append((c, d["sentences"][i]))
    det = G.build_detector(gloss) if gloss else None
    log(f"backward scoring {len(jobs)} candidates")
    scores = tr.score_answers([(fmt(bwd_t, target=c["text"]), src) for c, src in jobs], cfg["score_batch_size"],
                              adapter_name="backward" if tr.has_backward else "default")
    for (c, src), (s, n) in zip(jobs, scores):
        c["bwd"] = s / max(n, 1)
        c["len"] = math.log(max(len(c["text"]), 1) / max(len(src), 1))
        if det:
            comp = G.team_compliance(G.team_check(src, c["text"], det))   # 최윤서 rules (auto_enforce terms)
            c["glossary"] = 0.0 if comp is None else comp
        else:
            c["glossary"] = 0.0
        st = structure.check(src, c["text"], cfg["placeholder_regex"])
        c["structure_ok"] = st["structure_pass"]


def _context_features(docs, tr, cfg, chosen_by_doc, positions=None, fp=None):
    """fwd and lm for the given context path. positions=None -> all sentences.
    The forward prompt is rebuilt only when it depends on chosen translations; otherwise the
    prompt stored at generation time is reused."""
    k = cfg["context_size"]
    fwd_jobs, lm_jobs = [], []
    for d in docs:
        chosen = chosen_by_doc[d["doc_id"]]
        rng = range(len(d["sentences"])) if positions is None else [p for p in positions if p < len(d["sentences"])]
        for i in rng:
            _, ct = _ctx(d, i, chosen, k)
            user = fp(d, i, chosen) if fp is not None else d["prompts"][i]
            for c in d["cands"][i]:
                fwd_jobs.append((c, user))
                lm_jobs.append((c, ct))
    if fwd_jobs:
        fs = tr.score_answers([(u, c["text"]) for c, u in fwd_jobs], cfg["score_batch_size"])
        for (c, _), (s, n) in zip(fwd_jobs, fs):
            c["fwd"] = s / max(n, 1)
        ls = tr.score_continuations([(ct, c["text"]) for c, ct in lm_jobs], cfg["score_batch_size"],
                                    cfg["lm_max_context_tokens"], use_adapter=cfg["lm_use_adapter"])
        for (c, _), (s, n) in zip(lm_jobs, ls):
            c["lm"] = s / max(n, 1)


def score(docs, tr, cfg, gloss):
    _context_free_features(docs, tr, cfg, gloss)
    for d in docs:
        d["prompts"] = d.get("prompts") or [None] * len(d["sentences"])
        if any(p is None for p in d["prompts"]):
            raise SystemExit("candidates file has no stored prompts; regenerate with this version")
    draft = {d["doc_id"]: [c[0]["text"] for c in d["cands"]] for d in docs}
    log("forward + LM scoring with draft context")
    _context_features(docs, tr, cfg, draft)
    return docs


# ---------------------------------------------------------------- selection
def pick(cands, w):
    pool = [c for c in cands if c.get("structure_ok", True) and not c.get("abnormal")] or cands
    return max(pool, key=lambda c: sum(w.get(f, 0.0) * c.get(f, 0.0) for f in FEATURES))


def select_static(docs, w):
    """Selection with precomputed (draft-context) features; w=None -> greedy candidate."""
    return {d["doc_id"]: [(c[0] if w is None else pick(c, w))["text"] for c in d["cands"]] for d in docs}


# ---------------------------------------------------------------- objective
def objective(docs, chosen, gloss, glossary_weight):
    from sacrebleu.metrics import CHRF
    hyps = [" ".join(chosen[d["doc_id"]]) for d in docs]
    refs = [d["reference_ko"] for d in docs]
    chrf = CHRF().corpus_score(hyps, [refs]).score
    comp = None
    if gloss:
        det = G.build_detector(gloss)
        comp = G.team_aggregate([G.team_check(d["source_en"], h, det) for d, h in zip(docs, hyps)])["team_term_rate"]
    value = chrf + (glossary_weight * 100 * comp if comp is not None else 0.0)
    return value, chrf, comp


def tune(docs, gloss, trials, glossary_weight, seed):
    """Random search per experiment; fwd weight is fixed to 1 (scale invariance)."""
    rng = random.Random(seed)
    ranges = {"bwd": (0.0, 2.0), "lm": (0.0, 2.0), "glossary": (0.0, 5.0), "len": (-2.0, 2.0)}
    report = {}
    for name, feats in EXPERIMENTS.items():
        if feats is None:
            v, chrf, comp = objective(docs, select_static(docs, None), gloss, glossary_weight)
            report[name] = {"weights": None, "objective": v, "chrf": chrf, "glossary_compliance": comp}
            continue
        best = None
        grid = [{f: (1.0 if f == "fwd" else 0.0) for f in FEATURES}]
        for _ in range(trials if len(feats) > 1 else 0):
            w = {f: 0.0 for f in FEATURES}
            w["fwd"] = 1.0
            for f in feats:
                if f != "fwd":
                    w[f] = round(rng.uniform(*ranges[f]), 3)
            grid.append(w)
        for w in grid:
            v, chrf, comp = objective(docs, select_static(docs, w), gloss, glossary_weight)
            if best is None or v > best["objective"]:
                best = {"weights": w, "objective": v, "chrf": chrf, "glossary_compliance": comp}
        report[name] = best
        log(f"{name}: chrF={best['chrf']:.2f} glossary={best['glossary_compliance']} weights={best['weights']}")
    return report


# ---------------------------------------------------------------- decode
def decode(docs, tr, cfg, w, gloss=None):
    """Left-to-right reranking with the real chosen context (Yu et al. greedy/N-best search)."""
    chosen = {d["doc_id"]: [] for d in docs}
    picked = {d["doc_id"]: [] for d in docs}
    max_len = max(len(d["sentences"]) for d in docs)
    need_ctx = cfg["context_size"] > 0
    fp = ForwardPrompt(cfg, gloss) if C.forward_uses_target_context(cfg) else None
    for i in range(max_len):
        active = [d for d in docs if i < len(d["sentences"])]
        if need_ctx and i > 0 and (w.get("lm", 0) or (w.get("fwd", 0) and C.forward_uses_target_context(cfg))):
            _context_features(active, tr, cfg, chosen, positions=[i], fp=fp)
        for d in active:
            c = pick(d["cands"][i], w)
            chosen[d["doc_id"]].append(c["text"])
            picked[d["doc_id"]].append(c)
    return chosen, picked


def to_sentence_predictions(docs, chosen, picked, system, adapter):
    """One row per sentence; needs sentence-level references (jjha sentence-aligned data)."""
    rows = []
    for d in docs:
        if "sentence_refs" not in d:
            raise SystemExit("sentence-level output needs sentence-aligned data (current_en/target_ko rows)")
        for sid, src, ref, text, c in zip(d["sentence_ids"], d["sentences"], d["sentence_refs"],
                                          chosen[d["doc_id"]], picked[d["doc_id"]]):
            rows.append({"id": sid, "paper_id": d["doc_id"], "source_en": src, "reference_ko": ref,
                         "prediction_ko": text, "status": "ok", "hit_max_tokens": bool(c.get("hit_max_tokens")),
                         "retry_used": c.get("kind") == "retry", "system": system, "adapter_path": adapter})
    return rows


def to_predictions(docs, chosen, picked, system, adapter):
    rows = []
    for d in docs:
        rows.append({"id": d["doc_id"], "source_en": d["source_en"], "reference_ko": d["reference_ko"],
                     "prediction_ko": " ".join(chosen[d["doc_id"]]), "status": "ok",
                     "hit_max_tokens": any(c.get("hit_max_tokens") for c in picked[d["doc_id"]]),
                     "n_sentences": len(d["sentences"]), "system": system, "adapter_path": adapter})
    return rows
