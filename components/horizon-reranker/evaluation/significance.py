"""Paired bootstrap resampling (Koehn, 2004) for two systems on the same test items.

Reports, for chrF and (optionally) the team term rate: B - A, a 95% interval, and
p = share of resamples where B is not better than A (one-sided)."""
import random

import numpy as np


def _chrf_stats(hyps, refs):
    from sacrebleu.metrics import CHRF
    m = CHRF()
    return m, np.array(m._extract_corpus_statistics(hyps, [refs]))


def paired_bootstrap(a_rows, b_rows, n=1000, seed=12345, detector=None):
    ids = [r["id"] for r in a_rows]
    b_by = {r["id"]: r for r in b_rows}
    missing = [i for i in ids if i not in b_by]
    if missing or len(ids) != len(b_rows):
        raise ValueError(f"systems must cover the same ids (missing in B: {missing[:3]})")
    b_rows = [b_by[i] for i in ids]
    for x, y in zip(a_rows, b_rows):
        if x["source_en"].strip() != y["source_en"].strip() or x["reference_ko"] != y["reference_ko"]:
            raise ValueError(f"id {x['id']}: source or reference differs between systems")
    refs = [r["reference_ko"] for r in a_rows]
    m, sa = _chrf_stats([r["prediction_ko"] for r in a_rows], refs)
    _, sb = _chrf_stats([r["prediction_ko"] for r in b_rows], refs)
    score = lambda st: m._compute_score_from_stats(list(st.sum(axis=0))).score

    terms = None
    if detector is not None:
        from .glossary import team_check
        def counts(rows):
            out = []
            for r in rows:
                c = team_check(r["source_en"], r["prediction_ko"], detector)
                out.append((c["KEEP"][0] + c["TRANSLATE"][0], c["KEEP"][1] + c["TRANSLATE"][1]))
            return np.array(out, dtype=float)
        terms = (counts(a_rows), counts(b_rows))
    rate = lambda arr: arr[:, 0].sum() / arr[:, 1].sum() if arr[:, 1].sum() else float("nan")

    rng = random.Random(seed)
    k = len(ids)
    d_chrf, d_term = [], []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        d_chrf.append(score(sb[idx]) - score(sa[idx]))
        if terms:
            d_term.append(rate(terms[1][idx]) - rate(terms[0][idx]))

    def summarize(base_a, base_b, deltas):
        deltas = np.array([d for d in deltas if d == d])
        return {"A": round(base_a, 4), "B": round(base_b, 4), "delta": round(base_b - base_a, 4),
                "ci95": [round(float(np.percentile(deltas, 2.5)), 4), round(float(np.percentile(deltas, 97.5)), 4)],
                "p_B_not_better": round(float((deltas <= 0).mean()), 4)}

    out = {"n_items": k, "n_bootstrap": n, "chrf": summarize(score(sa), score(sb), d_chrf)}
    if terms:
        out["n_term_checks"] = int(terms[0][:, 1].sum())
        out["team_term_rate"] = summarize(rate(terms[0]), rate(terms[1]), d_term)
    return out
