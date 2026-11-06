"""Term metrics.

Two modes:
1. Reference-based (no glossary needed): which English terms from the source did the
   reference keep in English, and did the prediction keep the same ones?
2. Glossary-based (policy): KEEP / TRANSLATE / BILINGUAL labels from the team glossary.

Glossary: the team file glossary_curated.json (see evaluation/glossary.py).
"""
from .text import contains_term, latin_tokens, norm_term, term_set

from . import glossary as _g


def load_glossary(path):
    return _g.load(path)


def _pred_kept(pred_text):
    return {norm_term(t) for t in latin_tokens(pred_text)}


def reference_based(source, reference, prediction):
    src = term_set(source)
    ref_kept = src & _pred_kept(reference)   # same tokenizer on both sides keeps the metric symmetric
    pred_terms = _pred_kept(prediction)
    pred_kept = {t for t in src if t in pred_terms}
    agree = ref_kept & pred_kept
    return {
        "ref_kept": len(ref_kept),
        "pred_kept": len(pred_kept),
        "agree": len(agree),
        "missed_terms": sorted(ref_kept - pred_kept),       # reference kept English, prediction translated it
        "extra_terms": sorted(pred_kept - ref_kept),        # prediction left English, reference translated it
    }


def glossary_based(source, prediction, glossary):
    return _g.check(source, prediction, glossary)


def _ratio(a, b):
    return round(a / b, 4) if b else None


def aggregate_reference(per):
    agree = sum(p["agree"] for p in per)
    ref = sum(p["ref_kept"] for p in per)
    pred = sum(p["pred_kept"] for p in per)
    recall = agree / ref if ref else None
    precision = agree / pred if pred else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    return {
        "term_keep_recall_vs_ref": None if recall is None else round(recall, 4),
        "term_keep_precision_vs_ref": None if precision is None else round(precision, 4),
        "term_keep_f1_vs_ref": None if f1 is None else round(f1, 4),
        "terms_kept_by_ref": ref,
        "terms_kept_by_pred": pred,
    }


def aggregate_glossary(per, strict_per=None):
    out = _g.aggregate(per)
    if strict_per is not None:
        out.update(_g.team_aggregate(strict_per))      # same rules as 최윤서 evaluate_terminology.py
    return out
