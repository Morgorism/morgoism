"""Output health checks that explain bad scores (empty, untranslated, truncated, looping)."""
import re

from .text import hangul_ratio

REPEAT = re.compile(r"(.{8,}?)\1{3,}", re.S)


def check(source, prediction, record):
    pred = prediction or ""
    issues = []
    if not pred.strip():
        issues.append("empty")
    elif hangul_ratio(pred) < 0.3:
        issues.append("mostly_not_korean")
    if record.get("hit_max_tokens"):
        issues.append("hit_max_tokens")
    if record.get("status", "ok") != "ok":
        issues.append(f"status:{record.get('status')}")
    if REPEAT.search(pred):
        issues.append("repetition")
    ratio = len(pred) / max(1, len(source or ""))
    return {"len_ratio": round(ratio, 3), "issues": issues}


def aggregate(per):
    n = len(per) or 1
    flat = [i.split(":")[0] for p in per for i in p["issues"]]
    return {
        "empty": flat.count("empty"),
        "mostly_not_korean": flat.count("mostly_not_korean"),
        "hit_max_tokens": flat.count("hit_max_tokens"),
        "repetition": flat.count("repetition"),
        "non_ok_status": flat.count("status"),
        "clean_rate": round(sum(not p["issues"] for p in per) / n, 4),
    }
