"""Structure metrics: Markdown elements and spans that must survive translation unchanged."""
import re
from collections import Counter

DEFAULT_PLACEHOLDER = r"⟦[^⟧]+⟧"   # change with --placeholder-regex to match the preprocessing format

INVARIANT_PATTERNS = {
    "code_block": re.compile(r"```.*?```", re.S),
    "inline_code": re.compile(r"`[^`\n]+`"),
    "math_display": re.compile(r"\$\$.+?\$\$", re.S),
    "math_inline": re.compile(r"(?<!\$)\$(?!\$)[^$\n]+?\$(?!\$)"),
    "url": re.compile(r"https?://[^\s)\]>]+"),
    "citation": re.compile(r"\[\d+(?:\s*[,–-]\s*\d+)*\]"),
}

BLOCK_PATTERNS = {
    "heading": re.compile(r"^\s{0,3}#{1,6}\s", re.M),
    "bullet": re.compile(r"^\s*[-*+]\s+", re.M),
    "ordered": re.compile(r"^\s*\d+[.)]\s+", re.M),
    "table_row": re.compile(r"^\s*\|.*\|\s*$", re.M),
    "blockquote": re.compile(r"^\s*>\s?", re.M),
}

NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*%?")


def _spans(pattern, text):
    return Counter(m.group(0) for m in pattern.finditer(text or ""))


def check(source, prediction, placeholder_regex=DEFAULT_PLACEHOLDER):
    patterns = dict(INVARIANT_PATTERNS)
    patterns["placeholder"] = re.compile(placeholder_regex)

    errors = []
    invariant_total = invariant_kept = 0
    for name, pat in patterns.items():
        src = _spans(pat, source)
        if not src:
            continue
        pred = _spans(pat, prediction)
        for span, n in src.items():
            invariant_total += n
            kept = min(n, pred.get(span, 0))
            invariant_kept += kept
            if kept < n:
                errors.append(f"{name} changed/missing: {span[:40]}")
        extra = sum((pred - src).values())
        if extra:
            errors.append(f"{name} extra in prediction: {extra}")

    block_mismatch = 0
    for name, pat in BLOCK_PATTERNS.items():
        a, b = len(pat.findall(source or "")), len(pat.findall(prediction or ""))
        if a != b:
            block_mismatch += 1
            errors.append(f"{name} count {a}->{b}")

    src_nums = Counter(NUMBER.findall(source or ""))
    pred_nums = Counter(NUMBER.findall(prediction or ""))
    num_total = sum(src_nums.values())
    num_kept = sum(min(n, pred_nums.get(k, 0)) for k, n in src_nums.items())

    return {
        "invariant_total": invariant_total,
        "invariant_kept": invariant_kept,
        "block_mismatch": block_mismatch,
        "numbers_total": num_total,
        "numbers_kept": num_kept,
        "structure_pass": invariant_kept == invariant_total and block_mismatch == 0,
        "errors": errors,
    }


def aggregate(per):
    inv_t = sum(p["invariant_total"] for p in per)
    num_t = sum(p["numbers_total"] for p in per)
    return {
        "structure_pass_rate": round(sum(p["structure_pass"] for p in per) / len(per), 4) if per else None,
        "invariant_preservation": round(sum(p["invariant_kept"] for p in per) / inv_t, 4) if inv_t else None,
        "number_preservation": round(sum(p["numbers_kept"] for p in per) / num_t, 4) if num_t else None,
        "paragraphs_with_invariants": sum(p["invariant_total"] > 0 for p in per),
    }
