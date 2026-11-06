"""Team glossary (glossary_curated.json) loading and checking. Shared by evaluation and reranker.

Team format (JSON array):
  {"term": "fine-tuning", "action": "KEEP", "variants": ["fine tuning"], "translation": null,
   "translation_variants": [], "abbreviation": null, "context_sensitive": true, "auto_enforce": true, ...}
  {"term": "gradient descent", "action": "TRANSLATE", "translation": "경사하강법",
   "translation_variants": ["경사하강법", "경사 하강법"], ...}

Legacy JSONL with label/ko/forbidden is also accepted.
"""
import json
import re

ACTIONS = {"KEEP", "TRANSLATE", "BILINGUAL"}


def _entry(raw, where):
    action = str(raw.get("action") or raw.get("label") or "").upper()
    if not raw.get("term") or action not in ACTIONS:
        raise ValueError(f"{where}: need 'term' and action in {sorted(ACTIONS)}")
    ko = list(raw.get("translation_variants") or raw.get("ko") or [])
    if raw.get("translation") and raw["translation"] not in ko:
        ko.insert(0, raw["translation"])
    if action in {"TRANSLATE", "BILINGUAL"} and not ko:
        raise ValueError(f"{where}: {action} '{raw['term']}' has no Korean translation")
    surfaces = [raw["term"]] + [v for v in (raw.get("variants") or []) if v]
    return {
        "term": raw["term"],
        "action": action,
        "surfaces": surfaces,                       # English forms to look for
        "ko": ko,                                   # accepted Korean forms
        "forbidden": list(raw.get("forbidden") or []),
        "abbreviation": raw.get("abbreviation"),
        "context_sensitive": bool(raw.get("context_sensitive", False)),
        "auto_enforce": bool(raw.get("auto_enforce", True)),
        "domain": raw.get("domain"),
        "_raw": {**raw, "action": action},
    }


def load(path):
    text = open(path, encoding="utf-8").read().strip()
    if text.startswith("["):
        raws = [(r, f"{path}[{i}]") for i, r in enumerate(json.loads(text))]
    else:
        raws = [(json.loads(l), f"{path}:{n}") for n, l in enumerate(text.splitlines(), 1) if l.strip()]
    entries = [_entry(r, w) for r, w in raws]
    seen = {}
    for e in entries:
        k = e["term"].casefold()
        if k in seen:
            raise ValueError(f"duplicate glossary term: {e['term']}")
        seen[k] = e
    return entries


def _pattern(surface):
    # spaces and hyphens are interchangeable: 'fine-tuning' == 'fine tuning' == 'finetuning'
    parts = [re.escape(p) for p in re.split(r"[\s\-]+", surface.strip()) if p]
    body = r"[\s\-]?".join(parts) if len(parts) > 1 else parts[0]
    # acronyms (2+ capitals) are case-sensitive so 'AI' does not match 'ai' inside other text
    flags = 0 if re.fullmatch(r"[A-Z0-9+\-]{2,}", surface) else re.IGNORECASE
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?:s|es)?(?![A-Za-z0-9])", flags)


_CACHE = {}


def has_surface(text, surface):
    p = _CACHE.get(surface)
    if p is None:
        p = _CACHE[surface] = _pattern(surface)
    return p.search(text or "") is not None


def applicable(source, glossary):
    """Entries whose English term (or a variant) appears in the source.
    If a longer applicable term contains a shorter one ('cross-entropy loss' vs 'loss'),
    the shorter one is skipped to avoid double counting."""
    hits = [e for e in glossary if any(has_surface(source, s) for s in e["surfaces"])]
    keep = []
    for e in hits:
        t = e["term"].casefold()
        if any(o is not e and t != o["term"].casefold() and t in o["term"].casefold() for o in hits):
            continue
        keep.append(e)
    return keep


def check(source, prediction, glossary, strict_only=False):
    """Per-paragraph compliance. strict_only=True uses only auto_enforce entries."""
    out = {"keep_total": 0, "keep_ok": 0, "overtranslated": 0,
           "translate_total": 0, "translate_ok": 0,
           "bilingual_total": 0, "bilingual_ok": 0, "violations": []}
    for e in applicable(source, glossary):
        if strict_only and not e["auto_enforce"]:
            continue
        if e["action"] == "KEEP":
            out["keep_total"] += 1
            if any(has_surface(prediction, s) for s in e["surfaces"]):
                out["keep_ok"] += 1
            else:
                out["violations"].append(f"KEEP missing: {e['term']}")
            hit = next((f for f in e["forbidden"] if f in (prediction or "")), None)
            if hit:
                out["overtranslated"] += 1
                out["violations"].append(f"KEEP overtranslated: {e['term']}->{hit}")
        elif e["action"] == "TRANSLATE":
            out["translate_total"] += 1
            if any(k in (prediction or "") for k in e["ko"]):
                out["translate_ok"] += 1
            else:
                out["violations"].append(f"TRANSLATE missing ko: {e['term']}")
        else:
            out["bilingual_total"] += 1
            if _bilingual(prediction, e):
                out["bilingual_ok"] += 1
            else:
                out["violations"].append(f"BILINGUAL not paired: {e['term']}")
    return out


def _bilingual(prediction, e):
    for ko in e["ko"]:
        k = re.escape(ko)
        for s in e["surfaces"]:
            t = re.escape(s)
            if re.search(k + r"\s*\(\s*" + t, prediction or "", re.I) or re.search(t + r"\w*\s*\(\s*" + k, prediction or "", re.I):
                return True
    return False


def compliance(result):
    """Single number in [0,1] for reranking; None when no glossary entry applies."""
    total = result["keep_total"] + result["translate_total"] + result["bilingual_total"]
    if not total:
        return None
    ok = result["keep_ok"] + result["translate_ok"] + result["bilingual_ok"] - result["overtranslated"]
    return max(0.0, ok / total)


def _r(a, b):
    return round(a / b, 4) if b else None


def aggregate(per, prefix=""):
    s = lambda k: sum(p[k] for p in per)
    total = s("keep_total") + s("translate_total") + s("bilingual_total")
    ok = s("keep_ok") + s("translate_ok") + s("bilingual_ok")
    return {
        f"{prefix}glossary_compliance": _r(ok, total),
        f"{prefix}keep_preservation_rate": _r(s("keep_ok"), s("keep_total")),
        f"{prefix}overtranslation_rate": _r(s("overtranslated"), s("keep_total")),
        f"{prefix}translate_compliance_rate": _r(s("translate_ok"), s("translate_total")),
        f"{prefix}bilingual_accuracy": _r(s("bilingual_ok"), s("bilingual_total")),
        f"{prefix}glossary_hits": total,
    }


# ---------------------------------------------------------------------------
# Team rules, ported 1:1 from roooutan/horizon (최윤서):
#   detection + prompt rules : src/prepare_terminology_sft_direct.py (prepare_detector, find_terms, make_row)
#   scoring                  : src/evaluate_terminology.py (terminology_counts)
# Keep these identical to the team code so numbers are comparable across parts.
# ---------------------------------------------------------------------------
RULES_BASE = ("Translate the English academic text into natural Korean. "
              "Preserve technical proper names and any Markdown formatting.")


def _raw(e):
    return e.get("_raw", e)


def build_detector(glossary):
    lookup = {}
    for g in glossary:
        entry = _raw(g)
        options = [(entry["term"], entry["action"], False)]
        options += [(x, entry["action"], False) for x in entry.get("variants", []) or []]
        if entry.get("abbreviation"):
            options.append((entry["abbreviation"], "KEEP", True))
        for phrase, action, abbrev in options:
            if not phrase or re.search("[가-힣]", phrase):
                continue
            sensitive = abbrev or bool(re.fullmatch(r"[A-Z][A-Z0-9-]{1,9}", phrase))
            lookup.setdefault(phrase.casefold(), []).append((phrase, entry, action, sensitive))
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(x) for x in sorted(lookup, key=lambda x: (-len(x), x)))
        + r")(?![A-Za-z0-9_])", re.IGNORECASE)
    return pattern, lookup


def find_terms(source, detector):
    pattern, lookup = detector
    found, seen = [], set()
    for m in pattern.finditer(source):
        for original, entry, action, sensitive in lookup[m.group().casefold()]:
            if sensitive and m.group() != original:
                continue
            key = (m.group().casefold(), action, entry["term"])
            if key not in seen:
                seen.add(key)
                found.append({
                    "term": entry["term"], "matched_text": m.group(), "action": action,
                    "translation": entry.get("translation") if action == "TRANSLATE" else None,
                    "translation_variants": entry.get("translation_variants") or [],
                    "auto_enforce": entry.get("auto_enforce", True),
                })
            break
    return found


def rules_block(source, detector):
    """'' or '\\nFollow these terminology rules:\\n- ...' exactly as make_row builds the instruction."""
    lines, used = [], set()
    for hit in [x for x in find_terms(source, detector) if x["auto_enforce"]]:
        term = hit["matched_text"]
        key = (term.casefold(), hit["action"])
        if key in used:
            continue
        used.add(key)
        if hit["action"] == "KEEP":
            lines.append(f"- Keep '{term}' exactly in English.")
        else:
            options = list(dict.fromkeys(x for x in ([hit["translation"]] + hit["translation_variants"]) if x))
            if len(options) > 1:
                lines.append(f"- Translate '{term}' using a suitable Korean term from: {', '.join(options)}.")
            elif options:
                lines.append(f"- Translate '{term}' as '{options[0]}'.")
    return ("\nFollow these terminology rules:\n" + "\n".join(lines)) if lines else ""


def _boundary_contains(text, term):
    return bool(re.search(r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])", text, flags=re.I))


def team_check(source, prediction, detector):
    """terminology_counts for one paragraph: auto_enforce hits only; TRANSLATE also requires the
    English original to be gone."""
    out = {"KEEP": [0, 0], "TRANSLATE": [0, 0], "violations": []}
    for hit in find_terms(source, detector):
        if not hit.get("auto_enforce", True) or hit["action"] not in out:
            continue
        original = hit.get("matched_text") or hit.get("term")
        if hit["action"] == "KEEP":
            ok = _boundary_contains(prediction, original)
        else:
            cands = [c for c in [hit.get("translation")] + (hit.get("translation_variants") or [])
                     if isinstance(c, str) and c.strip()]
            if not cands:
                continue
            ok = any(c in prediction for c in cands) and not _boundary_contains(prediction, original)
        out[hit["action"]][1] += 1
        out[hit["action"]][0] += int(ok)
        if not ok:
            out["violations"].append(f"{hit['action']} fail: {original}")
    return out


def team_compliance(result):
    total = result["KEEP"][1] + result["TRANSLATE"][1]
    return None if not total else (result["KEEP"][0] + result["TRANSLATE"][0]) / total


def team_aggregate(per):
    out = {}
    for a in ("KEEP", "TRANSLATE"):
        c, t = sum(p[a][0] for p in per), sum(p[a][1] for p in per)
        out[f"team_{a.lower()}_rate"] = round(c / t, 4) if t else None
        out[f"team_{a.lower()}_n"] = t
    c = sum(p["KEEP"][0] + p["TRANSLATE"][0] for p in per)
    t = sum(p["KEEP"][1] + p["TRANSLATE"][1] for p in per)
    out["team_term_rate"] = round(c / t, 4) if t else None
    return out
