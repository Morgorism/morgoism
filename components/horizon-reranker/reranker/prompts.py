"""Forward-prompt builders.

style "template"     : cfg["prompts"]["forward"].format(source, context_source, context_target, rules)
                       {rules} = 최윤서 make_row rules (evaluation.glossary.rules_block)
style "jjha_stage4"  : 1:1 port of jjha0/horizon-context prompts.py `prompt(row, 'context', gloss)`
                       - context = previous English sentences only (prev_en, up to `context_size`)
                       - rules   = conservative subset: auto_enforce and not context_sensitive
                       - no prev_en -> plain prompt
Keep jjha_* functions identical to the team code (verified by tests/test_jjha_parity.py).
"""
import re

from evaluation import glossary as G

JJHA_BASE = ("Translate the English academic text into natural Korean. "
             "Preserve technical proper names and any Markdown formatting.")


def jjha_terms(source, gloss):
    hits = []
    seen = set()
    for e in gloss:
        if not e.get("auto_enforce", True) or e.get("context_sensitive", False):
            continue
        variants = [(e["term"], e["action"]), *((v, e["action"]) for v in e.get("variants", []))]
        if e.get("abbreviation"):
            variants.append((e["abbreviation"], "KEEP"))
        for phrase, action in variants:
            if not phrase or re.search("[가-힣]", phrase):
                continue
            sensitive = bool(re.fullmatch(r"[A-Z][A-Z0-9-]{1,9}", phrase))
            pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(phrase) + r"(?![A-Za-z0-9_])", 0 if sensitive else re.I)
            for m in pattern.finditer(source):
                key = (m.group().casefold(), action)
                if key in seen:
                    continue
                seen.add(key)
                allowed = list(dict.fromkeys(v for v in [e.get("translation"), *e.get("translation_variants", [])] if v))
                if action == "TRANSLATE" and not allowed:
                    continue
                hits.append({"source": m.group(), "action": action, "allowed": allowed})
    return sorted(hits, key=lambda h: (h["source"].casefold(), h["action"]))


def jjha_prompt(current_en, prev_en, gloss):
    rules = []
    for h in jjha_terms(current_en, gloss):
        if h["action"] == "KEEP":
            rules.append(f"- Keep '{h['source']}' exactly in English.")
        else:
            rules.append(f"- Translate '{h['source']}' using a suitable Korean term from: {', '.join(h['allowed'])}.")
    instruction = JJHA_BASE + ("\nFollow these terminology rules:\n" + "\n".join(rules) if rules else "")
    if not prev_en.strip():
        return instruction + "\n\n" + current_en
    return ("[Background Information]\n" + prev_en + "\n\n" + instruction +
            "\nUse the background only to resolve meaning. Translate only [Source Text]; "
            "do not translate or summarize the background. Output only the translation.\n"
            "[Source Text]\n" + current_en)


def raw_glossary(gloss):
    """jjha code expects the team JSON fields; evaluation.glossary entries keep them in _raw."""
    return [g.get("_raw", g) for g in gloss]


class ForwardPrompt:
    def __init__(self, cfg, gloss=None):
        self.style = cfg.get("prompt_style", "template")
        self.template = cfg["prompts"]["forward"]
        self.k = cfg["context_size"]
        self.gloss = gloss
        if self.style == "jjha_stage4" and not gloss:
            raise SystemExit("prompt_style jjha_stage4 needs --glossary (strict_glossary.json or glossary_curated.json)")
        if self.style == "template" and "{rules}" in self.template and not gloss:
            raise SystemExit("forward prompt uses {rules}: pass --glossary")
        self.raw = raw_glossary(gloss) if gloss else None
        self.det = G.build_detector(gloss) if gloss else None
        self._cache = {}

    def uses_target_context(self):
        return self.style == "template" and "{context_target}" in self.template

    def __call__(self, doc, i, chosen):
        lo = max(0, i - self.k)
        prev_src = " ".join(doc["sentences"][lo:i])
        if "prev_en" in doc:                               # jjha sentence data: use the given context as is
            prev_src = doc["prev_en"][i]
        if self.style == "jjha_stage4":
            return jjha_prompt(doc["sentences"][i], prev_src, self.raw)
        key = doc["sentences"][i]
        if key not in self._cache:
            self._cache[key] = G.rules_block(key, self.det) if self.det else ""
        return self.template.format(source=doc["sentences"][i], context_source=prev_src,
                                    context_target=" ".join(chosen[lo:i]), rules=self._cache[key], target="")


def abnormal_reasons(text, hit_cap):
    """1:1 port of jjha infer_stage4.abnormal_reasons (repeat-recovery trigger)."""
    from collections import Counter
    out = []
    if not text.strip():
        out.append("empty")
    if hit_cap:
        out.append("token_cap")
    if "�" in text:
        out.append("replacement_character")
    words = re.findall(r"\S+", text)
    counts = Counter(tuple(words[i:i + 6]) for i in range(max(0, len(words) - 5)))
    if counts and max(counts.values()) >= 3:
        out.append("repeated_six_word_phrase")
    return out
