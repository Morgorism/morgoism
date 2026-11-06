"""Shared text helpers."""
import re

LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*")  # hyphen/slash compounds are split into parts
HANGUL = re.compile(r"[가-힣]")

# Function words that are never technical terms. Kept short on purpose.
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "with", "by", "from", "as", "at",
    "is", "are", "was", "were", "be", "been", "this", "that", "these", "those", "it", "its", "we",
    "our", "which", "than", "such", "can", "also", "not", "but", "into", "using", "based", "et", "al",
}


def latin_tokens(text):
    return LATIN_TOKEN.findall(text or "")


def norm_term(token):
    """Case-fold and strip a simple English plural so 'Embeddings' == 'embedding'."""
    t = token.casefold()
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("es") and t[-3] in "sxz":
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def term_set(text):
    out = set()
    for tok in latin_tokens(text):
        n = norm_term(tok)
        if len(n) >= 2 and n not in STOPWORDS:
            out.add(n)
    return out


def contains_term(text, term):
    """English term present with English word boundaries; Korean particles may follow ('encoder를')."""
    pat = r"(?<![A-Za-z])" + re.escape(term) + r"(?:s|es)?(?![A-Za-z])"
    return re.search(pat, text or "", flags=re.IGNORECASE) is not None


def hangul_ratio(text):
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(HANGUL.match(c)) for c in letters) / len(letters)
