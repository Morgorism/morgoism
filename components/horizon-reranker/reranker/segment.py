"""Split each test/valid record (one abstract = one document) into sentences.

The current data has one paragraph per paper, so the document context the reranker uses is the
previous sentences of the same abstract. If multi-paragraph documents arrive later, records that
share a doc key are concatenated in 'order' and the same code applies.
"""
import json
import re

ABBREV = {"e.g", "i.e", "et al", "etc", "fig", "figs", "eq", "eqs", "vs", "cf", "no", "sec", "approx",
          "dr", "mr", "ms", "prof", "resp", "al", "ref", "refs", "tab", "ch", "vol", "pp", "inc", "ltd", "st"}
BOUNDARY = re.compile(r'(?<=[.!?])["\')\]]*\s+(?=["\'(\[]?[A-Z0-9])')


def split_sentences(text, min_words=4):
    text = re.sub(r"[ \t]+", " ", text.strip())
    out = []
    for para in [p for p in re.split(r"\n\s*\n", text) if p.strip()]:
        start = 0
        pieces = []
        for m in BOUNDARY.finditer(para):
            chunk = para[start:m.start()].strip()
            last = re.search(r"([A-Za-z][A-Za-z. ]{0,6})\.$", chunk)
            if last and last.group(1).strip().lower().rstrip(".") in ABBREV:
                continue                                    # 'e.g.' / 'et al.' is not a sentence end
            if re.search(r"\b[A-Z]\.$", chunk):
                continue                                    # initials such as 'J. Smith'
            pieces.append(chunk)
            start = m.end()
        pieces.append(para[start:].strip())
        for p in pieces:
            if out and len(p.split()) < min_words:          # glue very short fragments to the previous sentence
                out[-1] = out[-1] + " " + p
            elif p:
                out.append(p)
    return out


def build_sentence_docs(rows, limit=0):
    """jjha sentence-aligned format: one row per sentence with current_en, prev_en, target_ko, paper_id.
    Rows of the same paper become one document in file order (or by 'order'/'sent_index' if present).
    prev_en is used exactly as given; per-sentence references are kept for sentence-level scoring."""
    groups = {}
    for r in rows:
        groups.setdefault(r["paper_id"], []).append(r)
    docs = []
    for pid, rs in groups.items():
        key = next((k for k in ("order", "sent_index", "sentence_index") if k in rs[0]), None)
        if key:
            rs = sorted(rs, key=lambda r: r[key])
        docs.append({
            "doc_id": pid, "paper_id": pid, "domain": rs[0].get("domain"),
            "source_en": " ".join(r["current_en"] for r in rs),
            "reference_ko": " ".join(r["target_ko"] for r in rs),
            "sentences": [r["current_en"] for r in rs],
            "prev_en": [r.get("prev_en", "") for r in rs],
            "sentence_ids": [r["id"] for r in rs],
            "sentence_refs": [r["target_ko"] for r in rs],
        })
        if limit and len(docs) >= limit:
            break
    return docs


def build_docs(path, limit=0):
    with open(path, encoding="utf-8") as f:
        first = next((json.loads(l) for l in f if l.strip()), {})
    if "current_en" in first:
        with open(path, encoding="utf-8") as f:
            return build_sentence_docs([json.loads(l) for l in f if l.strip()], limit)
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            sents = split_sentences(r["source_en"])
            docs.append({
                "doc_id": r["id"],
                "paper_id": r.get("paper_id"),
                "domain": r.get("domain"),
                "source_en": r["source_en"],
                "reference_ko": r["target_ko"],
                "sentences": sents,
            })
            if limit and len(docs) >= limit:
                break
    return docs
