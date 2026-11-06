"""Turn a Markdown document into translation units and put translations back.

- Code fences (```), display math ($$ ... $$), tables (| ... |), HTML comments and blank lines are kept as is.
- Headings and list items are translated as one unit; the marker ('## ', '- ', '1. ', '> ') is kept.
- Plain paragraphs are split into sentences (same splitter as evaluation) and re-joined with spaces.
All units of the file form one document, so earlier sentences are context for later ones.
"""
import re

from .segment import split_sentences

PREFIX = re.compile(r"^(\s{0,3}(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s?)+)")


def parse(text):
    blocks, units = [], []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("```") or s == "$$" or s.startswith("$$"):
            fence = "```" if s.startswith("```") else "$$"
            j = i + 1
            single = fence == "$$" and s.endswith("$$") and len(s) > 2
            if not single:
                while j < len(lines) and not lines[j].strip().startswith(fence):
                    j += 1
                j += 1
            blocks.append(("raw", "\n".join(lines[i:j])))
            i = j
            continue
        if not s or s.startswith("|") or s.startswith("<!--"):
            blocks.append(("raw", line))
            i += 1
            continue
        m = PREFIX.match(line)
        if m:
            body = line[m.end():]
            blocks.append(("unit", m.group(1), [len(units)]))
            units.append(body)
            i += 1
            continue
        j = i
        para = []
        while j < len(lines) and lines[j].strip() and not PREFIX.match(lines[j]) \
                and not lines[j].strip().startswith(("```", "$$", "|", "<!--")):
            para.append(lines[j].strip())
            j += 1
        sents = split_sentences(" ".join(para), min_words=1)
        blocks.append(("unit", "", list(range(len(units), len(units) + len(sents)))))
        units.extend(sents)
        i = j
    return blocks, units


def render(blocks, translations):
    out = []
    for b in blocks:
        if b[0] == "raw":
            out.append(b[1])
        else:
            out.append(b[1] + " ".join(translations[k] for k in b[2]))
    return "\n".join(out) + "\n"
