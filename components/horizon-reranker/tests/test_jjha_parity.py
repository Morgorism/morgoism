"""Exact parity with jjha0/horizon-context prompts.py and infer_stage4.abnormal_reasons.
Set JJHA_REPO=/path/to/horizon-context and TEST_JSONL=/path/to/data/test.jsonl to run."""
import importlib.util
import json
import os
import sys

import pytest

REPO, TEST = os.environ.get("JJHA_REPO"), os.environ.get("TEST_JSONL")
pytestmark = pytest.mark.skipif(not (REPO and TEST), reason="set JJHA_REPO and TEST_JSONL")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_prompts_identical():
    from evaluation import glossary as G
    from reranker.prompts import jjha_prompt, raw_glossary
    from reranker.segment import build_docs
    theirs = _load("jjha_prompts", os.path.join(REPO, "prompts.py"))
    strict = json.load(open(os.path.join(REPO, "strict_glossary.json"), encoding="utf-8"))
    curated = raw_glossary(G.load(os.environ.get("CURATED", os.path.join(REPO, "strict_glossary.json"))))
    n = 0
    for d in build_docs(TEST):
        for i, s in enumerate(d["sentences"]):
            prev = " ".join(d["sentences"][max(0, i - 2):i])
            ref = theirs.prompt({"current_en": s, "prev_en": prev}, "context", strict)
            assert jjha_prompt(s, prev, strict) == ref
            assert jjha_prompt(s, prev, curated) == ref       # curated glossary gives the same rules
            n += 1
    assert n > 1000


def test_abnormal_identical():
    sys.modules.setdefault("prompts", _load("prompts", os.path.join(REPO, "prompts.py")))
    theirs = _load("jjha_infer", os.path.join(REPO, "infer_stage4.py"))
    from reranker.prompts import abnormal_reasons
    for text, cap in [("", False), ("a b c d e f " * 3, False), ("정상 번역입니다.", True), ("x�", False)]:
        assert abnormal_reasons(text, cap) == theirs.abnormal_reasons(text, cap)
