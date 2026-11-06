from evaluation import structure, term
from evaluation.__main__ import evaluate


def rec(src, ref, pred, **kw):
    return {"id": kw.pop("id", "x"), "source_en": src, "reference_ko": ref, "prediction_ko": pred, **kw}


def test_reference_as_prediction_is_perfect():
    src = "We use a Transformer encoder with 12 layers [3]. See `train.py` and $x^2$."
    ref = "12개 층의 Transformer encoder를 사용한다 [3]. `train.py`와 $x^2$를 참고한다."
    s, _ = evaluate([rec(src, ref, ref)], with_mt=False)
    assert s["term_keep_f1_vs_ref"] == 1.0
    assert s["structure_pass_rate"] == 1.0
    assert s["number_preservation"] == 1.0


def test_transliteration_lowers_keep_recall():
    src = "The encoder outputs embeddings."
    ref = "encoder는 embedding을 출력한다."
    t = term.reference_based(src, ref, "인코더는 임베딩을 출력한다.")
    assert t["ref_kept"] == 2 and t["agree"] == 0
    assert set(t["missed_terms"]) == {"encoder", "embedding"}


def test_particles_and_plural_match():
    t = term.reference_based("Embeddings are learned.", "embedding이 학습된다.", "embeddings가 학습된다.")
    assert t["agree"] == 1


def test_glossary_policy():
    from evaluation import glossary as G
    g = [G._entry(r, "t") for r in [
        {"term": "encoder", "action": "KEEP", "forbidden": ["인코더"]},
        {"term": "loss function", "action": "TRANSLATE", "translation": "손실 함수", "translation_variants": ["손실함수"]},
        {"term": "embedding", "action": "BILINGUAL", "translation": "임베딩"},
    ]]
    src = "The encoder minimizes the loss function of each embedding."
    good = term.glossary_based(src, "encoder는 각 임베딩(embedding)의 손실함수를 최소화한다.", g)
    bad = term.glossary_based(src, "인코더는 각 임베딩의 loss function을 최소화한다.", g)
    assert (good["keep_ok"], good["translate_ok"], good["bilingual_ok"]) == (1, 1, 1)
    assert (bad["keep_ok"], bad["overtranslated"], bad["translate_ok"], bad["bilingual_ok"]) == (0, 1, 0, 0)


def test_team_glossary_matching_rules():
    from evaluation import glossary as G
    g = [G._entry(r, "t") for r in [
        {"term": "AI", "action": "KEEP"},
        {"term": "fine-tuning", "action": "KEEP", "variants": ["fine tuning"]},
        {"term": "cross-entropy loss", "action": "TRANSLATE", "translation": "교차 엔트로피 손실"},
        {"term": "loss", "action": "TRANSLATE", "translation": "손실"},
    ]]
    assert G.applicable("We said ai is fine.", g) == []                 # acronym is case-sensitive
    hit = [e["term"] for e in G.applicable("Fine tuning with cross entropy loss.", g)]
    assert hit == ["fine-tuning", "cross-entropy loss"]                  # shorter 'loss' is not double counted
    r = G.check("Fine tuning with cross entropy loss.", "fine-tuning에 교차 엔트로피 손실을 쓴다.", g)
    assert G.compliance(r) == 1.0


def test_structure_detects_broken_math_and_list():
    src = "- item with $a+b$\n- second [2]"
    pred = "- $a + b$ 항목\n두 번째"
    st = structure.check(src, pred)
    assert not st["structure_pass"]
    assert any("math_inline" in e for e in st["errors"])
    assert any("bullet" in e for e in st["errors"])


def test_placeholder_regex_configurable():
    st = structure.check("Use <<M1>> here.", "<<M1>>를 사용한다.", placeholder_regex=r"<<M\d+>>")
    assert st["structure_pass"] and st["invariant_total"] == 1


def test_sanity_flags():
    s, per = evaluate([rec("Some English text here.", "참조", "Some English text here.", hit_max_tokens=True)], with_mt=False)
    assert "mostly_not_korean" in per[0]["issues"] and "hit_max_tokens" in per[0]["issues"]


def test_team_rules_and_scoring():
    from evaluation import glossary as G
    g = [G._entry(r, "t") for r in [
        {"term": "convolutional neural network", "action": "TRANSLATE", "translation": "합성곱 신경망",
         "abbreviation": "CNN", "auto_enforce": True},
        {"term": "dropout", "action": "KEEP", "auto_enforce": True},
        {"term": "training", "action": "TRANSLATE", "translation": "학습", "translation_variants": ["학습", "훈련"],
         "auto_enforce": False},
    ]]
    det = G.build_detector(g)
    src = "A convolutional neural network (CNN) with dropout improves training."
    assert G.rules_block(src, det) == (
        "\nFollow these terminology rules:\n- Translate 'convolutional neural network' as '합성곱 신경망'."
        "\n- Keep 'CNN' exactly in English.\n- Keep 'dropout' exactly in English.")   # 'training' not auto_enforce
    ok = G.team_check(src, "dropout을 쓰는 합성곱 신경망(CNN)이 학습을 개선한다.", det)
    assert ok["KEEP"] == [2, 2] and ok["TRANSLATE"] == [1, 1]
    bad = G.team_check(src, "dropout을 쓰는 합성곱 신경망(convolutional neural network)이 학습을 개선한다.", det)
    assert bad["TRANSLATE"] == [0, 1] and bad["KEEP"] == [1, 2]     # English left behind; CNN dropped


def test_paired_bootstrap_matches_corpus_chrf_and_detects_gain():
    from sacrebleu.metrics import CHRF
    from evaluation.significance import paired_bootstrap
    refs = ["모델은 encoder를 사용한다.", "결과가 크게 향상되었다.", "제안 방법은 빠르다.", "데이터를 수집하였다."] * 10
    A = [{"id": str(i), "source_en": "x", "reference_ko": r, "prediction_ko": "무관한 문장"} for i, r in enumerate(refs)]
    B = [{"id": str(i), "source_en": "x", "reference_ko": r, "prediction_ko": r} for i, r in enumerate(refs)]
    res = paired_bootstrap(A, B, n=200)
    assert abs(res["chrf"]["B"] - CHRF().corpus_score([r["prediction_ko"] for r in B], [refs]).score) < 1e-6
    assert res["chrf"]["p_B_not_better"] == 0.0 and res["chrf"]["ci95"][0] > 0
    same = paired_bootstrap(A, A, n=50)
    assert same["chrf"]["delta"] == 0 and same["chrf"]["p_B_not_better"] == 1.0
