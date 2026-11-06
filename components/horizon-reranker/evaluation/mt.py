"""BLEU / chrF with the same sacrebleu settings as the team's test.py (ko-mecab BLEU, default chrF)."""


def corpus_scores(hyps, refs):
    from sacrebleu.metrics import BLEU, CHRF
    bleu, chrf = BLEU(tokenize="ko-mecab"), CHRF()
    return {
        "bleu_ko_mecab": round(bleu.corpus_score(hyps, [refs]).score, 4),
        "chrf": round(chrf.corpus_score(hyps, [refs]).score, 4),
        "bleu_signature": str(bleu.get_signature()),
        "chrf_signature": str(chrf.get_signature()),
    }


def sentence_chrf(hyps, refs):
    from sacrebleu.metrics import CHRF
    chrf = CHRF()
    return [round(chrf.sentence_score(h, [r]).score, 2) for h, r in zip(hyps, refs)]
