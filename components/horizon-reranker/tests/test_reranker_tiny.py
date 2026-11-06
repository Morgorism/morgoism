"""End-to-end dry run of the reranker on a tiny random model that uses Hy-MT2's chat template.
Checks code paths (padding, masking, adapter on/off, CLI), not translation quality.
Skipped automatically when torch/transformers/peft are not installed."""
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

HY_TEMPLATE = (Path(__file__).parent / "hy_mt2_chat_template.jinja").read_text(encoding="utf-8")
SPECIALS = ["<|startoftext|>", "<|extra_0|>", "<|extra_4|>", "<|eos|>", "<|pad|>"]


def _tiny(tmp):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from peft import LoraConfig, get_peft_model

    corpus = ["Translate the following English text into Korean. Output only the translation.",
              "Translate the following Korean text into English. Output only the translation.",
              "We propose a new method for images. 우리는 이미지를 위한 새로운 방법을 제안한다.",
              "The encoder uses fine-tuning and dropout. encoder는 fine-tuning과 dropout을 사용한다."] * 20
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=SPECIALS,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<|startoftext|>", eos_token="<|eos|>",
                                  pad_token="<|pad|>")
    tok.chat_template = HY_TEMPLATE
    base = tmp / "base"
    tok.save_pretrained(base)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=2048,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(base)
    m = get_peft_model(LlamaForCausalLM.from_pretrained(base),
                       LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                  task_type="CAUSAL_LM", init_lora_weights=False))
    adapter = tmp / "adapter"
    m.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    return adapter


@pytest.fixture(scope="module")
def adapter(tmp_path_factory):
    return _tiny(tmp_path_factory.mktemp("tiny"))


def test_teacher_forcing_matches_generation(adapter):
    from reranker.model import Translator
    tr = Translator(str(adapter), device_map=None)
    users = ["Translate the following English text into Korean. Output only the translation.\n\nWe propose.",
             "Translate the following English text into Korean. Output only the translation.\n\nThe encoder uses dropout and a new method."]
    tr.tok.padding_side = "left"
    enc = tr.tok([tr.chat_prompt(u) for u in users], padding=True, add_special_tokens=False, return_tensors="pt")
    with torch.inference_mode():
        g = tr.model.generate(**enc, max_new_tokens=8, do_sample=False, output_scores=True,
                              return_dict_in_generate=True, pad_token_id=tr.tok.pad_token_id)
    trans = tr.model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True)
    start = enc["input_ids"].shape[1]
    for r, u in enumerate(users):
        new = g.sequences[r, start:].tolist()
        n = next((i + 1 for i, t in enumerate(new) if t in tr.eos_ids), len(new))
        gen_lp = float(trans[r, :n].sum())
        p = tr.tok.encode(tr.chat_prompt(u), add_special_tokens=False)
        (s, cnt), = tr._score_ids([(p, new[:n])], batch_size=4, use_adapter=True)
        assert cnt == n and math.isclose(s, gen_lp, rel_tol=1e-3, abs_tol=1e-3)
    # batching must not change scores
    pairs = [(u, "우리는 방법을 제안한다.") for u in users]
    one = [tr.score_answers([pr], 1)[0][0] for pr in pairs]
    many = [x[0] for x in tr.score_answers(pairs, 8)]
    assert all(math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-4) for a, b in zip(one, many))
    # adapter on/off gives different LM scores (disable_adapter works)
    on = tr.score_continuations([("우리는", "방법을 제안한다.")], 4, 64, use_adapter=True)[0][0]
    off = tr.score_continuations([("우리는", "방법을 제안한다.")], 4, 64, use_adapter=False)[0][0]
    assert not math.isclose(on, off)


def test_cli_end_to_end(adapter, tmp_path):
    from evaluation.__main__ import main as eval_main
    from reranker.__main__ import main as rr_main
    data = tmp_path / "valid.jsonl"
    rows = [{"id": f"p{i}_0001", "paper_id": f"p{i}", "domain": "cs",
             "source_en": "We propose a new method for images. The encoder uses fine-tuning and dropout. Results improve.",
             "target_ko": "우리는 이미지를 위한 새로운 방법을 제안한다. encoder는 fine-tuning과 dropout을 사용한다."} for i in range(3)]
    data.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    gloss = tmp_path / "g.json"
    gloss.write_text(json.dumps([{"term": "fine-tuning", "action": "KEEP"},
                                 {"term": "dropout", "action": "KEEP"}]), encoding="utf-8")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"adapter": str(adapter), "n_samples": 3, "max_new_tokens": 16,
                               "gen_batch_size": 2, "score_batch_size": 4,
                               "prompts": {"forward": "Translate the English academic text into natural Korean. "
                                           "Preserve technical proper names and any Markdown formatting."
                                           "{rules}\n\n{context_target}\n{source}"}}))
    run = tmp_path / "run"
    rr_main(["generate", "--config", str(cfg), "--glossary", str(gloss), "--data", str(data), "--out", str(run / "c.jsonl")])
    rr_main(["score", "--config", str(cfg), "--glossary", str(gloss), "--in", str(run / "c.jsonl"), "--out", str(run / "f.jsonl")])
    rr_main(["tune", "--glossary", str(gloss), "--in", str(run / "f.jsonl"), "--out", str(run / "w.json"), "--trials", "5"])
    rr_main(["decode", "--config", str(cfg), "--glossary", str(gloss), "--in", str(run / "f.jsonl"),
             "--weights", str(run / "w.json"), "--out-dir", str(run / "preds")])
    docs = [json.loads(l) for l in (run / "f.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(len(d["cands"]) == len(d["sentences"]) == 2 for d in docs)
    assert "Follow these terminology rules" not in docs[0]["prompts"][0]
    assert docs[0]["prompts"][1].startswith(
        "Translate the English academic text into natural Korean. Preserve technical proper names and any "
        "Markdown formatting.\nFollow these terminology rules:\n- Keep 'fine-tuning' exactly in English."
        "\n- Keep 'dropout' exactly in English.\n\n")
    for c in docs[0]["cands"][0]:
        assert {"fwd", "bwd", "lm", "glossary", "len", "structure_ok"} <= set(c)
    for name in ["A_greedy", "B_fwd", "C_fwd_bwd", "D_fwd_bwd_lm", "E_all"]:
        assert (run / "preds" / f"{name}.jsonl").is_file()
    eval_main(["run", "--pred", str(run / "preds" / "E_all.jsonl"), "--testset", str(data),
               "--glossary", str(gloss), "--out", str(run / "res"), "--no-mt"])
    assert (run / "res" / "summary.json").is_file()


def test_backward_adapter_switch(adapter):
    from reranker.model import Translator
    tr = Translator(str(adapter), device_map=None, backward_adapter=str(adapter))
    pair = [("Translate the following Korean text into English. Output only the translation.\n\n방법", "method")]
    a = tr.score_answers(pair, 2, adapter_name="default")[0][0]
    b = tr.score_answers(pair, 2, adapter_name="backward")[0][0]
    assert math.isclose(a, b, rel_tol=1e-5)          # same weights loaded twice -> same score
    assert tr.model.active_adapter == "default"      # switched back after scoring


def test_jjha_stage4_style_end_to_end(adapter, tmp_path):
    from reranker.__main__ import main as rr_main
    data = tmp_path / "t.jsonl"
    data.write_text(json.dumps({"id": "p_0001", "source_en": "We propose a new method for images. "
                                "The encoder uses fine-tuning and dropout now.", "target_ko": "방법"},
                               ensure_ascii=False), encoding="utf-8")
    gloss = tmp_path / "g.json"
    gloss.write_text(json.dumps([{"term": "dropout", "action": "KEEP", "auto_enforce": True,
                                  "context_sensitive": False}]), encoding="utf-8")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"adapter": str(adapter), "backward_adapter": str(adapter),
                               "prompt_style": "jjha_stage4", "repeat_recovery": True, "fixed_budget": True,
                               "n_samples": 2, "max_new_tokens": 12, "gen_batch_size": 2, "score_batch_size": 4}))
    rr_main(["generate", "--config", str(cfg), "--glossary", str(gloss), "--data", str(data), "--out", str(tmp_path / "c.jsonl")])
    d = json.loads((tmp_path / "c.jsonl").read_text(encoding="utf-8"))
    assert d["prompts"][0].endswith("\n\nWe propose a new method for images.")          # no context -> plain
    assert d["prompts"][1].startswith("[Background Information]\nWe propose a new method for images.\n\n")
    assert "- Keep 'dropout' exactly in English." in d["prompts"][1]
    assert d["prompts"][1].endswith("[Source Text]\nThe encoder uses fine-tuning and dropout now.")
    assert all("abnormal" in c for cs in d["cands"] for c in cs)
    rr_main(["score", "--config", str(cfg), "--glossary", str(gloss), "--in", str(tmp_path / "c.jsonl"), "--out", str(tmp_path / "f.jsonl")])
    rr_main(["tune", "--glossary", str(gloss), "--in", str(tmp_path / "f.jsonl"), "--out", str(tmp_path / "w.json"), "--trials", "3"])
    rr_main(["decode", "--config", str(cfg), "--in", str(tmp_path / "f.jsonl"), "--weights", str(tmp_path / "w.json"),
             "--out-dir", str(tmp_path / "preds")])
    assert (tmp_path / "preds" / "E_all.jsonl").is_file()


def test_sentence_aligned_input(adapter, tmp_path):
    from reranker.__main__ import main as rr_main
    rows = [{"id": "p1_s1", "paper_id": "p1", "current_en": "We propose a new method.", "prev_en": "",
             "target_ko": "새 방법을 제안한다."},
            {"id": "p1_s2", "paper_id": "p1", "current_en": "It uses dropout.", "prev_en": "Earlier text. We propose a new method.",
             "target_ko": "dropout을 사용한다."}]
    data = tmp_path / "s.jsonl"
    data.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    gloss = tmp_path / "g.json"
    gloss.write_text(json.dumps([{"term": "dropout", "action": "KEEP"}]), encoding="utf-8")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"adapter": str(adapter), "prompt_style": "jjha_stage4", "repeat_recovery": True,
                               "n_samples": 1, "max_new_tokens": 8, "score_batch_size": 4}))
    rr_main(["generate", "--config", str(cfg), "--glossary", str(gloss), "--data", str(data), "--out", str(tmp_path / "c.jsonl")])
    d = json.loads((tmp_path / "c.jsonl").read_text(encoding="utf-8"))
    assert d["prompts"][1].startswith("[Background Information]\nEarlier text. We propose a new method.\n\n")
    rr_main(["score", "--config", str(cfg), "--glossary", str(gloss), "--in", str(tmp_path / "c.jsonl"), "--out", str(tmp_path / "f.jsonl")])
    rr_main(["tune", "--in", str(tmp_path / "f.jsonl"), "--out", str(tmp_path / "w.json"), "--trials", "2"])
    rr_main(["decode", "--config", str(cfg), "--in", str(tmp_path / "f.jsonl"), "--weights", str(tmp_path / "w.json"),
             "--out-dir", str(tmp_path / "p")])
    sent = [json.loads(l) for l in (tmp_path / "p" / "A_greedy.sentences.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in sent] == ["p1_s1", "p1_s2"] and sent[1]["reference_ko"] == "dropout을 사용한다."


def test_markdown_roundtrip_and_translate(adapter, tmp_path):
    from reranker.markdown_io import parse, render
    md = ("# Introduction\n\nWe propose a method. It uses dropout.\n\n- first item here\n\n"
          "```python\nx = 1\n```\n\n$$\na+b\n$$\n\n| a | b |\n|---|---|\n")
    blocks, units = parse(md)
    assert units == ["Introduction", "We propose a method.", "It uses dropout.", "first item here"]
    out = render(blocks, [f"<{u}>" for u in units])
    assert out.startswith("# <Introduction>\n\n<We propose a method.> <It uses dropout.>\n\n- <first item here>")
    assert "```python\nx = 1\n```" in out and "$$\na+b\n$$" in out and "| a | b |" in out

    from reranker.__main__ import main as rr_main
    src = tmp_path / "paper.md"; src.write_text(md, encoding="utf-8")
    gloss = tmp_path / "g.json"; gloss.write_text(json.dumps([{"term": "dropout", "action": "KEEP"}]), encoding="utf-8")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"adapter": str(adapter), "prompt_style": "jjha_stage4", "repeat_recovery": True,
                               "n_samples": 1, "max_new_tokens": 8, "score_batch_size": 4}))
    w = tmp_path / "w.json"
    w.write_text(json.dumps({"E_all": {"weights": {"fwd": 1, "bwd": 0.5, "lm": 0.5, "glossary": 1, "len": 0}},
                             "A_greedy": {"weights": None}}))
    for exp in ("E_all", "A_greedy"):
        rr_main(["translate", "--config", str(cfg), "--glossary", str(gloss), "--weights", str(w),
                 "--experiment", exp, "--input", str(src), "--out", str(tmp_path / f"{exp}.md")])
        res = (tmp_path / f"{exp}.md").read_text(encoding="utf-8")
        assert res.startswith("# ") and "```python\nx = 1\n```" in res and "| a | b |" in res
