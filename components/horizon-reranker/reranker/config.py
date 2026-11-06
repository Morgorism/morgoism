"""Run configuration. Everything that depends on jjha's context-aware model lives in PROMPTS.

Template fields
  {source}          current English sentence
  {context_source}  previous English sentences of the same document (joined with a space)
  {context_target}  Korean translations already chosen for those sentences
  {target}          candidate Korean translation (backward template only)
  {rules}           '' or '\nFollow these terminology rules:\n- ...' built from the glossary exactly as
                    최윤서's prepare_terminology_sft_direct.py (use with the terminology adapter)

Default forward template = the exact training/test prompt in train.py / test.py (no context).
When jjha hands over the context-aware adapter, put the prompt used in its training data
into "forward" (for example with {context_source} / {context_target}) and nothing else changes.
"""
import copy
import json

DEFAULTS = {
    "adapter": None,                 # folder with adapter_config.json (required)
    "base_model": None,              # None = base_model_name_or_path from adapter_config.json
    "backward_adapter": None,        # LoRA for KO->EN scoring if the forward adapter was not trained on KO->EN
    "prompts": {
        "forward": "Translate the following English text into Korean. Output only the translation.\n\n{source}",
        # train.py added 25% KO->EN samples with this prompt, so the same adapter scores p(source | target)
        "backward": "Translate the following Korean text into English. Output only the translation.\n\n{target}",
    },
    "prompt_style": "template",      # "template" | "jjha_stage4" (see reranker/prompts.py)
    "context_size": 2,               # previous sentences given as context (forward prompt and LM)
    "repeat_recovery": False,        # jjha rule: abnormal greedy output -> one greedy retry with repetition controls
    "fixed_budget": False,           # True = always max_new_tokens (jjha), False = test.py length budget
    "lm_use_adapter": False,         # LM feature scores raw Korean text with the adapter disabled
    "lm_max_context_tokens": 512,
    "n_samples": 6,                  # sampled candidates per sentence (plus 1 greedy)
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 512,           # absolute cap; per-batch budget is derived from source length like test.py
    "gen_batch_size": 8,             # sentences per generate() call (x (n_samples) sequences)
    "score_batch_size": 16,
    "seed": 42,
    "placeholder_regex": "⟦[^⟧]+⟧",
}


def load(path=None, **overrides):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        for k, v in user.items():
            if k == "prompts":
                cfg["prompts"].update(v)
            else:
                cfg[k] = v
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    if "{source}" not in cfg["prompts"]["forward"] or "{target}" not in cfg["prompts"]["backward"]:
        raise ValueError("forward prompt needs {source}; backward prompt needs {target}")
    return cfg


def forward_uses_target_context(cfg):
    return cfg.get("prompt_style", "template") == "template" and "{context_target}" in cfg["prompts"]["forward"]


def forward_uses_context(cfg):
    t = cfg["prompts"]["forward"]
    return "{context_source}" in t or "{context_target}" in t
