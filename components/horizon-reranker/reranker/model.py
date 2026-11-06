"""Model loading, N-best generation and teacher-forced scoring, following test.py / train.py.

- Loading: PeftConfig -> base model (bf16 if supported) -> PeftModel, tokenizer from the adapter folder.
- Prompts: tokenizer.apply_chat_template(..., add_generation_prompt=True), exactly as test.py.
- Scored answer: the text the chat template appends after the prompt (content + EOS), exactly as
  train.py builds labels, so scores match what the model was trained on.
"""
import contextlib
import math
from pathlib import Path


class Translator:
    def __init__(self, adapter, base_model=None, device_map="auto", backward_adapter=None):
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        adapter = Path(adapter).expanduser().resolve()
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"adapter_config.json not found in {adapter}")
        base = base_model or PeftConfig.from_pretrained(str(adapter)).base_model_name_or_path
        tok_src = str(adapter) if (adapter / "tokenizer_config.json").is_file() else base
        try:
            self.tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
        except (OSError, ValueError):
            self.tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
            torch.float16 if torch.cuda.is_available() else torch.float32)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, device_map=device_map, trust_remote_code=True)
        self.model = PeftModel.from_pretrained(model, str(adapter), adapter_name="default")
        self.has_backward = False
        if backward_adapter:                     # optional second LoRA on the same base for p(source | target)
            self.model.load_adapter(str(Path(backward_adapter).expanduser().resolve()), adapter_name="backward")
            self.has_backward = True
        self.model.set_adapter("default")
        self.model.eval()
        self.model.config.use_cache = True
        self.adapter, self.base = str(adapter), base
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos is not None else set())
        if self.tok.eos_token_id is not None:
            self.eos_ids.add(self.tok.eos_token_id)

    @property
    def device(self):
        return self.model.device

    # ---------- prompts ----------
    def chat_prompt(self, user_text):
        return self.tok.apply_chat_template([{"role": "user", "content": user_text}],
                                            tokenize=False, add_generation_prompt=True)

    def answer_suffix(self, user_text, answer_text):
        prompt = self.chat_prompt(user_text)
        full = self.tok.apply_chat_template(
            [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer_text}],
            tokenize=False, add_generation_prompt=False)
        if not full.startswith(prompt):
            raise RuntimeError("chat template prompt is not a prefix of the full conversation")
        return prompt, full[len(prompt):]

    def n_tokens(self, text):
        return len(self.tok.encode(text, add_special_tokens=False))

    # ---------- generation ----------
    def generate(self, user_texts, n_samples, temperature, top_p, max_new_tokens, seed=None,
                 fixed_budget=False, retry=False):
        """Returns, per prompt, [greedy] + n_samples sampled outputs (deduplicated later).
        fixed_budget: always max_new_tokens (jjha infer_stage4) instead of the test.py length budget.
        retry: greedy with repetition_penalty 1.05 and no_repeat_ngram_size 12 only
               (jjha conditional_repeat_recovery_v1)."""
        torch = self.torch
        self.tok.padding_side = "left"           # required for batched decoder-only generation (test.py)
        prompts = [self.chat_prompt(u) for u in user_texts]
        enc = self.tok(prompts, padding=True, add_special_tokens=False, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        src_len = max(self.n_tokens(u) for u in user_texts)
        budget = max_new_tokens if fixed_budget else min(max_new_tokens, max(64, math.ceil(src_len * 3.5 + 32)))
        start = enc["input_ids"].shape[1]
        results = [[] for _ in prompts]
        with torch.inference_mode():
            extra = {"repetition_penalty": 1.05, "no_repeat_ngram_size": 12} if retry else {}
            g = self.model.generate(**enc, max_new_tokens=budget, do_sample=False, num_beams=1,
                                    pad_token_id=self.tok.pad_token_id, **extra)
            for i, ids in enumerate(g):
                results[i].append(self._decode(ids[start:], budget, "retry" if retry else "greedy"))
            if retry:
                return results
            if n_samples > 0:
                if seed is not None:
                    torch.manual_seed(seed)
                s = self.model.generate(**enc, max_new_tokens=budget, do_sample=True, temperature=temperature,
                                        top_p=top_p, num_return_sequences=n_samples,
                                        pad_token_id=self.tok.pad_token_id)
                for j, ids in enumerate(s):
                    results[j // n_samples].append(self._decode(ids[start:], budget, "sample"))
        return results

    def _decode(self, new_ids, budget, kind):
        ids = new_ids.tolist()
        stopped = any(t in self.eos_ids for t in ids)
        text = self.tok.decode(ids, skip_special_tokens=True).strip()
        return {"text": text, "kind": kind, "hit_max_tokens": len(ids) >= budget and not stopped}

    # ---------- scoring ----------
    def score_answers(self, pairs, batch_size, use_adapter=True, adapter_name="default"):
        """pairs: [(user_text, answer_text)] -> [(sum_logprob, n_answer_tokens)] under the chat template."""
        items = []
        for user, ans in pairs:
            prompt, suffix = self.answer_suffix(user, ans)
            items.append((self.tok.encode(prompt, add_special_tokens=False),
                          self.tok.encode(suffix, add_special_tokens=False)))
        return self._score_ids(items, batch_size, use_adapter, adapter_name)

    def score_continuations(self, pairs, batch_size, max_context_tokens, use_adapter=False):
        """Language-model feature. pairs: [(context_text, continuation_text)] scored as raw text."""
        items = []
        bos = [self.tok.bos_token_id] if self.tok.bos_token_id is not None else []
        for ctx, cont in pairs:
            ctx_ids = self.tok.encode(ctx, add_special_tokens=False)[-max_context_tokens:] if ctx else []
            cont_ids = self.tok.encode((" " if ctx else "") + cont, add_special_tokens=False)
            items.append((bos + ctx_ids, cont_ids))
        return self._score_ids(items, batch_size, use_adapter)

    def _score_ids(self, items, batch_size, use_adapter, adapter_name="default"):
        torch = self.torch
        out = [None] * len(items)
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]) + len(items[i][1]))
        if use_adapter:
            self.model.set_adapter(adapter_name)
        ctx = contextlib.nullcontext() if use_adapter else self.model.disable_adapter()
        pad = self.tok.pad_token_id
        with torch.inference_mode(), ctx:
            for b in range(0, len(order), batch_size):
                idx = order[b:b + batch_size]
                seqs = [items[i][0] + items[i][1] for i in idx]
                L = max(len(s) for s in seqs)
                ids = torch.full((len(seqs), L), pad, dtype=torch.long)
                mask = torch.zeros((len(seqs), L), dtype=torch.long)
                tgt = torch.full((len(seqs), L), -100, dtype=torch.long)
                for r, i in enumerate(idx):
                    p, a = items[i]
                    n = len(p) + len(a)
                    ids[r, :n] = torch.tensor(p + a)
                    mask[r, :n] = 1
                    if a:
                        tgt[r, len(p):n] = torch.tensor(a)
                ids, mask, tgt = ids.to(self.device), mask.to(self.device), tgt.to(self.device)
                logits = self.model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
                labels = tgt[:, 1:]
                logp = torch.log_softmax(logits, dim=-1)
                valid = labels != -100
                tok_lp = logp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1) * valid
                sums, counts = tok_lp.sum(-1).tolist(), valid.sum(-1).tolist()
                for r, i in enumerate(idx):
                    out[i] = (sums[r], int(counts[r]))
                del logits, logp
        self.model.set_adapter("default")
        return out
