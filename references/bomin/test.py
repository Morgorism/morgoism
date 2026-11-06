"""Resume-friendly batched EN->KO LoRA test evaluation.

Install: pip install -U 'sacrebleu[ko]' tqdm transformers peft accelerate
Run: python test_eval_fast.py --adapter /workspace/path/to/adapter

The script uses test data strictly for scoring, never in generation prompts.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def args_parser():
    p = argparse.ArgumentParser(description="Batched LoRA translation evaluation")
    p.add_argument("--adapter", required=True, help="Folder containing adapter_config.json")
    p.add_argument("--test", default="/workspace/data/test.jsonl")
    p.add_argument("--output", default="/workspace/outputs/hymt2_cs/checkpoint-760/adapter_config.json")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="0=all samples; 10=smoke test")
    p.add_argument("--max-new-tokens", type=int, default=1024, help="Absolute per-batch generation cap")
    p.add_argument("--max-input-tokens", type=int, default=4096, help="Skip prompts larger than this, no silent truncation")
    p.add_argument("--retry-truncated", action="store_true", help="Regenerate previously truncated outputs")
    return p.parse_args()


def fingerprint(row):
    fields = [str(row.get("id", "")), row["source_en"], row["target_ko"]]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode("utf-8")).hexdigest()


def read_test(path, limit):
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("source_en"), str) or not isinstance(row.get("target_ko"), str):
                raise ValueError(f"{path}:{lineno}: source_en/target_ko must be strings")
            row["_key"] = fingerprint(row)
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No test rows in {path}")
    return rows


def recover_and_load(path):
    """Read previous results; discard only a potentially partial final line."""
    saved = {}
    if not path.exists():
        return saved
    with open(path, "rb+") as f:
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = fingerprint({
                    "id": record.get("id"),
                    "source_en": record["source_en"],
                    "target_ko": record["reference_ko"],
                })
                if not isinstance(record.get("prediction_ko"), str):
                    raise ValueError("Missing prediction_ko")
                saved[key] = record
            except (ValueError, KeyError, UnicodeDecodeError) as e:
                if f.readline():
                    raise ValueError(f"Corrupted middle record in {path} at byte {start}: {e}") from e
                f.truncate(start)
                print(f"[Recovery] Removed incomplete trailing JSONL record at byte {start}")
                break
    return saved


def score(rows, saved, output_path):
    from sacrebleu.metrics import BLEU, CHRF
    completed = [saved[r["_key"]] for r in rows if r["_key"] in saved]
    usable = [r for r in completed if r.get("status", "ok") == "ok"]
    if not usable:
        print("No valid predictions yet. No metrics computed.")
        return
    hypotheses = [r["prediction_ko"] for r in usable]
    references = [r["reference_ko"] for r in usable]
    bleu = BLEU(tokenize="ko-mecab")
    chrf = CHRF()
    bleu_score = bleu.corpus_score(hypotheses, [references])
    chrf_score = chrf.corpus_score(hypotheses, [references])
    n_trunc = sum(bool(r.get("hit_max_tokens")) for r in usable)
    summary = {
        "requested": len(rows),
        "completed": len(completed),
        "scored": len(usable),
        "skipped": len(completed) - len(usable),
        "hit_max_tokens": n_trunc,
        "bleu_ko_mecab": round(bleu_score.score, 4),
        "chrf": round(chrf_score.score, 4),
        "bleu_signature": str(bleu.get_signature()),
        "chrf_signature": str(chrf.get_signature()),
    }
    metrics_path = output_path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n===== Test Evaluation =====")
    print(f"Completed: {len(completed)}/{len(rows)} | Scored: {len(usable)} | Skipped: {summary['skipped']}")
    print(f"BLEU (ko-mecab): {bleu_score.score:.2f} | chrF: {chrf_score.score:.2f}")
    print(f"Reached generation cap (possibly truncated): {n_trunc}")
    print(f"Predictions: {output_path}\nMetrics: {metrics_path}")
    if len(completed) < len(rows):
        print("[Note] Scores are preliminary because the test set is incomplete.")
    if n_trunc:
        print("[Note] Review hit_max_tokens=true rows; consider --retry-truncated --max-new-tokens 1536.")
    for r in usable[:2]:
        print("\nSOURCE:   ", r["source_en"][:200])
        print("REFERENCE:", r["reference_ko"][:200])
        print("PREDICT:  ", r["prediction_ko"][:200])


def main():
    args = args_parser()
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.max_input_tokens < 1:
        raise ValueError("batch-size, max-new-tokens and max-input-tokens must be positive")
    from tqdm import tqdm
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftConfig, PeftModel

    adapter_path = Path(args.adapter).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"LoRA adapter_config.json not found in {adapter_path}")
    test_rows = read_test(args.test, args.limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = recover_and_load(output_path)
    for rec in saved.values():
        if rec.get("adapter_path") and Path(rec["adapter_path"]).resolve() != adapter_path:
            raise ValueError("Existing output was produced using another adapter. Choose a different --output path.")
    if saved and any("adapter_path" not in rec for rec in saved.values()):
        print("[Note] Legacy predictions found without adapter metadata; verify they belong to this model.")

    pending = [r for r in test_rows if r["_key"] not in saved or
               (args.retry_truncated and saved[r["_key"]].get("hit_max_tokens", False))]
    print(f"Test set: {len(test_rows)} | Previously saved: {len(test_rows) - len(pending)} | Remaining: {len(pending)}")
    if not pending:
        score(test_rows, saved, output_path)
        return

    config = PeftConfig.from_pretrained(str(adapter_path))
    base_name = config.base_model_name_or_path
    tokenizer_source = str(adapter_path) if (adapter_path / "tokenizer_config.json").is_file() else base_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    except (OSError, ValueError):
        print("[Tokenizer] Adapter tokenizer incomplete; loading tokenizer from base model")
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    tokenizer.padding_side = "left"  # Mandatory for correct batched decoder-only generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer has no pad or EOS token")

    model = AutoModelForCausalLM.from_pretrained(
        base_name, dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    model.config.use_cache = True

    for row in pending:
        source = row["source_en"]
        messages = [{"role": "user", "content":
                     "Translate the following English text into Korean. Output only the translation.\n\n" + source}]
        row["_prompt"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        row["_prompt_len"] = len(tokenizer.encode(row["_prompt"], add_special_tokens=False))
        row["_source_len"] = len(tokenizer.encode(source, add_special_tokens=False))

    pending.sort(key=lambda r: r["_prompt_len"])  # Length buckets reduce padding waste
    eos_value = model.generation_config.eos_token_id
    eos_ids = set(eos_value) if isinstance(eos_value, (tuple, list)) else ({eos_value} if eos_value is not None else set())
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    def persist(records, f):
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            saved[fingerprint({"id": rec.get("id"), "source_en": rec["source_en"],
                               "target_ko": rec["reference_ko"]})] = rec
        f.flush()
        os.fsync(f.fileno())

    def generate_batch(batch):
        encoded = tokenizer([r["_prompt"] for r in batch], padding=True,
                            add_special_tokens=False, return_tensors="pt")
        if encoded["input_ids"].shape[1] > args.max_input_tokens:
            raise ValueError("Batch includes an oversized prompt")
        inputs = {k: v.to(model.device) for k, v in encoded.items()}
        # Source-based token budget, never based on reference translations (no target leakage)
        budget = min(
            args.max_new_tokens,
            max(256, math.ceil(max(r["_source_len"] for r in batch) * 3.5 + 128))
        )
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=budget, do_sample=False,
                num_beams=1, use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        start = inputs["input_ids"].shape[-1]
        results = []
        for r, output_ids in zip(batch, generated):
            new_ids = output_ids[start:].tolist()
            stopped = any(token in eos_ids for token in new_ids)
            prediction = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            results.append({
                "id": r.get("id"), "source_en": r["source_en"],
                "reference_ko": r["target_ko"], "prediction_ko": prediction,
                "hit_max_tokens": len(new_ids) >= budget and not stopped,
                "generated_token_budget": budget,
                "status": "ok", "adapter_path": str(adapter_path),
            })
        del generated, inputs, encoded
        return results

    print(f"Model: {base_name}\nAdapter: {adapter_path}\nBatch size: {args.batch_size} (auto-reduce on CUDA OOM)")
    i, batch_size = 0, args.batch_size
    with tqdm(total=len(pending), desc="Translation", unit="sample") as progress:
        with open(output_path, "a", encoding="utf-8") as out:
            while i < len(pending):
                if pending[i]["_prompt_len"] > args.max_input_tokens:
                    r = pending[i]
                    persist([{
                        "id": r.get("id"), "source_en": r["source_en"],
                        "reference_ko": r["target_ko"], "prediction_ko": "",
                        "hit_max_tokens": False, "status": "input_too_long",
                        "adapter_path": str(adapter_path),
                    }], out)
                    i += 1
                    progress.update(1)
                    continue
                # Keep similarly sized texts together; don't include too-long inputs
                batch = []
                for r in pending[i:i + batch_size]:
                    if r["_prompt_len"] > args.max_input_tokens:
                        break
                    batch.append(r)
                try:
                    records = generate_batch(batch)
                except torch.cuda.OutOfMemoryError:
                    if batch_size == 1:
                        raise RuntimeError("OOM even with batch-size 1; reduce --max-input-tokens")
                    batch_size = max(1, batch_size // 2)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"\n[CUDA OOM] Reducing batch size to {batch_size}; retrying")
                    continue
                persist(records, out)
                i += len(batch)
                progress.update(len(batch))
                progress.set_postfix(batch=batch_size)
    score(test_rows, saved, output_path)


if __name__ == "__main__":
    main()
