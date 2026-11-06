import argparse
import json
import random
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

MODEL_ID = "tencent/Hy-MT2-7B"

SOURCES = [
    ("train", "data/train.jsonl"),
    ("valid", "data/train.jsonl"),
    ("test", "data/train.jsonl"),
]

def prompt_for(direction, text):
    if direction == "en-ko":
        instruction = (
            "Translate the following CS technical text from "
            "English into Korean. Preserve technical meaning, "
            "code, and technical terminology. "
            "Output only the translation."
        )
    else:
        instruction = (
            "Translate the following CS technical text from "
            "Korean into English. Preserve technical meaning, "
            "code, and technical terminology. "
            "Output only the translation."
        )
    return instruction + "\n\n" + text

def normalized(text):
    text = unicodedata.normalize("NFC", str(text))
    return text.strip()


def key_for(text):
    return re.sub(r"\s+", " ", text).strip().casefold()

def korean_quality_issue(text):
    if "\ufffd" in text:
        return "replacement_character"

    hangul = sum("\uac00" <= c <= "\ud7a3" for c in text)
    hanja = sum("\u4e00" <= c <= "\u9fff" for c in text)
    questions = text.count("?")

    if hangul == 0:
        return "no_hangul"
    if hanja >= 4 and hanja / max(len(text), 1) > 0.05:
        return "possible_mojibake"
    if questions >= 3 and questions / len(text) > 0.04:
        return "possible_mojibake"

    return None


def find_member(names, suffix):
    matches = [
        n for n in names
        if n.lower().replace("\\", "/").endswith(
            "/" + suffix.lower()
        ) or n.lower() == suffix.lower()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one ZIP entry for {suffix}; "
            f"found {matches}"
        )
    return matches[0]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def make_sample(row, direction):
    if direction == "en-ko":
        src, tgt = row["en"], row["ko"]
    else:
        src, tgt = row["ko"], row["en"]

    return {
        "messages": [
            {
                "role": "user",
                "content": prompt_for(direction, src),
            },
            {"role": "assistant", "content": tgt},
        ],
        "direction": direction,
        "corpus": row["corpus"],
        "paper_id": row["paper_id"],
        "original_id": row["id"],
    }


def prepare(args):
    rng = random.Random(args.seed)
    raw = []
    counts = Counter()

    with zipfile.ZipFile(args.zip) as archive:
        names = archive.namelist()

        for corpus, split, suffix in SOURCES:
            member = find_member(names, suffix)

            with archive.open(member) as stream:
                for raw_line in stream:
                    if not raw_line.strip():
                        continue

                    counts[f"{corpus}_{split}_read"] += 1

                    try:
                        obj = json.loads(raw_line.decode("utf-8-sig"))
                    except (UnicodeError, ValueError):
                        counts["invalid_json_or_utf8"] += 1
                        continue

                    en = normalized(obj.get("source_en", ""))
                    ko = normalized(obj.get("target_ko", ""))

                    if not en or not ko:
                        counts["empty_pair"] += 1
                        continue

                    issue = korean_quality_issue(ko)
                    if issue:
                        counts[issue] += 1
                        continue

                    paper_id = obj.get("paper_id")
                    group = None

                    # OPUS paper_id는 문서 ID가 아니라 코퍼스 ID.
                    if corpus != "OPUS" and paper_id:
                        group = f"{corpus}:{paper_id}"

                    raw.append({
                        "id": obj.get("id", ""),
                        "corpus": corpus,
                        "split": split,
                        "paper_id": paper_id,
                        "group": group,
                        "en": en,
                        "ko": ko,
                    })

    # 문서가 여러 Split에 있으면 Test > Valid > Train
    # 우선순위에 따라 해당 문서의 낮은 우선순위
    # 샘플을 학습/평가 대상에서 제외한다.
    group_priority = {}
    for row in raw:
        if row["group"]:
            group_priority[row["group"]] = max(
                group_priority.get(row["group"], -1),
                PRIORITY[row["split"]],
            )

    retained = {"train": [], "valid": [], "test": []}
    seen_english = set()

    # Test를 우선적으로 보존하여 누출을 줄인다.
    ordered = sorted(
        raw,
        key=lambda r: -PRIORITY[r["split"]],
    )

    for row in ordered:
        group = row["group"]
        split = row["split"]

        if group and PRIORITY[split] != group_priority[group]:
            counts["document_split_conflict"] += 1
            continue

        key = key_for(row["en"])
        if key in seen_english:
            counts["duplicate_english_source"] += 1
            continue

        seen_english.add(key)
        retained[split].append(row)
        counts[f"{row['corpus']}_{split}_kept"] += 1

    if not retained["train"] or not retained["valid"]:
        raise RuntimeError(
            "Train or Validation is empty. "
            "Inspect document splitting and source files."
        )

    for split in retained:
        rng.shuffle(retained[split])

    # 전체 Train에서 reverse_rate=0.25이면
    # EN->KO : KO->EN = 약 80 : 20
    train_rows = [
        make_sample(row, "en-ko")
        for row in retained["train"]
    ]
    for row in retained["train"]:
        if rng.random() < args.reverse_rate:
            train_rows.append(make_sample(row, "ko-en"))

    rng.shuffle(train_rows)

    valid_rows = [
        make_sample(row, "en-ko")
        for row in retained["valid"]
    ]
    test_rows = [
        make_sample(row, "en-ko")
        for row in retained["test"]
    ]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "valid.jsonl", valid_rows)
    write_jsonl(out / "test.jsonl", test_rows)

    report = {
        "source_counts": dict(counts),
        "unique_train_pairs": len(retained["train"]),
        "train_samples_after_augmentation": len(train_rows),
        "valid_samples": len(valid_rows),
        "test_samples": len(test_rows),
        "train_corpus_distribution": dict(
            Counter(r["corpus"] for r in retained["train"])
        ),
    }

    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    rejected = sum(
        counts[k] for k in (
            "replacement_character",
            "no_hangul",
            "possible_mojibake",
        )
    )

    # 손상 의심 데이터가 많은 경우 학습을 중단해야 함.
    if rejected > 0.2 * max(len(raw) + rejected, 1):
        raise RuntimeError(
            "More than 20% of checked pairs failed "
            "Korean-quality checks. Inspect source encoding."
        )


def template_ids(tokenizer, messages, generation=False):
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation,
    )
    if isinstance(result, dict):
        return result["input_ids"]
    return result



class TranslationDataset:
    def __init__(
        self,
        path,
        tokenizer,
        max_length,
        is_train=False,
        reverse_fraction=0.25,
    ):
        self.examples = []
        self.skipped = 0
        rng = random.Random(42)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                row = json.loads(line)

                en = row["source_en"].strip()
                ko = row["target_ko"].strip()

                if not en or not ko:
                    self.skipped += 1
                    continue

                messages = [
                    {
                        "role": "user",
                        "content": f"Translate the following English text into Korean. Output only the translation.\n\n{en}"
                    },
                    {
                        "role": "assistant",
                        "content": ko
                    }
                ]

                self.add_sample(messages, tokenizer, max_length)

                if is_train and rng.random() < reverse_fraction:
                    reverse_messages = [
                        {
                            "role": "user",
                            "content": f"Translate the following Korean text into English. Output only the translation.\n\n{ko}"
                        },
                        {
                            "role": "assistant",
                            "content": en
                        }
                    ]
                    self.add_sample(reverse_messages, tokenizer, max_length)

        print(f"Loaded: {len(self.examples)}, Skipped: {self.skipped}")

    def add_sample(self, messages, tokenizer, max_length):
        # 1. 모델의 공식 Chat Template 적용
        prompt_text = tokenizer.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True,
        )

        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # 2. 문자열 수준에서 Prompt와 Answer 구분
        if not full_text.startswith(prompt_text):
            raise RuntimeError(
                "Chat template text mismatch.\n"
                f"Prompt ending: {prompt_text[-100:]!r}\n"
                f"Full beginning: {full_text[:100]!r}"
            )

        answer_text = full_text[len(prompt_text):]

        # 3. 각각 토큰화
        # add_special_tokens=False:
        # Chat Template에 이미 특수 토큰이 있기 때문
        prompt_ids = tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
        )

        answer_ids = tokenizer.encode(
            answer_text,
            add_special_tokens=False,
        )

        if not answer_ids:
            self.skipped += 1
            return

        # 4. 결합
        full_ids = prompt_ids + answer_ids

        # 5. Max Length 검사
        if len(full_ids) > max_length:
            self.skipped += 1
            return

        # 6. Prompt에 Loss가 적용되지 않도록 Masking
        labels = (
            [-100] * len(prompt_ids)
            + answer_ids.copy()
        )

        assert len(full_ids) == len(labels)
        assert any(x != -100 for x in labels)

        self.examples.append({
            "input_ids": full_ids,
            "labels": labels,
        })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

class TranslationCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        import torch

        max_len = max(len(x["input_ids"]) for x in examples)
        batch_ids, batch_labels, batch_masks = [], [], []

        for x in examples:
            n = len(x["input_ids"])
            padding = max_len - n

            batch_ids.append(
                x["input_ids"] + [self.pad_id] * padding
            )
            batch_labels.append(
                x["labels"] + [-100] * padding
            )
            batch_masks.append(
                [1] * n + [0] * padding
            )

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(
                batch_masks, dtype=torch.long
            ),
        }


def train(args):
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from peft import LoraConfig, TaskType, get_peft_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16.")

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_set = TranslationDataset(
        args.train,
        tokenizer,
        args.max_length,
        is_train=True,
    )

    valid_set = TranslationDataset(
        args.valid,
        tokenizer,
        args.max_length,
        is_train=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj"
        ],
    )

    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        optim="adamw_torch",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=100,

        save_strategy="epoch",
        save_total_limit=7,

        logging_strategy = "steps",
        logging_steps=20,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=TranslationCollator(
            tokenizer.pad_token_id
        ),
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    metrics = trainer.evaluate()
    print(f"Final Validation Loss: {metrics['eval_loss']:.4f}")

    adapter_dir = Path(args.output) / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    print(f"Saved adapter: {adapter_dir}")


def translate(args):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.adapter:
        model = PeftModel.from_pretrained(
            model, args.adapter
        )

    model.eval()

    messages = [{
        "role": "user",
        "content": prompt_for(args.direction, args.text),
    }]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    result = tokenizer.decode(
        outputs[0][prompt_len:],
        skip_special_tokens=True,
    )
    print(result.strip())


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(
        dest="command", required=True
    )

    p = commands.add_parser("prepare")
    p.add_argument("--zip", required=True)
    p.add_argument("--out", default="data")
    p.add_argument("--reverse_rate", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)

    t = commands.add_parser("train")
    t.add_argument("--model", default=MODEL_ID)
    t.add_argument("--train", default="data/train.jsonl")
    t.add_argument("--valid", default="data/valid.jsonl")
    t.add_argument("--output", default="outputs/hymt2_cs")
    t.add_argument("--max_length", type=int, default=1024)
    t.add_argument("--batch_size", type=int, default=2)
    t.add_argument("--grad_accum", type=int, default=8)
    t.add_argument("--epochs", type=float, default=2.0)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--resume", default=None)

    i = commands.add_parser("translate")
    i.add_argument("--model", default=MODEL_ID)
    i.add_argument("--adapter", default=None)
    i.add_argument(
        "--direction",
        choices=["en-ko", "ko-en"],
        default="en-ko",
    )
    i.add_argument("--text", required=True)
    i.add_argument("--max_new_tokens", type=int, default=512)

    args = parser.parse_args()

    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        train(args)
    else:
        translate(args)


if __name__ == "__main__":
    main()