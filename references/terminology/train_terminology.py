
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "kci_corpus" / "processed" / "final" / "sft_prepared"
ADAPTER = ROOT / "models" / "finetuning_Bomin" / "outputs" / "hymt2_cs" / "adapter"
MODEL_NAME = "tencent/Hy-MT2-7B"


def load_rows(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing: {path}")

    with path.open("r", encoding="utf-8-sig") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    for i, row in enumerate(rows, 1):
        required = ("id", "document_id", "instruction", "source", "target")
        if any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in required
        ):
            raise ValueError(f"Invalid record: {path.name}, line {i}")
    return rows


def preflight():
    groups = {
        split: load_rows(DATA / f"sft_{split}.jsonl")
        for split in ("train", "valid", "test")
    }

    for a, b in (
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ):
        for key in ("id", "document_id", "source"):
            first = {r[key] for r in groups[a]}
            second = {r[key] for r in groups[b]}
            if first & second:
                raise ValueError(f"Split overlap: {a}/{b}, {key}")

    config_path = ADAPTER / "adapter_config.json"
    weight_path = ADAPTER / "adapter_model.safetensors"

    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"Missing LoRA adapter files in {ADAPTER}")

    if weight_path.stat().st_size < 100_000:
        raise RuntimeError(
            "Adapter appears to be a Git LFS pointer. Run git lfs pull."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))

    if config.get("base_model_name_or_path") != MODEL_NAME:
        raise ValueError("Unexpected base model")

    if config.get("peft_type") != "LORA":
        raise ValueError("Adapter is not LoRA")

    print("===== SFT PRE-FLIGHT: PASS =====")
    print("Data:", {k: len(v) for k, v in groups.items()})
    print("Adapter bytes:", weight_path.stat().st_size)
    print("Model:", MODEL_NAME)
    print("LoRA rank:", config.get("r"))

    return groups


def messages_for(row, use_terminology_rules):
    instruction = row["instruction"].strip()

    # 일반 번역 성능 측정용 Validation에서는
    # 정답과 충돌할 수 있는 강제 용어 규칙을 제외
    if not use_terminology_rules:
        instruction = instruction.split(
            "\nFollow these terminology rules:", 1
        )[0]

    return [
        {
            "role": "user",
            "content": instruction + "\n\n" + row["source"],
        },
        {
            "role": "assistant",
            "content": row["target"],
        },
    ]


class SftDataset:
    def __init__(self, rows, tokenizer, max_length, use_rules):
        self.examples = []
        self.skipped_long = 0

        for row in rows:
            messages = messages_for(row, use_rules)

            prompt = tokenizer.apply_chat_template(
                messages[:1],
                tokenize=False,
                add_generation_prompt=True,
            )
            full = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not full.startswith(prompt):
                raise RuntimeError("Chat template boundary mismatch")

            prompt_ids = tokenizer.encode(
                prompt, add_special_tokens=False
            )
            answer_ids = tokenizer.encode(
                full[len(prompt):], add_special_tokens=False
            )

            if (
                not answer_ids
                or len(prompt_ids) + len(answer_ids) > max_length
            ):
                self.skipped_long += 1
                continue

            self.examples.append({
                "input_ids": prompt_ids + answer_ids,
                "labels": [-100] * len(prompt_ids) + answer_ids,
            })

        if not self.examples:
            raise ValueError(
                "No usable examples. Check input or max_length."
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        import torch

        length = max(len(x["input_ids"]) for x in examples)

        return {
            "input_ids": torch.tensor([
                x["input_ids"]
                + [self.pad_id] * (length - len(x["input_ids"]))
                for x in examples
            ], dtype=torch.long),

            "labels": torch.tensor([
                x["labels"]
                + [-100] * (length - len(x["labels"]))
                for x in examples
            ], dtype=torch.long),

            "attention_mask": torch.tensor([
                [1] * len(x["input_ids"])
                + [0] * (length - len(x["input_ids"]))
                for x in examples
            ], dtype=torch.long),
        }


def train(args, groups):
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from peft import PeftModel, prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for training")

    if args.train_limit < 0 or args.valid_limit < 0 or args.steps < 1:
        raise ValueError("Invalid limits or steps")

    set_seed(42)

    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no padding token")

    train_rows = (
        groups["train"][:args.train_limit]
        if args.train_limit else groups["train"]
    )
    valid_rows = (
        groups["valid"][:args.valid_limit]
        if args.valid_limit else groups["valid"]
    )

    train_set = SftDataset(
        train_rows, tokenizer, args.max_length, use_rules=True
    )
    valid_set = SftDataset(
        valid_rows, tokenizer, args.max_length, use_rules=False
    )

    print(
        f"Tokenized train={len(train_set)}, valid={len(valid_set)}"
    )
    print(
        "Skipped overlong:",
        train_set.skipped_long,
        valid_set.skipped_long,
    )

    # 메모리 절약용 4-bit 양자화
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        device_map={"": torch.cuda.current_device()},
        quantization_config=quant,
        trust_remote_code=True,
    )

    base.config.use_cache = False

    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },
    )

    # 핵심: 팀원이 학습한 LoRA를 이어서 학습
    model = PeftModel.from_pretrained(
        base,
        str(ADAPTER),
        is_trainable=True,
    )

    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    if trainable == 0:
        raise RuntimeError("No trainable LoRA parameters")

    print("Trainable LoRA parameters:", trainable)

    training_args = TrainingArguments(
        output_dir=args.output,
        max_steps=args.steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },
        optim="paged_adamw_8bit",
        eval_strategy="steps",
        eval_steps=max(1, args.steps),
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=Collator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )

    trainer.train()
    print("Validation:", trainer.evaluate())

    # 기존 팀원 Adapter를 덮어쓰지 않고 별도 저장
    adapter_out = Path(args.output) / "adapter"

    trainer.save_model(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))

    print("Saved terminology adapter:", adapter_out)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", choices=("check", "train"), default="check"
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--train-limit", type=int, default=32)
    parser.add_argument("--valid-limit", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--output",
        default=str(ROOT / "models" / "terminology_smoke"),
    )

    args = parser.parse_args()
    groups = preflight()

    if args.mode == "train":
        train(args, groups)


if __name__ == "__main__":
    main()
