"""Sequence-level knowledge distillation from a GPT teacher using LoRA.

The teacher model is never loaded here. We just train the student to reproduce
the GPT teacher's chain-of-thought text from the pre-generated JSONL files.
Only LoRA adapter weights are updated (less than 1% of total parameters).

Usage:
    python train_gpt_seq.py --dataset scibench
    python train_gpt_seq.py --dataset both
"""

import argparse

from transformers import TrainingArguments

from dataset import KDDataset, load_jsonl, tokenize_for_seq_kd
from kd_trainer import KDTrainer
from model import get_device, get_dtype, load_model, apply_lora


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--student",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace student model name or path",
    )
    p.add_argument(
        "--dataset",
        default="scibench",
        choices=["scibench", "theoremqa", "both"],
    )
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--output-dir", default="./output/gpt_seq_kd")

    # LoRA
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    return p.parse_args()


DATA_DIR = "gpt_teacher_data"


def gather_data(dataset_choice):
    paths = []
    if dataset_choice in ("scibench", "both"):
        paths.append(f"{DATA_DIR}/scibench_train_clean.jsonl")
    if dataset_choice in ("theoremqa", "both"):
        paths.append(f"{DATA_DIR}/theoremqa_train_clean.jsonl")

    data = []
    for p in paths:
        data.extend(load_jsonl(p))
    return data


def main():
    args = parse_args()
    device = get_device()
    dtype = get_dtype(device)
    print(f"Device: {device}  dtype: {dtype}")

    print(f"Loading student: {args.student}")
    tokenizer, student = load_model(args.student, device=device, dtype=dtype)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    student = apply_lora(
        student,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    print(f"Loading data ({args.dataset}) from {DATA_DIR}/")
    raw = gather_data(args.dataset)
    print(f"  {len(raw)} training examples")

    encodings = tokenize_for_seq_kd(tokenizer, raw, max_length=args.max_length)
    dataset = KDDataset(encodings)
    print(f"  Tokenised → {len(dataset)} samples, seq_len={args.max_length}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        warmup_steps=10,
        logging_steps=1,
        save_strategy="no",
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        use_cpu=(device.type == "cpu"),
    )

    trainer = KDTrainer(
        model=student,
        args=training_args,
        train_dataset=dataset,
        alpha=1.0,
    )

    print("Starting GPT sequence-KD training …")
    trainer.train()
    student.save_pretrained(args.output_dir + "/final")
    tokenizer.save_pretrained(args.output_dir + "/final")
    print(f"Adapter saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
