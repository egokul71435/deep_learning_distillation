"""Combined logit + sequence knowledge distillation from a Llama teacher using LoRA.

Requires pre-fetched top-k log-probabilities produced by fetch_logprobs.py.
Trains the student with a weighted combination of cross-entropy on the teacher's
text and sparse KL divergence on the teacher's top-k token distribution. Only
LoRA adapter weights are updated (less than 1% of total parameters).

Usage:
    # First pre-fetch logprobs (run once):
    python fetch_logprobs.py \
        --input  llama_teacher_data/scibench_train_clean.jsonl \
        --output llama_teacher_data/scibench_train_logprobs.jsonl

    # Then train:
    python train_llama_logit.py \
        --dataset scibench \
        --alpha 0.5 \
        --kd-temperature 2.0
"""

import argparse

from transformers import TrainingArguments

from dataset import KDDataset, load_jsonl, tokenize_for_logit_kd
from kd_trainer import KDTrainer
from model import get_device, get_dtype, load_model, apply_lora


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument(
        "--dataset",
        default="scibench",
        choices=["scibench", "theoremqa", "both"],
    )
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5,
                    help="Weight for sequence CE loss; (1-alpha) for logit KL loss")
    p.add_argument("--kd-temperature", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--output-dir", default="./output/llama_logit_kd")

    # LoRA
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    return p.parse_args()


DATA_DIR = "llama_teacher_data"


def gather_data_and_logprobs(dataset_choice):
    data_paths = []
    lp_paths = []
    if dataset_choice in ("scibench", "both"):
        data_paths.append(f"{DATA_DIR}/scibench_train_clean.jsonl")
        lp_paths.append(f"{DATA_DIR}/scibench_train_logprobs.jsonl")
    if dataset_choice in ("theoremqa", "both"):
        data_paths.append(f"{DATA_DIR}/theoremqa_train_clean.jsonl")
        lp_paths.append(f"{DATA_DIR}/theoremqa_train_logprobs.jsonl")

    data, logprobs = [], []
    for dp, lp in zip(data_paths, lp_paths):
        d = load_jsonl(dp)
        l = load_jsonl(lp)
        if len(d) != len(l):
            raise ValueError(
                f"Data/logprobs length mismatch: {dp}({len(d)}) vs {lp}({len(l)}). "
                "Re-run fetch_logprobs.py to regenerate."
            )
        data.extend(d)
        logprobs.extend(l)
    return data, logprobs


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

    print(f"Loading data + logprobs ({args.dataset}) from {DATA_DIR}/")
    raw_data, logprob_records = gather_data_and_logprobs(args.dataset)
    print(f"  {len(raw_data)} training examples with logprob data")

    encodings = tokenize_for_logit_kd(
        tokenizer, raw_data, logprob_records,
        max_length=args.max_length, top_k=args.top_k,
    )
    dataset = KDDataset(encodings)

    valid_positions = encodings["logit_kd_mask"].sum().item()
    total_positions = encodings["logit_kd_mask"].numel()
    pct = 100.0 * valid_positions / max(total_positions, 1)
    print(f"  Logit-KD coverage: {valid_positions}/{total_positions} positions ({pct:.1f}%)")

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
        alpha=args.alpha,
        kd_temperature=args.kd_temperature,
    )

    print(f"Starting Llama logit-KD training (alpha={args.alpha}, T={args.kd_temperature}) …")
    trainer.train()
    student.save_pretrained(args.output_dir + "/final")
    tokenizer.save_pretrained(args.output_dir + "/final")
    print(f"Adapter saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
