"""Quick smoke tests for the KD pipeline.

Uses GPT-2 as a tiny stand-in student to make sure nothing crashes before
committing to a real training run. Covers sequence KD, logit KD, and both
with LoRA wrapping.
"""

import torch
from transformers import TrainingArguments

from dataset import KDDataset
from kd_trainer import KDTrainer
from model import get_device, get_dtype, load_model, apply_lora

MODEL_NAME = "gpt2"
SEQ_LEN = 32
NUM_SAMPLES = 16
TOP_K = 5


def _make_seq_kd_dataset(tokenizer):
    """Plain sequence-KD dataset (labels = input_ids, question prefix masked)."""
    texts = ["What is 2+2? The answer is four."] * NUM_SAMPLES
    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=SEQ_LEN,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    labels[:, :5] = -100
    enc["labels"] = labels
    return enc


def _make_logit_kd_dataset(tokenizer):
    """Sequence-KD dataset augmented with fake sparse teacher logprobs."""
    enc = _make_seq_kd_dataset(tokenizer)
    B, S = enc["input_ids"].shape

    vocab_size = tokenizer.vocab_size
    topk_ids = torch.randint(0, vocab_size, (B, S, TOP_K))
    topk_lp = torch.full((B, S, TOP_K), -2.0)
    mask = torch.ones(B, S, dtype=torch.long)
    mask[:, :5] = 0

    enc["teacher_topk_ids"] = topk_ids
    enc["teacher_topk_logprobs"] = topk_lp
    enc["logit_kd_mask"] = mask
    return enc


def _training_args(output_dir):
    device = get_device()
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        max_steps=3,
        logging_steps=1,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        use_cpu=(device.type == "cpu"),
        report_to="none",
    )


def test_sequence_kd():
    """Offline sequence KD (alpha=1.0, no teacher logits)."""
    print("=" * 60)
    print("TEST: sequence KD (alpha=1.0)")
    device = get_device()
    dtype = get_dtype(device)
    tokenizer, student = load_model(MODEL_NAME, device=device, dtype=dtype)
    tokenizer.pad_token = tokenizer.eos_token

    ds = KDDataset(_make_seq_kd_dataset(tokenizer))
    trainer = KDTrainer(
        model=student,
        args=_training_args("./test_output/seq_kd"),
        train_dataset=ds,
        alpha=1.0,
    )
    trainer.train()
    print("  PASSED\n")


def test_logit_kd():
    """Sparse logit KD from pre-fetched top-k (alpha=0.5)."""
    print("=" * 60)
    print("TEST: logit KD (alpha=0.5)")
    device = get_device()
    dtype = get_dtype(device)
    tokenizer, student = load_model(MODEL_NAME, device=device, dtype=dtype)
    tokenizer.pad_token = tokenizer.eos_token

    ds = KDDataset(_make_logit_kd_dataset(tokenizer))
    trainer = KDTrainer(
        model=student,
        args=_training_args("./test_output/logit_kd"),
        train_dataset=ds,
        alpha=0.5,
        kd_temperature=2.0,
    )
    trainer.train()
    print("  PASSED\n")


def test_lora_sequence_kd():
    """LoRA + offline sequence KD."""
    print("=" * 60)
    print("TEST: LoRA + sequence KD (alpha=1.0)")
    device = get_device()
    dtype = get_dtype(device)
    tokenizer, student = load_model(MODEL_NAME, device=device, dtype=dtype)
    tokenizer.pad_token = tokenizer.eos_token

    # GPT-2 uses Conv1D (c_attn, c_proj) not q_proj/v_proj
    student = apply_lora(student, rank=4, alpha=8, target_modules=["c_attn"])

    ds = KDDataset(_make_seq_kd_dataset(tokenizer))
    trainer = KDTrainer(
        model=student,
        args=_training_args("./test_output/lora_seq_kd"),
        train_dataset=ds,
        alpha=1.0,
    )
    trainer.train()
    print("  PASSED\n")


def test_lora_logit_kd():
    """LoRA + sparse logit KD (alpha=0.5)."""
    print("=" * 60)
    print("TEST: LoRA + logit KD (alpha=0.5)")
    device = get_device()
    dtype = get_dtype(device)
    tokenizer, student = load_model(MODEL_NAME, device=device, dtype=dtype)
    tokenizer.pad_token = tokenizer.eos_token

    student = apply_lora(student, rank=4, alpha=8, target_modules=["c_attn"])

    ds = KDDataset(_make_logit_kd_dataset(tokenizer))
    trainer = KDTrainer(
        model=student,
        args=_training_args("./test_output/lora_logit_kd"),
        train_dataset=ds,
        alpha=0.5,
        kd_temperature=2.0,
    )
    trainer.train()
    print("  PASSED\n")


if __name__ == "__main__":
    print(f"Device: {get_device()}\n")
    test_sequence_kd()
    test_logit_kd()
    test_lora_sequence_kd()
    test_lora_logit_kd()
    print("=" * 60)
    print("All tests passed!")
