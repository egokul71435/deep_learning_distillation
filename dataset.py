"""Dataset utilities for knowledge distillation.

Provides tokenization functions for both sequence-level KD (where the student
just trains on the teacher's text) and logit-level KD (where the student also
gets the teacher's top-k log-probabilities). The logit KD path includes a
token alignment step that maps between teacher and student tokenizations by
character offset, since the two models may not tokenize identically.
"""

import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ---------------------------------------------------------------------------
# Tokenisation for offline sequence distillation
# ---------------------------------------------------------------------------

def tokenize_for_seq_kd(tokenizer, data, max_length=512):
    """Tokenize (question, teacher_cot) pairs for offline sequence KD.

    Labels are set so that loss is only computed on the teacher_cot portion
    (question prefix and padding are masked with -100).
    """
    questions = [d["problem_text"] for d in data]
    answers = [d["teacher_cot"] for d in data]
    full_texts = [q + "\n" + a for q, a in zip(questions, answers)]

    q_enc = tokenizer(questions, truncation=True, padding=False, max_length=max_length)
    q_lengths = [len(ids) for ids in q_enc["input_ids"]]

    encodings = tokenizer(
        full_texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    labels = encodings["input_ids"].clone()
    labels[encodings["attention_mask"] == 0] = -100
    for i, q_len in enumerate(q_lengths):
        labels[i, :q_len] = -100

    encodings["labels"] = labels
    return encodings


# ---------------------------------------------------------------------------
# Tokenisation for logit distillation (with cached top-k logprobs)
# ---------------------------------------------------------------------------

def _build_char_offsets(tokens):
    """Return a list of (start, end) character offsets for each token."""
    offsets = []
    pos = 0
    for t in tokens:
        offsets.append((pos, pos + len(t)))
        pos += len(t)
    return offsets


def _align_teacher_to_student(teacher_tokens, student_tokens):
    """Map each student-token position to the best-matching teacher-token position.

    Returns ``alignment`` (len = len(student_tokens)) where alignment[i] is the
    teacher-token index whose character span overlaps the start of student token i,
    or -1 when no good match exists.  Also returns a boolean list ``exact`` that is
    True when the teacher and student tokens at that position are the *same* string
    (i.e. they represent the same sub-word and logit KD is meaningful).
    """
    t_off = _build_char_offsets(teacher_tokens)
    s_off = _build_char_offsets(student_tokens)

    alignment = []
    exact = []
    ti = 0
    for si in range(len(s_off)):
        s_start, s_end = s_off[si]
        # advance teacher pointer to cover student start
        while ti < len(t_off) - 1 and t_off[ti][1] <= s_start:
            ti += 1
        if ti < len(t_off) and t_off[ti][0] <= s_start < t_off[ti][1]:
            alignment.append(ti)
            exact.append(teacher_tokens[ti] == student_tokens[si])
        else:
            alignment.append(-1)
            exact.append(False)
    return alignment, exact


def tokenize_for_logit_kd(
    tokenizer,
    data,
    logprob_records,
    max_length=512,
    top_k=20,
):
    """Build a dataset for combined sequence + logit distillation.

    ``logprob_records`` is a list (same length / order as ``data``) of dicts
    produced by ``fetch_logprobs.py``, each containing:
        tokens         – list[str]   teacher token strings (from generation)
        top_logprobs   – list[list[dict]]  per-position top-k {token, logprob}
        generated_text – str  the teacher's generated completion

    The sequence-KD labels use the teacher's *generated_text* (which aligns
    with the logprob tokens), falling back to the original ``teacher_cot``
    when ``generated_text`` is missing or empty.

    Returns the standard encodings dict *plus* three extra tensors:
        teacher_topk_ids      (N, seq_len, top_k)  – student-vocab token IDs
        teacher_topk_logprobs (N, seq_len, top_k)  – log-probabilities
        logit_kd_mask         (N, seq_len)          – 1 where logit KD is valid
    """
    # Use the teacher's generated text (aligned with logprobs) for seq KD
    augmented_data = []
    for row, rec in zip(data, logprob_records):
        entry = dict(row)
        gen = rec.get("generated_text", "")
        if gen:
            entry["teacher_cot"] = gen
        augmented_data.append(entry)

    encodings = tokenize_for_seq_kd(tokenizer, augmented_data, max_length=max_length)

    batch_size = encodings["input_ids"].size(0)
    seq_len = encodings["input_ids"].size(1)

    all_topk_ids = torch.zeros(batch_size, seq_len, top_k, dtype=torch.long)
    all_topk_logprobs = torch.full((batch_size, seq_len, top_k), -100.0)
    all_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)

    for i in range(batch_size):
        rec = logprob_records[i]
        teacher_tokens = rec["tokens"]
        teacher_top = rec["top_logprobs"]  # list of list-of-dicts

        # Decode the student token IDs back to strings for alignment
        student_ids = encodings["input_ids"][i].tolist()
        student_tokens = [tokenizer.decode([tid]) for tid in student_ids]

        alignment, exact = _align_teacher_to_student(teacher_tokens, student_tokens)

        for si in range(min(seq_len, len(alignment))):
            # Only use logit KD where tokens align exactly and labels are active
            if not exact[si]:
                continue
            ti = alignment[si]
            if ti < 0 or ti >= len(teacher_top):
                continue
            if teacher_top[ti] is None:
                continue

            entries = teacher_top[ti][:top_k]
            for ki, entry in enumerate(entries):
                tok_str = entry["token"]
                logprob = entry["logprob"]
                tok_id = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(tok_str)[:1] or [tokenizer.unk_token]
                )
                if isinstance(tok_id, list):
                    tok_id = tok_id[0] if tok_id else tokenizer.unk_token_id
                all_topk_ids[i, si, ki] = tok_id
                all_topk_logprobs[i, si, ki] = logprob

            all_mask[i, si] = 1

    encodings["teacher_topk_ids"] = all_topk_ids
    encodings["teacher_topk_logprobs"] = all_topk_logprobs
    encodings["logit_kd_mask"] = all_mask
    return encodings


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class KDDataset(Dataset):
    """Wraps a dict of tensors (from tokenize_for_*) into a map-style Dataset."""

    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
