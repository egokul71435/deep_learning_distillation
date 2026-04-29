"""Compute BERTScores for eval_results responses against teacher data.

Reads responses.jsonl from each model's eval_results folder, matches each
problem to its teacher_cot from the source teacher data, and computes
BERTScore using roberta-large embeddings directly. We do manual cosine
similarity over token embeddings instead of using the bert-score library,
which has compatibility issues with newer transformers versions.

Usage:
    python compute_bert_scores.py

    # Or specify which eval dirs and teacher dirs to compare:
    python compute_bert_scores.py \
        --eval-dirs eval_results/base eval_results/gpt_seq eval_results/llama_seq \
        --teacher-dirs gpt_teacher_data llama_teacher_data
"""

import argparse
import json
import os
import statistics

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_teacher_cots(teacher_dir):
    """Load all teacher test CoTs keyed by problemid."""
    cots = {}
    for ds in ("scibench", "theoremqa"):
        path = os.path.join(teacher_dir, f"{ds}_test_clean.jsonl")
        if not os.path.exists(path):
            continue
        for row in load_jsonl(path):
            pid = row.get("problemid", "").strip()
            cots[pid] = {"teacher_cot": row.get("teacher_cot", ""), "dataset": ds}
    return cots


@torch.inference_mode()
def encode_texts(model, tokenizer, texts, device, batch_size=32, max_length=512):
    """Encode texts into token-level embeddings using roberta-large."""
    all_embeddings = []
    all_masks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded)
        all_embeddings.append(outputs.last_hidden_state.cpu())
        all_masks.append(encoded["attention_mask"].cpu())
    return all_embeddings, all_masks


def bert_score_from_embeddings(cand_embs, cand_masks, ref_embs, ref_masks):
    """Compute BERTScore P/R/F1 from pre-computed embeddings."""
    precisions, recalls, f1s = [], [], []

    offset_c, offset_r = 0, 0
    ci, ri = 0, 0
    # Flatten batch tracking
    cand_list, ref_list = [], []
    for emb, mask in zip(cand_embs, cand_masks):
        for j in range(emb.size(0)):
            length = mask[j].sum().item()
            cand_list.append(F.normalize(emb[j, :length], dim=-1))
    for emb, mask in zip(ref_embs, ref_masks):
        for j in range(emb.size(0)):
            length = mask[j].sum().item()
            ref_list.append(F.normalize(emb[j, :length], dim=-1))

    for c_emb, r_emb in zip(cand_list, ref_list):
        sim = c_emb @ r_emb.T  # (cand_tokens, ref_tokens)
        precision = sim.max(dim=1).values.mean().item()
        recall = sim.max(dim=0).values.mean().item()
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return precisions, recalls, f1s


def compute_scores(model, tokenizer, student_cots, teacher_cots, device):
    """Compute BERTScore between paired student/teacher CoTs."""
    print(f"    Encoding {len(student_cots)} student responses …")
    c_embs, c_masks = encode_texts(model, tokenizer, student_cots, device)
    print(f"    Encoding {len(teacher_cots)} teacher responses …")
    r_embs, r_masks = encode_texts(model, tokenizer, teacher_cots, device)
    print(f"    Computing pairwise BERTScore …")
    return bert_score_from_embeddings(c_embs, c_masks, r_embs, r_masks)


def stats(vals):
    return {
        "mean": round(statistics.mean(vals), 4) if vals else 0.0,
        "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 4) if vals else 0.0,
        "max": round(max(vals), 4) if vals else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--eval-dirs", nargs="+",
        default=None,
        help="Paths to eval_results subdirs (auto-detected if omitted)",
    )
    parser.add_argument(
        "--teacher-dirs", nargs="+",
        default=["gpt_teacher_data", "llama_teacher_data"],
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Auto-detect eval dirs
    if args.eval_dirs is None:
        base = "eval_results"
        if os.path.isdir(base):
            args.eval_dirs = sorted(
                os.path.join(base, d) for d in os.listdir(base)
                if os.path.isfile(os.path.join(base, d, "responses.jsonl"))
            )
        else:
            args.eval_dirs = []
    if not args.eval_dirs:
        print("No eval_results dirs found with responses.jsonl. Exiting.")
        return

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load roberta-large with fast tokenizer
    print("Loading roberta-large …")
    tokenizer = AutoTokenizer.from_pretrained("roberta-large", use_fast=True)
    model = AutoModel.from_pretrained("roberta-large").to(device).eval()

    # Load all teacher CoTs
    all_teacher_cots = {}
    for td in args.teacher_dirs:
        if os.path.isdir(td):
            all_teacher_cots[td] = load_teacher_cots(td)
            print(f"  Loaded {len(all_teacher_cots[td])} teacher CoTs from {td}")

    # Process each eval dir
    for eval_dir in args.eval_dirs:
        responses_path = os.path.join(eval_dir, "responses.jsonl")
        if not os.path.isfile(responses_path):
            print(f"  Skipping {eval_dir} — no responses.jsonl")
            continue

        responses = load_jsonl(responses_path)
        model_name = os.path.basename(eval_dir)
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}  ({len(responses)} responses)")
        print(f"{'='*60}")

        for teacher_name, teacher_cots in all_teacher_cots.items():
            results_by_dataset = {}

            # Match responses to teacher CoTs by problemid
            paired = {"scibench": ([], []), "theoremqa": ([], [])}
            for resp in responses:
                pid = resp.get("problemid", "").strip()
                if pid in teacher_cots:
                    ds = teacher_cots[pid]["dataset"]
                    paired[ds][0].append(resp["student_cot"])
                    paired[ds][1].append(teacher_cots[pid]["teacher_cot"])

            overall_p, overall_r, overall_f1 = [], [], []

            for ds_name in ("scibench", "theoremqa"):
                s_cots, t_cots = paired[ds_name]
                if not s_cots:
                    continue

                print(f"\n  vs {teacher_name}/{ds_name} ({len(s_cots)} pairs)")
                p, r, f1 = compute_scores(model, tokenizer, s_cots, t_cots, device)
                overall_p.extend(p)
                overall_r.extend(r)
                overall_f1.extend(f1)

                results_by_dataset[ds_name] = {
                    "count": len(s_cots),
                    "precision": stats(p),
                    "recall": stats(r),
                    "f1": stats(f1),
                    "per_item_f1": [round(v, 4) for v in f1],
                }
                print(f"    BERTScore F1: {stats(f1)['mean']:.4f} "
                      f"(P={stats(p)['mean']:.4f}, R={stats(r)['mean']:.4f})")

            if overall_f1:
                print(f"\n  Overall vs {teacher_name}:")
                print(f"    BERTScore F1: {stats(overall_f1)['mean']:.4f} "
                      f"(P={stats(overall_p)['mean']:.4f}, R={stats(overall_r)['mean']:.4f})")

            # Write results
            out = {
                "model": model_name,
                "teacher": teacher_name,
                "overall": {
                    "count": len(overall_f1),
                    "precision": stats(overall_p),
                    "recall": stats(overall_r),
                    "f1": stats(overall_f1),
                },
                "datasets": results_by_dataset,
            }
            out_path = os.path.join(eval_dir, f"bert_scores_vs_{teacher_name}.json")
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()