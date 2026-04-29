"""Pre-fetch top-k log-probabilities from a Llama model via Together AI.

Uses the completions (not chat) endpoint since it reliably returns logprobs.
The problem text and a generation prompt are concatenated into a single text
prompt, and the model generates a completion whose per-token top-k logprobs
we capture. Results are saved to JSONL with resume support so you don't lose
progress if it crashes partway through.

The output JSONL can later be loaded by dataset.tokenize_for_logit_kd.

Usage:
    export TOGETHER_API_KEY="your-key-here"

    python fetch_logprobs.py \
        --input  llama_teacher_data/scibench_train_clean.jsonl \
        --output llama_teacher_data/scibench_train_logprobs.jsonl \
        --model  meta-llama/Llama-3.3-70B-Instruct-Turbo \
        --top-k  5
"""

import argparse
import json
import os
import sys
import time

import requests as _requests


API_BASE = "https://api.together.xyz/v1"


def get_api_key():
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        sys.exit("Set the TOGETHER_API_KEY environment variable first.")
    return key


def fetch_logprobs_for_example(api_key, model, problem_text,
                                top_k=5, max_tokens=4096, max_retries=3):
    """Generate a completion and collect per-token top-k logprobs.

    Uses the raw REST completions endpoint (``/v1/completions``) which
    reliably returns ``logprobs`` for both Turbo and non-Turbo models on
    Together AI.
    """
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{problem_text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "logprobs": top_k,
        "echo": False,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            resp = _requests.post(
                f"{API_BASE}/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )

            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

            data = resp.json()
            choice = data["choices"][0]
            generated_text = choice.get("text", "")
            lp = choice.get("logprobs")

            if lp is None:
                return _empty_record(generated_text)

            tokens = lp.get("tokens", [])
            token_logprobs = lp.get("token_logprobs", [])
            raw_top = lp.get("top_logprobs", [])

            top_logprobs = []
            for entry in raw_top:
                if entry is None:
                    top_logprobs.append([])
                    continue
                alts = []
                for tok_str, logprob_val in entry.items():
                    alts.append({"token": tok_str, "logprob": logprob_val})
                alts.sort(key=lambda x: x["logprob"], reverse=True)
                top_logprobs.append(alts[:top_k])

            return {
                "tokens": tokens,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
                "generated_text": generated_text,
            }

        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}/{max_retries}] {exc} — waiting {wait}s")
            time.sleep(wait)

    return None


def _empty_record(generated_text=""):
    return {
        "tokens": [],
        "token_logprobs": [],
        "top_logprobs": [],
        "generated_text": generated_text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to *_clean.jsonl")
    parser.add_argument("--output", required=True, help="Output JSONL for logprobs")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        help="Together AI model identifier",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens for the teacher's generated completion")
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds to wait between API calls (rate-limit courtesy)",
    )
    args = parser.parse_args()

    data = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    api_key = get_api_key()
    out_path = args.output
    already_done = 0
    if os.path.exists(out_path):
        with open(out_path) as f:
            already_done = sum(1 for _ in f)
        print(f"Resuming — {already_done}/{len(data)} already fetched.")

    with open(out_path, "a") as fout:
        for idx in range(already_done, len(data)):
            row = data[idx]
            problem = row["problem_text"]
            print(f"[{idx+1}/{len(data)}] fetching logprobs …", end=" ", flush=True)

            record = fetch_logprobs_for_example(
                api_key, args.model, problem,
                top_k=args.top_k, max_tokens=args.max_tokens,
            )

            if record is None:
                print("FAILED — writing empty record")
                record = _empty_record()

            n_tokens = len(record["tokens"])
            record["problemid"] = row.get("problemid", "")

            if idx == 0 and n_tokens > 0:
                print(f"ok ({n_tokens} tokens) — first 3 tokens: {record['tokens'][:3]}")
            elif idx == 0 and n_tokens == 0:
                print(f"WARNING: 0 tokens returned. API may not support logprobs for this model.")
            else:
                print(f"ok ({n_tokens} tokens)")

            fout.write(json.dumps(record) + "\n")
            fout.flush()

            if args.delay > 0:
                time.sleep(args.delay)

    print(f"Done. Wrote {len(data)} records to {out_path}")


if __name__ == "__main__":
    main()
