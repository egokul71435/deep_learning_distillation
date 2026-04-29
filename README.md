# Knowledge Distillation for Scientific Reasoning

This project explores knowledge distillation as a way to transfer the reasoning ability of large language models into a much smaller one. The idea is straightforward: take a big teacher model that's good at solving science and math problems, have it generate chain-of-thought solutions, and then train a small student model to imitate that reasoning.

We tried two different distillation approaches, sequence-level and logit-level, across two different teachers, and compared what each one actually buys you. The interesting finding was that plain sequence KD (just training on the teacher's text) works surprisingly well, while logit KD (using the teacher's token-level probability distribution) adds value but is harder to get right due to tokenizer misalignment between models.

Everything runs on Apple Silicon. No cloud GPUs were used for training.

## Overview

| | |
|---|---|
| **Student** | Llama-3.2-3B-Instruct |
| **Teachers** | GPT-oss-120B (via OpenRouter), Llama-3.3-70B-Instruct-Turbo (via Together AI) |
| **Methods** | Offline sequence KD, sparse logit KD |
| **Training** | LoRA adapters only, 0.17% of parameters trainable |
| **Benchmarks** | SciBench (university science), TheoremQA (math/theorems) |
| **Evaluation** | Numeric accuracy + BERTScore F1 against teacher CoTs |
| **Hardware** | Apple Silicon (MPS / CPU) |

---

## Distillation Approaches

### Sequence KD

The simpler of the two. Each teacher generates CoT solutions offline, saved as JSONL. The student trains on these with cross-entropy. It's basically supervised fine-tuning on teacher-written text. The teacher model is never loaded during training, which keeps the whole thing runnable on a laptop.

The loss only applies to the answer portion. We mask out the question prefix so the student isn't wasting capacity learning to reproduce the prompt.

### Logit KD

This goes a step further. On top of the teacher's text, we also collect the teacher's per-token top-k log-probabilities through the Together AI completions API. During training, these get turned into a sparse teacher distribution, and the student is trained to match it via KL divergence:

```
L_total = α × CE(student, teacher_text) + (1 - α) × T² × KL(teacher_topk || student)
```

The tricky part is token alignment. The 70B teacher and 3B student don't tokenize identically, so we wrote an alignment step that maps student tokens to teacher tokens by character offset and only applies the KL loss where they agree exactly. Everywhere else, the student just gets the sequence-level signal. In practice this means logit KD coverage is partial, which probably explains why it doesn't blow sequence KD out of the water.

### Why sparse KL?

Inference APIs don't give you the full vocabulary distribution. You only get the top-k logprobs back. So we reconstruct a teacher distribution by exponentiating and renormalizing over just those top-k entries, then compute KL on that shared support. It's an approximation, but it captures the teacher's confidence over the tokens that actually matter.

---

## Experiments

We ran four configurations, keeping everything else constant (same student, same LoRA config, same training hyperparameters) so the comparison is fair:

| Name | Teacher | Strategy | Alpha |
|---|---|---|---|
| **base** | none | no adapter, raw Llama-3.2-3B | n/a |
| **gpt_seq** | GPT-oss-120B | sequence KD | 1.0 |
| **llama_seq** | Llama-3.3-70B | sequence KD | 1.0 |
| **llama_logit** | Llama-3.3-70B | sequence + logit KD | 0.5 |

LoRA was the same everywhere: rank 16, alpha 32, targeting `q_proj` and `v_proj`, 5% dropout. Training was 1 epoch, AdamW at 2e-4, batch size 4 with 4 gradient accumulation steps.

---

## Results

### BERTScore F1

This measures how closely the student's generated reasoning matches the teacher's reference CoT, using roberta-large embeddings. It captures whether the student learned to reason like the teacher, not just whether it got the right number at the end.

**Compared to GPT teacher CoTs:**

| Model | SciBench | TheoremQA | Overall |
|---|---|---|---|
| base | 0.9503 | 0.9487 | 0.9494 |
| gpt_seq | **0.9648** | **0.9590** | **0.9613** |
| llama_seq | 0.9519 | 0.9513 | 0.9516 |
| llama_logit | 0.9491 | 0.9499 | 0.9498 |

**Compared to Llama teacher CoTs:**

| Model | SciBench | TheoremQA | Overall |
|---|---|---|---|
| base | 0.9648 | 0.9631 | 0.9639 |
| gpt_seq | 0.9616 | 0.9576 | 0.9595 |
| llama_seq | **0.9696** | **0.9691** | **0.9693** |
| llama_logit | 0.9667 | 0.9643 | 0.9662 |

### Numeric Accuracy

Extracted the final number from each response and compared to ground truth with 5% tolerance.

| Model | SciBench | TheoremQA | Overall |
|---|---|---|---|
| base | 9.8% | 15.7% | 13.3% |
| gpt_seq | 11.1% | 15.7% | 13.8% |
| llama_seq | 9.2% | 13.3% | 11.5% |
| llama_logit | **14.7%** | **19.1%** | **17.2%** |

These numbers are low across the board, which is expected for university-level problems on a 3B model. That said, logit KD produced the best accuracies by a clear margin (17.2% overall vs 13.3% for the base), even though it didn't score the highest BERTScores. This suggests the token-level probability signal from the teacher helps the student get to the right answer more often, even when the overall reasoning style doesn't match the teacher as closely as sequence KD does.

### What we took away from the results

Each distilled model scores highest against its own teacher, which makes sense. The student learns to reason in the teacher's style. GPT-seq produces the most GPT-like CoTs, Llama-seq produces the most Llama-like CoTs.

Sequence KD beat logit KD in BERTScore across the board. We think this is mostly a tokenizer alignment problem. The logit KD signal only applies at positions where the teacher and student tokenize a word identically, and for a 70B vs 3B model pair that coverage ends up being partial. The sparse KL on those positions does help (llama_logit beats the base model) but it doesn't overcome the coverage gap compared to sequence KD, which supervises every token.

If we could get full-vocabulary distributions from the teacher (not just top-k), or if both models shared a tokenizer, logit KD would likely do better. That's the main bottleneck.

---

## Project Structure

```
deep_learning_distillation/
├── README.md
│
│   # Core code
├── model.py                     # model loading, device detection, LoRA wrapper
├── dataset.py                   # tokenization + teacher-student token alignment
├── kd_trainer.py                # custom Trainer: CE loss + sparse KL loss
├── train_gpt_seq.py             # train with GPT teacher (sequence KD)
├── train_llama_seq.py           # train with Llama teacher (sequence KD)
├── train_llama_logit.py         # train with Llama teacher (logit + sequence KD)
├── fetch_logprobs.py            # fetch top-k logprobs from Together AI
├── evaluate.py                  # generate + score student answers
├── compute_bert_scores.py       # BERTScore computation (roberta-large)
├── test_kd.py                   # smoke tests for all code paths
├── requirements.txt
│
│   # Teacher data
├── gpt_teacher_data/            # GPT-oss-120B CoTs (SciBench + TheoremQA, train/test)
├── llama_teacher_data/          # Llama-70B CoTs + pre-fetched logprobs
│
│   # Results
├── eval_results/
│   ├── base/                    # baseline (no adapter)
│   ├── gpt_seq/                 # GPT sequence-KD model
│   ├── llama_seq/               # Llama sequence-KD model
│   └── llama_logit/             # Llama logit-KD model
│
│   # Trained adapters
└── output/
    ├── gpt_seq/final/           # LoRA weights (safetensors)
    ├── llama_seq/final/
    └── llama_logit_kd/final/
```

Each `eval_results/` subfolder contains `responses.jsonl` (per-problem predictions) and `bert_scores_vs_*.json` files (BERTScore breakdowns against each teacher's CoTs). Some also have `metrics.json` with accuracy numbers.

---

## How the Code Fits Together

**`kd_trainer.py`** is the center of it all. It subclasses HuggingFace's Trainer and handles three modes depending on what's in the batch: pure sequence KD (alpha=1, no logprobs), pure logit KD (alpha=0), or a mix. The sparse KL computation temperature-scales both sides, gathers the student's log-softmax at the teacher's top-k positions, renormalizes the teacher's top-k into a proper distribution, and computes forward KL with the standard T² correction.

**`dataset.py`** handles the fiddly tokenization work. For sequence KD it's straightforward: tokenize, mask the question prefix. For logit KD it also needs to align teacher tokens to student tokens by character offset. The alignment function walks both token sequences in parallel using character positions and flags exact matches. Only those positions get the KL loss.

**`fetch_logprobs.py`** hits the Together AI `/v1/completions` endpoint (not chat, since the chat endpoint doesn't reliably return logprobs) and saves per-token top-k logprobs to JSONL. It supports resume, so if it crashes partway through you don't lose progress.

**`evaluate.py`** generates a CoT for each test problem, tries to extract a numeric answer (checking for `\boxed{}`, "the answer is", trailing equations, last number in sequence), and compares against ground truth. It also computes BERTScore inline.

**`compute_bert_scores.py`** is a standalone version that re-computes BERTScores across all eval directories and teacher datasets. It uses roberta-large embeddings directly with manual cosine similarity rather than the `bert-score` library, since the library has compatibility issues with newer transformers versions.

---

## Running It

### Install dependencies

```bash
pip install -r requirements.txt
```

### Smoke test

```bash
python test_kd.py
```

Runs quick sanity checks, just makes sure nothing crashes before you commit to a real training run.

### Training

```bash
python train_gpt_seq.py --dataset both
python train_llama_seq.py --dataset both
python train_llama_logit.py --dataset both --alpha 0.5 --kd-temperature 2.0
```

### Fetching logprobs (needed for logit KD)

```bash
export TOGETHER_API_KEY="your-key"
python fetch_logprobs.py \
    --input  llama_teacher_data/scibench_train_clean.jsonl \
    --output llama_teacher_data/scibench_train_logprobs.jsonl
```

### Evaluation

```bash
python evaluate.py \
    --adapter-path ./output/gpt_seq/final \
    --base-model meta-llama/Llama-3.2-3B-Instruct \
    --teacher-dir gpt_teacher_data \
    --output-dir ./eval_results/gpt_seq
```

---

## What We'd Do Differently

**More data.** SciBench and TheoremQA together only give us a few hundred training examples, which is really not a lot for fine-tuning even with LoRA. We could have pulled in more math and science datasets (MATH, GSM8K, ARC, etc.) and generated teacher CoTs for those as well. More volume of teacher data would almost certainly improve both BERTScore and accuracy, especially for logit KD where the model needs enough examples to learn from the probability-level signal.

**More compute, full fine-tuning.** We used LoRA throughout because of time and compute constraints. Training on a single Apple Silicon machine meant we couldn't realistically do full fine-tuning on a 3B model, and each experiment needed to finish in a reasonable timeframe. LoRA let us train only ~0.17% of the student's parameters, which made the whole thing feasible but also limited how much the student could actually change its behavior. With access to GPUs and more time, full fine-tuning (or at least higher-rank LoRA with more target modules) would likely give the student more room to internalize the teacher's reasoning patterns.

**Longer training.** Every experiment was capped at 1 epoch. The loss was still decreasing at the end of each run, which means we left performance on the table. Multiple epochs with a proper learning rate schedule (cosine decay, warmup) would help, especially since the datasets are small enough that the model hasn't fully converged after one pass.

**Tokenizer alignment in logit KD.** This is probably the biggest technical limitation. The teacher and student use different tokenizers, so the logit-level supervision only applies at positions where both tokenizers produce the exact same token. A lot of positions get skipped. If both models shared a tokenizer, or if we implemented a subword-merging strategy to aggregate logprobs across token boundaries, the logit KD signal would be much denser and cleaner.

**Hyperparameter tuning.** The alpha=0.5 and temperature=2.0 for logit KD were set by intuition, not tuned. A sweep over these could find a better balance between the sequence and logit losses. Same goes for LoRA rank and learning rate. We picked reasonable defaults but didn't have the compute budget to search properly.

**Larger student model.** A 3B model is fundamentally limited on university-level math and science. The accuracy numbers reflect this. With more compute we'd try a 7B or 13B student, where there's more capacity to absorb the teacher's reasoning and the accuracy gap would likely shrink significantly.

**Teacher quality evaluation.** We never directly measured how accurate the teachers themselves are on these benchmarks. Knowing the teacher's accuracy would help contextualize the student results, since the student can't do better than the data it's trained on.

---

## Dependencies

```
torch>=2.1.0
transformers>=4.38.0
accelerate>=0.27.0
peft>=0.10.0
openai>=1.12.0
bert-score>=0.3.13
```
