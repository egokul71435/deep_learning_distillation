"""Custom HuggingFace Trainer for knowledge distillation.

Supports three training modes in a single class: pure sequence KD (cross-entropy
on teacher text), pure logit KD (sparse KL divergence from cached top-k teacher
logprobs), or a weighted combination of both controlled by the alpha parameter.
The sparse KL loss reconstructs a teacher distribution from pre-fetched top-k
log-probabilities and computes KL divergence over that shared support, with
temperature scaling and the standard T-squared gradient correction.
"""

import torch
import torch.nn.functional as F
from transformers import Trainer


class KDTrainer(Trainer):
    """Knowledge-distillation trainer supporting three modes.

    1. **Offline sequence KD** (``alpha = 1.0``, no logprob tensors in batch):
       Standard cross-entropy on teacher-generated text.  No teacher model is
       needed — the labels already encode the teacher's output.

    2. **Sparse logit KD** (``alpha = 0.0``, logprob tensors present):
       KL divergence between the student's distribution and a sparse teacher
       distribution reconstructed from pre-fetched top-k log-probabilities
       (e.g. from Together AI).

    3. **Combined** (``0 < alpha < 1``, logprob tensors present):
       Weighted sum of (1) and (2).

    Parameters
    ----------
    alpha : float
        Weight for the sequence (CE) loss.  ``1 - alpha`` goes to logit KD.
    kd_temperature : float
        Temperature applied when computing the KL term.
    """

    def __init__(self, *args, alpha=1.0, kd_temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.kd_temperature = kd_temperature

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_topk_ids = inputs.pop("teacher_topk_ids", None)
        teacher_topk_logprobs = inputs.pop("teacher_topk_logprobs", None)
        logit_kd_mask = inputs.pop("logit_kd_mask", None)

        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # --- Sequence KD: cross-entropy on teacher text -----------------
        seq_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # --- Logit KD (sparse, from cached top-k) ----------------------
        has_logit_kd = (
            teacher_topk_ids is not None
            and teacher_topk_logprobs is not None
            and self.alpha < 1.0
        )

        if has_logit_kd:
            logit_loss = self._sparse_kl_loss(
                shift_logits,
                shift_labels,
                teacher_topk_ids,
                teacher_topk_logprobs,
                logit_kd_mask,
            )
            total_loss = self.alpha * seq_loss + (1.0 - self.alpha) * logit_loss
        else:
            total_loss = seq_loss

        return (total_loss, outputs) if return_outputs else total_loss

    # ------------------------------------------------------------------

    def _sparse_kl_loss(
        self,
        student_logits,
        labels,
        teacher_topk_ids,
        teacher_topk_logprobs,
        logit_kd_mask,
    ):
        """KL divergence from a sparse teacher distribution (top-k only).

        The teacher distribution is constructed by exponentiating the cached
        log-probs (optionally temperature-scaled) and renormalising over the
        top-k support.  The student distribution is the full softmax (also
        temperature-scaled), but we only gather the entries at the teacher's
        top-k token IDs so the KL is computed over the shared support.
        """
        # Shift teacher tensors to match the autoregressive shift
        t_ids = teacher_topk_ids[:, 1:, :].contiguous()
        t_lp = teacher_topk_logprobs[:, 1:, :].contiguous()

        if logit_kd_mask is not None:
            kd_mask = logit_kd_mask[:, 1:].contiguous()
        else:
            kd_mask = (labels != -100).long()

        # Also require that label positions are active
        label_mask = (labels != -100).long()
        mask = (kd_mask * label_mask).float()  # (B, S)

        T = self.kd_temperature

        # Student log-probs at temperature T
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)  # (B, S, V)

        # Gather student log-probs at the teacher's top-k positions
        student_topk_lp = student_log_probs.gather(-1, t_ids)  # (B, S, K)

        # Teacher probs at temperature T, renormalised over top-k
        teacher_topk_probs = (t_lp / T).exp()
        teacher_topk_probs = teacher_topk_probs / teacher_topk_probs.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)

        # KL(teacher || student) = Σ_k p_t * (log p_t − log p_s)
        kl = teacher_topk_probs * (
            teacher_topk_probs.clamp(min=1e-8).log() - student_topk_lp
        )
        kl = kl.sum(dim=-1)  # (B, S)

        kl = (kl * mask).sum() / mask.sum().clamp(min=1)
        kl = kl * (T ** 2)
        return kl
