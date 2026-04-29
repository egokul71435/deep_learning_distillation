"""Model loading and LoRA setup utilities.

Handles device detection (CUDA/MPS/CPU), dtype selection, and wrapping a
pretrained causal LM with LoRA adapters. The default LoRA config targets
q_proj and v_proj at rank 16, which keeps trainable params under 0.2% for
3B-class models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_dtype(device=None):
    if device is None:
        device = get_device()
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_model(model_name, device=None, dtype=None):
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = get_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
    ).to(device)
    return tokenizer, model


def apply_lora(model, rank=16, alpha=32, target_modules=None, dropout=0.05):
    """Wrap *model* with LoRA adapters and freeze the base weights.

    Default ``rank=16`` on q_proj + v_proj keeps trainable params well under
    1 % for 3 B-class models (Llama-3.2-3B ≈ 0.17 %).

    Returns the PEFT-wrapped model and prints a trainable-parameter summary.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100.0 * trainable / total
    print(f"  LoRA applied  —  trainable: {trainable:,} / {total:,}  ({pct:.3f} %)")
    return model
