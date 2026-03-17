"""
Minimal LoRA (Low-Rank Adaptation) implementation for nanochat GPT models.
No external dependencies — just wraps nn.Linear with low-rank adapters.

Usage:
    model, tokenizer, meta = load_model("rl", device, phase="train", model_tag="d24")
    num_lora = apply_lora(model, rank=8, alpha=8.0)
    # ... train only LoRA params ...
    merged_sd = get_merged_state_dict(model)  # for saving eval-compatible checkpoint
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear layer with low-rank adapter. Base weights are frozen."""

    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.scaling = alpha / rank
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        # Match device and dtype of the base model weights
        device = base_linear.weight.device
        dtype = base_linear.weight.dtype
        # A: (in_features, rank) — projects input down to rank
        self.lora_A = nn.Parameter(torch.empty(in_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B: (rank, out_features) — projects back up; zero init so LoRA starts as identity
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device, dtype=dtype))
        # Freeze the base weights
        base_linear.weight.requires_grad = False
        if base_linear.bias is not None:
            base_linear.bias.requires_grad = False

    def forward(self, x):
        base_out = self.base_linear(x)
        # Keep master LoRA weights in fp32 for optimization precision, but run
        # the low-rank matmuls in the activation dtype just like nanochat.Linear.
        lora_A = self.lora_A.to(dtype=x.dtype)
        lora_B = self.lora_B.to(dtype=x.dtype)
        lora_out = (x @ lora_A) @ lora_B
        return base_out + lora_out * self.scaling

    @property
    def merged_weight(self):
        """Return base weight + LoRA contribution, shape (out_features, in_features)."""
        # base weight is (out, in), lora contribution is (A @ B).T = B.T @ A.T which is (out, in)
        return self.base_linear.weight + (self.lora_B.T @ self.lora_A.T) * self.scaling


def apply_lora(model, rank=8, alpha=8.0):
    """
    Replace target nn.Linear layers with LoRA-wrapped versions. Freezes all base params.

    Targets (matching HF pipeline's all-linear-layers approach):
      - c_q, c_k, c_v  — attention Q/K/V projections
      - c_proj          — attention output + MLP down projection
      - c_fc            — MLP up projection

    Returns the number of LoRA adapters applied.
    """
    target_names = {"c_q", "c_k", "c_v", "c_proj", "c_fc"}

    # Freeze all params first
    for param in model.parameters():
        param.requires_grad = False

    # Replace matching linear layers with LoRA wrappers
    count = 0
    for module_name, module in list(model.named_modules()):
        attr_name = module_name.split(".")[-1]
        if isinstance(module, nn.Linear) and attr_name in target_names:
            # Navigate to parent module
            parts = module_name.rsplit(".", 1)
            if len(parts) == 2:
                parent = model.get_submodule(parts[0])
            else:
                parent = model
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, attr_name, lora_layer)
            count += 1

    return count


def get_merged_state_dict(model):
    """
    Return a state dict with LoRA weights merged back into base weights.
    Keys use original names (e.g. transformer.h.0.attn.c_q.weight) so the
    checkpoint is compatible with the unmodified GPT model for evaluation.
    """
    result = {}
    raw_sd = model.state_dict()
    for key, value in raw_sd.items():
        if ".lora_A" in key or ".lora_B" in key:
            # Skip LoRA parameters (they're folded into the merged weight)
            continue
        elif ".base_linear." in key:
            # Replace base_linear.weight with merged weight using original key name
            module_path = key.rsplit(".base_linear.", 1)[0]
            original_key = key.replace(".base_linear.", ".")
            if original_key not in result:
                # Get the LoRA module and compute merged weight
                module = model.get_submodule(module_path)
                result[original_key] = module.merged_weight.detach().clone()
        else:
            result[key] = value
    return result


def get_lora_state_dict(model):
    """
    Return LoRA-only parameters (A/B matrices) from a model wrapped with LoRA.
    """
    raw_sd = model.state_dict()
    return {
        key: value
        for key, value in raw_sd.items()
        if ".lora_A" in key or ".lora_B" in key
    }


def count_parameters(model):
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
