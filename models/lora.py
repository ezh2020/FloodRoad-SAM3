from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scale
        return base_out + lora


def apply_lora(
    module: nn.Module,
    target_keywords: Iterable[str],
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> List[str]:
    """Replace matching Linear layers with LoRA adapters.

    Returns the fully qualified names that were adapted. The function is deliberately
    conservative: a layer is adapted only when its name contains one of the target
    keywords. This keeps it compatible with varied SAM-like transformer layouts.
    """

    keywords = tuple(target_keywords)
    changed: List[str] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(k in full_name for k in keywords):
                setattr(parent, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                changed.append(full_name)
            else:
                visit(child, full_name)

    visit(module)
    return changed


def lora_parameters(module: nn.Module):
    for name, param in module.named_parameters():
        if "lora_" in name and param.requires_grad:
            yield param

