import torch
from torch.nn import Module


class HadamardLifting(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # exp(x) / sum(exp(x)), computed stably (naive exp overflows for small softmax_temp)
        return torch.softmax(x.flatten(), dim=0).view_as(x)
    

def get_DiT_depth(DiT_model: str):
    if DiT_model.startswith("DiT-XL"):
        return 28
    if DiT_model.startswith("DiT-L"):
        return 24
    if DiT_model.startswith("DiT-B") or DiT_model.startswith("DiT-S"):
        return 12
    raise ValueError(f"Unknown DiT model: {DiT_model}")
