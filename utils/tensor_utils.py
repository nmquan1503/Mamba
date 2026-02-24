import torch

def ensure_contiguous(x: torch.Tensor | None):
    if x is None or x.is_contiguous():
        return x
    return x.contiguous()
