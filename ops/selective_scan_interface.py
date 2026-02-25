import torch
import torch.nn.functional as F
import selective_scan

from ..utils.tensor_utils import ensure_contiguous

class SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        u: torch.Tensor, 
        delta: torch.Tensor, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        C: torch.Tensor, 
        D: torch.Tensor, 
        delta_bias: torch.Tensor | None = None, 
        use_delta_softplus=True, 
    ):
        """
        Args:
            u: [batch_size, num_channels, seq_len]
            delta: [batch_size, num_channels, seq_len]
            A: [num_channels, state_dim]
            B: [batch_size, num_groups, state_dim, seq_len]
            C: [batch_size, num_groups, state_dim, seq_len]
            D: [num_channels]
            delta_bias: [num_channels]
        """

        u = ensure_contiguous(u);
        delta = ensure_contiguous(delta)
        A = ensure_contiguous(A)
        B = ensure_contiguous(B)
        C = ensure_contiguous(C)
        D = ensure_contiguous(D)
        delta_bias = ensure_contiguous(delta_bias)
        
        out, x = selective_scan.forward(u, delta, A, B, C, D, delta_bias, use_delta_softplus)
        ctx.use_delta_softplus = use_delta_softplus
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    def backward(ctx, dout):
        dout = ensure_contiguous(dout)
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        du, ddelta, dA, dB, dC, dD, ddelta_bias = selective_scan.backward(u, delta, A, B, C, D, delta_bias, dout, x, ctx.use_delta_softplus)
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None