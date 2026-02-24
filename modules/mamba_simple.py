import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..ops.selective_scan_interface import SelectiveScanFn

class MambaSimple(nn.Module):
    def __init__(
        self,
        model_dim: int,
        state_dim: int = 16,
        conv_kernel: int = 4,
        expansion_factor: int = 2,
        delta_rank: str | int = "auto",
        delta_min: float = 0.001,
        delta_max: float = 0.1,
        delta_init: str = "random",
        delta_scale: float = 1.0,
        delta_init_floor: float = 1e-4,
        conv_bias: bool = False,
        bias: bool = False,
        device: str | None = None
    ):
        super().__init__()

        self.model_dim = model_dim
        self.state_dim = state_dim
        self.conv_kernel = conv_kernel
        self.expansion_factor = expansion_factor
        self.inner_dim = int(expansion_factor * model_dim)
        self.delta_rank = math.ceil(model_dim / 16) if delta_rank == "auto" else delta_rank

        self.input_proj = nn.Linear(
            self.model_dim, 
            self.inner_dim * 2, 
            bias=bias,
        )

        self.short_conv = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=self.inner_dim,
            bias=conv_bias,
        )

        self.act = nn.SiLU()

        self.ssm_param_proj = nn.Linear(
            self.inner_dim, 
            self.delta_rank + self.state_dim * 2, 
            bias=False,
        )
        self.delta_proj = nn.Linear(
            self.delta_rank, 
            self.inner_dim, 
            bias=True, 
        )

        delta_std = self.delta_rank ** -0.5 * delta_scale
        if delta_init == "constant":
            nn.init.constant_(self.delta_proj.weight, delta_std)
        elif delta_init == "random":
            nn.init.uniform_(self.delta_proj.weight, -delta_std, delta_std)
        else:
            raise NotImplementedError
    
        delta = torch.exp(
            torch.rand(self.inner_dim, device=device, dtype=torch.float32) 
            * (math.log(delta_max) - math.log(delta_min))
            + math.log(delta_min)
        ).clamp(min=delta_init_floor)

        inv_delta = delta + torch.log(-torch.expm1(-delta))
        with torch.no_grad():
            self.delta_proj.bias.copy_(inv_delta)

        A = torch.arange(1, state_dim + 1, device=device, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.inner_dim, state_dim).contiguous()
        self.log_A = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.inner_dim, device=device))

        self.output_proj = nn.Linear(self.inner_dim, model_dim, bias=bias)
    
    def forward(self, hidden_states):
        """
        Args: (batch_size, seq_len, model_dim)
        Returns: (batch_size, seq_len, model_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # (batch_size, inner_dim, seq_len)
        state_inputs, gate_logits = torch.chunk(
            self.input_proj(hidden_states).permute(0, 2, 1).contiguous(), 
            chunks=2, 
            dim=1
        )
        state_inputs = self.short_conv(state_inputs)[..., :seq_len]
        state_inputs = self.act(state_inputs)

        ssm_params = self.ssm_param_proj(
            state_inputs.permute(0, 2, 1)
                .contiguous()
                .view(batch_size * seq_len, self.inner_dim)
        )
        
        # delta: (batch_size * seq_len, delta_rank)
        # B, C: (batch_size * seq_len, state_dim)
        delta, B, C = torch.split(
            ssm_params,
            [self.delta_rank, self.state_dim, self.state_dim],
            dim=-1
        )

        # (inner_dim, batch_size * seq_len)
        delta = self.delta_proj.weight @ delta.t()

        # (batch_size, inner_dim, seq_len)
        delta = delta.view(self.inner_dim, batch_size, seq_len).permute(1, 0, 2).contiguous()

        # (batch_size, num_groups, state_dim, seq_len)
        B = B.view(batch_size, seq_len, self.state_dim).permute(0, 2, 1).contiguous().unsqueeze(1)
        C = C.view(batch_size, seq_len, self.state_dim).permute(0, 2, 1).contiguous().unsqueeze(1)

        # (inner_dim, state_dim)
        A = -torch.exp(self.log_A.float())

        # (batch_size, inner_dim, seq_len)
        state_outputs = SelectiveScanFn.apply(
            state_inputs,
            delta,
            A,
            B,
            C,
            self.D.float(),
            self.delta_proj.bias.float(),
            True
        )

        # (batch_size, seq_len, inner_dim)
        hidden_states = state_outputs.permute(0, 2, 1).contiguous()
        hidden_states *= self.act(
            gate_logits.permute(0, 2, 1).contiguous()
        )

        # (batch_size, seq_len, model_dim)
        hidden_states = self.output_proj(hidden_states)

        return hidden_states