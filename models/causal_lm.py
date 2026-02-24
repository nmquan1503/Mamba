import torch
import torch.nn as nn
from dataclasses import dataclass

from ..modules import RMSNorm, MambaSimple

@dataclass
class CausalLMConfig:
    vocab_size: int = 32000
    model_dim: int = 512
    state_dim: int = 16
    conv_kernel: int = 4
    expansion_factor: int = 2
    dropout_rate: float = 0.15
    num_layers: int = 5
    tie_embeddings: bool = True

class Block(nn.Module):
    def __init__(self, config: CausalLMConfig):
        super().__init__()
        self.norm = RMSNorm(config.model_dim)
        self.mamba = MambaSimple(
            model_dim=config.model_dim,
            state_dim=config.state_dim,
            conv_kernel=config.conv_kernel,
            expansion_factor=config.expansion_factor
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mamba(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += residual

        return hidden_states

class CausalLM(nn.Module):
    def __init__(self, config: CausalLMConfig):
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.layers = nn.ModuleList([
            Block(config)
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.model_dim)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
    
    def forward(self, input_ids):
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        pass