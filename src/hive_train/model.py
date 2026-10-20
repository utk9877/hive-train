"""Eager FP32 decoder stages with stable global parameter names."""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, device):
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        for name in ("query", "key", "value", "output"):
            setattr(self, name, nn.Linear(config.width, config.width,
                                          device=device, dtype=torch.float32))

    def forward(self, x):
        b, t, d = x.shape
        def heads(projection):
            return projection(x).reshape(b, t, self.heads, self.head_width).transpose(1, 2)
        q, k, v = heads(self.query), heads(self.key), heads(self.value)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        probabilities = F.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
        hidden = (probabilities @ v).transpose(1, 2).contiguous().reshape(b, t, d)
        return self.output(hidden)


class Block(nn.Module):
    def __init__(self, config: ModelConfig, device):
        super().__init__()
        options = dict(device=device, dtype=torch.float32)
        self.norm1 = nn.LayerNorm(config.width, eps=1e-5, **options)
        self.attention = Attention(config, device)
        self.norm2 = nn.LayerNorm(config.width, eps=1e-5, **options)
        self.expand = nn.Linear(config.width, config.mlp_width, **options)
        self.contract = nn.Linear(config.mlp_width, config.width, **options)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.contract(F.gelu(self.expand(self.norm2(x)), approximate="none"))


class DecoderStage(nn.Module):
    """Only constructs owned blocks; [0,L) is the unsplit reference.

    Default meta construction allocates no parameter storage and consumes no CPU
    initialization RNG. Use build_stage to materialize and initialize parameters.
    """
    def __init__(self, config: ModelConfig, start: int, end: int, *, device="meta"):
        super().__init__()
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= config.layers:
            raise ValueError("invalid contiguous block interval")
        self.config, self.start, self.end = config, start, end
        options = dict(device=device, dtype=torch.float32)
        if start == 0:
            self.token_embedding = nn.Embedding(config.vocab_size, config.width, **options)
            self.position_embedding = nn.Embedding(config.max_sequence_length, config.width, **options)
        self.blocks = nn.ModuleDict({str(i): Block(config, device) for i in range(start, end)})
        if end == config.layers:
            self.final_norm = nn.LayerNorm(config.width, eps=1e-5, **options)
            self.output_head = nn.Linear(config.width, config.vocab_size, bias=False, **options)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        if self.start == 0:
            if x.dtype != torch.int64 or x.ndim != 2:
                raise ValueError("prefix input must be int64 [batch, sequence]")
            if not 0 < x.shape[1] <= self.config.max_sequence_length or x.shape[0] == 0:
                raise ValueError("invalid input dimensions")
            positions = torch.arange(x.shape[1], device=x.device)
            x = self.token_embedding(x) + self.position_embedding(positions)
        elif (x.dtype != torch.float32 or x.ndim != 3 or x.shape[2] != self.config.width
              or x.shape[0] == 0 or not 0 < x.shape[1] <= self.config.max_sequence_length):
            raise ValueError("suffix input must be float32 [batch, sequence, width]")
        for block in self.blocks.values():
            x = block(x)
        if self.end == self.config.layers:
            x = self.output_head(self.final_norm(x))
        return x


def build_stage(config=ModelConfig(), start=0, end=None, *, device="cpu", seed=2026):
    from .initialization import initialize_named
    stage = DecoderStage(config, start, config.layers if end is None else end)
    stage.to_empty(device=device)
    initialize_named(stage, seed)
    return stage
