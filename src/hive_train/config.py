"""Portable decoder workload and versioned numerical evaluation semantics."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math


MODEL_ARCHITECTURE = "decoder_v2"


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 4
    width: int = 128
    heads: int = 4
    mlp_width: int = 512
    vocab_size: int = 256
    max_sequence_length: int = 128

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.layers < 2 or self.width % self.heads:
            raise ValueError("require layers >= 2 and width divisible by heads")


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.lr <= 0 or self.eps <= 0 or self.weight_decay < 0:
            raise ValueError("invalid AdamW lr, eps or weight_decay")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW betas must be in [0, 1)")


def json_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
