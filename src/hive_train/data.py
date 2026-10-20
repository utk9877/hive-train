"""Small deterministic CPU batches; token shifting happens exactly here."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Batch:
    inputs: torch.Tensor
    targets: torch.Tensor


def synthetic_batches(config, *, count=1, batch_size=2, length=32, seed=7):
    if any(type(v) is not int or v <= 0 for v in (count, batch_size, length)):
        raise ValueError("count, batch_size and length must be positive integers")
    if length > config.max_sequence_length:
        raise ValueError("sequence too long")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batches = []
    for _ in range(count):
        tokens = torch.randint(config.vocab_size, (batch_size, length + 1), generator=generator)
        batches.append(Batch(tokens[:, :-1].clone(), tokens[:, 1:].clone()))
    return batches


def validate_batches(config, batches):
    if not batches:
        raise ValueError("step requires at least one microbatch")
    valid = 0
    for batch in batches:
        x, y = batch.inputs, batch.targets
        if x.device.type != "cpu" or y.device.type != "cpu":
            raise ValueError("data inputs and targets must originate on CPU")
        if x.dtype != torch.int64 or y.dtype != torch.int64 or x.ndim != 2 or x.shape != y.shape:
            raise ValueError("inputs and targets must be matching int64 [b,T]")
        if x.shape[0] <= 0 or not 0 < x.shape[1] <= config.max_sequence_length:
            raise ValueError("invalid batch dimensions")
        if not bool(((x >= 0) & (x < config.vocab_size)).all()):
            raise ValueError("input token out of vocabulary")
        if not bool((((y >= 0) & (y < config.vocab_size)) | (y == -100)).all()):
            raise ValueError("target token out of vocabulary")
        valid += int((y != -100).sum())
    if valid == 0:
        raise ValueError("all-empty optimizer step")
    return valid
