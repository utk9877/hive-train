"""Small-model local correctness reference. No sockets or recovery semantics yet."""

from dataclasses import dataclass
import torch
from torch.nn import functional as F

from .boundary import BoundaryKey, LocalBoundary
from .config import OptimizerConfig
from .data import validate_batches


def optimizer_for(module, config=OptimizerConfig()):
    return torch.optim.AdamW(module.parameters(), lr=config.lr,
        betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay,
        foreach=False, fused=False, amsgrad=False)


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def finite_gradients(module):
    for name, p in module.named_parameters():
        if p.grad is None or not bool(torch.isfinite(p.grad).all()):
            raise ValueError(f"missing/nonfinite gradient: {name}")


def validate_ownership(stages):
    if len(stages) != 2:
        raise ValueError("M1 requires two stages")
    a, b = stages
    if a.config != b.config or a.start != 0 or a.end != b.start or b.end != a.config.layers:
        raise ValueError("stages do not cover one contiguous decoder")
    names_a, names_b = set(dict(a.named_parameters())), set(dict(b.named_parameters()))
    if names_a & names_b:
        raise ValueError("duplicate parameter ownership")


@dataclass
class StepResult:
    loss: float
    valid_tokens: int
    logits: list[torch.Tensor]
    boundary_gradients: list[torch.Tensor]


class ReferenceTrainer:
    """Owns either a whole decoder or two stages for a local correctness run.

    Validation failure poisons this trainer; create a new one from the fixture.
    This is deliberately not a fault-tolerant/distributed training application.
    """
    def __init__(self, stages, *, split, optimizer=OptimizerConfig()):
        self.stages = list(stages)
        self.split = split
        config = self.stages[0].config
        if type(split) is not int or not 1 <= split < config.layers:
            raise ValueError("invalid split")
        if len(self.stages) == 2:
            validate_ownership(self.stages)
            if self.stages[0].end != split:
                raise ValueError("split does not match stage ownership")
        elif len(self.stages) != 1 or self.stages[0].start != 0 or self.stages[0].end != config.layers:
            raise ValueError("expected a whole decoder or two contiguous stages")
        self.optimizers = [optimizer_for(s, optimizer) for s in self.stages]
        self.version = 0
        self.failed = False

    def named_parameters(self):
        return {n: p for stage in self.stages for n, p in stage.named_parameters()}

    def snapshot(self):
        """CPU snapshots are only for this bounded tiny-model test harness."""
        parameters, gradients, moments = {}, {}, {}
        for stage, optimizer in zip(self.stages, self.optimizers):
            for name, p in stage.named_parameters():
                parameters[name] = p.detach().to("cpu", copy=True)
                if p.grad is not None:
                    gradients[name] = p.grad.detach().to("cpu", copy=True)
                for key, tensor in optimizer.state.get(p, {}).items():
                    moments[f"{name}/{key}"] = tensor.detach().to("cpu", copy=True)
        return {"parameters": parameters, "gradients": gradients, "optimizer": moments}

    def step(self, batches):
        if self.failed:
            raise RuntimeError("trainer failed; restore a new instance, do not retry")
        # Validate all data before zero_grad or forward, including the N=0 case.
        total = validate_batches(self.stages[0].config, batches)
        # Poison first: KeyboardInterrupt/SystemExit can occur between optimizer
        # applications too. Only a fully successful step makes this usable again.
        self.failed = True
        result = self._step(batches, total)
        self.failed = False
        return result

    def _step(self, batches, total):
        for stage, optimizer in zip(self.stages, self.optimizers):
            stage.train()
            optimizer.zero_grad(set_to_none=True)
        result = StepResult(0.0, total, [], [])
        for index, batch in enumerate(batches):
            first, last = self.stages[0], self.stages[-1]
            if len(self.stages) == 2:
                hidden = first(batch.inputs.to(first.device))
                key = BoundaryKey(self.version, index, self.version)
                boundary = LocalBoundary(key, hidden)
                leaf = boundary.leaf(last.device)
                logits = last(leaf)
            else:
                captured = []
                def capture(_module, _args, output):
                    output.retain_grad()
                    captured.append(output)
                handle = first.blocks[str(self.split - 1)].register_forward_hook(capture)
                try:
                    logits = first(batch.inputs.to(first.device))
                finally:
                    handle.remove()
                leaf = captured[0]
            loss_sum = F.cross_entropy(logits.reshape(-1, first.config.vocab_size),
                batch.targets.to(last.device).reshape(-1), reduction="sum", ignore_index=-100)
            if not bool(torch.isfinite(loss_sum)):
                raise ValueError("nonfinite loss")
            (loss_sum / total).backward()
            if leaf.grad is None:
                raise ValueError("missing boundary derivative")
            result.boundary_gradients.append(leaf.grad.detach().to("cpu", copy=True))
            if len(self.stages) == 2:
                boundary.backward(key, leaf.grad)
            result.logits.append(logits.detach().to("cpu", copy=True))
            result.loss += float(loss_sum.detach().cpu()) / total
            del leaf, logits, loss_sum
            if len(self.stages) == 2:
                del hidden, boundary
            else:
                del captured
        # Local analogue of all-ready: check every stage before either updates.
        for stage in self.stages:
            finite_gradients(stage)
        for stage, optimizer in zip(self.stages, self.optimizers):
            optimizer.step()
            synchronize(stage.device)
            for name, p in stage.named_parameters():
                if not bool(torch.isfinite(p).all()):
                    raise ValueError(f"nonfinite updated parameter: {name}")
                for key, value in optimizer.state[p].items():
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError(f"nonfinite optimizer state: {name}/{key}")
        self.version += 1
        return result
