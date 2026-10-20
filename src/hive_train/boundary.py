"""One local detached boundary, with fail-stop identity/lifetime checks.

This simulates CPU staging in one process; it is not the M2 wire protocol.
"""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class BoundaryKey:
    step: int
    microbatch: int
    version: int

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in (self.step, self.microbatch, self.version)):
            raise ValueError("boundary IDs must be nonnegative integers")


class LocalBoundary:
    def __init__(self, key: BoundaryKey, output: torch.Tensor):
        if output.ndim != 3 or output.dtype != torch.float32 or not output.requires_grad:
            raise ValueError("boundary output must be a gradient-requiring FP32 [b,T,d]")
        if not bool(torch.isfinite(output).all()):
            raise ValueError("nonfinite activation")
        self.key, self._output = key, output
        self._opened = False
        self.consumed = False

    def leaf(self, device):
        if self._opened or self.consumed:
            raise RuntimeError("boundary already opened/consumed")
        self._opened = True
        # copy=True prevents a CPU-only test from sharing the original storage.
        payload = self._output.detach().to("cpu", copy=True).contiguous()
        return payload.to(device, copy=True).detach().requires_grad_(True)

    def backward(self, key, gradient):
        if self.consumed:
            raise RuntimeError("boundary already consumed/aborted")
        self.consumed = True  # any error is terminal; never replay a partial backward
        output, self._output = self._output, None
        if not self._opened or key != self.key:
            raise ValueError("missing, stale or wrong boundary identity")
        if (gradient is None or gradient.dtype != output.dtype or gradient.shape != output.shape
                or not bool(torch.isfinite(gradient).all())):
            raise ValueError("missing, invalid or nonfinite boundary gradient")
        payload = gradient.detach().to("cpu", copy=True).contiguous()
        output.backward(payload.to(output.device))
