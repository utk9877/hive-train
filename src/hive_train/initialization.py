"""Partition-independent CPU initialization and small-fixture fingerprints."""

import ctypes
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys
import torch
from torch import nn

from .config import MODEL_ARCHITECTURE, json_hash

INIT_ALGORITHM = "named_cpu_v1"


def parameter_seed(name: str, master_seed: int = 2026) -> int:
    if type(master_seed) is not int or not 0 <= master_seed < 2**63:
        raise ValueError("seed must be a nonnegative uint63")
    digest = hashlib.sha256(f"hive-train:init:v1:{master_seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@torch.no_grad()
def initialize_named(module: nn.Module, master_seed=2026):
    """Stage already lives on its owner; at most one CPU parameter is staged."""
    norms = {f"{name}.weight" for name, child in module.named_modules()
             if isinstance(child, nn.LayerNorm)}
    for name, parameter in module.named_parameters():
        generator = torch.Generator(device="cpu").manual_seed(parameter_seed(name, master_seed))
        staging = torch.empty(parameter.shape, dtype=torch.float32, device="cpu")
        if name in norms:
            staging.fill_(1)
        elif name.endswith(".bias"):
            staging.zero_()
        else:
            staging.normal_(0, 0.02, generator=generator)
        parameter.copy_(staging)
        del staging


def cpu_state(module):
    """Full small fixture only. Never use this to gather a large distributed model."""
    return {name: p.detach().to("cpu", copy=True) for name, p in module.named_parameters()}


@torch.no_grad()
def load_fixture(module, state):
    """Load only this stage's named tensors from a trusted in-memory CPU fixture."""
    for name, parameter in module.named_parameters():
        source = state[name]
        if source.dtype != torch.float32 or source.device.type != "cpu" or source.shape != parameter.shape:
            raise ValueError(f"invalid fixture tensor {name}")
        parameter.copy_(source)


def tensor_hash(tensor):
    """Hash contiguous little-endian CPU bytes without NumPy or pickle."""
    if sys.byteorder != "little":
        raise RuntimeError("fixture byte hashing currently requires a little-endian host")
    cpu = tensor.detach().to("cpu").contiguous()
    # Pointer belongs to the live contiguous CPU tensor, never network input.
    payload = ctypes.string_at(cpu.data_ptr(), cpu.numel() * cpu.element_size())
    return hashlib.sha256(payload).hexdigest()


def fixture_manifest(module, seed=2026):
    tensors = [{"name": name, "shape": list(p.shape), "dtype": "float32", "sha256": tensor_hash(p)}
               for name, p in module.named_parameters()]
    specification = [{k: t[k] for k in ("name", "shape", "dtype")} for t in tensors]
    return {"algorithm": INIT_ALGORITHM, "master_seed": seed, "architecture": MODEL_ARCHITECTURE,
            "torch_version": torch.__version__, "tensors": tensors,
            "parameter_spec_hash": json_hash(specification), "fixture_hash": json_hash(tensors)}


def export_fixture(module, directory, seed=2026):
    """Export a small fixture, not a resumable training checkpoint.

    Refuse an existing directory; binary payloads and JSON never use pickle.
    The manifest hash describes named values independently of numbered filenames.
    """
    manifest = fixture_manifest(module, seed)
    manifest["model"] = asdict(module.config)
    manifest["format"] = "m1_fixture_f32le_v1"
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    for index, (name, parameter) in enumerate(module.named_parameters()):
        cpu = parameter.detach().to("cpu").contiguous()
        filename = f"{index:04d}.bin"
        payload = ctypes.string_at(cpu.data_ptr(), cpu.numel() * cpu.element_size())
        (destination / filename).write_bytes(payload)
        manifest["tensors"][index]["file"] = filename
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
