"""Numerical comparison and reproducible, small-model M1 diagnostic reports."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
import platform
from pathlib import Path
import sys
import torch

from .config import ModelConfig, OptimizerConfig, json_hash
from .data import Batch, synthetic_batches
from .initialization import cpu_state, fixture_manifest, load_fixture, tensor_hash
from .model import build_stage
from .reference import ReferenceTrainer

TOLERANCES = {
    "cpu": {"output": (1e-6, 1e-5), "state": (1e-6, 1e-4), "l2": 1e-3},
    "accelerator": {"output": (1e-5, 1e-4), "state": (1e-5, 1e-3), "l2": 5e-3},
    "cross_device": {"output": (1e-4, 1e-3), "state": (1e-4, 1e-2), "l2": 2e-2},
}


def require_device(name):
    if name not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be cpu, cuda or mps")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback was performed")
    if name == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
            raise RuntimeError("MPS fallback must be disabled before importing torch")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; no CPU fallback was performed")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device(name)


def environment():
    source = Path(__file__).parent
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))}
    return {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_tree_hash": json_hash(source_hashes), "source_files": source_hashes,
            "python": sys.version, "platform": platform.platform(),
            "architecture": platform.machine(), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
            "packages": dict(sorted((d.metadata["Name"], d.version)
                             for d in importlib.metadata.distributions())),
            "planned_environment": {"python_minor": "3.13", "torch": "2.11.0"},
            "matches_planned_environment": (sys.version_info[:2] == (3, 13)
                                             and torch.__version__.split("+")[0] == "2.11.0")}


def assert_tensor_equal(actual, expected, *, name, tier="cpu", output=False, l2=False):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: shape/dtype mismatch")
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise AssertionError(f"{name}: nonfinite tensor")
    atol, rtol = TOLERANCES[tier]["output" if output else "state"]
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=lambda m: f"{name}: {m}")
    difference = (actual.double() - expected.double()).reshape(-1)
    norm = float(expected.double().norm())
    relative = float(difference.norm()) / max(norm, 1e-6)
    if l2 and norm > 1e-6 and relative > TOLERANCES[tier]["l2"]:
        raise AssertionError(f"{name}: normalized L2 {relative} exceeds {TOLERANCES[tier]['l2']}")
    return {"max_abs": float(difference.abs().max()), "normalized_l2": relative}


def compare_step(actual_result, expected_result, actual, expected, before_actual, before_expected, *, tier):
    metrics = {}
    if actual_result.valid_tokens != expected_result.valid_tokens:
        raise AssertionError("valid-token count mismatch")
    metrics["loss"] = assert_tensor_equal(torch.tensor(actual_result.loss, dtype=torch.float64),
        torch.tensor(expected_result.loss, dtype=torch.float64), name="loss", tier=tier, output=True)
    for kind in ("logits", "boundary_gradients"):
        left, right = getattr(actual_result, kind), getattr(expected_result, kind)
        if len(left) != len(right):
            raise AssertionError(f"{kind}: microbatch count mismatch")
        for i, (a, b) in enumerate(zip(left, right)):
            name = f"{kind}/{i}"
            metrics[name] = assert_tensor_equal(a, b, name=name, tier=tier,
                                               output=kind == "logits", l2=kind != "logits")
    for kind in ("parameters", "gradients", "optimizer"):
        if actual[kind].keys() != expected[kind].keys():
            raise AssertionError(f"{kind}: parameter/state coverage mismatch")
        for key in actual[kind]:
            name = f"{kind}/{key}"
            a, b = actual[kind][key], expected[kind][key]
            if kind == "optimizer" and key.endswith("/step"):
                if not torch.equal(a, b):
                    raise AssertionError(f"{name}: update counter mismatch")
            metrics[name] = assert_tensor_equal(a, b, name=name, tier=tier, l2=kind == "gradients")
    total_delta = 0.0
    for key in actual["parameters"]:
        a = actual["parameters"][key] - before_actual[key]
        b = expected["parameters"][key] - before_expected[key]
        total_delta += float(a.double().square().sum())
        metrics[f"update/{key}"] = assert_tensor_equal(a, b, name=f"update/{key}", tier=tier, l2=True)
    if total_delta == 0:
        raise AssertionError("no parameter update")
    return metrics


def check_placement(trainer, expected_device=None):
    for stage, optimizer in zip(trainer.stages, trainer.optimizers):
        device = stage.device
        if expected_device is not None and device.type != torch.device(expected_device).type:
            raise AssertionError(f"stage on {device}, expected {expected_device}")
        for name, p in stage.named_parameters():
            if p.device != device or p.dtype != torch.float32 or p.grad is None or p.grad.device != device:
                raise AssertionError(f"{name}: parameter/gradient placement mismatch")
            for key in ("exp_avg", "exp_avg_sq"):
                if optimizer.state[p][key].device != device:
                    raise AssertionError(f"{name}/{key}: moment placement mismatch")


def run_equivalence(*, device="cpu", reference_device=None, steps=10, splits=(1, 2, 3),
                    scenarios=("single", "masked", "empty_microbatch"), config=ModelConfig()):
    """All compared instances load the same bounded CPU initial fixture.

    reference_device defaults to the same backend; an explicit CPU reference
    exercises a wider cross-device tolerance. No sockets or performance claims.
    """
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be positive")
    splits, scenarios = tuple(splits), tuple(scenarios)
    if not splits or not scenarios:
        raise ValueError("at least one split and scenario required")
    if any(type(k) is not int or not 1 <= k < config.layers for k in splits):
        raise ValueError("invalid split")
    target = require_device(device)
    reference_device = device if reference_device is None else reference_device
    reference = require_device(reference_device)
    tier = "cross_device" if target != reference else ("cpu" if device == "cpu" else "accelerator")
    # This reference tool intentionally keeps multiple tiny models, not a capacity path.
    initial = build_stage(config)
    fixture = cpu_state(initial)
    manifest = fixture_manifest(initial)
    del initial
    report = {"schema": "m1_equivalence_v1", "status": "pass", "environment": environment(),
              "config": asdict(config), "optimizer": asdict(OptimizerConfig()),
              "fixture": manifest, "device": device, "reference_device": reference_device,
              "tier": tier, "tolerances": TOLERANCES[tier], "steps_per_case": steps,
              "parameter_count": sum(t.numel() for t in fixture.values()), "cases": []}
    for split in splits:
        for scenario in scenarios:
            if scenario not in ("single", "masked", "empty_microbatch"):
                raise ValueError(f"unknown scenario {scenario}")
            whole = build_stage(config, device=reference)
            prefix = build_stage(config, 0, split, device=target)
            suffix = build_stage(config, split, config.layers, device=target)
            for stage in (whole, prefix, suffix):
                load_fixture(stage, fixture)
            expected = ReferenceTrainer([whole], split=split)
            actual = ReferenceTrainer([prefix, suffix], split=split)
            case = {"split": split, "scenario": scenario,
                    "batch_size": 2, "sequence_length": 32, "batch_seed_base": 7, "steps": []}
            for step in range(steps):
                count = 1 if scenario == "single" else 2
                batches = synthetic_batches(config, count=count, seed=7 + step)
                if count == 2:
                    y = batches[1].targets.clone()
                    y[:, 11:] = -100
                    if scenario == "empty_microbatch":
                        y.fill_(-100)
                    batches[1] = Batch(batches[1].inputs, y)
                before_a, before_b = actual.snapshot()["parameters"], expected.snapshot()["parameters"]
                ref_result, split_result = expected.step(batches), actual.step(batches)
                check_placement(expected, reference)
                check_placement(actual, target)
                metrics = compare_step(split_result, ref_result, actual.snapshot(), expected.snapshot(),
                                       before_a, before_b, tier=tier)
                worst = max(metrics, key=lambda name: metrics[name]["max_abs"])
                case["steps"].append({"step": step, "loss": split_result.loss,
                    "reference_loss": ref_result.loss, "valid_tokens": split_result.valid_tokens,
                    "batch_hashes": [{"inputs": tensor_hash(b.inputs), "targets": tensor_hash(b.targets)}
                                     for b in batches],
                    "worst_absolute_error": {"name": worst, **metrics[worst]}, "errors": metrics})
            report["cases"].append(case)
            del whole, prefix, suffix, expected, actual
    report["config_hash"] = json_hash({"model": report["config"], "optimizer": report["optimizer"]})
    return report
