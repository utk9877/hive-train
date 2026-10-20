"""Shared serial-runtime contracts. The coordinator creates metadata, not weights."""

from dataclasses import asdict
import hashlib
import platform
import sys
import time

from .config import ModelConfig, OptimizerConfig, json_hash
from .protocol import ProtocolError, MAX_TENSOR_BYTES, validate_metadata
from .state import STEP, CHECKPOINT
from .transport import Timeouts, metadata

WORKERS = ("worker-0", "worker-1")


def model_config(config):
    return ModelConfig(**{key: config["model"][key] for key in ModelConfig.__dataclass_fields__})


def manifest(config, start=0, end=None):
    from .model import DecoderStage
    stage = DecoderStage(config, start, config.layers if end is None else end)
    return [{"name": n, "shape": list(p.shape), "dtype": "float32"}
            for n, p in stage.named_parameters()]


def parameter_name_hash(spec):
    return hashlib.sha256("\n".join(p["name"] for p in spec).encode("utf-8")).hexdigest()


def make_assignments(*, split=2, batch_size=2, length=32, model=ModelConfig(),
                     dataset_hash=None, seed=2026):
    if type(split) is not int or not 0 < split < model.layers:
        raise ValueError("split must be an interior block boundary")
    config = {"model": {**asdict(model), "architecture": "decoder_v1", "dtype": "float32"},
              "optimizer": {**asdict(OptimizerConfig()), "name": "adamw", "foreach": False,
                            "fused": False, "amsgrad": False},
              "data": {"dataset_hash": dataset_hash or json_hash({"source": "synthetic_v1"}),
                       "tokenizer": "synthetic_v1", "max_microbatch_size": batch_size,
                       "sequence_length": length, "ignore_index": -100},
              "init": {"algorithm": "named_cpu_v1", "master_seed": seed},
              "parameter_spec_hash": json_hash(manifest(model))}
    timeouts = {"connect_ms": 10000, "handshake_ms": 10000, "idle_ms": 30000,
                "status_interval_ms": 5000, "compute_ms": 300000, "checkpoint_ms": 600000,
                "transfer_no_progress_ms": 30000, "min_directional_bytes_per_second": 1048576,
                "transfer_multiplier": 5.0}
    result = {}
    for i, worker in enumerate(WORKERS):
        assignment = {"config": config, "config_hash": json_hash(config), "stage_id": i,
                      "block_start": 0 if i == 0 else split,
                      "block_end": split if i == 0 else model.layers,
                      "checkpoint_id": None, "starting_step": 0, "starting_version": 0,
                      "max_tensor_bytes": MAX_TENSOR_BYTES, "timeouts": timeouts.copy()}
        validate_metadata(metadata("RUN_CONFIG", "00000000-0000-4000-8000-000000000000",
                                   "coordinator", worker, 2, version=0, body=assignment), 0)
        result[worker] = assignment
    return result


def abort(gate, sessions):
    for session in sessions:
        session.close()
    try:
        gate.disconnect()
    except ProtocolError:
        pass


class Channel:
    """Single owner consumes gates before sends and after authentication."""
    def __init__(self, session, gate):
        self.session, self.gate = session, gate

    def configure_transport(self, assignment):
        t = assignment["timeouts"]
        self.session.stream.timeouts = Timeouts(idle=t["idle_ms"] / 1000,
            handshake=t["handshake_ms"] / 1000, no_progress=t["transfer_no_progress_ms"] / 1000,
            minimum_bytes_per_second=t["min_directional_bytes_per_second"],
            multiplier=t["transfer_multiplier"])
        if assignment["max_tensor_bytes"] > self.session.stream.max_tensor_bytes:
            raise ProtocolError("CONFIG_MISMATCH", "run cap exceeds admitted cap")
        self.session.stream.max_tensor_bytes = assignment["max_tensor_bytes"]

    def deadline(self):
        g = self.gate
        g.check_timeout()
        if g.phase == "CHECKPOINTING":
            return g.checkpoint_at + g.timeouts["checkpoint_ms"] / 1000
        key = "idle_ms" if g.phase in ("IDLE", "CONFIGURING") else "compute_ms"
        return g.progress_at + g.timeouts[key] / 1000

    def send(self, kind, *, body=None, tensor=None, payload=b"", microbatch=None):
        try:
            g, s = self.gate, self.session
            step = g.step_id if kind in STEP | CHECKPOINT else None
            version = g.parameter_version
            if kind == "RUN_CONFIG":
                version = body["starting_version"]
            elif kind == "STEP_APPLIED":
                version += 1
            elif kind == "PING" and s.local == "coordinator" and g.phase == "APPLYING" and s.peer in g.acked:
                version += 1
            m = metadata(kind, s.run_id, s.local, s.peer, s.send_seq, step=step,
                         microbatch=microbatch, version=version, body=body, tensor=tensor)
            deadline = self.deadline()
            g.accept(m)
            s.send(kind, step=step, microbatch=microbatch, version=version, body=body,
                   tensor=tensor, payload=payload, deadline=deadline)
        except BaseException:
            abort(self.gate, [self.session])
            raise

    def receive(self, expected_type=None):
        try:
            deadline = self.deadline()
            frame = self.session.receive(config=getattr(self.gate, "config", None),
                expected_type=expected_type, deadline=deadline,
                wait_timeout=max(0.001, deadline - time.monotonic()))
            self.gate.accept(frame.metadata)
            return frame
        except BaseException:
            abort(self.gate, [self.session])
            raise


def make_capabilities(device):
    import torch
    from .validation import require_device
    from .reference import synchronize
    selected = require_device(str(device))
    # A real update diagnostic, independent of global initialization RNG.
    with torch.random.fork_rng(devices=[]):
        layer = torch.nn.Linear(4, 4, device=selected, dtype=torch.float32)
        x = torch.ones(2, 4, device=selected, requires_grad=True)
        before = layer.weight.detach().clone()
        optimizer = torch.optim.AdamW(layer.parameters(), foreach=False, fused=False)
        layer(x).square().sum().backward()
        if x.grad is None or not all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                    for p in layer.parameters()):
            raise RuntimeError("device diagnostic backward failed")
        optimizer.step()
        synchronize(selected)
        if torch.equal(before, layer.weight) or not bool(torch.isfinite(layer.weight).all()):
            raise RuntimeError("device diagnostic update failed")
    memory = dict.fromkeys(("total_bytes", "available_bytes", "allocated_bytes", "reserved_bytes"))
    name = platform.processor() or "CPU"
    if selected.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        memory.update(total_bytes=total, available_bytes=free,
                      allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())
        name = torch.cuda.get_device_name()
    elif selected.type == "mps":
        memory.update(allocated_bytes=torch.mps.current_allocated_memory(),
                      reserved_bytes=torch.mps.driver_allocated_memory())
        name = "Apple MPS (unified memory)"
    return {"platform": {"os": platform.system(), "architecture": platform.machine(),
            "python": platform.python_version(), "torch": str(torch.__version__),
            "backend": selected.type, "device_name": name, "cuda_driver": None,
            "cuda_runtime": torch.version.cuda, "macos_version": platform.mac_ver()[0] or None},
            "diagnostics_passed": True, "fallback_enabled": False, "memory": memory}
