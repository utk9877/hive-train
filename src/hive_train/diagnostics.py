"""Tiny-fixture snapshots for M3 validation, never training checkpoint storage."""

import json
from pathlib import Path

from .config import ModelConfig
from .tensor_codec import encode_tensor, decode_tensor


class SnapshotWriter:
    """Overwrite one bounded snapshot after each step; private diagnostic path."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.before = {}
        self.micro = []

    def __call__(self, stage, optimizer, event, details):
        if stage.config != ModelConfig():
            raise ValueError("snapshots restricted to the tiny diagnostic fixture")
        if event == "before_step":
            self.before = {n: p.detach().cpu().clone() for n, p in stage.named_parameters()}
            self.micro = []
        elif event == "micro_done":
            if len(self.micro) >= 4:
                raise ValueError("diagnostic microbatch bound exceeded")
            self.micro.append({k: v.detach().cpu().clone() if hasattr(v, "detach") else v
                               for k, v in details.items()})
        elif event == "after_step":
            flat = {}
            for n, p in stage.named_parameters():
                flat[f"parameters:{n}"] = p.detach()
                flat[f"gradients:{n}"] = p.grad.detach()
                flat[f"before:{n}"] = self.before[n]
                for key, value in optimizer.state[p].items():
                    flat[f"optimizer:{n}/{key}"] = value.detach()
            micro_scalars = []
            for i, detail in enumerate(self.micro):
                scalars = {}
                for key, value in detail.items():
                    if hasattr(value, "detach"):
                        flat[f"micro:{i}:{key}"] = value
                    else:
                        scalars[key] = value
                micro_scalars.append(scalars)
            entries = []
            for i, (name, tensor) in enumerate(flat.items()):
                scalar = tensor.ndim == 0
                desc, payload = encode_tensor(tensor.reshape(1) if scalar else tensor, "snapshot")
                filename = f"{i:04d}.bin"
                (self.directory / filename).write_bytes(payload)
                entries.append({"key": name, "file": filename, "scalar": scalar, "tensor": desc})
            index = {"step_id": details["step_id"], "run_id": details["run_id"],
                     "config_hash": details["config_hash"], "stage_id": details["stage_id"],
                     "entries": entries, "micro": micro_scalars}
            (self.directory / "index.json").write_text(json.dumps(index, allow_nan=False), encoding="utf-8")
            self.before, self.micro = {}, []


def read_snapshot(directory, step_id, *, run_id=None, config_hash=None, stage_id=None):
    """Read only our private, bounded fixture output after its completion barrier."""
    directory = Path(directory)
    path = directory / "index.json"
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("oversized diagnostic index")
    index = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(index.get("micro"), list) or not 1 <= len(index["micro"]) <= 4:
        raise ValueError("diagnostic microbatch bound exceeded")
    if index["step_id"] != step_id or len(index["entries"]) > 1024:
        raise ValueError("wrong diagnostic step/entry count")
    for key, expected in (("run_id", run_id), ("config_hash", config_hash), ("stage_id", stage_id)):
        if expected is not None and index.get(key) != expected:
            raise ValueError(f"diagnostic {key} mismatch")
    result = {"parameters": {}, "gradients": {}, "optimizer": {}, "before": {},
              "micro": index["micro"]}
    total = 0
    for i, entry in enumerate(index["entries"]):
        filename = f"{i:04d}.bin"
        if entry["file"] != filename:
            raise ValueError("invalid diagnostic filename")
        size = (directory / filename).stat().st_size
        total += size
        if total > 64 * 1024**2 or size != entry["tensor"]["nbytes"]:
            raise ValueError("diagnostic byte bound/length mismatch")
        tensor = decode_tensor(entry["tensor"], bytearray((directory / filename).read_bytes()))
        if entry["scalar"]:
            tensor = tensor.reshape(())
        group, key = entry["key"].split(":", 1)
        if group == "micro":
            micro, name = key.split(":", 1)
            result["micro"][int(micro)][name] = tensor
        else:
            result[group][key] = tensor
    return result
