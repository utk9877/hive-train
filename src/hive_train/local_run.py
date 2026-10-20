"""M3 process-isolated CPU verification over inherited socketpairs (POSIX)."""

import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
from uuid import uuid4


def batches_for_step(step, *, count=1, scenario="single"):
    import torch
    from .config import ModelConfig
    from .data import Batch, synthetic_batches
    if type(count) is not int or not 1 <= count <= 4:
        raise ValueError("diagnostic microbatch count must be in 1..4")
    if type(step) is not int or step < 0:
        raise ValueError("invalid diagnostic step")
    if scenario not in ("single", "masked", "empty_microbatch", "pattern"):
        raise ValueError("unknown scenario")
    if scenario in ("masked", "empty_microbatch") and count < 2:
        raise ValueError("mask scenario needs two microbatches")
    batches = synthetic_batches(ModelConfig(), count=count, seed=7 + step)
    if scenario == "pattern":
        for i in range(count):
            offset = (step * 7 + i * 13) % 256
            # Reserve this exact sequence start for fixed diagnostic evaluation.
            if offset == 157:
                offset = 158
            tokens = (torch.arange(33).repeat(2, 1) + offset) % 256
            batches[i] = Batch(tokens[:, :-1].clone(), tokens[:, 1:].clone())
    elif scenario in ("masked", "empty_microbatch"):
        targets = batches[-1].targets.clone()
        targets[:, 11:] = -100
        if scenario == "empty_microbatch":
            targets.fill_(-100)
        batches[-1] = Batch(batches[-1].inputs, targets)
    return batches


def verify_processes(*, steps=10, split=2, scenario="single", count=1):
    if os.name != "posix":
        raise RuntimeError("inherited-socket diagnostic requires POSIX; Windows uses the TCP launch commands")
    if type(steps) is not int or not 1 <= steps <= 100 or count not in (1, 2, 3, 4):
        raise ValueError("diagnostic requires 1..100 steps and K1..4")
    import torch
    from .coordinator import CoordinatorRuntime
    from .diagnostics import read_snapshot
    from .model import build_stage
    from .reference import ReferenceTrainer, StepResult
    from .runtime import make_assignments, WORKERS
    from .transport import SocketStream, server_handshake
    from .validation import compare_step, environment
    torch.set_num_threads(1)
    reference = ReferenceTrainer([build_stage()], split=split)
    records, learning = [], []
    def evaluate(parameters=None):
        # Evaluation of the tiny diagnostic fixture only. The training scheduler
        # never gathers weights; this separate verifier deliberately can.
        from .initialization import load_fixture
        from torch.nn import functional as F
        model = reference.stages[0] if parameters is None else build_stage()
        if parameters is not None:
            load_fixture(model, parameters)
        tokens = (torch.arange(33).repeat(2, 1) + 157) % 256
        with torch.no_grad():
            model.eval()
            return float(F.cross_entropy(model(tokens[:, :-1]).reshape(-1, 256),
                                         tokens[:, 1:].reshape(-1)))
    if scenario == "pattern":
        learning.append({"after_steps": 0, "reference_loss": evaluate(), "distributed_loss": None})
    processes, sessions, files, runtime = [], {}, [], None
    with tempfile.TemporaryDirectory(prefix="hive-m3-") as temporary:
        root = Path(temporary)
        key = secrets.token_bytes(32)
        secret_path = root / "secret.key"
        descriptor = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as file:
            file.write(key)
        run_id = str(uuid4())
        try:
            # Worker processes own separate interpreters, models and optimizer state.
            for worker in WORKERS:
                parent, child = socket.socketpair()
                log = (root / f"{worker}.log").open("w")
                files.append(log)
                try:
                    command = [sys.executable, "-m", "hive_train.train_cli", "worker",
                        "--worker-id", worker, "--device", "cpu", "--socket-fd", str(child.fileno()),
                        "--secret-file", str(secret_path), "--output", str(root / f"{worker}.json"),
                        "--diagnostic-directory", str(root / worker)]
                    process = subprocess.Popen(command, pass_fds=(child.fileno(),), stdout=log, stderr=log)
                    processes.append(process)
                    sessions[worker] = server_handshake(SocketStream(parent), key, run_id)
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
            runtime = CoordinatorRuntime(sessions, make_assignments(split=split))
            runtime.configure()
            for step in range(steps):
                batches = batches_for_step(step, count=count, scenario=scenario)
                before_expected = reference.snapshot()["parameters"]
                expected = reference.step(batches)
                summary = runtime.step(batches)
                shards = [read_snapshot(root / worker, step) for worker in WORKERS]
                actual = {k: {n: v for shard in shards for n, v in shard[k].items()}
                          for k in ("parameters", "gradients", "optimizer", "before")}
                if scenario == "pattern" and step == 0:
                    learning[0]["distributed_loss"] = evaluate(actual["before"])
                result = StepResult(summary["loss"], summary["valid_tokens"],
                    [m["logits"] for m in shards[1]["micro"]],
                    [m["boundary_grad"] for m in shards[0]["micro"]])
                errors = compare_step(result, expected, actual, reference.snapshot(),
                                      actual["before"], before_expected, tier="cpu")
                worst = max(errors, key=lambda key: errors[key]["max_abs"])
                records.append({**summary, "reference_loss": expected.loss,
                                "worst_error": {"name": worst, **errors[worst]}})
                if scenario == "pattern" and ((step + 1) % 10 == 0 or step + 1 == steps):
                    learning.append({"after_steps": step + 1, "reference_loss": evaluate(),
                                     "distributed_loss": evaluate(actual["parameters"])})
            if scenario == "pattern" and steps >= 100:
                if learning[-1]["distributed_loss"] >= learning[0]["distributed_loss"]:
                    raise AssertionError("100-step pattern validation loss did not improve")
                if abs(learning[-1]["distributed_loss"] - learning[-1]["reference_loss"]) > max(0.02, 0.02 * learning[-1]["reference_loss"]):
                    raise AssertionError("pattern validation differs from reference")
            stopped = runtime.stop()
            workers = {}
            for worker, process in zip(WORKERS, processes):
                code = process.wait(timeout=15)
                if code != 0:
                    raise RuntimeError(f"{worker} exited with {code}")
                workers[worker] = json.loads((root / f"{worker}.json").read_text())
            return {**stopped, "schema": "m3_process_equivalence_v1", "environment": environment(),
                    "transport": "authenticated inherited socketpair; no TCP/Wi-Fi claim",
                    "workers": workers, "split": split, "scenario": scenario,
                    "microbatches": count, "cases": records, "pattern_validation": learning,
                    "scope": "tiny-fixture CPU validation; snapshots/reference add memory and time overhead"}
        except BaseException as error:
            logs = {w: (root / f"{w}.log").read_text()[-4000:] for w in WORKERS
                    if (root / f"{w}.log").exists()}
            error.add_note(json.dumps(logs))
            raise
        finally:
            if runtime is not None and runtime.gate.phase != "STOPPED":
                runtime.close()
            for session in sessions.values():
                session.close()
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for file in files:
                file.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--split", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--scenario", choices=("single", "masked", "empty_microbatch", "pattern"), default="single")
    parser.add_argument("--microbatches", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    try:
        report = verify_processes(steps=args.steps, split=args.split, scenario=args.scenario,
                                  count=args.microbatches)
    except (Exception, KeyboardInterrupt) as error:
        report = {"status": "fail", "error": f"{type(error).__name__}: {error}",
                  "details": getattr(error, "__notes__", [])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output), "error": report.get("error")}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
