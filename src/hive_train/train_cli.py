"""M3 coordinator/worker launchers. Fresh runs only; no checkpoint resume."""

import argparse
import json
import os
from pathlib import Path
import socket
from uuid import uuid4


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="role", required=True)
    co = commands.add_parser("coordinator")
    co.add_argument("--bind", default="0.0.0.0")
    co.add_argument("--steps", type=int, default=10)
    co.add_argument("--microbatches", type=int, choices=(1, 2, 3, 4), default=1)
    co.add_argument("--split", type=int, choices=(1, 2, 3), default=2)
    co.add_argument("--scenario", choices=("single", "masked", "empty_microbatch", "pattern"), default="single")
    co.add_argument("--allow-cpu", action="store_true", help="permit explicit CPU diagnostic workers")
    co.add_argument("--adam-eps", type=float, default=1e-8,
                    help="explicit optimizer experiment sent to both workers; default 1e-8")
    worker = commands.add_parser("worker")
    worker.add_argument("--host", default="127.0.0.1")
    worker.add_argument("--worker-id", choices=("worker-0", "worker-1"), required=True)
    worker.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    worker.add_argument("--diagnostic-directory", type=Path, help="tiny-fixture local snapshots; adds overhead")
    worker.add_argument("--socket-fd", type=int, help=argparse.SUPPRESS)
    for sub in (co, worker):
        sub.add_argument("--port", type=int, default=29550)
        sub.add_argument("--secret-file", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    report = {"schema": "m3_training_run_v1", "role": args.role, "status": "fail",
              "cpu_threads": args.threads}
    sessions, runtime, sock = {}, None, None
    try:
        import torch
        from .runtime import make_assignments
        from .transport import SocketStream, accept_members, client_handshake
        from .validation import environment
        if args.threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(args.threads)
        report["environment"] = environment()
        secret = args.secret_file.read_bytes()
        if len(secret) != 32:
            raise ValueError("secret file must contain exactly 32 bytes")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.role == "worker":
            from .worker import WorkerRuntime
            from .diagnostics import SnapshotWriter
            sock = (socket.socket(fileno=args.socket_fd) if args.socket_fd is not None
                    else socket.create_connection((args.host, args.port), timeout=10))
            session = client_handshake(SocketStream(sock), secret, args.worker_id)
            sessions[args.worker_id] = session
            writer = SnapshotWriter(args.diagnostic_directory) if args.diagnostic_directory else None
            runtime = WorkerRuntime(session, device=args.device, diagnostic=writer is not None, observer=writer)
            report.update(runtime.run())
        else:
            from .coordinator import CoordinatorRuntime
            from .local_run import batches_for_step
            from .config import OptimizerConfig, json_hash
            if not 1 <= args.steps <= 100000:
                raise ValueError("steps must be in 1..100000")
            if args.scenario in ("masked", "empty_microbatch") and args.microbatches < 2:
                raise ValueError("masked scenarios require at least two microbatches")
            assignments = make_assignments(split=args.split, optimizer=OptimizerConfig(eps=args.adam_eps), dataset_hash=json_hash(
                {"source": "synthetic_v1", "seed": 7, "scenario": args.scenario}))
            run_id = str(uuid4())
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind((args.bind, args.port))
                listener.listen(2)
                print(f"Listening on {args.bind}:{args.port}; start both workers within 30 seconds", flush=True)
                sessions = accept_members(listener, secret, run_id)
            # Logs stream to disk; no per-step tensor or telemetry backlog in RAM.
            trace_path = args.output.with_suffix(".steps.jsonl")
            with trace_path.open("w", encoding="utf-8") as log:
                def record(row):
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                runtime = CoordinatorRuntime(sessions, assignments, observer=record)
                runtime.configure()
                backends = {w: c["platform"]["backend"] for w, c in runtime.capabilities.items()}
                if not args.allow_cpu and backends != {"worker-0": "cuda", "worker-1": "mps"}:
                    raise ValueError("expected CUDA prefix and MPS suffix; use --allow-cpu for diagnostics")
                last = None
                for step in range(args.steps):
                    last = runtime.step(batches_for_step(step, count=args.microbatches, scenario=args.scenario))
                report.update(runtime.stop())
            report.update(last_step=last, capabilities=runtime.capabilities,
                          assignments=assignments, trace=str(trace_path), checkpoint_saved=False,
                          workload={"scenario": args.scenario, "microbatches": args.microbatches,
                                    "seed_base": 7})
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="fail", error=f"{type(error).__name__}: {error}")
    finally:
        for session in sessions.values():
            session.close()
        if sock is not None:
            sock.close()
        if runtime is not None and report.get("status") != "pass":
            # Worker.run owns its cleanup; coordinator aborts both local links.
            if hasattr(runtime, "close"):
                runtime.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output), "error": report.get("error")}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
