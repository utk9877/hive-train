"""Run with python -m hive_train; every command is local-only in M1."""

import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hive Train M1 local decoder/split diagnostics")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="report environment/device availability; no training")
    doctor.add_argument("--output", type=Path)
    fixture = commands.add_parser("fixture", help="export tiny CPU initialization as binary tensors + JSON")
    fixture.add_argument("--output-directory", type=Path, required=True)
    verify = commands.add_parser("verify", help="compare full vs local split forward/backward/AdamW")
    verify.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    verify.add_argument("--reference-device", choices=("cpu", "cuda", "mps"))
    verify.add_argument("--steps", type=int, default=10)
    verify.add_argument("--split", type=int, action="append", help="repeat for multiple cuts; default 1,2,3")
    verify.add_argument("--threads", type=int, default=1, help="CPU intra-op threads for tiny test")
    verify.add_argument("--adam-eps", type=float, default=1e-8,
                        help="explicit matched optimizer experiment; default 1e-8")
    verify.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    # Must precede torch import. Explicit fallback=1 is rejected by require_device.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    from .validation import EquivalenceFailure, environment, run_equivalence
    import torch
    try:
        if args.command == "doctor":
            report = environment()
        elif args.command == "fixture":
            from .initialization import export_fixture
            from .model import build_stage
            manifest = export_fixture(build_stage(), args.output_directory)
            report = {"status": "pass", "fixture_hash": manifest["fixture_hash"],
                      "directory": str(args.output_directory), "environment": environment()}
        else:
            from .config import OptimizerConfig
            if args.threads <= 0:
                raise ValueError("--threads must be positive")
            torch.set_num_threads(args.threads)
            report = run_equivalence(device=args.device, reference_device=args.reference_device,
                                     steps=args.steps, splits=args.split or (1, 2, 3),
                                     optimizer=OptimizerConfig(eps=args.adam_eps))
    except EquivalenceFailure as error:
        report = error.report
    except (ValueError, RuntimeError, AssertionError, OSError) as error:
        report = {"status": "fail", "error": str(error), "environment": environment()}
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {k: v for k, v in report.items() if k not in ("fixture", "cases", "incomplete_case")}
    if "cases" in report:
        summary["cases_passed"] = len(report["cases"])
        summary["fixture_hash"] = report["fixture"]["fixture_hash"]
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 1 if report.get("status") == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
