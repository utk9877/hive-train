"""Compare privately collected final tiny-fixture snapshots to the CPU reference."""

import argparse
import json
from pathlib import Path


def check(report, stage0, stage1):
    import torch
    from .config import ModelConfig, OptimizerConfig, json_hash
    from .diagnostics import read_snapshot
    from .local_run import batches_for_step
    from .model import build_stage
    from .reference import ReferenceTrainer, StepResult
    from .runtime import model_config
    from .protocol import validate_metadata
    from .transport import metadata
    from .validation import compare_step, environment
    torch.set_num_threads(1)
    if report.get("status") != "pass" or type(report.get("steps")) is not int or not 1 <= report["steps"] <= 100:
        raise ValueError("require a completed 1..100 step diagnostic run")
    assignments = report["assignments"]
    for worker in ("worker-0", "worker-1"):
        a = assignments[worker]
        validate_metadata(metadata("RUN_CONFIG", report["run_id"], "coordinator", worker, 2,
                                   version=0, body=a), 0)
        if a["config_hash"] != json_hash(a["config"]):
            raise ValueError("diagnostic config hash mismatch")
    if assignments["worker-0"]["config"] != assignments["worker-1"]["config"]:
        raise ValueError("worker configurations differ")
    config = assignments["worker-0"]["config"]
    if model_config(config) != ModelConfig() or config["init"]["master_seed"] != 2026:
        raise ValueError("snapshot comparison restricted to the standard tiny fixture")
    versions = {str(c["platform"]["torch"]).split("+")[0] for c in report["capabilities"].values()}
    if versions != {str(torch.__version__).split("+")[0]}:
        raise ValueError("use the workers' PyTorch base release for CPU reference initialization")
    split = assignments["worker-0"]["block_end"]
    if (assignments["worker-0"]["block_start"] != 0 or assignments["worker-1"]["block_start"] != split
            or assignments["worker-1"]["block_end"] != 4):
        raise ValueError("invalid diagnostic partition")
    if config["data"]["max_microbatch_size"] != 2 or config["data"]["sequence_length"] != 32:
        raise ValueError("unsupported diagnostic batch shape")
    if report["last_step"]["step_id"] != report["steps"] - 1 or report["parameter_version"] != report["steps"]:
        raise ValueError("diagnostic counters differ")
    opt = OptimizerConfig(**{k: config["optimizer"][k] for k in OptimizerConfig.__dataclass_fields__})
    reference = ReferenceTrainer([build_stage()], split=split, optimizer=opt)
    workload = report["workload"]
    if type(workload.get("microbatches")) is not int or not 1 <= workload["microbatches"] <= 4:
        raise ValueError("diagnostic microbatches must be in 1..4")
    if workload["seed_base"] != 7:
        raise ValueError("unsupported diagnostic data seed")
    for step in range(report["steps"]):
        before = reference.snapshot()["parameters"]
        expected = reference.step(batches_for_step(step, count=workload["microbatches"],
                                                 scenario=workload["scenario"]))
    shards = [read_snapshot(path, step, run_id=report["run_id"],
              config_hash=assignments[f"worker-{i}"]["config_hash"], stage_id=i)
              for i, path in enumerate((stage0, stage1))]
    actual = {k: {} for k in ("parameters", "gradients", "optimizer", "before")}
    for shard in shards:
        for kind in actual:
            if actual[kind].keys() & shard[kind].keys():
                raise ValueError("duplicate parameter ownership in diagnostic snapshots")
            actual[kind].update(shard[kind])
    final = report["last_step"]
    result = StepResult(final["loss"], final["valid_tokens"],
        [m["logits"] for m in shards[1]["micro"]],
        [m["boundary_grad"] for m in shards[0]["micro"]])
    tier = "cpu" if all(c["platform"]["backend"] == "cpu" for c in report["capabilities"].values()) else "cross_device"
    metrics = compare_step(result, expected, actual, reference.snapshot(), actual["before"], before, tier=tier)
    worst = max(metrics, key=lambda key: metrics[key]["max_abs"])
    return {"status": "pass", "schema": "m3_final_snapshot_comparison_v1", "run_id": report["run_id"],
            "step_id": step, "tier": tier, "environment": environment(),
            "worst_error": {"name": worst, **metrics[worst]},
            "scope": "final diagnostic step only; no recovery, capacity or throughput claim"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator-report", type=Path, required=True)
    parser.add_argument("--stage0", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = check(json.loads(args.coordinator_report.read_text()), args.stage0, args.stage1)
    except (Exception, KeyboardInterrupt) as error:
        report = {"status": "fail", "error": f"{type(error).__name__}: {error}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output), "error": report.get("error")}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
