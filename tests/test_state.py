"""Pure protocol-state fixtures, deliberately independent of torch and sockets."""

from copy import deepcopy
import hashlib
import json
import math
import unittest

from hive_train.protocol import ProtocolError
from hive_train.state import CoordinatorState, WorkerState, apply_id_for_step


RUN = "b6c0317d-fcda-4c4b-8d2d-68c5836e5da5"
APPLY = apply_id_for_step(RUN, 0)
CHECKPOINT = "dd3a246f-122f-4dfb-9e3a-f54e5b51fb32"
HASH = "a" * 64
WORKERS = ("worker-0", "worker-1")


def config_body(stage=0):
    config = {
        "model": {"layers": 4, "width": 128, "heads": 4, "mlp_width": 512,
                  "vocab_size": 256, "max_sequence_length": 128,
                  "architecture": "decoder_v2", "dtype": "float32"},
        "optimizer": {"name": "adamw", "lr": 0.001, "beta1": 0.9, "beta2": 0.999,
                      "eps": 1e-8, "weight_decay": 0.01, "foreach": False,
                      "fused": False, "amsgrad": False},
        "data": {"dataset_hash": HASH, "tokenizer": "synthetic_v1",
                 "max_microbatch_size": 2, "sequence_length": 32, "ignore_index": -100},
        "init": {"algorithm": "named_cpu_v1", "master_seed": 2026},
        "parameter_spec_hash": HASH,
    }
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    return {"config": config, "config_hash": digest, "stage_id": stage,
            "block_start": 0 if stage == 0 else 2, "block_end": 2 if stage == 0 else 4,
            "checkpoint_id": None, "starting_step": 0, "starting_version": 0,
            "max_tensor_bytes": 67108864,
            "timeouts": {"connect_ms": 10000, "handshake_ms": 10000, "idle_ms": 30000,
                         "status_interval_ms": 5000, "compute_ms": 300000,
                         "checkpoint_ms": 600000, "transfer_no_progress_ms": 30000,
                         "min_directional_bytes_per_second": 1048576, "transfer_multiplier": 5.0}}


def capabilities():
    return {"platform": {"os": "test", "architecture": "test", "python": "3.11",
                          "torch": "none", "backend": "cpu", "device_name": "fake",
                          "cuda_driver": None, "cuda_runtime": None, "macos_version": None},
            "diagnostics_passed": True, "fallback_enabled": False,
            "memory": {k: None for k in ("total_bytes", "available_bytes", "allocated_bytes", "reserved_bytes")}}


def metadata(typ, sender="coordinator", receiver="worker-0", body=None,
             step=None, micro=None, version=0, tensor=None):
    return {"type": typ, "run_id": RUN, "sender": sender, "receiver": receiver,
            "seq": 2, "step_id": step, "microbatch_id": micro,
            "parameter_version": version, "body": {} if body is None else deepcopy(body),
            "tensor": deepcopy(tensor)}


def tensor(typ, b=2):
    name, dtype, shape, itemsize = {
        "TARGET": ("targets", "int64", [b, 32], 8),
        "INPUT": ("input_ids", "int64", [b, 32], 8),
        "ACTIVATION": ("hidden", "float32", [b, 32, 128], 4),
        "BOUNDARY_GRAD": ("hidden_grad", "float32", [b, 32, 128], 4),
    }[typ]
    return {"name": name, "dtype": dtype, "shape": shape, "byte_order": "little",
            "nbytes": math.prod(shape) * itemsize}


class Trace:
    """Reusable two-worker event fixture; emit returns recorded wire metadata."""

    def __init__(self, clock=lambda: 0, callbacks=None):
        self.coordinator = CoordinatorState(RUN, clock=clock, parameter_hashes={w: HASH for w in WORKERS})
        callbacks = callbacks or {}
        self.workers = {w: WorkerState(RUN, w, clock=clock, apply_update=callbacks.get(w),
                                       expected_parameter_name_hash=HASH) for w in WORKERS}
        self.events = []
        self.seq = {(s, r): (2 if s == "coordinator" else 1)
                    for w in WORKERS for s, r in (("coordinator", w), (w, "coordinator"))}

    def emit(self, typ, worker="worker-0", outgoing=True, body=None,
             step=None, micro=None, version=0, descriptor=None):
        sender, receiver = ("coordinator", worker) if outgoing else (worker, "coordinator")
        m = metadata(typ, sender, receiver, body, step, micro, version, descriptor)
        m["seq"] = self.seq[sender, receiver]
        self.seq[sender, receiver] += 1
        # Sender gates precede receiver gates; a failed send requires disconnect.
        if outgoing:
            self.coordinator.accept(m)
            self.workers[worker].accept(m)
        else:
            self.workers[worker].accept(m)
            self.coordinator.accept(m)
        self.events.append(deepcopy(m))
        return m

    def setup(self):
        for w in WORKERS:
            self.emit("CAPABILITIES", w, False, capabilities(), version=None)
        for i, w in enumerate(WORKERS):
            cfg = config_body(i)
            self.emit("RUN_CONFIG", w, body=cfg)
            self.emit("RUN_READY", w, False,
                      {"config_hash": cfg["config_hash"], "stage_id": i,
                       "loaded_version": 0, "parameter_name_hash": HASH})
        return self

    def begin(self, count=2, total=82):
        for w in WORKERS:
            self.emit("STEP_BEGIN", w, body={"microbatch_count": count,
                      "total_valid_tokens": total, "batch_plan_hash": HASH}, step=0)
            self.emit("STEP_STARTED", w, False, step=0)
        return self

    def micro(self, index=0, valid=64):
        self.emit("TARGET", "worker-1", body={"valid_tokens": valid}, step=0,
                  micro=index, descriptor=tensor("TARGET"))
        self.emit("TARGET_READY", "worker-1", False, step=0, micro=index)
        self.emit("INPUT", step=0, micro=index, descriptor=tensor("INPUT"))
        self.emit("ACTIVATION", outgoing=False, step=0, micro=index, descriptor=tensor("ACTIVATION"))
        self.emit("ACTIVATION", "worker-1", step=0, micro=index, descriptor=tensor("ACTIVATION"))
        grad = {"loss_sum": float(valid) * 5.0, "valid_tokens": valid}
        self.emit("BOUNDARY_GRAD", "worker-1", False, grad, step=0,
                  micro=index, descriptor=tensor("BOUNDARY_GRAD"))
        self.emit("BOUNDARY_GRAD", body=grad, step=0, micro=index, descriptor=tensor("BOUNDARY_GRAD"))
        self.emit("MICRO_DONE", outgoing=False, step=0, micro=index)
        return self

    def prepare(self):
        for w in WORKERS:
            self.emit("STEP_PREPARE", w, step=0)
            self.emit("STEP_READY", w, False,
                      {"gradient_finite": True, "microbatches_completed": self.coordinator.count}, step=0)
        return self

    def apply(self):
        for w in WORKERS:
            self.emit("STEP_APPLY", w, body={"apply_id": APPLY, "next_version": 1}, step=0)
            self.emit("STEP_APPLIED", w, False, {"apply_id": APPLY, "next_version": 1}, step=0, version=1)
        return self

    def commit(self):
        for w in WORKERS:
            self.emit("STEP_COMMIT", w, body={"apply_id": APPLY, "next_version": 1}, step=0, version=1)
            self.emit("STEP_COMMITTED", w, False, {"apply_id": APPLY}, step=0, version=1)
        return self

    def prepared(self):
        return self.setup().begin().micro(0, 64).micro(1, 18).prepare()

    def complete(self):
        return self.prepared().apply().commit()


class StateTests(unittest.TestCase):
    def rejected(self, gate, m, code=None):
        with self.assertRaises(ProtocolError) as result:
            gate.accept(m)
        self.assertEqual(gate.phase, "ABORTED")
        if code:
            self.assertEqual(result.exception.code, code)

    def test_complete_two_microbatch_trace_updates_once_and_commits(self):
        calls = []
        trace = Trace(callbacks={w: lambda w=w: calls.append(w) for w in WORKERS}).complete()
        self.assertEqual(calls, list(WORKERS))
        for gate in [trace.coordinator, *trace.workers.values()]:
            self.assertEqual((gate.phase, gate.step_id, gate.parameter_version), ("IDLE", 1, 1))
            self.assertEqual(gate.valid_seen, 82)
        self.assertEqual(len([m for m in trace.events if m["type"] == "ACTIVATION"]), 4)

    def test_zero_valid_microbatch_allowed_in_nonempty_step(self):
        trace = Trace().setup().begin(total=64).micro(0, 64).micro(1, 0).prepare().apply().commit()
        self.assertEqual(trace.coordinator.valid_seen, 64)

    def test_no_target_until_both_started(self):
        trace = Trace().setup()
        b = {"microbatch_count": 2, "total_valid_tokens": 82, "batch_plan_hash": HASH}
        trace.emit("STEP_BEGIN", body=b, step=0)
        trace.emit("STEP_STARTED", outgoing=False, step=0)
        self.rejected(trace.coordinator, metadata("TARGET", receiver="worker-1", body={"valid_tokens": 64},
                      step=0, micro=0, tensor=tensor("TARGET")))

    def test_input_requires_target_ack_and_activation_relay(self):
        for skip in ("target_ack", "relay"):
            trace = Trace().setup().begin()
            trace.emit("TARGET", "worker-1", body={"valid_tokens": 64}, step=0,
                       micro=0, descriptor=tensor("TARGET"))
            if skip == "target_ack":
                bad = metadata("INPUT", step=0, micro=0, tensor=tensor("INPUT"))
            else:
                trace.emit("TARGET_READY", "worker-1", False, step=0, micro=0)
                trace.emit("INPUT", step=0, micro=0, descriptor=tensor("INPUT"))
                trace.emit("ACTIVATION", outgoing=False, step=0, micro=0, descriptor=tensor("ACTIVATION"))
                bad = metadata("BOUNDARY_GRAD", "worker-1", "coordinator",
                               {"loss_sum": 1.0, "valid_tokens": 64}, step=0, micro=0,
                               tensor=tensor("BOUNDARY_GRAD"))
            self.rejected(trace.coordinator, bad)

    def test_prepare_requires_completed_backward(self):
        trace = Trace().setup().begin().micro(0, 64)
        for gate in (trace.coordinator, trace.workers["worker-0"]):
            self.rejected(gate, metadata("STEP_PREPARE", step=0))

    def test_apply_requires_both_ready(self):
        trace = Trace().setup().begin().micro(0, 64).micro(1, 18)
        trace.emit("STEP_PREPARE", step=0)
        trace.emit("STEP_READY", outgoing=False, body={"gradient_finite": True, "microbatches_completed": 2}, step=0)
        self.rejected(trace.coordinator, metadata("STEP_APPLY", body={"apply_id": APPLY, "next_version": 1}, step=0))

    def test_duplicate_apply_cannot_invoke_fake_update_twice(self):
        calls = []
        trace = Trace(callbacks={"worker-0": lambda: calls.append(1)}).prepared()
        event = trace.emit("STEP_APPLY", body={"apply_id": APPLY, "next_version": 1}, step=0)
        self.rejected(trace.workers["worker-0"], event, "DUPLICATE")
        self.assertEqual(calls, [1])
        self.rejected(trace.coordinator, event, "DUPLICATE")

    def test_interrupt_during_apply_poisoned_before_side_effect(self):
        for exc in (RuntimeError, KeyboardInterrupt, SystemExit):
            trace = Trace().prepared()
            worker = trace.workers["worker-0"]
            def fail():
                self.assertEqual(worker.phase, "APPLYING")
                raise exc("simulated interruption")
            worker.apply_update = fail
            event = metadata("STEP_APPLY", body={"apply_id": APPLY, "next_version": 1}, step=0)
            with self.assertRaises(exc):
                worker.accept(event)
            self.assertEqual(worker.phase, "ABORTED")
            self.rejected(worker, event, "WRONG_STATE")

    def test_missing_applied_reply_never_authorizes_commit(self):
        trace = Trace().prepared()
        trace.emit("STEP_APPLY", body={"apply_id": APPLY, "next_version": 1}, step=0)
        self.rejected(trace.coordinator, metadata("STEP_COMMIT", body={"apply_id": APPLY, "next_version": 1}, step=0, version=1))

    def test_partial_update_disconnect_aborts_every_gate(self):
        trace = Trace().prepared()
        trace.emit("STEP_APPLY", body={"apply_id": APPLY, "next_version": 1}, step=0)
        trace.emit("STEP_APPLIED", outgoing=False, body={"apply_id": APPLY, "next_version": 1}, step=0, version=1)
        for gate in (trace.coordinator, *trace.workers.values()):
            with self.assertRaises(ProtocolError) as result:
                gate.disconnect()
            self.assertEqual(result.exception.code, "PEER_LOST")
            self.assertEqual(gate.phase, "ABORTED")

    def test_next_step_waits_for_both_committed(self):
        trace = Trace().prepared().apply()
        trace.emit("STEP_COMMIT", body={"apply_id": APPLY, "next_version": 1}, step=0, version=1)
        trace.emit("STEP_COMMITTED", outgoing=False, body={"apply_id": APPLY}, step=0, version=1)
        self.rejected(trace.coordinator, metadata("STEP_BEGIN", body={"microbatch_count": 1,
                      "total_valid_tokens": 64, "batch_plan_hash": HASH}, step=1, version=1))

    def test_stale_run_step_version_and_microbatch_are_terminal(self):
        for field, value in (("run_id", CHECKPOINT), ("step_id", 1), ("parameter_version", 1), ("microbatch_id", 1)):
            trace = Trace().setup().begin()
            event = metadata("TARGET", receiver="worker-1", body={"valid_tokens": 64}, step=0,
                             micro=0, tensor=tensor("TARGET"))
            event[field] = value
            self.rejected(trace.coordinator, event)
            self.rejected(trace.workers["worker-1"], event)

    def test_target_and_hidden_shape_must_match_configuration(self):
        for dimension, value in ((0, 3), (1, 31)):
            trace = Trace().setup().begin()
            desc = tensor("TARGET")
            desc["shape"][dimension] = value
            desc["nbytes"] = math.prod(desc["shape"]) * 8
            event = metadata("TARGET", receiver="worker-1", body={"valid_tokens": 1}, step=0, micro=0, tensor=desc)
            self.rejected(trace.coordinator, event, "SHAPE_MISMATCH")
            self.rejected(trace.workers["worker-1"], event, "SHAPE_MISMATCH")

    def test_input_batch_must_match_target_batch(self):
        trace = Trace().setup().begin()
        trace.emit("TARGET", "worker-1", body={"valid_tokens": 32}, step=0, micro=0, descriptor=tensor("TARGET", b=1))
        trace.emit("TARGET_READY", "worker-1", False, step=0, micro=0)
        self.rejected(trace.coordinator, metadata("INPUT", step=0, micro=0, tensor=tensor("INPUT", b=2)), "SHAPE_MISMATCH")

    def test_wrong_token_total_rejected_before_prepare(self):
        trace = Trace().setup().begin(total=81).micro(0, 64)
        with self.assertRaises(ProtocolError):
            trace.micro(1, 18)
        self.assertEqual(trace.workers["worker-1"].phase, "ABORTED")

    def test_gradient_token_count_must_match_target(self):
        trace = Trace().setup().begin()
        trace.emit("TARGET", "worker-1", body={"valid_tokens": 64}, step=0, micro=0, descriptor=tensor("TARGET"))
        trace.emit("TARGET_READY", "worker-1", False, step=0, micro=0)
        trace.emit("INPUT", step=0, micro=0, descriptor=tensor("INPUT"))
        trace.emit("ACTIVATION", outgoing=False, step=0, micro=0, descriptor=tensor("ACTIVATION"))
        trace.emit("ACTIVATION", "worker-1", step=0, micro=0, descriptor=tensor("ACTIVATION"))
        bad = metadata("BOUNDARY_GRAD", "worker-1", "coordinator", {"valid_tokens": 63, "loss_sum": 1.0},
                       step=0, micro=0, tensor=tensor("BOUNDARY_GRAD"))
        self.rejected(trace.coordinator, bad, "CONFIG_MISMATCH")
        self.rejected(trace.workers["worker-1"], bad, "CONFIG_MISMATCH")

    def test_ready_count_and_finiteness(self):
        for count, finite in ((1, True), (2, False)):
            trace = Trace().setup().begin().micro(0, 64).micro(1, 18)
            trace.emit("STEP_PREPARE", step=0)
            bad = metadata("STEP_READY", "worker-0", "coordinator",
                           {"gradient_finite": finite, "microbatches_completed": count}, step=0)
            self.rejected(trace.coordinator, bad)
            self.rejected(trace.workers["worker-0"], bad)

    def test_status_cannot_extend_compute_timeout(self):
        now = [0.0]
        trace = Trace(clock=lambda: now[0]).setup().begin()
        for timestamp in (100.0, 200.0, 299.0):
            now[0] = timestamp
            trace.emit("STATUS", outgoing=False, body={"phase": "ACCUMULATING",
                       "last_completed_microbatch": None, "last_progress_age_ms": int(timestamp * 1000)})
        now[0] = 300.0
        for gate in (trace.coordinator, trace.workers["worker-0"]):
            with self.assertRaises(ProtocolError) as result:
                gate.check_timeout()
            self.assertEqual(result.exception.code, "TIMEOUT")
            self.assertEqual(gate.phase, "ABORTED")

    def test_checkpoint_requires_all_shards_and_acks(self):
        trace = Trace().complete()
        for w in WORKERS:
            trace.emit("CHECKPOINT_BEGIN", w, body={"checkpoint_id": CHECKPOINT}, step=1, version=1)
            trace.emit("CHECKPOINT_READY", w, False, {"checkpoint_id": CHECKPOINT,
                       "shard_index_hash": HASH, "shard_bytes": 1000}, step=1, version=1)
        for w in WORKERS:
            trace.emit("CHECKPOINT_COMMIT", w, body={"checkpoint_id": CHECKPOINT,
                       "manifest_hash": HASH}, step=1, version=1)
            trace.emit("CHECKPOINT_ACK", w, False, {"checkpoint_id": CHECKPOINT}, step=1, version=1)
            if w == "worker-0":
                self.assertEqual(trace.coordinator.phase, "CHECKPOINTING")
        self.assertEqual(trace.coordinator.phase, "IDLE")

    def test_next_step_before_checkpoint_second_ack_rejected(self):
        trace = Trace().complete()
        for w in WORKERS:
            trace.emit("CHECKPOINT_BEGIN", w, body={"checkpoint_id": CHECKPOINT}, step=1, version=1)
            trace.emit("CHECKPOINT_READY", w, False, {"checkpoint_id": CHECKPOINT,
                       "shard_index_hash": HASH, "shard_bytes": 1000}, step=1, version=1)
        trace.emit("CHECKPOINT_COMMIT", body={"checkpoint_id": CHECKPOINT, "manifest_hash": HASH}, step=1, version=1)
        trace.emit("CHECKPOINT_ACK", outgoing=False, body={"checkpoint_id": CHECKPOINT}, step=1, version=1)
        self.rejected(trace.coordinator, metadata("STEP_BEGIN", body={"microbatch_count": 1,
                      "total_valid_tokens": 64, "batch_plan_hash": HASH}, step=1, version=1))

    def test_checkpoint_status_does_not_extend_absolute_deadline(self):
        now = [0.0]
        trace = Trace(clock=lambda: now[0]).setup()
        trace.emit("CHECKPOINT_BEGIN", body={"checkpoint_id": CHECKPOINT}, step=0)
        now[0] = 599.0
        trace.emit("STATUS", outgoing=False, body={"phase": "CHECKPOINTING", "last_completed_microbatch": None,
                                                 "last_progress_age_ms": 599000})
        now[0] = 600.0
        for gate in (trace.coordinator, trace.workers["worker-0"]):
            with self.assertRaises(ProtocolError) as result:
                gate.check_timeout()
            self.assertEqual(result.exception.code, "TIMEOUT")

    def test_partition_gap_and_wrong_loaded_name_hash(self):
        trace = Trace()
        for w in WORKERS:
            trace.emit("CAPABILITIES", w, False, capabilities(), version=None)
        trace.emit("RUN_CONFIG", body=config_body(0))
        cfg = config_body(1)
        cfg["block_start"] = 3
        self.rejected(trace.coordinator, metadata("RUN_CONFIG", receiver="worker-1", body=cfg), "CONFIG_MISMATCH")
        worker = trace.workers["worker-0"]
        self.rejected(worker, metadata("RUN_READY", "worker-0", "coordinator",
                      {"config_hash": config_body()["config_hash"], "stage_id": 0,
                       "loaded_version": 0, "parameter_name_hash": "b" * 64}), "CONFIG_MISMATCH")

    def test_malformed_step_counts_and_duplicate_ack(self):
        for count, total in ((0, 1), (1, 0), (1, 65), (True, 1)):
            gate = Trace().setup().coordinator
            self.rejected(gate, metadata("STEP_BEGIN", body={"microbatch_count": count,
                          "total_valid_tokens": total, "batch_plan_hash": HASH}, step=0))
        trace = Trace().prepared()
        bad = metadata("STEP_READY", "worker-0", "coordinator",
                       {"gradient_finite": True, "microbatches_completed": 2}, step=0)
        self.rejected(trace.coordinator, bad, "DUPLICATE")

    def test_stop_handshake_fences_new_work(self):
        trace = Trace().setup()
        trace.emit("STOP")
        event = metadata("STEP_BEGIN", body={"microbatch_count": 1,
                         "total_valid_tokens": 64, "batch_plan_hash": HASH}, step=0)
        self.rejected(trace.coordinator, event)
        self.rejected(trace.workers["worker-0"], event)
        trace = Trace().complete()
        for worker in WORKERS:
            trace.emit("STOP", worker, version=1)
            trace.emit("STOPPED", worker, False, version=1)
        self.assertTrue(all(g.phase == "STOPPED" for g in [trace.coordinator, *trace.workers.values()]))

    def test_early_operational_message_is_protocol_error(self):
        for event in (metadata("STEP_APPLIED", "worker-0", "coordinator",
                               {"apply_id": APPLY, "next_version": 1}, step=0, version=1),
                      metadata("TARGET", receiver="worker-1", body={"valid_tokens": 64},
                               step=0, micro=0, tensor=tensor("TARGET"))):
            self.rejected(CoordinatorState(RUN), event)
            worker = event["sender"] if event["sender"] != "coordinator" else event["receiver"]
            self.rejected(WorkerState(RUN, worker), event)

    def test_relay_must_preserve_loss_and_shape(self):
        trace = Trace().setup().begin()
        trace.emit("TARGET", "worker-1", body={"valid_tokens": 64}, step=0, micro=0, descriptor=tensor("TARGET"))
        trace.emit("TARGET_READY", "worker-1", False, step=0, micro=0)
        trace.emit("INPUT", step=0, micro=0, descriptor=tensor("INPUT"))
        desc = tensor("ACTIVATION")
        desc["shape"][2] = 64
        desc["nbytes"] //= 2
        bad = metadata("ACTIVATION", "worker-0", "coordinator", step=0, micro=0, tensor=desc)
        self.rejected(trace.coordinator, bad, "SHAPE_MISMATCH")
        self.rejected(trace.workers["worker-0"], bad, "SHAPE_MISMATCH")

    def test_slow_apply_cannot_reset_expired_deadline(self):
        now = [0.0]
        trace = Trace(clock=lambda: now[0]).prepared()
        def update():
            now[0] = 301.0
        worker = trace.workers["worker-0"]
        worker.apply_update = update
        self.rejected(worker, metadata("STEP_APPLY", body={"apply_id": APPLY, "next_version": 1}, step=0), "TIMEOUT")

    def test_checkpoint_cannot_publish_with_only_one_shard(self):
        trace = Trace().setup()
        trace.emit("CHECKPOINT_BEGIN", body={"checkpoint_id": CHECKPOINT}, step=0)
        trace.emit("CHECKPOINT_READY", outgoing=False, body={"checkpoint_id": CHECKPOINT,
                   "shard_index_hash": HASH, "shard_bytes": 1000}, step=0)
        self.rejected(trace.coordinator, metadata("CHECKPOINT_COMMIT",
                      body={"checkpoint_id": CHECKPOINT, "manifest_hash": HASH}, step=0))

    def test_next_step_succeeds_with_new_version_and_apply_id(self):
        trace = Trace().complete()
        training_events = [deepcopy(m) for m in trace.events if m["step_id"] == 0]
        for event in training_events:
            event["step_id"] += 1
            event["parameter_version"] += 1
            if "apply_id" in event["body"]:
                event["body"]["apply_id"] = apply_id_for_step(RUN, 1)
            if "next_version" in event["body"]:
                event["body"]["next_version"] += 1
            sender, receiver = event["sender"], event["receiver"]
            event["seq"] = trace.seq[sender, receiver]
            trace.seq[sender, receiver] += 1
            worker = receiver if sender == "coordinator" else sender
            trace.coordinator.accept(event)
            trace.workers[worker].accept(event)
        self.assertEqual((trace.coordinator.step_id, trace.coordinator.parameter_version), (2, 2))

    def test_restore_nonzero_counters_and_fresh_run(self):
        trace = Trace()
        for w in WORKERS:
            trace.emit("CAPABILITIES", w, False, capabilities(), version=None)
        for stage, w in enumerate(WORKERS):
            config = config_body(stage)
            config.update(starting_step=7, starting_version=7, checkpoint_id=CHECKPOINT)
            trace.emit("RUN_CONFIG", w, body=config, version=7)
            trace.emit("RUN_READY", w, False, body={"config_hash": config["config_hash"],
                       "stage_id": stage, "loaded_version": 7, "parameter_name_hash": HASH}, version=7)
        self.assertEqual((trace.coordinator.step_id, trace.coordinator.parameter_version), (7, 7))
        trace.emit("STEP_BEGIN", body={"microbatch_count": 1,
                   "total_valid_tokens": 64, "batch_plan_hash": HASH}, step=7, version=7)

    def test_invalid_direction_rejected(self):
        trace = Trace().setup().begin()
        bad = metadata("TARGET", "worker-0", "worker-1", {"valid_tokens": 64},
                       step=0, micro=0, tensor=tensor("TARGET"))
        self.rejected(trace.coordinator, bad)
        self.rejected(trace.workers["worker-1"], bad)

    def test_crossed_ping_and_apply_ack_versions_are_liveness_only(self):
        now = [0.0]
        trace = Trace(clock=lambda: now[0]).prepared()
        trace.emit("STEP_APPLY", body={"apply_id": APPLY, "next_version": 1}, step=0)
        # Coordinator sends a v0 PING before the v1 APPLIED crosses it in the
        # opposite stream. Delivery at the worker happens after its update.
        ping = metadata("PING", body={"nonce": 3}, version=0)
        trace.coordinator.accept(ping)
        trace.emit("STEP_APPLIED", outgoing=False, body={"apply_id": APPLY, "next_version": 1}, step=0, version=1)
        now[0] = 299.0
        trace.workers["worker-0"].accept(ping)
        trace.emit("PONG", outgoing=False, body={"nonce": 3}, version=1)
        # Coordinator still stores global v0 while worker-0 is at v1, but
        # new PINGs must use the worker version established by its APPLIED ack.
        trace.emit("PING", body={"nonce": 4}, version=1)
        trace.emit("PONG", outgoing=False, body={"nonce": 4}, version=1)
        self.assertEqual(trace.coordinator.phase, "APPLYING")
        self.assertEqual(trace.workers["worker-0"].phase, "COMMITTING")
        now[0] = 300.0
        for gate in (trace.coordinator, trace.workers["worker-0"]):
            with self.assertRaises(ProtocolError) as error:
                gate.check_timeout()
            self.assertEqual(error.exception.code, "TIMEOUT")

    def test_commit_phase_adjacent_heartbeats_do_not_relax_other_messages(self):
        trace = Trace().prepared().apply()
        for version in (0, 1):
            # Only incoming liveness events accept adjacent versions; new
            # outgoing events always reflect observed/current local state.
            trace.workers["worker-0"].accept(metadata("PING", body={"nonce": version}, version=version))
            trace.coordinator.accept(metadata("PONG", "worker-0", "coordinator", {"nonce": version}, version=version))
        bad = metadata("STEP_COMMIT", body={"apply_id": APPLY, "next_version": 1}, step=0, version=0)
        self.rejected(trace.coordinator, bad, "VERSION_MISMATCH")
        self.rejected(trace.workers["worker-0"], bad, "VERSION_MISMATCH")
        trace = Trace().prepared().apply()
        status = metadata("STATUS", "worker-0", "coordinator", {"phase": "COMMITTING",
                          "last_completed_microbatch": 1, "last_progress_age_ms": 0}, version=0)
        self.rejected(trace.coordinator, status, "VERSION_MISMATCH")
        self.rejected(trace.workers["worker-0"], status, "VERSION_MISMATCH")

    def test_heartbeat_versions_are_bounded_and_idle_stays_exact(self):
        trace = Trace().prepared().apply()
        bad = metadata("PING", body={"nonce": 1}, version=2)
        self.rejected(trace.coordinator, bad, "VERSION_MISMATCH")
        self.rejected(trace.workers["worker-0"], bad, "VERSION_MISMATCH")
        trace = Trace().complete()
        bad["parameter_version"] = 0
        self.rejected(trace.coordinator, bad, "VERSION_MISMATCH")
        self.rejected(trace.workers["worker-0"], bad, "VERSION_MISMATCH")

    def test_apply_id_is_bound_to_run_and_step_without_history(self):
        self.assertEqual(apply_id_for_step(RUN, 0), APPLY)
        self.assertNotEqual(APPLY, apply_id_for_step(RUN, 1))
        self.assertNotEqual(APPLY, apply_id_for_step(CHECKPOINT, 0))
        for bad_step in (-1, True, 2**63, "0"):
            with self.assertRaises(ValueError):
                apply_id_for_step(RUN, bad_step)
        for wrong_id in (apply_id_for_step(RUN, 1), apply_id_for_step(CHECKPOINT, 0)):
            calls = []
            trace = Trace(callbacks={"worker-0": lambda: calls.append(1)}).prepared()
            bad = metadata("STEP_APPLY", body={"apply_id": wrong_id, "next_version": 1}, step=0)
            self.rejected(trace.coordinator, bad, "VERSION_MISMATCH")
            self.rejected(trace.workers["worker-0"], bad, "VERSION_MISMATCH")
            self.assertFalse(calls)
        trace = Trace().complete()
        self.assertFalse(hasattr(trace.coordinator, "used_apply_ids"))
        self.assertFalse(hasattr(trace.workers["worker-0"], "used_apply_ids"))

    def test_ping_after_first_commit_ack_uses_that_workers_version(self):
        trace = Trace().prepared().apply()
        trace.emit("STEP_COMMIT", body={"apply_id": APPLY, "next_version": 1}, step=0, version=1)
        trace.emit("STEP_COMMITTED", outgoing=False, body={"apply_id": APPLY}, step=0, version=1)
        self.assertEqual(trace.coordinator.phase, "COMMITTING")
        self.assertEqual(trace.workers["worker-0"].phase, "IDLE")
        trace.emit("PING", body={"nonce": 1}, version=1)
        trace.emit("PONG", outgoing=False, body={"nonce": 1}, version=1)
        old_ping = metadata("PING", body={"nonce": 2}, version=0)
        self.rejected(trace.coordinator, old_ping, "VERSION_MISMATCH")
        self.rejected(trace.workers["worker-0"], old_ping, "VERSION_MISMATCH")

    def test_new_ping_and_pong_do_not_emit_old_update_versions(self):
        trace = Trace().prepared()
        trace.emit("STEP_APPLY", body={"apply_id": APPLY, "next_version": 1}, step=0)
        trace.emit("STEP_APPLIED", outgoing=False, body={"apply_id": APPLY, "next_version": 1}, step=0, version=1)
        self.rejected(trace.coordinator, metadata("PING", body={"nonce": 1}, version=0), "VERSION_MISMATCH")
        self.rejected(trace.workers["worker-0"], metadata("PONG", "worker-0", "coordinator",
                      {"nonce": 1}, version=0), "VERSION_MISMATCH")


if __name__ == "__main__":
    unittest.main()
