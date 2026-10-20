"""Worker-local failure and lifecycle checks with scripted authenticated frames.

Socket authentication and actual two-worker numerical equivalence are exercised
separately in test_runtime. These tests isolate payload semantics and cleanup.
"""

from copy import deepcopy
import gc
import os
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch

from hive_train.config import ModelConfig, json_hash
from hive_train.data import synthetic_batches
from hive_train.model import build_stage
from hive_train.protocol import Frame, ProtocolError, validate_metadata
from hive_train.runtime import make_assignments
from hive_train.state import apply_id_for_step
from hive_train.tensor_codec import decode_tensor, encode_tensor
from hive_train.transport import Timeouts, metadata
from hive_train.worker import WorkerRuntime


RUN = "ea717f46-9da6-489f-bfa0-340851e97b57"


class ScriptedSession:
    """Already-authenticated local test double; records only tiny test frames."""

    def __init__(self, worker="worker-1"):
        self.local, self.peer, self.run_id = worker, "coordinator", RUN
        self.send_seq, self.receive_seq = 1, 2
        self.stream = SimpleNamespace(timeouts=Timeouts(), max_tensor_bytes=67108864)
        self.incoming = []
        self.sent = []
        self.closed = False

    def add(self, kind, *, body=None, step=None, microbatch=None, version=0, tensor=None):
        desc, payload = (None, bytearray()) if tensor is None else encode_tensor(tensor[0], tensor[1])
        m = metadata(kind, RUN, "coordinator", self.local, 2 + len(self.incoming),
                     step=step, microbatch=microbatch, version=version, body=body, tensor=desc)
        validate_metadata(m, len(payload))
        self.incoming.append(Frame(m, payload))
        return self

    def send(self, kind, *, step=None, microbatch=None, version=None, body=None,
             tensor=None, payload=b"", deadline=None):
        m = metadata(kind, RUN, self.local, self.peer, self.send_seq, step=step,
                     microbatch=microbatch, version=version, body=body, tensor=tensor)
        validate_metadata(m, len(payload))
        self.sent.append(Frame(m, bytearray(payload)))
        self.send_seq += 1

    def receive(self, *, config=None, expected_type=None, deadline=None, wait_timeout=None):
        if not self.incoming:
            raise ProtocolError("PEER_LOST", "scripted peer disconnected")
        frame = self.incoming.pop(0)
        if expected_type is not None and frame.metadata["type"] != expected_type:
            raise ProtocolError("WRONG_STATE", "unexpected scripted message")
        self.receive_seq += 1
        return frame

    def close(self):
        self.closed = True


def suffix_session(*, ignored=False):
    session = ScriptedSession()
    assignment = make_assignments()["worker-1"]
    batch = synthetic_batches(ModelConfig())[0]
    prefix = build_stage(end=2)
    hidden = prefix(batch.inputs).detach()
    if ignored:
        batch.targets[:, 9:] = -100
    valid = int(batch.targets.ne(-100).sum())
    session.add("RUN_CONFIG", body=assignment)
    session.add("STEP_BEGIN", step=0, body={"microbatch_count": 1,
                "total_valid_tokens": valid, "batch_plan_hash": "c" * 64})
    session.add("TARGET", step=0, microbatch=0, body={"valid_tokens": valid}, tensor=(batch.targets, "targets"))
    session.add("ACTIVATION", step=0, microbatch=0, tensor=(hidden, "hidden"))
    session.add("STEP_PREPARE", step=0)
    session.add("STEP_APPLY", step=0, body={"apply_id": apply_id_for_step(RUN, 0), "next_version": 1})
    session.add("STEP_COMMIT", step=0, version=1, body={"apply_id": apply_id_for_step(RUN, 0), "next_version": 1})
    session.add("STOP", version=1)
    return session


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_failed_cleanup(self, worker, session):
        self.assertTrue(session.closed)
        self.assertEqual(worker.gate.phase, "ABORTED")
        self.assertIsNone(worker.stage)
        self.assertIsNone(worker.optimizer)
        self.assertIsNone(worker._hidden)
        self.assertIsNone(worker._targets)
        self.assertNotIn("STEP_APPLIED", [f.metadata["type"] for f in session.sent])
        with self.assertRaises(ProtocolError):
            worker.run()

    def test_real_suffix_update_and_clean_stop(self):
        session = suffix_session(ignored=True)
        observed = []
        def observer(stage, optimizer, event, details):
            observed.append((event, details))
            if event == "after_step":
                self.assertEqual(worker.gate.phase, "APPLYING")
                for p in stage.parameters():
                    self.assertIsNotNone(p.grad)
                    self.assertEqual(int(optimizer.state[p]["step"]), 1)
        worker = WorkerRuntime(session, diagnostic=True, observer=observer)
        result = worker.run()
        self.assertEqual(result["steps_committed"], 1)
        self.assertEqual(result["parameter_version"], 1)
        self.assertEqual(result["microbatches_completed"], 1)
        self.assertGreater(result["last_loss"], 0)
        self.assertEqual([e for e, _ in observed], ["before_step", "micro_done", "after_step"])
        micro = observed[1][1]
        self.assertEqual(micro["valid_tokens"], 18)
        self.assertEqual(micro["boundary_grad"].shape, (2, 32, 128))
        self.assertEqual(micro["logits"].shape, (2, 32, 256))
        self.assertFalse(micro["boundary_grad"].requires_grad)
        self.assertEqual(worker.gate.phase, "STOPPED")
        self.assertTrue(session.closed)
        self.assertIsNone(worker.stage)

    def test_only_owned_stage_materialized(self):
        for worker_id, start, end in (("worker-0", 0, 2), ("worker-1", 2, 4)):
            session = ScriptedSession(worker_id)
            session.add("RUN_CONFIG", body=make_assignments()[worker_id]).add("STOP")
            with patch("hive_train.worker.build_stage", wraps=build_stage) as materialize:
                WorkerRuntime(session).run()
            materialize.assert_called_once()
            self.assertEqual(materialize.call_args.args[1:], (start, end))

    def test_actual_target_count_cannot_lie(self):
        session = suffix_session()
        session.incoming[2].metadata["body"]["valid_tokens"] = 63
        worker = WorkerRuntime(session)
        with self.assertRaises(ProtocolError) as error:
            worker.run()
        self.assertEqual(error.exception.code, "CONFIG_MISMATCH")
        self.assert_failed_cleanup(worker, session)
        self.assertNotIn("TARGET_READY", [f.metadata["type"] for f in session.sent])

    def test_target_index_validation_precedes_embedding_loss(self):
        session = suffix_session()
        tensor = torch.zeros((2, 32), dtype=torch.int64)
        tensor[0, 0] = 256
        descriptor, payload = encode_tensor(tensor, "targets")
        session.incoming[2].metadata["tensor"] = descriptor
        session.incoming[2].payload = payload
        worker = WorkerRuntime(session)
        with self.assertRaises(ProtocolError) as error:
            worker.run()
        self.assertEqual(error.exception.code, "SHAPE_MISMATCH")
        self.assert_failed_cleanup(worker, session)

    def test_apply_interruption_never_emits_ack_or_retries(self):
        for failure in (KeyboardInterrupt, SystemExit, RuntimeError):
            session = suffix_session()
            called = []
            def observer(stage, optimizer, event, details):
                if event == "before_step":
                    original = optimizer.step
                    def interrupted():
                        self.assertEqual(worker.gate.phase, "APPLYING")
                        called.append(1)
                        original()
                        raise failure("simulated interruption after optimizer update")
                    optimizer.step = interrupted
            worker = WorkerRuntime(session, diagnostic=True, observer=observer)
            with self.assertRaises(failure):
                worker.run()
            self.assertEqual(called, [1])
            self.assert_failed_cleanup(worker, session)

    def test_missing_gradient_is_rejected_at_prepare(self):
        session = suffix_session()
        def observer(stage, optimizer, event, details):
            if event == "micro_done":
                next(stage.parameters()).grad = None
        worker = WorkerRuntime(session, diagnostic=True, observer=observer)
        with self.assertRaises(ProtocolError) as error:
            worker.run()
        self.assertEqual(error.exception.code, "DEVICE_ERROR")
        self.assert_failed_cleanup(worker, session)
        self.assertNotIn("STEP_READY", [f.metadata["type"] for f in session.sent])

    def test_global_manifest_hash_verified_before_materialization(self):
        assignment = deepcopy(make_assignments()["worker-1"])
        assignment["config"]["parameter_spec_hash"] = "f" * 64
        assignment["config_hash"] = json_hash(assignment["config"])
        session = ScriptedSession().add("RUN_CONFIG", body=assignment)
        worker = WorkerRuntime(session)
        with patch("hive_train.worker.build_stage") as materialize:
            with self.assertRaises(ProtocolError) as error:
                worker.run()
            materialize.assert_not_called()
        self.assertEqual(error.exception.code, "CONFIG_MISMATCH")
        self.assert_failed_cleanup(worker, session)

    def test_checkpoint_restore_rejected_explicitly(self):
        assignment = make_assignments()["worker-1"]
        assignment.update(checkpoint_id=RUN, starting_step=1, starting_version=1)
        session = ScriptedSession().add("RUN_CONFIG", body=assignment, version=1)
        worker = WorkerRuntime(session)
        with self.assertRaises(ProtocolError) as error:
            worker.run()
        self.assertEqual(error.exception.code, "CHECKPOINT_ERROR")
        self.assert_failed_cleanup(worker, session)

    def test_observers_require_bounded_fixture(self):
        with self.assertRaises(ValueError):
            WorkerRuntime(ScriptedSession(), observer=lambda *_: None)
        assignment = make_assignments(length=64)["worker-1"]
        session = ScriptedSession().add("RUN_CONFIG", body=assignment)
        worker = WorkerRuntime(session, diagnostic=True, observer=lambda *_: None)
        with self.assertRaises(ProtocolError) as error:
            worker.run()
        self.assertEqual(error.exception.code, "CONFIG_MISMATCH")
        self.assert_failed_cleanup(worker, session)

    def test_observer_microbatch_cap_rejects_before_zero_grad(self):
        session = ScriptedSession().add("RUN_CONFIG", body=make_assignments()["worker-1"])
        session.add("STEP_BEGIN", step=0, body={"microbatch_count": 5,
                    "total_valid_tokens": 320, "batch_plan_hash": "c" * 64})
        worker = WorkerRuntime(session, diagnostic=True, observer=lambda *_: None)
        with self.assertRaises(ProtocolError) as error:
            worker.run()
        self.assertEqual(error.exception.code, "CONFIG_MISMATCH")
        self.assert_failed_cleanup(worker, session)
        self.assertNotIn("STEP_STARTED", [f.metadata["type"] for f in session.sent])

    def test_device_failure_closes_session_without_cpu_fallback(self):
        session = ScriptedSession()
        worker = WorkerRuntime(session, device="cuda")
        with patch("hive_train.worker.require_device", side_effect=RuntimeError("CUDA unavailable")):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                worker.run()
        self.assert_failed_cleanup(worker, session)
        self.assertFalse(session.sent)

    def test_mps_fallback_environment_rejected_before_device_probe(self):
        session = ScriptedSession()
        worker = WorkerRuntime(session, device="mps")
        with patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            with self.assertRaisesRegex(RuntimeError, "fallback must be disabled"):
                worker.run()
        self.assert_failed_cleanup(worker, session)
        self.assertFalse(session.sent)

    def test_nonfinite_boundary_consumes_prefix_graph_and_aborts(self):
        session = ScriptedSession("worker-0")
        batch = synthetic_batches(ModelConfig())[0]
        session.add("RUN_CONFIG", body=make_assignments()["worker-0"])
        session.add("STEP_BEGIN", step=0, body={"microbatch_count": 1,
                    "total_valid_tokens": 64, "batch_plan_hash": "c" * 64})
        session.add("INPUT", step=0, microbatch=0, tensor=(batch.inputs, "input_ids"))
        session.add("BOUNDARY_GRAD", step=0, microbatch=0,
                    body={"valid_tokens": 64, "loss_sum": 1.0},
                    tensor=(torch.zeros(2, 32, 128), "hidden_grad"))
        # Simulate a schema-valid, authenticated peer sending a NaN payload.
        struct.pack_into("<f", session.incoming[-1].payload, 0, float("nan"))
        worker = WorkerRuntime(session)
        with self.assertRaises(ProtocolError) as error:
            worker.run()
        self.assertEqual(error.exception.code, "NONFINITE")
        sent = [f.metadata["type"] for f in session.sent]
        self.assertIn("ACTIVATION", sent)
        self.assertNotIn("MICRO_DONE", sent)
        self.assert_failed_cleanup(worker, session)

    def test_corrupt_optimizer_moment_after_update_never_acknowledged(self):
        session = suffix_session()
        counters = []
        def observer(stage, optimizer, event, details):
            if event == "before_step":
                original = optimizer.step
                def corrupt():
                    original()
                    state = optimizer.state[next(stage.parameters())]
                    counters.append(int(state["step"]))
                    state["exp_avg"].fill_(float("nan"))
                optimizer.step = corrupt
        worker = WorkerRuntime(session, diagnostic=True, observer=observer)
        with self.assertRaises(ProtocolError) as error:
            worker.run()
        self.assertEqual(error.exception.code, "NONFINITE")
        self.assertEqual(counters, [1])
        self.assert_failed_cleanup(worker, session)

    def test_observer_tensor_copies_cannot_corrupt_transmitted_gradient(self):
        session = suffix_session()
        def observer(stage, optimizer, event, details):
            if event == "micro_done":
                details["boundary_grad"].fill_(float("nan"))
                details["logits"].zero_()
        worker = WorkerRuntime(session, diagnostic=True, observer=observer)
        result = worker.run()
        self.assertEqual(result["status"], "pass")
        frame = next(f for f in session.sent if f.metadata["type"] == "BOUNDARY_GRAD")
        gradient = decode_tensor(frame.metadata["tensor"], frame.payload)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_normal_stop_releases_module_optimizer_and_graph_references(self):
        session = suffix_session()
        references = []
        def observer(stage, optimizer, event, details):
            if event == "before_step":
                references.extend((weakref.ref(stage), weakref.ref(optimizer)))
        worker = WorkerRuntime(session, diagnostic=True, observer=observer)
        result = worker.run()
        gc.collect()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["phase"], "STOPPED")
        self.assertTrue(all(reference() is None for reference in references))
        self.assertIsNone(worker._targets)
        self.assertIsNone(worker._hidden)


if __name__ == "__main__":
    unittest.main()
