"""Real CPU autograd/AdamW over authenticated socketpairs, not target LAN tests.

Snapshots here are bounded tiny-fixture diagnostics. The coordinator runtime
does not gather model/optimizer tensors; each worker's test observer owns them.
"""

from concurrent.futures import ThreadPoolExecutor
import socket
import struct
import threading
import unittest
from uuid import uuid4

import torch

from hive_train.config import ModelConfig, OptimizerConfig
from hive_train.coordinator import CoordinatorRuntime
from hive_train.data import synthetic_batches
from hive_train.model import build_stage
from hive_train.protocol import ProtocolError
from hive_train.reference import ReferenceTrainer, StepResult
from hive_train.runtime import make_assignments
from hive_train.transport import SocketStream, client_handshake, server_handshake
from hive_train.validation import assert_tensor_equal, compare_step
from hive_train.worker import WorkerRuntime


WORKERS = ("worker-0", "worker-1")


def snapshot(stage, optimizer):
    result = {kind: {} for kind in ("parameters", "gradients", "optimizer")}
    for name, parameter in stage.named_parameters():
        result["parameters"][name] = parameter.detach().cpu().clone()
        if parameter.grad is not None:
            result["gradients"][name] = parameter.grad.detach().cpu().clone()
        for key, tensor in optimizer.state.get(parameter, {}).items():
            result["optimizer"][f"{name}/{key}"] = tensor.detach().cpu().clone()
    return result


class LocalRun:
    """Run the production session/runtime APIs in a bounded local test harness."""

    def __init__(self, case, *, split=2, observer=None, diagnostic=True, optimizer=OptimizerConfig()):
        self.case = case
        self.sessions = {}
        self.clients = {}
        self.workers = {}
        self.threads = {}
        self.errors = {}
        self.results = {}
        self.run_id = str(uuid4())
        self.assignment = make_assignments(split=split, batch_size=2, length=32, model=ModelConfig(), optimizer=optimizer)
        case.addCleanup(self.close)
        with ThreadPoolExecutor(max_workers=1) as executor:
            for worker in WORKERS:
                server_sock, worker_sock = socket.socketpair()
                server_stream, client_stream = SocketStream(server_sock), SocketStream(worker_sock)
                future = executor.submit(server_handshake, server_stream, b"r" * 32, self.run_id)
                self.clients[worker] = client_handshake(client_stream, b"r" * 32, worker)
                self.sessions[worker] = future.result(timeout=5)
        for worker in WORKERS:
            self.workers[worker] = WorkerRuntime(self.clients[worker], device="cpu",
                                                  diagnostic=diagnostic, observer=observer)
            thread = threading.Thread(target=self._run_worker, args=(worker,), daemon=True,
                                      name=f"hive-test-{worker}")
            self.threads[worker] = thread
            thread.start()
        self.coordinator = CoordinatorRuntime(self.sessions, self.assignment)
        try:
            self.coordinator.configure()
        except BaseException as error:
            self.close()
            raise RuntimeError(f"runtime setup failed; worker failures: {self.errors}") from error

    def _run_worker(self, worker):
        try:
            self.results[worker] = self.workers[worker].run()
        except BaseException as error:
            self.errors[worker] = error

    def join(self):
        for thread in self.threads.values():
            thread.join(timeout=5)
            self.case.assertFalse(thread.is_alive(), "worker failed to stop after terminal event")

    def close(self):
        for session in (*self.sessions.values(), *self.clients.values()):
            session.close()
        self.join()

    def stop(self):
        result = self.coordinator.stop()
        self.join()
        self.case.assertFalse(self.errors, self.errors)
        return result


class Capture:
    """One step of small diagnostic tensors, discarded after each comparison."""

    def __init__(self):
        self.clear()
        self.references = {}
        self.update_counts = {worker: 0 for worker in WORKERS}

    def clear(self):
        self.before = {}
        self.after = {}
        self.micro = {worker: [] for worker in WORKERS}

    def __call__(self, stage, optimizer, event, details):
        worker = "worker-0" if stage.start == 0 else "worker-1"
        self.references[worker] = (stage, optimizer)
        if event == "before_step":
            self.before[worker] = snapshot(stage, optimizer)["parameters"]
        elif event == "after_step":
            self.after[worker] = snapshot(stage, optimizer)
            self.update_counts[worker] += 1
        elif event == "micro_done":
            self.micro[worker].append({key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                                       for key, value in details.items()})

    def merged(self):
        result = {kind: {} for kind in ("parameters", "gradients", "optimizer")}
        for worker in WORKERS:
            for kind in result:
                overlap = result[kind].keys() & self.after[worker][kind].keys()
                if overlap:
                    raise AssertionError(f"duplicate global ownership: {overlap}")
                result[kind].update(self.after[worker][kind])
        return result


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_explicit_epsilon_reaches_both_workers_and_matches_reference(self):
        opt = OptimizerConfig(eps=1e-6)
        capture = Capture()
        run = LocalRun(self, split=1, observer=capture, optimizer=opt)
        reference = ReferenceTrainer([build_stage()], split=1, optimizer=opt)
        batches = synthetic_batches(ModelConfig(), count=2)
        batches[1].targets[:, 11:] = -100
        before = reference.snapshot()["parameters"]
        expected = reference.step(batches)
        actual = run.coordinator.step(batches)
        result = StepResult(actual["loss"], actual["valid_tokens"],
            [m["logits"] for m in capture.micro["worker-1"]],
            [m["boundary_grad"] for m in capture.micro["worker-0"]])
        before_actual = {name: value for shard in capture.before.values() for name, value in shard.items()}
        compare_step(result, expected, capture.merged(), reference.snapshot(), before_actual, before, tier="cpu")
        defaults = make_assignments(split=1)
        for worker in WORKERS:
            self.assertEqual(run.workers[worker].optimizer.param_groups[0]["eps"], 1e-6)
            self.assertNotEqual(run.assignment[worker]["config_hash"], defaults[worker]["config_hash"])
            self.assertEqual(defaults[worker]["config"]["optimizer"]["eps"], 1e-8)
        run.stop()

    def test_all_cuts_ten_steps_match_unsplit_gradients_updates_and_adam(self):
        config = ModelConfig()
        for split in (1, 2, 3):
            with self.subTest(split=split):
                capture = Capture()
                run = LocalRun(self, split=split, observer=capture)
                reference = ReferenceTrainer([build_stage(config)], split=split)
                for step in range(10):
                    capture.clear()
                    count = 1 if step % 3 == 0 else 2
                    batches = synthetic_batches(config, count=count, seed=200 + step)
                    if count == 2:
                        batches[1].targets[:, 9:] = -100
                        if step % 3 == 2:
                            batches[0].targets[:] = -100
                    before_reference = reference.snapshot()["parameters"]
                    expected_result = reference.step(batches)
                    actual = run.coordinator.step(batches)
                    self.assertEqual((actual["step_id"], actual["parameter_version"]), (step, step + 1))
                    self.assertEqual(actual["valid_tokens"], expected_result.valid_tokens)
                    for worker in WORKERS:
                        self.assertEqual(capture.update_counts[worker], step + 1)
                        self.assertEqual(len(capture.micro[worker]), count)
                        for index, event in enumerate(capture.micro[worker]):
                            self.assertEqual((event["step_id"], event["microbatch_id"]), (step, index))
                            assert_tensor_equal(event["boundary_grad"], expected_result.boundary_gradients[index],
                                                name=f"{worker}/boundary/{index}", l2=True)
                    suffix = capture.micro["worker-1"]
                    actual_result = StepResult(actual["loss"], actual["valid_tokens"],
                                               [event["logits"] for event in suffix],
                                               [event["boundary_grad"] for event in suffix])
                    before = {name: tensor for worker in WORKERS for name, tensor in capture.before[worker].items()}
                    compare_step(actual_result, expected_result, capture.merged(), reference.snapshot(),
                                 before, before_reference, tier="cpu")
                    self.assertEqual(sum(event["valid_tokens"] for event in suffix), actual["valid_tokens"])
                    self.assertAlmostEqual(sum(event["loss_sum"] for event in suffix) / actual["valid_tokens"],
                                           actual["loss"], places=7)
                run.stop()
                for worker in WORKERS:
                    self.assertEqual(run.workers[worker].gate.phase, "STOPPED")

    def assert_aborted(self, run, capture, *, updates=0):
        run.join()
        self.assertEqual(run.coordinator.gate.phase, "ABORTED")
        self.assertTrue(run.errors, "fault unexpectedly completed as graceful stop")
        for worker in WORKERS:
            self.assertEqual(run.workers[worker].gate.phase, "ABORTED")
            self.assertLessEqual(capture.update_counts[worker], updates)
            if worker in capture.references:
                stage, optimizer = capture.references[worker]
                if updates == 0:
                    self.assertFalse(optimizer.state, "malformed input created optimizer moments")
                    for name, parameter in stage.named_parameters():
                        self.assertTrue(torch.equal(parameter, capture.before[worker][name]), name)
                else:
                    for parameter in stage.parameters():
                        if parameter in optimizer.state:
                            self.assertLessEqual(int(optimizer.state[parameter]["step"]), updates)
        with self.assertRaises((ProtocolError, RuntimeError)):
            run.coordinator.step(synthetic_batches(ModelConfig()))

    def test_authenticated_bad_target_token_is_rejected_before_any_update(self):
        capture = Capture()
        run = LocalRun(self, observer=capture)
        original_send = run.sessions["worker-1"].send

        def corrupt_target(kind, **kwargs):
            if kind == "TARGET":
                payload = bytearray(kwargs["payload"])
                struct.pack_into("<q", payload, 0, ModelConfig().vocab_size)
                kwargs["payload"] = payload
            return original_send(kind, **kwargs)

        run.sessions["worker-1"].send = corrupt_target
        with self.assertRaises((ProtocolError, ValueError)):
            run.coordinator.step(synthetic_batches(ModelConfig()))
        self.assert_aborted(run, capture)

    def test_authenticated_nan_activation_is_rejected_before_any_update(self):
        capture = Capture()
        run = LocalRun(self, observer=capture)
        original_send = run.clients["worker-0"].send

        def corrupt_activation(kind, **kwargs):
            if kind == "ACTIVATION":
                payload = bytearray(kwargs["payload"])
                struct.pack_into("<f", payload, 0, float("nan"))
                kwargs["payload"] = payload
            return original_send(kind, **kwargs)

        run.clients["worker-0"].send = corrupt_activation
        with self.assertRaises((ProtocolError, ValueError)):
            run.coordinator.step(synthetic_batches(ModelConfig()))
        self.assert_aborted(run, capture)

    def test_authenticated_target_count_mismatch_cannot_change_normalization(self):
        capture = Capture()
        run = LocalRun(self, observer=capture)
        original_send = run.sessions["worker-1"].send

        def alter_ignored_count(kind, **kwargs):
            if kind == "TARGET":
                payload = bytearray(kwargs["payload"])
                struct.pack_into("<q", payload, 0, -100)
                kwargs["payload"] = payload  # Header's valid_tokens deliberately unchanged.
            return original_send(kind, **kwargs)

        run.sessions["worker-1"].send = alter_ignored_count
        with self.assertRaises((ProtocolError, ValueError)):
            run.coordinator.step(synthetic_batches(ModelConfig()))
        self.assert_aborted(run, capture)

    def test_disconnect_after_real_update_before_applied_is_terminal(self):
        capture = Capture()
        run_holder = {}

        def disconnect(stage, optimizer, event, details):
            capture(stage, optimizer, event, details)
            if event == "after_step" and stage.start == 0:
                run_holder["run"].clients["worker-0"].close()

        run = LocalRun(self, observer=disconnect)
        run_holder["run"] = run
        with self.assertRaises((ProtocolError, ValueError)):
            run.coordinator.step(synthetic_batches(ModelConfig()))
        self.assertEqual(capture.update_counts["worker-0"], 1, "failure must follow a real optimizer update")
        self.assert_aborted(run, capture, updates=1)

    def test_all_ignored_step_is_rejected_before_forward_or_update(self):
        capture = Capture()
        run = LocalRun(self, observer=capture)
        batches = synthetic_batches(ModelConfig())
        batches[0].targets[:] = -100
        with self.assertRaises((ProtocolError, ValueError)):
            run.coordinator.step(batches)
        self.assertFalse(capture.before, "invalid step started worker computation")
        self.assertEqual(capture.update_counts, {worker: 0 for worker in WORKERS})
        self.assert_aborted(run, capture)


if __name__ == "__main__":
    unittest.main()
