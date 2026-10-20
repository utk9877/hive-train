"""Verify diagnostic evidence and the shared runtime heartbeat adapter."""

import json
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import torch

from hive_train.config import ModelConfig, OptimizerConfig
from hive_train.data import synthetic_batches
from hive_train.diagnostics import SnapshotWriter
from hive_train.runtime import Channel
from hive_train.snapshot_check import check
from hive_train.local_run import batches_for_step
from test_runtime import LocalRun
from test_state import Trace, RUN, APPLY


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_final_snapshot_comparison_and_wrong_run_rejection(self):
        with tempfile.TemporaryDirectory(prefix="hive-snapshot-test-") as temp:
            root = Path(temp)
            writers = [SnapshotWriter(root / f"stage-{i}") for i in (0, 1)]
            def observe(stage, optimizer, event, details):
                writers[details["stage_id"]](stage, optimizer, event, details)
            run = LocalRun(self, observer=observe, optimizer=OptimizerConfig(eps=1e-6))
            for step in range(2):
                last = run.coordinator.step(batches_for_step(step, count=2, scenario="masked"))
            report = {**run.stop(), "assignments": run.assignment,
                      "capabilities": run.coordinator.capabilities,
                      "last_step": last, "workload": {"scenario": "masked", "microbatches": 2, "seed_base": 7}}
            result = check(report, root / "stage-0", root / "stage-1")
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["step_id"], 1)
            malformed = deepcopy(report)
            malformed["workload"]["microbatches"] = 100000000
            with self.assertRaisesRegex(ValueError, "microbatches"):
                check(malformed, root / "stage-0", root / "stage-1")
            index_path = root / "stage-1/index.json"
            index = json.loads(index_path.read_text())
            index["run_id"] = "00000000-0000-4000-8000-000000000000"
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "run_id mismatch"):
                check(report, root / "stage-0", root / "stage-1")

    def test_pattern_evaluation_start_is_reserved_from_training(self):
        # Complete modular cycle, including old collision at step59.
        for step in range(256):
            for batch in batches_for_step(step, count=4, scenario="pattern"):
                self.assertNotEqual(int(batch.inputs[0, 0]), 157)

    def test_channel_ping_uses_already_applied_peer_version(self):
        trace = Trace().prepared()
        trace.emit("STEP_APPLY", body={"apply_id": APPLY, "next_version": 1}, step=0)
        trace.emit("STEP_APPLIED", outgoing=False, body={"apply_id": APPLY, "next_version": 1}, step=0, version=1)
        session = SimpleNamespace(local="coordinator", peer="worker-0", run_id=RUN,
                                  send_seq=50, send=Mock(), close=Mock())
        Channel(session, trace.coordinator).send("PING", body={"nonce": 11})
        self.assertEqual(session.send.call_args.kwargs["version"], 1)
        self.assertEqual(trace.coordinator.phase, "APPLYING")
        session.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
