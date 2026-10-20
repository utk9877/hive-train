import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from hive_train.__main__ import main
from hive_train.config import ModelConfig
from hive_train.config import OptimizerConfig, json_hash
from hive_train.data import synthetic_batches
from hive_train.model import build_stage
from hive_train.validation import TensorComparisonError, assert_tensor_equal, compare_step


class ComparisonDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_explicit_epsilon_is_recorded_hashed_and_validated_by_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "experiment.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["verify", "--split", "1", "--steps", "1", "--adam-eps", "1e-6", "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(report["optimizer"]["eps"], 1e-6)
            self.assertEqual(report["config_hash"], json_hash({"model": report["config"], "optimizer": report["optimizer"]}))
            default_config = {"model": report["config"], "optimizer": {**report["optimizer"], "eps": 1e-8}}
            self.assertNotEqual(report["config_hash"], json_hash(default_config))
            self.assertEqual(OptimizerConfig().eps, 1e-8)
            for invalid in ("0", "nan"):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["verify", "--adam-eps", invalid, "--output", str(output)]), 1)

    def test_adam_near_zero_gradient_sensitivity_is_an_optimizer_configuration_effect(self):
        from hive_train.reference import optimizer_for
        differences = {}
        for epsilon in (1e-8, 1e-6):
            values = []
            for gradient in (-3e-9, -7e-10):
                module = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    module.weight.fill_(0.0006363041466102004)
                module.weight.grad = torch.full_like(module.weight, gradient)
                optimizer_for(module, OptimizerConfig(eps=epsilon)).step()
                values.append(float(module.weight.detach()))
            differences[epsilon] = abs(values[0] - values[1])
        self.assertGreater(differences[1e-8], 1e-4)
        self.assertLess(differences[1e-6], 3e-6)

    def test_small_updates_still_fail_relative_gate_and_expose_absolute_error(self):
        expected = torch.full((128,), 3e-6)
        actual = expected + 2e-7
        # The elementwise absolute gate passes, but update direction/size still matters.
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-3)
        with self.assertRaises(TensorComparisonError) as caught:
            assert_tensor_equal(actual, expected, name="update/blocks.0.attention.key.bias",
                                tier="accelerator", l2=True)
        diagnostic = caught.exception.diagnostic
        self.assertEqual(diagnostic["criterion"], "normalized_l2")
        self.assertEqual(diagnostic["limit"], 0.005)
        self.assertAlmostEqual(diagnostic["max_abs"], 2e-7, delta=1e-12)
        self.assertAlmostEqual(diagnostic["normalized_l2"], 1 / 15, places=6)
        self.assertGreater(diagnostic["reference_norm"], 1e-6)
        json.dumps(diagnostic, allow_nan=False)

    def test_elementwise_failure_has_values_and_location(self):
        with self.assertRaises(TensorComparisonError) as caught:
            assert_tensor_equal(torch.tensor([1., 3.]), torch.tensor([1., 2.]), name="logits/0")
        diagnostic = caught.exception.diagnostic
        self.assertEqual(diagnostic["criterion"], "elementwise")
        self.assertEqual(diagnostic["worst_flat_index"], 1)
        self.assertEqual(diagnostic["actual_at_worst"], 3.)
        self.assertEqual(diagnostic["reference_at_worst"], 2.)

    def test_cli_keeps_context_and_partial_progress_but_exits_failed(self):
        calls = 0

        def inject_second_step_failure(*args, **kwargs):
            nonlocal calls
            metrics = compare_step(*args, **kwargs)
            calls += 1
            if calls == 2:
                assert_tensor_equal(torch.full((128,), 3.2e-6), torch.full((128,), 3e-6),
                    name="update/blocks.0.attention.key.bias", tier="accelerator", l2=True)
            return metrics

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failure.json"
            with patch("hive_train.validation.compare_step", side_effect=inject_second_step_failure):
                with contextlib.redirect_stdout(io.StringIO()) as captured:
                    code = main(["verify", "--split", "1", "--steps", "3", "--output", str(output)])
            report = json.loads(output.read_text())
            summary = json.loads(captured.getvalue())
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["failure"]["step"], 1)
        self.assertEqual(report["failure"]["split"], 1)
        self.assertEqual(report["failure"]["scenario"], "single")
        self.assertEqual(len(report["incomplete_case"]["steps"]), 1)
        self.assertEqual(summary["cases_passed"], 0)
        self.assertEqual(report["parameter_count"], 875264)
        self.assertEqual(report["config"]["architecture"], "decoder_v2")
        self.assertEqual(report["environment"]["model_architecture"], "decoder_v2")
        self.assertIn("fixture", report)
        self.assertIn("config_hash", report)
        for side in ("actual", "reference"):
            evidence = report["failure"]["parameter_evidence"][side]
            for kind in ("gradient", "update", "exp_avg", "exp_avg_sq", "step"):
                self.assertGreaterEqual(evidence[kind]["norm"], 0.)
            self.assertIn("gradient", evidence["at_worst_index"])
            self.assertEqual(evidence["at_worst_index"]["flat_index"], report["failure"]["comparison"]["worst_flat_index"])

    def test_key_bias_is_a_structural_zero_in_high_precision(self):
        model = build_stage().double()
        batch = synthetic_batches(ModelConfig())[0]
        logits = model(batch.inputs)
        F.cross_entropy(logits.reshape(-1, 256), batch.targets.reshape(-1)).backward()
        attention = model.blocks["0"].attention
        self.assertLess(float(attention.key.bias.grad.norm()), 1e-14)
        self.assertGreater(float(attention.key.weight.grad.norm()), 1e-6)
        with torch.no_grad():
            attention.key.bias.add_(0.25)
            shifted = model(batch.inputs)
        torch.testing.assert_close(shifted, logits, atol=1e-12, rtol=1e-12)
