import unittest
from unittest.mock import patch
import torch

from hive_train.boundary import BoundaryKey, LocalBoundary
from hive_train.config import ModelConfig
from hive_train.data import synthetic_batches
from hive_train.model import build_stage
from hive_train.reference import ReferenceTrainer
from hive_train.validation import run_equivalence


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_cuts_multiple_updates_and_masked_accumulation(self):
        report = run_equivalence(steps=10)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(len(report["cases"]), 9)

    def test_boundary_is_an_independent_leaf_and_chain_rule_resumes(self):
        source = torch.randn(2, 3, 4, requires_grad=True)
        hidden = source * 2
        key = BoundaryKey(0, 0, 0)
        boundary = LocalBoundary(key, hidden)
        leaf = boundary.leaf("cpu")
        self.assertTrue(leaf.is_leaf)
        self.assertIsNone(leaf.grad_fn)
        self.assertNotEqual(leaf.data_ptr(), hidden.data_ptr())
        leaf.square().sum().backward()
        self.assertIsNone(source.grad)
        boundary.backward(key, leaf.grad)
        torch.testing.assert_close(source.grad, 8 * source)
        with self.assertRaises(RuntimeError):
            boundary.backward(key, leaf.grad)

    def test_bad_boundary_identity_shape_dtype_missing_and_nan_are_terminal(self):
        key = BoundaryKey(0, 0, 0)
        for received_key, gradient in (
            (BoundaryKey(1, 0, 0), torch.ones(1, 2, 3)),
            (BoundaryKey(0, 1, 0), torch.ones(1, 2, 3)),
            (BoundaryKey(0, 0, 1), torch.ones(1, 2, 3)),
            (key, None), (key, torch.ones(1, 3, 3)),
            (key, torch.ones(1, 2, 3, dtype=torch.float64)),
            (key, torch.full((1, 2, 3), float("nan"))),
        ):
            with self.subTest(key=received_key, gradient=gradient):
                source = torch.ones(1, 2, 3, requires_grad=True)
                boundary = LocalBoundary(key, source * 2)
                boundary.leaf("cpu")
                with self.assertRaises(ValueError):
                    boundary.backward(received_key, gradient)
                self.assertIsNone(source.grad)
                with self.assertRaises(RuntimeError):
                    boundary.backward(key, torch.ones_like(source))

    def test_ownership_and_no_early_optimizer_update(self):
        with self.assertRaises(ValueError):
            ReferenceTrainer([build_stage(end=2), build_stage(start=1)], split=2)
        trainer = ReferenceTrainer([build_stage(end=2), build_stage(start=2)], split=2)
        before = trainer.snapshot()["parameters"]
        batches = synthetic_batches(ModelConfig())
        parameter = next(trainer.stages[1].parameters())
        handle = parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
        with self.assertRaises(ValueError):
            trainer.step(batches)
        handle.remove()
        for name, p in trainer.named_parameters().items():
            self.assertTrue(torch.equal(before[name], p), name)
        self.assertTrue(all(not optimizer.state for optimizer in trainer.optimizers))
        with self.assertRaises(RuntimeError):
            trainer.step(batches)

    def test_interruption_between_updates_poisoned_before_retry(self):
        trainer = ReferenceTrainer([build_stage(end=2), build_stage(start=2)], split=2)
        batches = synthetic_batches(ModelConfig())
        with patch.object(trainer.optimizers[1], "step", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                trainer.step(batches)
        self.assertTrue(trainer.failed)
        self.assertEqual(trainer.version, 0)
        self.assertTrue(trainer.optimizers[0].state)
        self.assertFalse(trainer.optimizers[1].state)
        before_retry = trainer.snapshot()
        with self.assertRaises(RuntimeError):
            trainer.step(batches)
        for kind, tensors in before_retry.items():
            for name, tensor in tensors.items():
                self.assertTrue(torch.equal(tensor, trainer.snapshot()[kind][name]))
