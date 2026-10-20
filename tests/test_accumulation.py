import unittest
import torch
from torch.nn import functional as F

from hive_train.config import ModelConfig
from hive_train.data import Batch, synthetic_batches
from hive_train.model import build_stage
from hive_train.reference import ReferenceTrainer


class AccumulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_token_weighted_gradients_match_one_effective_batch(self):
        batches = synthetic_batches(ModelConfig(), count=2)
        batches[1].targets[:, 9:] = -100
        full = build_stage()
        trainer = ReferenceTrainer([build_stage(end=2), build_stage(start=2)], split=2)
        x = torch.cat([b.inputs for b in batches])
        y = torch.cat([b.targets for b in batches])
        loss = F.cross_entropy(full(x).reshape(-1, 256), y.reshape(-1), ignore_index=-100)
        loss.backward()
        result = trainer.step(batches)
        self.assertAlmostEqual(result.loss, float(loss.detach()), places=5)
        self.assertEqual(result.valid_tokens, 82)
        for name, p in full.named_parameters():
            torch.testing.assert_close(trainer.named_parameters()[name].grad, p.grad,
                                       atol=1e-6, rtol=1e-4, msg=name)

    def test_empty_step_rejected_before_mutation(self):
        trainer = ReferenceTrainer([build_stage()], split=2)
        batches = synthetic_batches(ModelConfig())
        batches[0].targets.fill_(-100)
        before = trainer.snapshot()["parameters"]
        with self.assertRaisesRegex(ValueError, "all-empty"):
            trainer.step(batches)
        self.assertEqual(trainer.version, 0)
        self.assertFalse(trainer.optimizers[0].state)
        for name, p in trainer.named_parameters().items():
            self.assertIsNone(p.grad)
            self.assertTrue(torch.equal(p, before[name]))

    def test_invalid_tokens_rejected_before_forward(self):
        for which, value in (("inputs", -1), ("inputs", 256), ("targets", -1), ("targets", 256)):
            trainer = ReferenceTrainer([build_stage()], split=2)
            batches = synthetic_batches(ModelConfig())
            getattr(batches[0], which)[0, 0] = value
            with self.assertRaises(ValueError):
                trainer.step(batches)
            self.assertFalse(trainer.optimizers[0].state)

    def test_next_token_shift_occurs_once(self):
        b = synthetic_batches(ModelConfig())[0]
        self.assertTrue(torch.equal(b.inputs[:, 1:], b.targets[:, :-1]))
