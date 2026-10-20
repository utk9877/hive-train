import unittest
from unittest.mock import patch
import torch

from hive_train.config import ModelConfig
from hive_train.initialization import cpu_state, parameter_seed
from hive_train.model import Block, DecoderStage, build_stage


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parameter_count_names_and_untied_endpoints(self):
        model = build_stage()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 875264)
        self.assertIsNot(model.token_embedding.weight, model.output_head.weight)
        self.assertIn("blocks.3.attention.key.bias", dict(model.named_parameters()))

    def test_partition_initialization_is_identical_and_preserves_rng(self):
        before = torch.random.get_rng_state().clone()
        whole = cpu_state(build_stage())
        for cut in (1, 2, 3):
            a, b = build_stage(start=0, end=cut), build_stage(start=cut, end=4)
            combined = dict(a.named_parameters()) | dict(b.named_parameters())
            self.assertEqual(set(whole), set(combined))
            for name, p in combined.items():
                self.assertTrue(torch.equal(whole[name], p), name)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_stage_constructs_only_owned_blocks(self):
        with patch("hive_train.model.Block", wraps=Block) as constructor:
            stage = DecoderStage(ModelConfig(), 2, 4)
        self.assertEqual(constructor.call_count, 2)
        self.assertEqual(list(stage.blocks), ["2", "3"])
        self.assertFalse(hasattr(stage, "token_embedding"))
        self.assertTrue(all(p.device.type == "meta" for p in stage.parameters()))

    def test_causal_attention_and_output_shape(self):
        model = build_stage()
        x = torch.arange(16).reshape(2, 8)
        y = x.clone()
        y[:, 5:] = 99
        with torch.no_grad():
            first, second = model(x), model(y)
        self.assertEqual(first.shape, (2, 8, 256))
        torch.testing.assert_close(first[:, :5], second[:, :5], atol=0, rtol=0)

    def test_invalid_configuration_and_seed(self):
        for kwargs in ({"layers": 1}, {"heads": 3}, {"width": True}, {"mlp_width": 0}):
            with self.assertRaises(ValueError):
                ModelConfig(**kwargs)
        for start, end in ((-1, 2), (2, 2), (0, 5)):
            with self.assertRaises(ValueError):
                build_stage(start=start, end=end)
        with self.assertRaises(ValueError):
            parameter_seed("x", -1)

    def test_seed_encoding_golden_vectors(self):
        self.assertEqual(parameter_seed("token_embedding.weight"), 7790418495509202419)
        self.assertEqual(parameter_seed("blocks.3.attention.key.bias"), 906336757702146182)
