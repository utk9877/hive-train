import copy
import unittest
from unittest.mock import patch
import torch

from hive_train.config import ModelConfig
from hive_train.initialization import cpu_state, parameter_seed
from hive_train.model import Block, DecoderStage, build_stage
from hive_train.data import synthetic_batches
from hive_train.reference import ReferenceTrainer


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

    def test_cancelled_key_bias_matches_conventional_attention_and_derivatives(self):
        config = ModelConfig(layers=2, width=16, heads=4, mlp_width=32)
        stable = build_stage(config).blocks["0"].attention.double()
        with torch.no_grad():
            stable.key.bias.copy_(torch.linspace(-0.25, 0.25, config.width))
        conventional = copy.deepcopy(stable)
        conventional.key = torch.nn.Linear(config.width, config.width, dtype=torch.float64)
        conventional.key.load_state_dict(stable.key.state_dict())
        generator = torch.Generator().manual_seed(2718)
        x = torch.randn(2, 7, config.width, generator=generator, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        actual, expected = stable(x), conventional(y)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, atol=1e-12, rtol=1e-12)
        for (name, parameter), (other_name, reference) in zip(stable.named_parameters(), conventional.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(parameter.grad, reference.grad, atol=1e-12, rtol=1e-12, msg=name)
        self.assertEqual(float(stable.key.bias.grad.norm()), 0.)
        self.assertLess(float(conventional.key.bias.grad.norm()), 1e-14)
        self.assertGreater(float(stable.key.weight.grad.norm()), 1e-6)

    def test_structural_zero_keeps_adam_state_and_decoupled_weight_decay(self):
        config = ModelConfig(layers=2, width=16, heads=4, mlp_width=32)
        model = build_stage(config)
        with torch.no_grad():
            for block in model.blocks.values():
                block.attention.key.bias.fill_(0.25)
        trainer = ReferenceTrainer([model], split=1)
        before = {name: p.detach().clone() for name, p in model.named_parameters() if name.endswith("key.bias")}
        for step in range(3):
            trainer.step(synthetic_batches(config, seed=7 + step))
            for name, p in model.named_parameters():
                self.assertTrue(p.requires_grad, name)
                self.assertIsNotNone(p.grad, name)
                if name in before:
                    self.assertEqual(float(p.grad.norm()), 0., name)
                    state = trainer.optimizers[0].state[p]
                    self.assertEqual(float(state["exp_avg"].norm()), 0.)
                    self.assertEqual(float(state["exp_avg_sq"].norm()), 0.)
                    self.assertEqual(int(state["step"]), step + 1)
                    torch.testing.assert_close(p, before[name] * (1 - 1e-3 * 0.01) ** (step + 1))

    def test_fixture_values_unchanged_by_attention_evaluation_revision(self):
        from hive_train.initialization import fixture_manifest
        fixture = fixture_manifest(build_stage())
        with patch("hive_train.model.KeyProjection", torch.nn.Linear):
            conventional = fixture_manifest(build_stage())
        self.assertEqual(fixture["architecture"], "decoder_v2")
        self.assertEqual(fixture["fixture_hash"], conventional["fixture_hash"])
        self.assertEqual(fixture["parameter_spec_hash"], conventional["parameter_spec_hash"])

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
