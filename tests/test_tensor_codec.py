import struct
import unittest
import torch

from hive_train.protocol import ProtocolError
from hive_train.tensor_codec import decode_tensor, encode_tensor


class TensorCodecTests(unittest.TestCase):
    def test_dtypes_noncontiguity_and_signed_zero_roundtrip(self):
        for value in (torch.tensor([[0., -0., 1., -2.]], dtype=torch.float32),
                      torch.tensor([[-100, 0, 255, 2**62]], dtype=torch.int64),
                      torch.tensor([[0, 255, 3, 127]], dtype=torch.uint8)):
            for tensor in (value, value.reshape(2, 2).t()):
                descriptor, payload = encode_tensor(tensor, "fixture")
                restored = decode_tensor(descriptor, payload)
                self.assertTrue(torch.equal(tensor, restored))
                self.assertEqual(encode_tensor(restored, "fixture")[1], payload)
        descriptor, payload = encode_tensor(torch.tensor([0., -0.]), "fixture")
        self.assertEqual(payload, struct.pack("<2f", 0., -0.))

    def test_nonfinite_and_bad_tokens_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ProtocolError):
                encode_tensor(torch.tensor([value]), "hidden")
        descriptor, payload = encode_tensor(torch.tensor([0.]), "hidden")
        payload[:] = struct.pack("<f", float("nan"))
        with self.assertRaises(ProtocolError):
            decode_tensor(descriptor, payload)
        descriptor, payload = encode_tensor(torch.tensor([[-100, 256]]), "targets")
        with self.assertRaises(ProtocolError):
            decode_tensor(descriptor, payload, vocab_size=256)
        descriptor, payload = encode_tensor(torch.tensor([[-100, 255]]), "targets")
        self.assertTrue(torch.equal(decode_tensor(descriptor, payload, vocab_size=256), torch.tensor([[-100, 255]])))

    def test_received_tensor_owns_backing_storage_after_frame_release(self):
        descriptor, payload = encode_tensor(torch.arange(4, dtype=torch.float32), "hidden")
        tensor = decode_tensor(descriptor, payload)
        del payload
        self.assertEqual(tensor.tolist(), [0., 1., 2., 3.])
        with self.assertRaises(ProtocolError):
            decode_tensor(descriptor, bytes(16))
