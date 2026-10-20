"""Explicit CPU tensor bytes, no NumPy or pickle. Not a checkpoint loader."""

import array
import ctypes
import sys

from .protocol import ProtocolError, validate_tensor_descriptor


def _byteswap(payload, itemsize):
    if itemsize == 1:
        return payload
    values = array.array("I" if itemsize == 4 else "Q")
    values.frombytes(payload)
    values.byteswap()
    return bytearray(values.tobytes())


def encode_tensor(tensor, name):
    import torch
    types = {torch.float32: "float32", torch.int64: "int64", torch.uint8: "uint8"}
    if tensor.dtype not in types or tensor.layout != torch.strided:
        raise ProtocolError("SHAPE_MISMATCH", "unsupported tensor dtype/layout")
    descriptor = {"name": name, "dtype": types[tensor.dtype], "shape": list(tensor.shape),
                  "byte_order": "little", "nbytes": tensor.numel() * tensor.element_size()}
    validate_tensor_descriptor(descriptor, descriptor["nbytes"])
    if tensor.dtype == torch.float32 and not bool(torch.isfinite(tensor).all()):
        raise ProtocolError("NONFINITE", "nonfinite outgoing tensor")
    # Blocking copy and owning bytearray separate accelerator and buffer lifetime.
    cpu = tensor.detach().to("cpu").contiguous()
    payload = bytearray(cpu.numel() * cpu.element_size())
    if payload:
        ctypes.memmove((ctypes.c_char * len(payload)).from_buffer(payload), cpu.data_ptr(), len(payload))
    if sys.byteorder != "little":
        payload = _byteswap(payload, cpu.element_size())
    return descriptor, payload


def decode_tensor(descriptor, payload, *, vocab_size=None):
    """Returns a CPU tensor referencing the owned writable bytearray.

    Call only after frame authentication. Tensor keeps the backing buffer alive;
    caller must not mutate it until accelerator copying/consumption finishes.
    """
    import torch
    validate_tensor_descriptor(descriptor, len(payload))
    if not isinstance(payload, bytearray):
        raise ProtocolError("BAD_FRAME", "receive tensor buffer must be an owned bytearray")
    types = {"float32": torch.float32, "int64": torch.int64, "uint8": torch.uint8}
    dtype = types[descriptor["dtype"]]
    itemsize = {"float32": 4, "int64": 8, "uint8": 1}[descriptor["dtype"]]
    buffer = payload if sys.byteorder == "little" else _byteswap(payload, itemsize)
    tensor = torch.frombuffer(buffer, dtype=dtype).reshape(descriptor["shape"])
    if dtype == torch.float32 and not bool(torch.isfinite(tensor).all()):
        raise ProtocolError("NONFINITE", "nonfinite incoming tensor")
    if vocab_size is not None and descriptor["name"] in ("input_ids", "targets"):
        allowed = (tensor >= 0) & (tensor < vocab_size)
        if descriptor["name"] == "targets":
            allowed |= tensor == -100
        if not bool(allowed.all()):
            raise ProtocolError("SHAPE_MISMATCH", "token outside vocabulary")
    return tensor
