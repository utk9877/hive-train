"""Adversarial schema tests, runnable without PyTorch or network access."""

import copy
import hashlib
import json
import unittest

from hive_train.protocol import (
    MAX_METADATA_BYTES, MAX_TENSOR_BYTES, PREFIX, ProtocolError,
    decode_metadata, encode_metadata, validate_metadata, validate_prefix,
    validate_tensor_contract, validate_tensor_descriptor,
)


RUN = "7d57ecc2-b181-4d4d-a4c3-1b366a1e7d4e"
HASH = "a" * 64


def message(kind="PING", sender="coordinator", receiver="worker-0", body=None,
            step=None, micro=None, version=0, tensor=None):
    return dict(type=kind, run_id=RUN, sender=sender, receiver=receiver, seq=2,
                step_id=step, microbatch_id=micro, parameter_version=version,
                body={"nonce": 0} if body is None else body, tensor=tensor)


def configuration():
    config = {
        "model": dict(layers=4, width=128, heads=4, mlp_width=512, vocab_size=256,
                      max_sequence_length=128, architecture="decoder_v1", dtype="float32"),
        "optimizer": dict(name="adamw", lr=.001, beta1=.9, beta2=.999, eps=1e-8,
                          weight_decay=.01, foreach=False, fused=False, amsgrad=False),
        "data": dict(dataset_hash=HASH, tokenizer="synthetic_v1", max_microbatch_size=2,
                     sequence_length=32, ignore_index=-100),
        "init": dict(algorithm="named_cpu_v1", master_seed=2026),
        "parameter_spec_hash": HASH,
    }
    body = dict(config=config, config_hash="", stage_id=0, block_start=0, block_end=2,
                checkpoint_id=None, starting_step=0, starting_version=0,
                max_tensor_bytes=MAX_TENSOR_BYTES,
                timeouts=dict(connect_ms=10000, handshake_ms=10000, idle_ms=30000,
                              status_interval_ms=5000, compute_ms=300000, checkpoint_ms=600000,
                              transfer_no_progress_ms=30000, min_directional_bytes_per_second=1048576,
                              transfer_multiplier=5.0))
    refresh_hash(body)
    return message("RUN_CONFIG", body=body)


def refresh_hash(body):
    raw = json.dumps(body["config"], sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    body["config_hash"] = hashlib.sha256(raw.encode()).hexdigest()


def descriptor(name="hidden", dtype="float32", shape=None):
    shape = [2, 32, 128] if shape is None else shape
    size = {"float32": 4, "int64": 8, "uint8": 1}[dtype]
    for extent in shape:
        size *= extent
    return dict(name=name, dtype=dtype, shape=shape, byte_order="little", nbytes=size)


class ProtocolTests(unittest.TestCase):
    def reject(self, metadata, payload_length=0):
        with self.assertRaises(ProtocolError):
            validate_metadata(metadata, payload_length)

    def test_prefix_checks_before_allocation(self):
        self.assertEqual(PREFIX.size, 24)
        valid = [b"HTRN", 1, 0, 200, 0, 1024]
        self.assertEqual(validate_prefix(PREFIX.pack(*valid)), (200, 1024))
        for index, value in [(0, b"WRNG"), (1, 2), (2, 1), (3, 0),
                             (3, MAX_METADATA_BYTES + 1), (4, 1), (5, MAX_TENSOR_BYTES + 1),
                             (5, (1 << 64) - 1)]:
            with self.subTest(index=index, value=value):
                bad = valid.copy()
                bad[index] = value
                with self.assertRaises(ProtocolError):
                    validate_prefix(PREFIX.pack(*bad))
        for bad in (b"", b"x" * 23, b"x" * 25):
            with self.assertRaises(ProtocolError):
                validate_prefix(bad)
        with self.assertRaises(ProtocolError):
            validate_prefix(PREFIX.pack(*valid), max_tensor_bytes=512)
        with self.assertRaises(ProtocolError):
            validate_prefix(PREFIX.pack(b"HTRN", 1, 0, 16385, 0, 0), handshake=True)
        with self.assertRaises(ProtocolError):
            validate_prefix(PREFIX.pack(*valid), handshake=True)

    def test_configuration_canonical_round_trip(self):
        metadata = configuration()
        raw = encode_metadata(metadata, 0)
        self.assertEqual(decode_metadata(raw, 0), metadata)
        self.assertNotIn(b" ", raw)

    def test_strict_common_keys_integer_bool_and_uuid(self):
        for field, value in [("seq", True), ("seq", -1), ("seq", 1 << 63), ("seq", 2.0),
                             ("run_id", RUN.upper()), ("run_id", RUN.replace("-", "")),
                             ("run_id", "../../checkpoint"), ("run_id", []),
                             ("step_id", 0), ("microbatch_id", 0), ("parameter_version", None)]:
            with self.subTest(field=field, value=value):
                bad = message()
                bad[field] = value
                self.reject(bad)
        bad = message()
        bad["extension"] = 0
        self.reject(bad)
        del bad["extension"]
        del bad["tensor"]
        self.reject(bad)
        bad = message()
        bad["body"]["extension"] = 0
        self.reject(bad)

    def test_json_adversaries(self):
        valid = encode_metadata(message(), 0)
        bad_inputs = [b"\xff", b"[]", b"null", b"{} trailing", b"\xef\xbb\xbf{}",
                      b"[" * 5000 + b"]" * 5000, b" " * (MAX_METADATA_BYTES + 1),
                      valid.replace(b'"seq":2', b'"seq":2,"seq":2'),
                      valid.replace(b'"nonce":0', b'"nonce":0,"nonce":1'),
                      valid.replace(b'"nonce":0', b'"nonce":NaN'),
                      valid.replace(b'"nonce":0', b'"nonce":Infinity'),
                      valid.replace(b'"nonce":0', b'"nonce":1e999'),
                      valid.replace(b'"nonce":0', b'"nonce":' + b"9" * 10000),
                      valid.replace(b'"type":"PING"', b'"type":"\\ud800"')]
        for raw in bad_inputs:
            with self.subTest(raw=raw[:60]):
                with self.assertRaises(ProtocolError):
                    decode_metadata(raw, 0)

    def test_unhashable_json_values_are_protocol_errors(self):
        for field in ("type", "sender", "receiver", "run_id"):
            for value in ([], {}, ["PING"]):
                bad = message()
                bad[field] = value
                self.reject(bad)
        for field in ("phase", "code"):
            bad = (message("STATUS", sender="worker-0", receiver="coordinator",
                           body=dict(phase=[], last_completed_microbatch=None, last_progress_age_ms=0))
                   if field == "phase" else message("ERROR", body=dict(code=[], detail="failed")))
            self.reject(bad)
        for field in ("dtype", "shape", "byte_order", "name", "nbytes"):
            bad = descriptor()
            bad[field] = {}
            with self.assertRaises(ProtocolError):
                validate_tensor_descriptor(bad, 32768)

    def test_tensor_checked_shape_products_and_generic_dtypes(self):
        for dtype in ("float32", "int64", "uint8"):
            good = descriptor(name="example", dtype=dtype, shape=[2, 3])
            validate_tensor_descriptor(good, good["nbytes"])
        for field, value in [("shape", []), ("shape", [1] * 5), ("shape", [0]),
                             ("shape", [-1]), ("shape", [True]), ("shape", [1.0]),
                             ("shape", [1 << 31]), ("shape", [(1 << 31) - 1] * 4),
                             ("dtype", "float16"), ("byte_order", "big"), ("nbytes", True),
                             ("nbytes", 4), ("strides", [4])]:
            with self.subTest(field=field, value=value):
                bad = descriptor()
                bad[field] = value
                with self.assertRaises(ProtocolError):
                    validate_tensor_descriptor(bad, 32768)
        with self.assertRaises(ProtocolError):
            validate_tensor_descriptor(descriptor(), 32767)

    def test_tensor_contract_and_token_bounds(self):
        config = configuration()["body"]["config"]
        activation = message("ACTIVATION", sender="worker-0", receiver="coordinator", body={},
                             step=0, micro=0, tensor=descriptor())
        validate_metadata(activation, 32768)
        validate_tensor_contract(activation, config)
        for shape in ([3, 32, 128], [2, 31, 128], [2, 32, 64]):
            bad = copy.deepcopy(activation)
            bad["tensor"] = descriptor(shape=shape)
            validate_metadata(bad, bad["tensor"]["nbytes"])
            with self.assertRaises(ProtocolError):
                validate_tensor_contract(bad, config)
        for field, value in [("name", "arbitrary"), ("dtype", "uint8")]:
            bad = copy.deepcopy(activation)
            bad["tensor"][field] = value
            self.reject(bad, 32768)
        target = message("TARGET", receiver="worker-1", body={"valid_tokens": 65}, step=0, micro=0,
                         tensor=descriptor(name="targets", dtype="int64", shape=[2, 32]))
        self.reject(target, 512)
        target["body"]["valid_tokens"] = 0
        validate_metadata(target, 512)  # Empty microbatch is legal inside nonempty step.

    def test_direction_scope_and_versions(self):
        for kind, sender, receiver in [("PING", "worker-0", "coordinator"),
                                       ("INPUT", "coordinator", "worker-1"),
                                       ("ACTIVATION", "worker-1", "coordinator"),
                                       ("PONG", "worker-0", "worker-1")]:
            self.reject(message(kind, sender, receiver))
        apply = message("STEP_APPLY", step=0, body=dict(apply_id=RUN, next_version=1))
        validate_metadata(apply, 0)
        for field, value in [("step_id", None), ("microbatch_id", 0), ("parameter_version", None),
                             ("parameter_version", 1)]:
            bad = copy.deepcopy(apply)
            bad[field] = value
            self.reject(bad)
        applied = message("STEP_APPLIED", sender="worker-0", receiver="coordinator", step=0,
                          version=1, body=dict(apply_id=RUN, next_version=1))
        validate_metadata(applied, 0)
        applied["body"]["next_version"] = 2
        self.reject(applied)

    def test_handshake_scope(self):
        cases = [("CHALLENGE", "coordinator", "pending", {"server_nonce": HASH}, 0),
                 ("HELLO", "worker-0", "coordinator", {"worker_nonce": HASH, "proof": HASH}, 0),
                 ("WELCOME", "coordinator", "worker-1", {"proof": HASH, "max_tensor_bytes": MAX_TENSOR_BYTES}, 1)]
        for kind, sender, receiver, body, seq in cases:
            good = message(kind, sender, receiver, body, version=None)
            good["seq"] = seq
            validate_metadata(good, 0)
            for field, value in [("step_id", 0), ("parameter_version", 0), ("seq", seq + 1)]:
                bad = copy.deepcopy(good)
                bad[field] = value
                self.reject(bad)

    def test_run_configuration_rejects_inconsistent_and_unknown_fields(self):
        for field, value in [("config_hash", "0" * 64), ("stage_id", 1), ("block_start", 1),
                             ("block_end", 4), ("starting_step", 1), ("starting_version", 1),
                             ("checkpoint_id", "../root"), ("max_tensor_bytes", 100), ("extra", 1)]:
            bad = configuration()
            bad["body"][field] = value
            self.reject(bad)
        for group, field, value in [("model", "width", 127), ("model", "layers", 1),
                                    ("optimizer", "fused", 0), ("optimizer", "beta1", 1),
                                    ("optimizer", "eps", 0), ("optimizer", "lr", True),
                                    ("data", "ignore_index", 0), ("data", "sequence_length", 129),
                                    ("init", "master_seed", -1), ("init", "extra", 0)]:
            bad = configuration()
            bad["body"]["config"][group][field] = value
            refresh_hash(bad["body"])
            self.reject(bad)
        restored = configuration()
        restored["body"].update(checkpoint_id=RUN, starting_step=7, starting_version=7)
        restored["parameter_version"] = 7
        validate_metadata(restored, 0)

    def test_capabilities_nested_schema(self):
        capabilities = message("CAPABILITIES", sender="worker-1", receiver="coordinator", version=None,
                               body=dict(platform=dict(os="Darwin", architecture="arm64", python="3.13",
                                                       torch="2.11", backend="mps", device_name="Apple M4",
                                                       cuda_driver=None, cuda_runtime=None, macos_version="15"),
                                         diagnostics_passed=True, fallback_enabled=False,
                                         memory=dict(total_bytes=16 << 30, available_bytes=None,
                                                     allocated_bytes=0, reserved_bytes=None)))
        validate_metadata(capabilities, 0)
        for group, field, value in [("platform", "backend", "unknown"), ("platform", "os", None),
                                    ("platform", "os", "é" * 129), ("platform", "other", "x"),
                                    ("memory", "allocated_bytes", True), ("memory", "allocated_bytes", -1)]:
            bad = copy.deepcopy(capabilities)
            bad["body"][group][field] = value
            self.reject(bad)

    def test_error_sanitization_and_control_payload(self):
        error = message("ERROR", body=dict(code="OOM", detail="Device memory exhausted"), version=None)
        validate_metadata(error, 0)
        for detail in ("line\nforged log", "x\x1b[31m", "x" * 1025, "\ud800", "\u202eSpoof"):
            bad = copy.deepcopy(error)
            bad["body"]["detail"] = detail
            self.reject(bad)
        self.reject(message(), 1)
        bad = message()
        bad["tensor"] = descriptor(dtype="uint8", shape=[1])
        self.reject(bad, 1)

    def test_remaining_valid_message_schemas_and_wrong_body_keys(self):
        cases = [
            message("RUN_READY", "worker-0", "coordinator", dict(config_hash=HASH, stage_id=0,
                    loaded_version=0, parameter_name_hash=HASH)),
            message("STEP_BEGIN", body=dict(microbatch_count=2, total_valid_tokens=80, batch_plan_hash=HASH), step=0),
            message("STEP_STARTED", "worker-0", "coordinator", {}, step=0),
            message("TARGET_READY", "worker-1", "coordinator", {}, step=0, micro=0),
            message("INPUT", body={}, step=0, micro=0,
                    tensor=descriptor("input_ids", "int64", [2, 32])),
            message("ACTIVATION", receiver="worker-1", body={}, step=0, micro=0, tensor=descriptor()),
            message("BOUNDARY_GRAD", "worker-1", "coordinator", dict(loss_sum=55.0, valid_tokens=64),
                    step=0, micro=0, tensor=descriptor("hidden_grad")),
            message("BOUNDARY_GRAD", body=dict(loss_sum=55.0, valid_tokens=64), step=0, micro=0,
                    tensor=descriptor("hidden_grad")),
            message("MICRO_DONE", "worker-0", "coordinator", {}, step=0, micro=0),
            message("STEP_PREPARE", body={}, step=0),
            message("STEP_READY", "worker-0", "coordinator", dict(gradient_finite=True, microbatches_completed=2), step=0),
            message("STEP_COMMIT", body=dict(apply_id=RUN, next_version=1), step=0, version=1),
            message("STEP_COMMITTED", "worker-0", "coordinator", dict(apply_id=RUN), step=0, version=1),
            message("CHECKPOINT_BEGIN", body=dict(checkpoint_id=RUN), step=1, version=1),
            message("CHECKPOINT_READY", "worker-0", "coordinator",
                    dict(checkpoint_id=RUN, shard_index_hash=HASH, shard_bytes=32), step=1, version=1),
            message("CHECKPOINT_COMMIT", body=dict(checkpoint_id=RUN, manifest_hash=HASH), step=1, version=1),
            message("CHECKPOINT_ACK", "worker-0", "coordinator", dict(checkpoint_id=RUN), step=1, version=1),
            message("STATUS", "worker-0", "coordinator",
                    dict(phase="ACCUMULATING", last_completed_microbatch=0, last_progress_age_ms=9)),
            message("PONG", "worker-0", "coordinator"),
            message("ABORT", body=dict(code="PEER_LOST", detail="Peer disconnected")),
            message("STOP", body={}),
            message("STOPPED", "worker-0", "coordinator", {}),
        ]
        for good in cases:
            with self.subTest(kind=good["type"], sender=good["sender"]):
                length = 0 if good["tensor"] is None else good["tensor"]["nbytes"]
                self.assertEqual(decode_metadata(encode_metadata(good, length), length), good)
                bad = copy.deepcopy(good)
                bad["body"]["unexpected"] = 0
                self.reject(bad, length)

    def test_recursive_type_mutations_never_escape_protocol_error(self):
        # Every parser-visible node is independently replaced, including nested
        # object/list values that can trigger accidental Python TypeError.
        def paths(value, prefix=()):
            yield prefix
            if isinstance(value, dict):
                for key, child in value.items():
                    yield from paths(child, prefix + (key,))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    yield from paths(child, prefix + (index,))

        activation = message("ACTIVATION", "worker-0", "coordinator", {}, step=0, micro=0, tensor=descriptor())
        for good, length in ((configuration(), 0), (activation, 32768)):
            for path in paths(good):
                for value in (None, [], {}, True, -1, 1 << 63, "unexpected"):
                    bad = copy.deepcopy(good)
                    if path:
                        node = bad
                        for key in path[:-1]:
                            node = node[key]
                        node[path[-1]] = value
                    else:
                        bad = value
                    try:
                        validate_metadata(bad, length)
                    except ProtocolError:
                        pass
                    except Exception as exc:
                        self.fail(f"Non-protocol exception at {path}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    unittest.main()
