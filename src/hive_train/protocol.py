"""Bounded, strict protocol-1 schemas. No tensor framework or network side effects.

This module checks structural validity; authentication, sequence/run fencing and
state transitions belong to transport/state respectively. A valid schema alone
never grants permission to apply an optimizer update.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct
import unicodedata
import uuid

from .config import MODEL_ARCHITECTURE


PREFIX = struct.Struct("!4sHHIIQ")
MAX_METADATA_BYTES = 65536
MAX_TENSOR_BYTES = 67108864
TAG_BYTES = 32
UINT63_MAX = (1 << 63) - 1
HANDSHAKES = frozenset({"CHALLENGE", "HELLO", "WELCOME"})
ERROR_CODES = frozenset({
    "BAD_FRAME", "AUTH_FAILED", "VERSION_MISMATCH", "WRONG_STATE",
    "DUPLICATE", "SHAPE_MISMATCH", "NONFINITE", "DEVICE_ERROR", "OOM",
    "TIMEOUT", "PEER_LOST", "CHECKPOINT_ERROR", "CONFIG_MISMATCH",
})
PHASES = frozenset({
    "ADMITTING", "CONFIGURING", "IDLE", "ACCUMULATING", "PREPARING",
    "READY", "APPLYING", "COMMITTING", "CHECKPOINTING", "ABORTED", "STOPPED",
})
_COMMON = frozenset({"type", "run_id", "sender", "receiver", "seq",
                     "step_id", "microbatch_id", "parameter_version", "body", "tensor"})
_MICRO = frozenset({"TARGET", "TARGET_READY", "INPUT", "ACTIVATION", "BOUNDARY_GRAD", "MICRO_DONE"})
_STEP = frozenset({"STEP_BEGIN", "STEP_STARTED", "STEP_PREPARE", "STEP_READY",
                   "STEP_APPLY", "STEP_APPLIED", "STEP_COMMIT", "STEP_COMMITTED"})
_CHECKPOINT = frozenset({"CHECKPOINT_BEGIN", "CHECKPOINT_READY", "CHECKPOINT_COMMIT", "CHECKPOINT_ACK"})
_TENSORS = {"TARGET": ("targets", "int64", 2), "INPUT": ("input_ids", "int64", 2),
            "ACTIVATION": ("hidden", "float32", 3), "BOUNDARY_GRAD": ("hidden_grad", "float32", 3)}
_BODIES = {
    "CHALLENGE": "server_nonce", "HELLO": "worker_nonce proof",
    "WELCOME": "proof max_tensor_bytes",
    "CAPABILITIES": "platform diagnostics_passed fallback_enabled memory",
    "RUN_CONFIG": "config_hash config stage_id block_start block_end checkpoint_id starting_step starting_version max_tensor_bytes timeouts",
    "RUN_READY": "config_hash stage_id loaded_version parameter_name_hash",
    "STEP_BEGIN": "microbatch_count total_valid_tokens batch_plan_hash",
    "STEP_STARTED": "", "TARGET": "valid_tokens", "TARGET_READY": "", "INPUT": "",
    "ACTIVATION": "", "BOUNDARY_GRAD": "loss_sum valid_tokens", "MICRO_DONE": "",
    "STEP_PREPARE": "", "STEP_READY": "gradient_finite microbatches_completed",
    "STEP_APPLY": "apply_id next_version", "STEP_APPLIED": "apply_id next_version",
    "STEP_COMMIT": "apply_id next_version", "STEP_COMMITTED": "apply_id",
    "CHECKPOINT_BEGIN": "checkpoint_id", "CHECKPOINT_READY": "checkpoint_id shard_index_hash shard_bytes",
    "CHECKPOINT_COMMIT": "checkpoint_id manifest_hash", "CHECKPOINT_ACK": "checkpoint_id",
    "STATUS": "phase last_completed_microbatch last_progress_age_ms", "PING": "nonce", "PONG": "nonce",
    "ERROR": "code detail", "ABORT": "code detail", "STOP": "", "STOPPED": "",
}
_C_TO_W = frozenset({"WELCOME", "RUN_CONFIG", "STEP_BEGIN", "STEP_PREPARE", "STEP_APPLY",
                      "STEP_COMMIT", "CHECKPOINT_BEGIN", "CHECKPOINT_COMMIT", "PING", "ABORT", "STOP"})
_W_TO_C = frozenset({"HELLO", "CAPABILITIES", "RUN_READY", "STEP_STARTED", "STEP_READY",
                      "STEP_APPLIED", "STEP_COMMITTED", "CHECKPOINT_READY", "CHECKPOINT_ACK",
                      "STATUS", "PONG", "STOPPED"})
_WORKERS = frozenset({"worker-0", "worker-1"})


class ProtocolError(Exception):
    """Terminal error; detail contains diagnostics, never peer-supplied payloads."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass
class Frame:
    metadata: dict
    payload: bytearray


def _require(condition, detail, code="BAD_FRAME"):
    if not condition:
        raise ProtocolError(code, detail)


def _keys(value, names):
    expected = set(names.split()) if isinstance(names, str) else set(names)
    _require(type(value) is dict and set(value) == expected, "object fields do not match schema")


def _uint(value, positive=False):
    _require(type(value) is int and (1 if positive else 0) <= value <= UINT63_MAX,
             "expected bounded nonnegative integer")


def _nullable_uint(value):
    if value is not None:
        _uint(value)


def _string(value, limit=256):
    _require(type(value) is str, "expected string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ProtocolError("BAD_FRAME", "invalid Unicode string") from exc
    _require(len(encoded) <= limit, "string exceeds byte limit")


def _hash(value):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             "expected lowercase 32-byte hex")


def _uuid(value):
    try:
        _require(type(value) is str and str(uuid.UUID(value)) == value, "expected canonical UUID")
    except (ValueError, AttributeError) as exc:
        raise ProtocolError("BAD_FRAME", "expected canonical UUID") from exc


def _number(value):
    _require(type(value) in (int, float), "expected finite number")
    if type(value) is int:
        _uint(value)
    _require(math.isfinite(value), "expected finite number", "NONFINITE")


def _limit(value):
    _uint(value, positive=True)
    _require(value <= MAX_TENSOR_BYTES, "tensor limit exceeds protocol maximum")


def validate_prefix(prefix: bytes, max_tensor_bytes=MAX_TENSOR_BYTES, handshake=False):
    """Validate fixed fields before allocating metadata or payload."""
    _limit(max_tensor_bytes)
    _require(len(prefix) == PREFIX.size, "incorrect prefix length")
    magic, version, flags, metadata_length, reserved, payload_length = PREFIX.unpack(prefix)
    _require(magic == b"HTRN", "incorrect frame magic")
    _require(version == 1, "unsupported protocol version", "VERSION_MISMATCH")
    _require(flags == reserved == 0, "unsupported flags or reserved fields")
    _require(0 < metadata_length <= (16384 if handshake else MAX_METADATA_BYTES), "invalid metadata length")
    _require(payload_length <= max_tensor_bytes, "payload exceeds tensor limit")
    if handshake:
        _require(payload_length == 0, "handshake cannot contain a payload")
    return metadata_length, payload_length


def _tree(value, depth=0, path=()):
    # Checking before json.dumps also bounds caller-supplied/cyclic objects.
    if type(value) in (dict, list):
        _require(depth < 8, "JSON nesting exceeds eight containers")
        if type(value) is dict:
            for key, child in value.items():
                _string(key)
                _tree(child, depth + 1, path + (key,))
        else:
            for child in value:
                _tree(child, depth + 1, path + (None,))
    elif type(value) is int:
        if path == ("body", "config", "data", "ignore_index"):
            _require(value == -100, "ignore_index must be -100")
        else:
            _uint(value)
    elif type(value) is float:
        _number(value)
    elif type(value) is str:
        _string(value, 4096 if path == ("body", "detail") else 256)
    else:
        _require(value is None or type(value) is bool, "unsupported JSON value")


def validate_tensor_descriptor(descriptor, payload_length, max_tensor_bytes=MAX_TENSOR_BYTES):
    _limit(max_tensor_bytes)
    _uint(payload_length)
    _keys(descriptor, "name dtype shape byte_order nbytes")
    _string(descriptor["name"])
    sizes = {"float32": 4, "int64": 8, "uint8": 1}
    _require(type(descriptor["dtype"]) is str and descriptor["dtype"] in sizes,
             "unsupported tensor dtype", "SHAPE_MISMATCH")
    _require(descriptor["byte_order"] == "little", "unsupported tensor byte order", "SHAPE_MISMATCH")
    shape = descriptor["shape"]
    _require(type(shape) is list and 1 <= len(shape) <= 4, "invalid tensor rank", "SHAPE_MISMATCH")
    size = sizes[descriptor["dtype"]]
    for dimension in shape:
        _uint(dimension, positive=True)
        _require(dimension <= (1 << 31) - 1, "tensor dimension too large", "SHAPE_MISMATCH")
        _require(dimension <= max_tensor_bytes // size, "tensor product exceeds limit", "SHAPE_MISMATCH")
        size *= dimension
    _uint(descriptor["nbytes"], positive=True)
    _require(size == descriptor["nbytes"] == payload_length <= max_tensor_bytes,
             "tensor shape and byte lengths disagree", "SHAPE_MISMATCH")


def _config(config):
    _keys(config, "model optimizer data init parameter_spec_hash")
    _hash(config["parameter_spec_hash"])
    model, optimizer, data, init = (config[k] for k in ("model", "optimizer", "data", "init"))
    _keys(model, "layers width heads mlp_width vocab_size max_sequence_length architecture dtype")
    for name in ("layers", "width", "heads", "mlp_width", "vocab_size", "max_sequence_length"):
        _uint(model[name], positive=True)
    _require(model["architecture"] == MODEL_ARCHITECTURE and model["dtype"] == "float32", "unsupported model")
    _require(model["layers"] >= 2 and model["width"] % model["heads"] == 0, "invalid layer/head configuration")
    _keys(optimizer, "name lr beta1 beta2 eps weight_decay foreach fused amsgrad")
    _require(optimizer["name"] == "adamw", "unsupported optimizer")
    for name in ("lr", "beta1", "beta2", "eps", "weight_decay"):
        _number(optimizer[name])
    _require(optimizer["lr"] > 0 and optimizer["eps"] > 0 and optimizer["weight_decay"] >= 0,
             "invalid optimizer scalars")
    _require(0 <= optimizer["beta1"] < 1 and 0 <= optimizer["beta2"] < 1, "invalid optimizer beta")
    _require(all(optimizer[k] is False for k in ("foreach", "fused", "amsgrad")), "unsupported optimizer mode")
    _keys(data, "dataset_hash tokenizer max_microbatch_size sequence_length ignore_index")
    _hash(data["dataset_hash"])
    _require(data["tokenizer"] in ("byte_v1", "synthetic_v1"), "unsupported tokenizer")
    _uint(data["max_microbatch_size"], positive=True)
    _uint(data["sequence_length"], positive=True)
    _require(type(data["ignore_index"]) is int and data["ignore_index"] == -100, "invalid ignore_index")
    _require(data["sequence_length"] <= model["max_sequence_length"], "sequence exceeds model limit")
    _keys(init, "algorithm master_seed")
    _require(init["algorithm"] == "named_cpu_v1", "unsupported initialization")
    _uint(init["master_seed"])


def _run_config(body, metadata, max_tensor_bytes):
    config = body["config"]
    _config(config)
    _hash(body["config_hash"])
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                       allow_nan=False).encode("utf-8")).hexdigest()
    _require(body["config_hash"] == digest, "configuration hash mismatch", "CONFIG_MISMATCH")
    for name in ("stage_id", "block_start", "block_end", "starting_step", "starting_version"):
        _uint(body[name])
    stage, start, end, layers = body["stage_id"], body["block_start"], body["block_end"], config["model"]["layers"]
    _require(stage in (0, 1) and metadata["receiver"] == f"worker-{stage}", "stage identity mismatch", "CONFIG_MISMATCH")
    _require((stage == 0 and start == 0 and 0 < end < layers) or
             (stage == 1 and 0 < start < end == layers), "invalid contiguous stage partition", "CONFIG_MISMATCH")
    if body["checkpoint_id"] is not None:
        _uuid(body["checkpoint_id"])
    else:
        _require(body["starting_step"] == body["starting_version"] == 0, "fresh run must start at zero", "CONFIG_MISMATCH")
    _require(metadata["parameter_version"] == body["starting_version"], "starting version mismatch", "VERSION_MISMATCH")
    _limit(body["max_tensor_bytes"])
    _require(body["max_tensor_bytes"] <= max_tensor_bytes, "configured tensor limit exceeds negotiated limit")
    b, t, d = config["data"]["max_microbatch_size"], config["data"]["sequence_length"], config["model"]["width"]
    _require(b * t * max(8, 4 * d) <= body["max_tensor_bytes"], "configured tensor exceeds transport limit", "CONFIG_MISMATCH")
    timeouts = body["timeouts"]
    _keys(timeouts, "connect_ms handshake_ms idle_ms status_interval_ms compute_ms checkpoint_ms transfer_no_progress_ms min_directional_bytes_per_second transfer_multiplier")
    for name, value in timeouts.items():
        if name == "transfer_multiplier":
            _number(value)
            _require(value > 0, "transfer multiplier must be positive")
        else:
            _uint(value, positive=True)


def validate_metadata(metadata, payload_length, max_tensor_bytes=MAX_TENSOR_BYTES):
    """Check all keys, types, directions and intrinsic scope before payload read."""
    _limit(max_tensor_bytes)
    _uint(payload_length)
    _require(payload_length <= max_tensor_bytes, "payload exceeds tensor limit")
    _tree(metadata)
    _keys(metadata, _COMMON)
    kind = metadata["type"]
    _require(type(kind) is str and kind in _BODIES, "unknown message type")
    _uuid(metadata["run_id"])
    _uint(metadata["seq"])
    sender, receiver = metadata["sender"], metadata["receiver"]
    _require(type(sender) is str and type(receiver) is str, "invalid endpoint identity")
    cw = sender == "coordinator" and receiver in _WORKERS
    wc = sender in _WORKERS and receiver == "coordinator"
    if kind == "CHALLENGE":
        valid_direction = sender == "coordinator" and receiver == "pending"
    elif kind in _C_TO_W:
        valid_direction = cw
    elif kind in _W_TO_C:
        valid_direction = wc
    elif kind == "ERROR":
        valid_direction = cw or wc
    else:
        valid_direction = (sender, receiver) in {
            "TARGET": {("coordinator", "worker-1")}, "TARGET_READY": {("worker-1", "coordinator")},
            "INPUT": {("coordinator", "worker-0")}, "MICRO_DONE": {("worker-0", "coordinator")},
            "ACTIVATION": {("worker-0", "coordinator"), ("coordinator", "worker-1")},
            "BOUNDARY_GRAD": {("worker-1", "coordinator"), ("coordinator", "worker-0")},
        }.get(kind, set())
    _require(valid_direction, "message direction is not permitted")
    step, micro, version = (metadata[k] for k in ("step_id", "microbatch_id", "parameter_version"))
    for value in (step, micro, version):
        _nullable_uint(value)
    if kind in _MICRO:
        _require(step is not None and micro is not None, "microbatch message lacks scope")
    else:
        _require(micro is None, "unexpected microbatch scope")
        if kind in _STEP | _CHECKPOINT:
            _require(step is not None, "step message lacks scope")
        elif kind not in ("ERROR", "ABORT"):
            _require(step is None, "unexpected step scope")
    if kind in HANDSHAKES or kind == "CAPABILITIES":
        _require(version is None, "unexpected preconfiguration version")
    elif kind not in ("ERROR", "ABORT"):
        _require(version is not None, "operational message lacks version")
    if kind in HANDSHAKES:
        _require(metadata["seq"] == (1 if kind == "WELCOME" else 0), "invalid handshake sequence")
    body = metadata["body"]
    _keys(body, _BODIES[kind])
    if kind in _TENSORS:
        descriptor = metadata["tensor"]
        validate_tensor_descriptor(descriptor, payload_length, max_tensor_bytes)
        name, dtype, rank = _TENSORS[kind]
        _require((descriptor["name"], descriptor["dtype"], len(descriptor["shape"])) == (name, dtype, rank),
                 "message tensor contract mismatch", "SHAPE_MISMATCH")
    else:
        _require(metadata["tensor"] is None and payload_length == 0, "control message has tensor data")
    for name in ("server_nonce", "worker_nonce", "proof", "config_hash", "parameter_name_hash", "batch_plan_hash",
                 "shard_index_hash", "manifest_hash"):
        if name in body:
            _hash(body[name])
    for name in ("apply_id", "checkpoint_id"):
        if name in body and not (kind == "RUN_CONFIG" and body[name] is None):
            _uuid(body[name])
    for name in ("valid_tokens", "next_version", "stage_id", "loaded_version", "last_progress_age_ms", "nonce"):
        if name in body:
            _uint(body[name])
    for name in ("microbatch_count", "total_valid_tokens", "microbatches_completed", "shard_bytes"):
        if name in body:
            _uint(body[name], positive=True)
    if kind == "WELCOME":
        _limit(body["max_tensor_bytes"])
        _require(body["max_tensor_bytes"] <= max_tensor_bytes, "welcome exceeds configured tensor cap")
    elif kind == "CAPABILITIES":
        platform = body["platform"]
        _keys(platform, "os architecture python torch backend device_name cuda_driver cuda_runtime macos_version")
        for name, value in platform.items():
            if name not in ("cuda_driver", "cuda_runtime", "macos_version") or value is not None:
                _string(value)
        _require(platform["backend"] in ("cpu", "cuda", "mps"), "unsupported backend")
        _require(type(body["diagnostics_passed"]) is bool and type(body["fallback_enabled"]) is bool, "invalid diagnostics flags")
        _keys(body["memory"], "total_bytes available_bytes allocated_bytes reserved_bytes")
        for value in body["memory"].values():
            _nullable_uint(value)
    elif kind == "RUN_CONFIG":
        _run_config(body, metadata, max_tensor_bytes)
    elif kind == "RUN_READY":
        _require(body["stage_id"] in (0, 1) and sender == f"worker-{body['stage_id']}", "ready stage mismatch")
        _require(version == body["loaded_version"], "loaded version mismatch", "VERSION_MISMATCH")
    elif kind == "BOUNDARY_GRAD":
        _number(body["loss_sum"])
    elif kind == "STEP_READY":
        _require(body["gradient_finite"] is True, "nonfinite gradients", "NONFINITE")
    elif kind in ("STEP_APPLY", "STEP_APPLIED", "STEP_COMMIT"):
        expected = version + 1 if kind == "STEP_APPLY" else version
        _require(body["next_version"] == expected, "update version mismatch", "VERSION_MISMATCH")
    elif kind == "STATUS":
        _require(type(body["phase"]) is str and body["phase"] in PHASES, "unknown phase")
        _nullable_uint(body["last_completed_microbatch"])
    elif kind in ("ERROR", "ABORT"):
        _require(type(body["code"]) is str and body["code"] in ERROR_CODES, "unknown error code")
        detail = body["detail"]
        _require(type(detail) is str and len(detail) <= 1024 and
                 all(unicodedata.category(c)[0] != "C" for c in detail), "unsanitized error detail")
    if kind in ("TARGET", "BOUNDARY_GRAD"):
        shape = metadata["tensor"]["shape"]
        _require(body["valid_tokens"] <= shape[0] * shape[1], "valid tokens exceed tensor extent")


def validate_tensor_contract(metadata, config: dict):
    """Apply the run's b/T/d limit before allocation; paired b is state-owned."""
    if metadata["type"] not in _TENSORS:
        return
    shape = metadata["tensor"]["shape"]
    _require(1 <= shape[0] <= config["data"]["max_microbatch_size"] and
             shape[1] == config["data"]["sequence_length"], "tensor differs from configured b/T", "SHAPE_MISMATCH")
    if len(shape) == 3:
        _require(shape[2] == config["model"]["width"], "tensor differs from configured width", "SHAPE_MISMATCH")


def encode_metadata(metadata: dict, payload_length: int, max_tensor_bytes=MAX_TENSOR_BYTES) -> bytes:
    validate_metadata(metadata, payload_length, max_tensor_bytes)
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    _require(len(encoded) <= (16384 if metadata["type"] in HANDSHAKES else MAX_METADATA_BYTES), "metadata exceeds limit")
    return encoded


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _constant(_):
    raise ProtocolError("NONFINITE", "nonfinite JSON constant")


def _check_json_depth(raw):
    # Bound nesting before the recursive JSON parser. Brackets in strings and
    # escaped quote/backslash characters do not contribute to container depth.
    depth = 0
    quoted = escaped = False
    for character in raw:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            _require(depth <= 8, "JSON nesting exceeds eight containers")
        elif character in "]}":
            depth -= 1


def decode_metadata(raw: bytes, payload_length: int, max_tensor_bytes=MAX_TENSOR_BYTES) -> dict:
    _require(0 < len(raw) <= MAX_METADATA_BYTES, "invalid metadata length")
    try:
        source = raw.decode("utf-8", errors="strict")
        _check_json_depth(source)
        metadata = json.loads(source, object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ProtocolError("BAD_FRAME", "invalid UTF-8 or JSON") from exc
    validate_metadata(metadata, payload_length, max_tensor_bytes)
    if metadata["type"] in HANDSHAKES:
        _require(len(raw) <= 16384, "handshake metadata exceeds limit")
    return metadata
