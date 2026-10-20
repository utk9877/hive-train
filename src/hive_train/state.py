"""Fail-stop protocol-1 state gates; no sockets, tensors, or training execution.

Call ``accept`` for every received *authenticated* frame and before every send.
An outgoing completion reply is a local completion event: the caller must finish
its device work before emitting it. Coordinator relay sends are separate events,
so an incoming activation alone never authorizes the next boundary operation.
These objects are compute-thread owned, not thread safe. A failed send must call
``disconnect``; no command can be retried using the same gate.
"""

from copy import deepcopy
import hashlib
import json
import time
from uuid import UUID, uuid5

from .protocol import ProtocolError, validate_metadata


WORKERS = frozenset(("worker-0", "worker-1"))
MICRO = frozenset(("TARGET", "TARGET_READY", "INPUT", "ACTIVATION",
                   "BOUNDARY_GRAD", "MICRO_DONE"))
STEP = MICRO | frozenset(("STEP_BEGIN", "STEP_STARTED", "STEP_PREPARE",
                         "STEP_READY", "STEP_APPLY", "STEP_APPLIED",
                         "STEP_COMMIT", "STEP_COMMITTED"))
CHECKPOINT = frozenset(("CHECKPOINT_BEGIN", "CHECKPOINT_READY",
                        "CHECKPOINT_COMMIT", "CHECKPOINT_ACK"))
TERMINAL = frozenset(("ABORTED", "STOPPED"))


def apply_id_for_step(run_id, step_id):
    """Deterministic per-run/per-step apply UUID; no unbounded history needed."""
    if type(step_id) is not int or not 0 <= step_id < 2**63:
        raise ValueError("step_id must be a uint63")
    namespace = UUID(run_id)
    if str(namespace) != run_id:
        raise ValueError("run_id must be a canonical UUID")
    return str(uuid5(namespace, f"step:{step_id}"))


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


class _Gate:
    def __init__(self, run_id, *, clock=time.monotonic):
        self.run_id = run_id
        self.clock = clock
        self.phase = "CONFIGURING"
        self.step_id = 0
        self.microbatch = 0
        self.parameter_version = None
        self.progress_at = clock()
        self.timeouts = {"compute_ms": 300000, "checkpoint_ms": 600000,
                         "idle_ms": 30000}
        self.checkpoint_at = None
        self.failure = None

    def _fail(self, code, detail):
        self.phase = "ABORTED"
        self.failure = (code, detail)
        raise ProtocolError(code, detail)

    def _require(self, condition, detail, code="WRONG_STATE"):
        if not condition:
            self._fail(code, detail)

    def disconnect(self):
        """Poison live state even when the last update's outcome is unknown."""
        self._fail("PEER_LOST", "restore a complete checkpoint under a fresh run ID")

    def check_timeout(self):
        if self.phase in TERMINAL:
            self._fail("WRONG_STATE", "terminal gate cannot resume")
        if self.phase == "CHECKPOINTING":
            since, budget = self.checkpoint_at, self.timeouts["checkpoint_ms"]
        else:
            since = self.progress_at
            budget = self.timeouts["idle_ms" if self.phase in ("IDLE", "CONFIGURING")
                                   else "compute_ms"]
        if self.clock() - since >= budget / 1000:
            self._fail("TIMEOUT", "absolute phase deadline exceeded; no step retry")

    def accept(self, metadata):
        """Validate and consume one decoded event; errors permanently abort."""
        try:
            self.check_timeout()
            payload_length = 0 if metadata.get("tensor") is None else metadata["tensor"]["nbytes"]
            validate_metadata(metadata, payload_length)
            self._require(metadata["run_id"] == self.run_id, "stale run", "VERSION_MISMATCH")
            self._accept(metadata)
        except BaseException:
            self.phase = "ABORTED"
            raise

    def _scope(self, m, version=None):
        typ = m["type"]
        expected_step = self.step_id if typ in STEP | CHECKPOINT else None
        self._require(m["step_id"] == expected_step, "wrong step", "VERSION_MISMATCH")
        if typ in MICRO:
            self._require(m["microbatch_id"] == self.microbatch, "wrong microbatch", "DUPLICATE")
        else:
            self._require(m["microbatch_id"] is None, "unexpected microbatch")
        expected_version = self.parameter_version if version is None else version
        local_identity = getattr(self, "worker_id", "coordinator")
        if (typ in ("PING", "PONG") and m["receiver"] == local_identity and
                self.phase in ("APPLYING", "COMMITTING")):
            # The two ordered streams can cross an update acknowledgment. These
            # messages prove liveness only; they cannot authorize work or extend
            # a compute deadline. Training messages still require exact versions.
            self._require(m["parameter_version"] in (self.step_input_version, self.step_input_version + 1),
                          "heartbeat outside active update versions", "VERSION_MISMATCH")
            return
        self._require(m["parameter_version"] == expected_version,
                      "wrong parameter version", "VERSION_MISMATCH")

    def _config(self, m, worker):
        b = m["body"]
        stage = int(worker[-1])
        self._require(b["stage_id"] == stage, "worker/stage mismatch", "CONFIG_MISMATCH")
        self._require(_hash(b["config"]) == b["config_hash"], "config hash mismatch", "CONFIG_MISMATCH")
        layers = b["config"]["model"]["layers"]
        start, end = b["block_start"], b["block_end"]
        valid = 0 == start < end < layers if stage == 0 else 0 < start < end == layers
        self._require(valid, "invalid contiguous ownership", "CONFIG_MISMATCH")
        self._require(m["parameter_version"] == b["starting_version"] and
                      m["step_id"] is None and m["microbatch_id"] is None,
                      "invalid configuration counters", "CONFIG_MISMATCH")
        self._require(b["starting_step"] == b["starting_version"],
                      "step/version counter mismatch", "CONFIG_MISMATCH")
        if b["checkpoint_id"] is None:
            self._require(b["starting_step"] == 0, "nonzero start requires checkpoint", "CONFIG_MISMATCH")
        return deepcopy(b)

    def _tensor(self, m, *, batch_size=None):
        descriptor = m["tensor"]
        shape = descriptor["shape"]
        data, model = self.config["data"], self.config["model"]
        self._require(1 <= shape[0] <= data["max_microbatch_size"] and
                      shape[1] == data["sequence_length"], "configured tensor shape", "SHAPE_MISMATCH")
        if batch_size is not None:
            self._require(shape[0] == batch_size, "microbatch batch size mismatch", "SHAPE_MISMATCH")
        if m["type"] in ("ACTIVATION", "BOUNDARY_GRAD"):
            self._require(shape[2] == model["width"], "hidden width mismatch", "SHAPE_MISMATCH")
        self._require(descriptor["nbytes"] <= self.max_tensor_bytes,
                      "negotiated tensor limit exceeded", "SHAPE_MISMATCH")
        if "valid_tokens" in m["body"]:
            self._require(m["body"]["valid_tokens"] <= shape[0] * shape[1],
                          "valid tokens exceed tensor size", "CONFIG_MISMATCH")
        return shape[0]

    def _begin(self, b):
        self.step_input_version = self.parameter_version
        self.count = b["microbatch_count"]
        self.total = b["total_valid_tokens"]
        self._require(self.count > 0 and self.total > 0, "empty step", "CONFIG_MISMATCH")
        self._require(self.total <= self.count * self.config["data"]["max_microbatch_size"] *
                      self.config["data"]["sequence_length"], "impossible token total", "CONFIG_MISMATCH")
        self.microbatch = 0
        self.valid_seen = 0
        self.phase = "ACCUMULATING"
        self.progress_at = self.clock()

    def _completed(self):
        self.progress_at = self.clock()


class CoordinatorState(_Gate):
    """Observe both links, including coordinator sends and tensor relay sends.

    Session authentication/sequence checks precede this gate. ``parameter_hashes``
    optionally binds worker RUN_READY reports to locally generated M1 manifests;
    a real runtime must supply both expected hashes before accepting model state.
    """

    def __init__(self, run_id, *, clock=time.monotonic, parameter_hashes=None):
        super().__init__(run_id, clock=clock)
        self.capabilities = set()
        self.configs = {}
        self.ready = set()
        self.parameter_hashes = parameter_hashes
        self.sent = set()
        self.acked = set()
        self.apply_id = None

    def _accept(self, m):
        typ, b = m["type"], m["body"]
        outgoing = m["sender"] == "coordinator"
        worker = m["receiver"] if outgoing else m["sender"]
        self._require(worker in WORKERS and (outgoing or m["receiver"] == "coordinator"),
                      "invalid coordinator link direction")
        if typ == "RUN_CONFIG":
            self._require(outgoing and self.phase == "CONFIGURING" and self.capabilities == WORKERS,
                          "configuration requires both capabilities")
            self._require(worker not in self.configs, "duplicate configuration", "DUPLICATE")
            cfg = self._config(m, worker)
            if self.configs:
                other = next(iter(self.configs.values()))
                for key in ("config_hash", "starting_step", "starting_version", "checkpoint_id",
                            "max_tensor_bytes", "timeouts"):
                    self._require(cfg[key] == other[key], "worker configuration mismatch", "CONFIG_MISMATCH")
                self._require((cfg["block_start"] == other["block_end"] if cfg["stage_id"] == 1
                               else cfg["block_end"] == other["block_start"]),
                              "partition gap or overlap", "CONFIG_MISMATCH")
            self.configs[worker] = cfg
            self.config = cfg["config"]
            self.timeouts = cfg["timeouts"]
            self.max_tensor_bytes = cfg["max_tensor_bytes"]
            self.step_id, self.parameter_version = cfg["starting_step"], cfg["starting_version"]
            self._completed()
            return
        # A worker not configured yet still has a null pre-configuration version.
        if typ == "CAPABILITIES":
            self._require(not outgoing and self.phase == "CONFIGURING" and not self.configs,
                          "capabilities after configuration")
            self._require(worker not in self.capabilities, "duplicate capabilities", "DUPLICATE")
            self._scope(m)
            self._require(b["diagnostics_passed"] and not b["fallback_enabled"],
                          "device diagnostics not clean", "DEVICE_ERROR")
            self.capabilities.add(worker)
            self._completed()
            return
        expected_version = self.parameter_version
        if typ == "STEP_APPLIED" or (typ in ("STATUS", "PING", "PONG", "ERROR") and
                                      self.phase == "APPLYING" and worker in self.acked):
            self._require(expected_version is not None, "update before configuration")
            expected_version += 1
        self._scope(m, expected_version)
        if typ in ("ERROR", "ABORT"):
            self._fail(b["code"], b["detail"])
        if typ in ("PING", "PONG", "STATUS"):
            self._require(outgoing == (typ == "PING"), "invalid heartbeat direction")
            if self.phase == "IDLE":
                self._completed()
            return
        self._require(not getattr(self, "stopping", False) or typ in ("STOP", "STOPPED"),
                      "work scheduled after stop command")
        if typ == "RUN_READY":
            self._require(not outgoing and self.phase == "CONFIGURING" and worker in self.configs,
                          "unexpected readiness")
            self._require(worker not in self.ready, "duplicate readiness", "DUPLICATE")
            cfg = self.configs[worker]
            self._require(b["stage_id"] == cfg["stage_id"] and b["config_hash"] == cfg["config_hash"] and
                          b["loaded_version"] == self.parameter_version, "loaded state mismatch", "CONFIG_MISMATCH")
            if self.parameter_hashes is not None:
                self._require(b["parameter_name_hash"] == self.parameter_hashes[worker],
                              "parameter ownership mismatch", "CONFIG_MISMATCH")
            self.ready.add(worker)
            if self.ready == WORKERS:
                self.phase = "IDLE"
            self._completed()
            return
        if typ == "STEP_BEGIN":
            self._require(outgoing and (self.phase == "IDLE" or
                          (self.phase == "ACCUMULATING" and self.flow == "START")), "step before commit barrier")
            if self.phase == "IDLE":
                self._begin(b)
                self.flow = "START"
                self.begin_body = deepcopy(b)
                self.sent, self.acked = set(), set()
            self._require(b == self.begin_body and worker not in self.sent,
                          "mismatched or duplicate step start", "DUPLICATE")
            self.sent.add(worker)
            return
        if typ == "STEP_STARTED":
            self._require(not outgoing and self.phase == "ACCUMULATING" and self.flow == "START",
                          "unexpected step started")
            self._reply(worker)
            if self.acked == WORKERS:
                self.flow = "TARGET"
            return
        if typ in MICRO:
            self._micro(m, worker, outgoing)
            return
        if typ in ("STEP_PREPARE", "STEP_READY", "STEP_APPLY", "STEP_APPLIED", "STEP_COMMIT", "STEP_COMMITTED"):
            self._barrier(m, worker, outgoing)
            return
        if typ in CHECKPOINT:
            self._checkpoint(m, worker, outgoing)
            return
        if typ == "STOP":
            self._require(outgoing and self.phase == "IDLE", "stop outside idle boundary")
            if not getattr(self, "stopping", False):
                self.sent, self.acked = set(), set()
            self._require(worker not in self.sent, "duplicate stop", "DUPLICATE")
            self.sent.add(worker)
            self.stopping = True
            return
        if typ == "STOPPED":
            self._require(not outgoing and self.phase == "IDLE" and getattr(self, "stopping", False),
                          "unexpected stopped")
            self._reply(worker)
            if self.acked == WORKERS:
                self.phase = "STOPPED"
            return
        self._fail("WRONG_STATE", "unexpected message for coordinator")

    def _reply(self, worker):
        self._require(worker in self.sent and worker not in self.acked, "unsolicited or duplicate reply", "DUPLICATE")
        self.acked.add(worker)
        self._completed()

    def _micro(self, m, worker, outgoing):
        self._require(self.phase == "ACCUMULATING", "microbatch outside accumulation")
        trace = {
            "TARGET": ("TARGET", "worker-1", True, "TARGET_READY"),
            "TARGET_READY": ("TARGET_READY", "worker-1", False, "INPUT"),
            "INPUT": ("INPUT", "worker-0", True, "ACTIVATION_IN"),
            "ACTIVATION_IN": ("ACTIVATION", "worker-0", False, "ACTIVATION_OUT"),
            "ACTIVATION_OUT": ("ACTIVATION", "worker-1", True, "GRAD_IN"),
            "GRAD_IN": ("BOUNDARY_GRAD", "worker-1", False, "GRAD_OUT"),
            "GRAD_OUT": ("BOUNDARY_GRAD", "worker-0", True, "MICRO_DONE"),
            "MICRO_DONE": ("MICRO_DONE", "worker-0", False, "TARGET"),
        }
        expected = trace.get(self.flow)
        self._require(expected is not None and (m["type"], worker, outgoing) == expected[:3],
                      "out-of-order microbatch event")
        typ = m["type"]
        if typ == "TARGET":
            self.batch_size = self._tensor(m)
            self.target_valid = m["body"]["valid_tokens"]
        elif m["tensor"] is not None:
            self._tensor(m, batch_size=self.batch_size)
        if typ == "BOUNDARY_GRAD":
            self._require(m["body"]["valid_tokens"] == self.target_valid,
                          "gradient/target token count mismatch", "CONFIG_MISMATCH")
        if self.flow in ("ACTIVATION_IN", "GRAD_IN"):
            self.relay = deepcopy((m["tensor"], m["body"]))
        if self.flow in ("ACTIVATION_OUT", "GRAD_OUT"):
            self._require((m["tensor"], m["body"]) == self.relay, "relay metadata changed", "CONFIG_MISMATCH")
            self.relay = None
        self.flow = expected[3]
        if typ == "MICRO_DONE":
            self.microbatch += 1
            self.valid_seen += self.target_valid
            self._require(self.valid_seen <= self.total, "token total exceeded", "CONFIG_MISMATCH")
            if self.microbatch == self.count:
                self._require(self.valid_seen == self.total, "final token total mismatch", "CONFIG_MISMATCH")
                self.flow = "DONE"
        if not outgoing:
            self._completed()

    def _barrier(self, m, worker, outgoing):
        typ, b = m["type"], m["body"]
        if typ == "STEP_PREPARE":
            if self.phase == "ACCUMULATING":
                self._require(self.flow == "DONE", "prepare before all microbatches")
                self.phase, self.sent, self.acked = "PREPARING", set(), set()
            self._require(outgoing and self.phase == "PREPARING", "unexpected prepare")
        elif typ == "STEP_READY":
            self._require(not outgoing and self.phase == "PREPARING", "unexpected step ready")
            self._require(b["gradient_finite"], "nonfinite gradients", "NONFINITE")
            self._require(b["microbatches_completed"] == self.count, "completed count mismatch", "CONFIG_MISMATCH")
            self._reply(worker)
            return
        elif typ == "STEP_APPLY":
            if self.phase == "PREPARING":
                self._require(self.acked == WORKERS, "apply before both ready")
                self._require(b["apply_id"] == apply_id_for_step(self.run_id, self.step_id),
                              "apply ID does not identify current run/step", "VERSION_MISMATCH")
                self.phase, self.sent, self.acked = "APPLYING", set(), set()
                self.apply_id = b["apply_id"]
            self._require(outgoing and self.phase == "APPLYING", "unexpected apply")
            self._require(b["apply_id"] == self.apply_id and b["next_version"] == self.parameter_version + 1,
                          "apply ID/version mismatch", "VERSION_MISMATCH")
        elif typ == "STEP_APPLIED":
            self._require(not outgoing and self.phase == "APPLYING", "unexpected applied")
            self._require(b["apply_id"] == self.apply_id and b["next_version"] == self.parameter_version + 1,
                          "applied ID/version mismatch", "VERSION_MISMATCH")
            self._reply(worker)
            if self.acked == WORKERS:
                self.parameter_version += 1
                self.phase, self.sent, self.acked = "COMMITTING", set(), set()
            return
        elif typ == "STEP_COMMIT":
            self._require(outgoing and self.phase == "COMMITTING", "commit before both applied")
            self._require(b["apply_id"] == self.apply_id and b["next_version"] == self.parameter_version,
                          "commit ID/version mismatch", "VERSION_MISMATCH")
        elif typ == "STEP_COMMITTED":
            self._require(not outgoing and self.phase == "COMMITTING", "unexpected committed")
            self._require(b["apply_id"] == self.apply_id, "commit ID mismatch", "VERSION_MISMATCH")
            self._reply(worker)
            if self.acked == WORKERS:
                self.phase = "IDLE"
                self.step_id += 1
            return
        self._require(worker not in self.sent, "duplicate barrier command", "DUPLICATE")
        self.sent.add(worker)

    def _checkpoint(self, m, worker, outgoing):
        typ, b = m["type"], m["body"]
        if typ == "CHECKPOINT_BEGIN" and self.phase == "IDLE":
            self.phase, self.sent, self.acked = "CHECKPOINTING", set(), set()
            self.checkpoint_id, self.checkpoint_flow = b["checkpoint_id"], "READY"
            self.checkpoint_at = self.clock()
        self._require(self.phase == "CHECKPOINTING" and b["checkpoint_id"] == self.checkpoint_id,
                      "checkpoint boundary/ID mismatch", "CHECKPOINT_ERROR")
        if typ == "CHECKPOINT_BEGIN":
            self._require(outgoing and self.checkpoint_flow == "READY", "unexpected checkpoint begin")
        elif typ == "CHECKPOINT_READY":
            self._require(not outgoing and self.checkpoint_flow == "READY", "unexpected checkpoint ready")
            self._reply(worker)
            return
        elif typ == "CHECKPOINT_COMMIT":
            if self.checkpoint_flow == "READY":
                self._require(self.acked == WORKERS, "checkpoint commit before both shards")
                self.checkpoint_flow, self.sent, self.acked = "ACK", set(), set()
                self.manifest_hash = b["manifest_hash"]
            self._require(outgoing and b["manifest_hash"] == self.manifest_hash, "manifest mismatch", "CHECKPOINT_ERROR")
        elif typ == "CHECKPOINT_ACK":
            self._require(not outgoing and self.checkpoint_flow == "ACK", "unexpected checkpoint ack")
            self._reply(worker)
            if self.acked == WORKERS:
                self.phase = "IDLE"
            return
        self._require(worker not in self.sent, "duplicate checkpoint command", "DUPLICATE")
        self.sent.add(worker)


class WorkerState(_Gate):
    """One stage's control gate; optional ``apply_update`` is a fake-test hook.

    The callback executes after APPLYING is recorded; failure including process
    interruption poisons state. Its return means the caller synchronized and
    checked the update. Without a callback, ``STEP_APPLIED`` is the explicit
    completion event after externally executed work. Never emit it speculatively.
    """

    def __init__(self, run_id, worker_id, *, clock=time.monotonic,
                 apply_update=None, expected_parameter_name_hash=None):
        if worker_id not in WORKERS:
            raise ValueError("worker_id must be worker-0 or worker-1")
        super().__init__(run_id, clock=clock)
        self.worker_id = worker_id
        self.apply_update = apply_update
        self.expected_parameter_name_hash = expected_parameter_name_hash
        self.configured = False
        self.capabilities_sent = False
        self.apply_id = None

    def _accept(self, m):
        typ, b = m["type"], m["body"]
        outgoing = m["sender"] == self.worker_id
        self._require((outgoing and m["receiver"] == "coordinator") or
                      (m["sender"] == "coordinator" and m["receiver"] == self.worker_id),
                      "wrong worker link direction")
        if typ == "RUN_CONFIG":
            self._require(not outgoing and self.phase == "CONFIGURING" and
                          self.capabilities_sent and not self.configured, "unexpected configuration")
            cfg = self._config(m, self.worker_id)
            self.config = cfg["config"]
            self.assignment = cfg
            self.timeouts = cfg["timeouts"]
            self.max_tensor_bytes = cfg["max_tensor_bytes"]
            self.parameter_version, self.step_id = cfg["starting_version"], cfg["starting_step"]
            self.configured = True
            self._completed()
            return
        next_version = self.parameter_version
        if typ == "STEP_APPLIED":
            self._require(next_version is not None, "update before configuration")
            next_version += 1
        self._scope(m, next_version)
        if typ in ("ERROR", "ABORT"):
            self._fail(b["code"], b["detail"])
        if typ in ("PING", "PONG", "STATUS"):
            self._require(outgoing == (typ != "PING"), "invalid heartbeat direction")
            if self.phase == "IDLE":
                self._completed()
            return
        self._require(not getattr(self, "stop_received", False) or typ == "STOPPED",
                      "work received after stop command")
        if typ == "CAPABILITIES":
            self._require(outgoing and not self.capabilities_sent and not self.configured,
                          "duplicate capabilities", "DUPLICATE")
            self._require(b["diagnostics_passed"] and not b["fallback_enabled"],
                          "device diagnostics not clean", "DEVICE_ERROR")
            self.capabilities_sent = True
        elif typ == "RUN_READY":
            self._require(outgoing and self.configured and self.phase == "CONFIGURING", "unexpected run ready")
            self._require(b["config_hash"] == self.assignment["config_hash"] and
                          b["stage_id"] == self.assignment["stage_id"] and
                          b["loaded_version"] == self.parameter_version,
                          "loaded configuration mismatch", "CONFIG_MISMATCH")
            if self.expected_parameter_name_hash is not None:
                self._require(b["parameter_name_hash"] == self.expected_parameter_name_hash,
                              "parameter ownership mismatch", "CONFIG_MISMATCH")
            self.phase = "IDLE"
        elif typ == "STEP_BEGIN":
            self._require(not outgoing and self.phase == "IDLE", "step before commit")
            self._begin(b)
            self.flow = "START"
        elif typ == "STEP_STARTED":
            self._require(outgoing and self.phase == "ACCUMULATING" and self.flow == "START", "unexpected started")
            self.flow = "INPUT" if self.worker_id == "worker-0" else "TARGET"
        elif typ in MICRO:
            self._micro(m, outgoing)
        elif typ == "STEP_PREPARE":
            self._require(not outgoing and self.phase == "ACCUMULATING" and self.microbatch == self.count and
                          self.valid_seen == self.total and self.flow == "DONE", "incomplete accumulation")
            self.phase = "PREPARING"
        elif typ == "STEP_READY":
            self._require(outgoing and self.phase == "PREPARING", "unexpected ready")
            self._require(b["gradient_finite"], "nonfinite gradients", "NONFINITE")
            self._require(b["microbatches_completed"] == self.count, "wrong completed count", "CONFIG_MISMATCH")
            self.phase = "READY"
        elif typ == "STEP_APPLY":
            self._require(not outgoing and self.phase == "READY", "duplicate or premature apply", "DUPLICATE")
            self._require(b["next_version"] == self.parameter_version + 1,
                          "apply version reused", "VERSION_MISMATCH")
            self._require(b["apply_id"] == apply_id_for_step(self.run_id, self.step_id),
                          "apply ID does not identify current run/step", "VERSION_MISMATCH")
            self.phase, self.apply_id = "APPLYING", b["apply_id"]
            self._completed()
            if self.apply_update is not None:
                self.apply_update()
                self.check_timeout()
        elif typ == "STEP_APPLIED":
            self._require(outgoing and self.phase == "APPLYING", "unexpected applied")
            self._require(b["apply_id"] == self.apply_id and b["next_version"] == self.parameter_version + 1,
                          "applied ID/version mismatch", "VERSION_MISMATCH")
            self.parameter_version += 1
            self.phase = "COMMITTING"
            self.commit_received = False
        elif typ == "STEP_COMMIT":
            self._require(not outgoing and self.phase == "COMMITTING" and not self.commit_received,
                          "unexpected or duplicate commit")
            self._require(b["apply_id"] == self.apply_id and b["next_version"] == self.parameter_version,
                          "commit ID/version mismatch", "VERSION_MISMATCH")
            self.commit_received = True
        elif typ == "STEP_COMMITTED":
            self._require(outgoing and self.phase == "COMMITTING" and self.commit_received,
                          "commit acknowledgment before commit")
            self._require(b["apply_id"] == self.apply_id, "commit ID mismatch", "VERSION_MISMATCH")
            self.phase = "IDLE"
            self.step_id += 1
        elif typ in CHECKPOINT:
            self._checkpoint(m, outgoing)
        elif typ == "STOP":
            self._require(not outgoing and self.phase == "IDLE", "stop outside idle")
            self.stop_received = True
        elif typ == "STOPPED":
            self._require(outgoing and self.phase == "IDLE" and getattr(self, "stop_received", False),
                          "unsolicited stopped")
            self.phase = "STOPPED"
        else:
            self._fail("WRONG_STATE", "unexpected message for worker")
        if outgoing or typ in ("STEP_BEGIN", "STEP_APPLY"):
            self._completed()

    def _micro(self, m, outgoing):
        self._require(self.phase == "ACCUMULATING", "microbatch outside accumulation")
        trace = ({"INPUT": ("INPUT", False, "ACTIVATION"),
                  "ACTIVATION": ("ACTIVATION", True, "BOUNDARY_GRAD"),
                  "BOUNDARY_GRAD": ("BOUNDARY_GRAD", False, "MICRO_DONE"),
                  "MICRO_DONE": ("MICRO_DONE", True, "INPUT")}
                 if self.worker_id == "worker-0" else
                 {"TARGET": ("TARGET", False, "TARGET_READY"),
                  "TARGET_READY": ("TARGET_READY", True, "ACTIVATION"),
                  "ACTIVATION": ("ACTIVATION", False, "BOUNDARY_GRAD"),
                  "BOUNDARY_GRAD": ("BOUNDARY_GRAD", True, "TARGET")})
        expected = trace.get(self.flow)
        self._require(expected is not None and (m["type"], outgoing) == expected[:2], "out-of-order local microbatch")
        typ = m["type"]
        if typ in ("TARGET", "INPUT"):
            self.batch_size = self._tensor(m)
            if typ == "TARGET":
                self.target_valid = m["body"]["valid_tokens"]
        elif m["tensor"] is not None:
            self._tensor(m, batch_size=self.batch_size)
        if typ == "BOUNDARY_GRAD":
            if self.worker_id == "worker-1":
                self._require(m["body"]["valid_tokens"] == self.target_valid,
                              "target/gradient count mismatch", "CONFIG_MISMATCH")
            self.valid_seen += m["body"]["valid_tokens"]
            self._require(self.valid_seen <= self.total, "token total exceeded", "CONFIG_MISMATCH")
        self.flow = expected[2]
        if typ == "MICRO_DONE" or (typ == "BOUNDARY_GRAD" and self.worker_id == "worker-1"):
            self.microbatch += 1
            if self.microbatch == self.count:
                self._require(self.valid_seen == self.total, "token total mismatch", "CONFIG_MISMATCH")
                self.flow = "DONE"

    def _checkpoint(self, m, outgoing):
        typ, b = m["type"], m["body"]
        if typ == "CHECKPOINT_BEGIN":
            self._require(not outgoing and self.phase == "IDLE", "checkpoint outside idle")
            self.phase, self.checkpoint_id, self.checkpoint_flow = "CHECKPOINTING", b["checkpoint_id"], "READY"
            self.checkpoint_at = self.clock()
            return
        self._require(self.phase == "CHECKPOINTING" and b["checkpoint_id"] == self.checkpoint_id,
                      "checkpoint ID mismatch", "CHECKPOINT_ERROR")
        if typ == "CHECKPOINT_READY":
            self._require(outgoing and self.checkpoint_flow == "READY", "unexpected shard readiness")
            self.checkpoint_flow = "COMMIT"
        elif typ == "CHECKPOINT_COMMIT":
            self._require(not outgoing and self.checkpoint_flow == "COMMIT", "unexpected checkpoint commit")
            self.checkpoint_flow = "ACK"
        elif typ == "CHECKPOINT_ACK":
            self._require(outgoing and self.checkpoint_flow == "ACK", "unexpected checkpoint ack")
            self.phase = "IDLE"
