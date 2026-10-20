"""Serial two-worker training scheduler; no parameter or optimizer ownership."""

from copy import deepcopy
import time

from .config import json_hash
from .data import validate_batches
from .initialization import tensor_hash
from .protocol import ProtocolError
from .runtime import Channel, WORKERS, abort, manifest, model_config, parameter_name_hash
from .state import CoordinatorState, apply_id_for_step
from .tensor_codec import encode_tensor
from .transport import relay_one


class CoordinatorRuntime:
    def __init__(self, sessions, assignments, observer=None):
        if set(sessions) != set(WORKERS) or set(assignments) != set(WORKERS):
            raise ValueError("exactly two fixed workers required")
        run_ids = {s.run_id for s in sessions.values()}
        if len(run_ids) != 1:
            raise ValueError("sessions must belong to one run")
        self.sessions, self.assignments = sessions, deepcopy(assignments)
        hashes = {w: parameter_name_hash(manifest(model_config(a["config"]), a["block_start"], a["block_end"]))
                  for w, a in assignments.items()}
        self.gate = CoordinatorState(next(iter(run_ids)), parameter_hashes=hashes)
        self.channels = {w: Channel(s, self.gate) for w, s in sessions.items()}
        self.observer = observer
        self.capabilities = {}

    def close(self):
        abort(self.gate, self.sessions.values())

    def configure(self):
        try:
            for w in WORKERS:
                self.capabilities[w] = self.channels[w].receive("CAPABILITIES").metadata["body"]
            versions = {c["platform"]["torch"].split("+")[0] for c in self.capabilities.values()}
            if len(versions) != 1:
                raise ProtocolError("CONFIG_MISMATCH", "worker PyTorch base releases differ")
            for w in WORKERS:
                a = self.assignments[w]
                if a["checkpoint_id"] is not None or a["starting_step"] != 0:
                    raise ProtocolError("CHECKPOINT_ERROR", "M3 supports fresh initialization only")
                self.channels[w].send("RUN_CONFIG", body=a)
                self.channels[w].configure_transport(a)
                self.channels[w].receive("RUN_READY")
        except BaseException:
            self.close()
            raise

    def _barrier(self, command, reply, body=None):
        for w in WORKERS:
            self.channels[w].send(command, body=body)
            self.channels[w].receive(reply)

    def step(self, batches):
        try:
            if self.gate.phase != "IDLE":
                raise ProtocolError("WRONG_STATE", "step requires configured idle workers")
            config = self.gate.config
            total = validate_batches(model_config(config), batches)
            if len(batches) > 4:
                raise ValueError("M3 serial runtime currently limits K to four")
            for batch in batches:
                if batch.inputs.shape[0] > config["data"]["max_microbatch_size"] or batch.inputs.shape[1] != config["data"]["sequence_length"]:
                    raise ValueError("batch does not match admitted run configuration")
            plan = [{"inputs": tensor_hash(b.inputs), "targets": tensor_hash(b.targets)} for b in batches]
            step_id, version = self.gate.step_id, self.gate.parameter_version
            started = time.monotonic()
            self._barrier("STEP_BEGIN", "STEP_STARTED", {"microbatch_count": len(batches),
                          "total_valid_tokens": total, "batch_plan_hash": json_hash(plan)})
            loss_sum = 0.0
            for index, batch in enumerate(batches):
                descriptor, payload = encode_tensor(batch.targets, "targets")
                valid = int((batch.targets != -100).sum())
                self.channels["worker-1"].send("TARGET", microbatch=index,
                    body={"valid_tokens": valid}, tensor=descriptor, payload=payload)
                del payload
                self.channels["worker-1"].receive("TARGET_READY")
                descriptor, payload = encode_tensor(batch.inputs, "input_ids")
                self.channels["worker-0"].send("INPUT", microbatch=index, tensor=descriptor, payload=payload)
                del payload
                relay_one(self.sessions["worker-0"], self.sessions["worker-1"], "ACTIVATION",
                          config=config, state=self.gate, deadline=self.channels["worker-0"].deadline())
                # Relay helper returns the small body only, never a retained tensor.
                body = relay_one(self.sessions["worker-1"], self.sessions["worker-0"], "BOUNDARY_GRAD",
                          config=config, state=self.gate, deadline=self.channels["worker-1"].deadline())
                loss_sum += body["loss_sum"]
                self.channels["worker-0"].receive("MICRO_DONE")
            self._barrier("STEP_PREPARE", "STEP_READY")
            update = {"apply_id": apply_id_for_step(self.gate.run_id, step_id), "next_version": version + 1}
            self._barrier("STEP_APPLY", "STEP_APPLIED", update)
            for w in WORKERS:
                self.channels[w].send("STEP_COMMIT", body=update)
                self.channels[w].receive("STEP_COMMITTED")
            result = {"step_id": step_id, "parameter_version": self.gate.parameter_version,
                      "loss": loss_sum / total, "valid_tokens": total,
                      "batch_plan_hash": json_hash(plan), "seconds": time.monotonic() - started}
            if self.observer is not None:
                self.observer(result)
            return result
        except BaseException:
            self.close()
            raise

    def stop(self):
        try:
            self._barrier("STOP", "STOPPED")
            return {"status": "pass", "run_id": self.gate.run_id,
                    "steps": self.gate.step_id, "parameter_version": self.gate.parameter_version}
        except BaseException:
            self.close()
            raise
        finally:
            for session in self.sessions.values():
                session.close()
