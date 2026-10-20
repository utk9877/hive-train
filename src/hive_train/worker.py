"""Serial real-autograd worker on one authenticated coordinator connection.

Only the assigned stage and its optimizer are materialized. Any failed operation
ends this instance; M3 supports fresh deterministic runs, not checkpoint recovery.
The optional observer is restricted to the tiny diagnostic fixture and executes
synchronously on the compute thread. It is never a tensor-gathering wire API.
"""

import torch
from torch.nn import functional as F

from .config import ModelConfig, OptimizerConfig, json_hash
from .model import build_stage
from .protocol import ProtocolError
from .reference import optimizer_for, synchronize
from .runtime import Channel, make_capabilities, manifest, model_config, parameter_name_hash
from .state import WorkerState
from .tensor_codec import decode_tensor, encode_tensor
from .validation import require_device


class WorkerRuntime:
    """One-use, compute-thread-owned stage runtime; ``run`` ends at STOP.

    ``observer(stage, optimizer, event, details)`` may inspect before_step,
    micro_done and after_step. Tensor details are independent detached CPU copies.
    The caller must bound any retained snapshots. Normal runs retain only scalar
    counters, one live microbatch graph, and one target or boundary buffer.
    """

    def __init__(self, session, device="cpu", diagnostic=False, observer=None):
        if session.local not in ("worker-0", "worker-1") or session.peer != "coordinator":
            raise ValueError("worker requires an authenticated coordinator session")
        if observer is not None and not diagnostic:
            raise ValueError("observation requires the bounded diagnostic mode")
        self.session = session
        self.device_name = device
        self.diagnostic = diagnostic
        self.observer = observer
        self.gate = WorkerState(session.run_id, session.local)
        self.channel = Channel(session, self.gate)
        self.stage = self.optimizer = None
        self._hidden = self._targets = None
        self._started = False
        self._step_loss_sum = 0.0
        self._last_loss = None
        self._steps_committed = self._microbatches_completed = 0

    def run(self):
        """Train until coordinator STOP, then close; errors are never retried."""
        if self._started:
            raise ProtocolError("WRONG_STATE", "worker runtime cannot be restarted")
        self._started = True
        try:
            self.device = require_device(self.device_name)
            capabilities = make_capabilities(self.device_name)
            self.channel.send("CAPABILITIES", body=capabilities)
            frame = self.channel.receive("RUN_CONFIG")
            self._configure(frame.metadata["body"])
            del frame
            while True:
                frame = self.channel.receive()
                done = self._dispatch(frame)
                del frame  # release an incoming payload before the next read
                if done:
                    return {"status": "pass", "phase": "STOPPED", "worker_id": self.session.local,
                            "device": str(self.device), "steps_committed": self._steps_committed,
                            "parameter_version": self.gate.parameter_version,
                            "microbatches_completed": self._microbatches_completed,
                            "last_loss": self._last_loss}
        except BaseException:
            try:
                self.gate.disconnect()
            except ProtocolError:
                pass
            raise
        finally:
            self.session.close()
            self._hidden = self._targets = None
            self.optimizer = self.stage = None

    def _configure(self, assignment):
        self.channel.configure_transport(assignment)
        if assignment["checkpoint_id"] is not None or assignment["starting_step"] != 0:
            raise ProtocolError("CHECKPOINT_ERROR", "checkpoint loading is not implemented in M3")
        cfg = self.gate.config
        self.model_config = model_config(cfg)
        if self.diagnostic and (self.model_config != ModelConfig() or
                                cfg["data"]["max_microbatch_size"] > 2 or
                                cfg["data"]["sequence_length"] > 32):
            raise ProtocolError("CONFIG_MISMATCH", "diagnostics require the bounded M1 fixture")
        global_spec = manifest(self.model_config, 0, self.model_config.layers)
        if json_hash(global_spec) != cfg["parameter_spec_hash"]:
            raise ProtocolError("CONFIG_MISMATCH", "global parameter specification mismatch")
        del global_spec
        start, end = assignment["block_start"], assignment["block_end"]
        names = manifest(self.model_config, start, end)
        self.gate.expected_parameter_name_hash = parameter_name_hash(names)
        self.stage = build_stage(self.model_config, start, end, device=self.device,
                                 seed=cfg["init"]["master_seed"])
        actual = [{"name": name, "shape": list(p.shape), "dtype": "float32"}
                  for name, p in self.stage.named_parameters()]
        if actual != names:
            raise ProtocolError("CONFIG_MISMATCH", "materialized stage ownership mismatch")
        del actual, names
        opt = cfg["optimizer"]
        self.optimizer = optimizer_for(self.stage, OptimizerConfig(**{
            k: opt[k] for k in ("lr", "beta1", "beta2", "eps", "weight_decay")}))
        self.stage.train()
        self._check_parameters()
        synchronize(self.device)
        self.gate.check_timeout()
        # WorkerState records APPLYING before this callback can touch optimizer.
        self.gate.apply_update = self._apply
        self.channel.send("RUN_READY", body={"config_hash": assignment["config_hash"],
                          "stage_id": assignment["stage_id"], "loaded_version": 0,
                          "parameter_name_hash": self.gate.expected_parameter_name_hash})

    def _dispatch(self, frame):
        m = frame.metadata
        typ, body = m["type"], m["body"]
        if typ == "PING":
            self.channel.send("PONG", body={"nonce": body["nonce"]})
        elif typ == "STEP_BEGIN":
            if self.diagnostic and body["microbatch_count"] > 4:
                raise ProtocolError("CONFIG_MISMATCH", "diagnostic microbatch cap is four")
            if self._hidden is not None or self._targets is not None:
                raise ProtocolError("WRONG_STATE", "graph or targets survived committed boundary")
            self.optimizer.zero_grad(set_to_none=True)
            self._step_loss_sum = 0.0
            synchronize(self.device)
            self.gate.check_timeout()
            self._notify("before_step")
            self.channel.send("STEP_STARTED")
        elif typ == "TARGET":
            target = self._device_tensor(frame)
            count = int(target.ne(-100).sum().item())
            if count != body["valid_tokens"]:
                raise ProtocolError("CONFIG_MISMATCH", "target payload count disagrees with metadata")
            self._targets = target
            self._target_count = count
            synchronize(self.device)
            self.gate.check_timeout()
            self.channel.send("TARGET_READY", microbatch=m["microbatch_id"])
        elif typ == "INPUT":
            self._prefix_forward(frame)
        elif typ == "ACTIVATION":
            self._suffix_backward(frame)
        elif typ == "BOUNDARY_GRAD":
            self._prefix_backward(frame)
        elif typ == "STEP_PREPARE":
            if self._hidden is not None or self._targets is not None:
                raise ProtocolError("WRONG_STATE", "live graph at prepare barrier")
            self._check_parameters(gradients=True)
            synchronize(self.device)
            self.gate.check_timeout()
            self.channel.send("STEP_READY", body={"gradient_finite": True,
                              "microbatches_completed": self.gate.microbatch})
        elif typ == "STEP_APPLY":
            # Channel.receive has already executed the gate's guarded callback.
            self.channel.send("STEP_APPLIED", body={"apply_id": body["apply_id"],
                              "next_version": body["next_version"]})
        elif typ == "STEP_COMMIT":
            self.channel.send("STEP_COMMITTED", body={"apply_id": body["apply_id"]})
            self._steps_committed += 1
            self._last_loss = self._step_loss_sum / self.gate.total
        elif typ == "STOP":
            self.channel.send("STOPPED")
            return True
        elif typ.startswith("CHECKPOINT_"):
            raise ProtocolError("CHECKPOINT_ERROR", "checkpoint writing is not implemented in M3")
        else:
            raise ProtocolError("WRONG_STATE", "unsupported worker operation")
        return False

    def _device_tensor(self, frame):
        cpu = decode_tensor(frame.metadata["tensor"], frame.payload,
                            vocab_size=self.model_config.vocab_size)
        # copy=True also isolates CPU diagnostics from the receive buffer.
        tensor = cpu.to(self.device, copy=True)
        if tensor.dtype == torch.float32 and not bool(torch.isfinite(tensor).all()):
            raise ProtocolError("NONFINITE", "received nonfinite device tensor")
        return tensor

    def _send_tensor(self, kind, tensor, name, *, microbatch, body=None):
        descriptor, payload = encode_tensor(tensor, name)
        self.gate.check_timeout()
        self.channel.send(kind, body=body, tensor=descriptor, payload=payload, microbatch=microbatch)

    def _prefix_forward(self, frame):
        if self._hidden is not None:
            raise ProtocolError("WRONG_STATE", "second prefix graph in flight")
        inputs = self._device_tensor(frame)
        self._hidden = self.stage(inputs)
        if not self._hidden.requires_grad:
            raise ProtocolError("DEVICE_ERROR", "prefix output lost autograd")
        synchronize(self.device)
        self._send_tensor("ACTIVATION", self._hidden.detach(), "hidden",
                          microbatch=frame.metadata["microbatch_id"])

    def _suffix_backward(self, frame):
        if self._targets is None:
            raise ProtocolError("WRONG_STATE", "activation without validated targets")
        # Detached transmitted storage becomes a DEVICE leaf, retaining .grad.
        leaf = self._device_tensor(frame).detach().requires_grad_(True)
        logits = self.stage(leaf)
        if not bool(torch.isfinite(logits).all()):
            raise ProtocolError("NONFINITE", "nonfinite logits")
        loss_sum = F.cross_entropy(logits.reshape(-1, self.model_config.vocab_size),
                                  self._targets.reshape(-1), reduction="sum", ignore_index=-100)
        if not bool(torch.isfinite(loss_sum)):
            raise ProtocolError("NONFINITE", "nonfinite loss")
        # Exactly one normalization by valid tokens in the whole optimizer step.
        (loss_sum / self.gate.total).backward()
        gradient = leaf.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise ProtocolError("NONFINITE", "missing or nonfinite boundary gradient")
        synchronize(self.device)
        value = float(loss_sum.detach().item())
        self._step_loss_sum += value
        microbatch = frame.metadata["microbatch_id"]
        self._notify("micro_done", microbatch=microbatch, logits=logits,
                     boundary_grad=gradient, loss_sum=value, valid_tokens=self._target_count)
        descriptor, payload = encode_tensor(gradient, "hidden_grad")
        del leaf, logits, loss_sum, gradient
        self._targets = None
        self.gate.check_timeout()
        self.channel.send("BOUNDARY_GRAD", body={"loss_sum": value, "valid_tokens": self._target_count},
                          tensor=descriptor, payload=payload, microbatch=microbatch)
        self._microbatches_completed += 1

    def _prefix_backward(self, frame):
        hidden, self._hidden = self._hidden, None  # consume exactly once, even on failure
        if hidden is None:
            raise ProtocolError("WRONG_STATE", "boundary gradient has no retained graph")
        gradient = self._device_tensor(frame)
        if gradient.shape != hidden.shape or gradient.dtype != hidden.dtype:
            raise ProtocolError("SHAPE_MISMATCH", "boundary gradient differs from retained activation")
        hidden.backward(gradient)  # already normalized by suffix; do not divide again
        synchronize(self.device)
        body = frame.metadata["body"]
        self._step_loss_sum += body["loss_sum"]
        microbatch = frame.metadata["microbatch_id"]
        self._notify("micro_done", microbatch=microbatch, logits=None,
                     boundary_grad=gradient, loss_sum=body["loss_sum"], valid_tokens=body["valid_tokens"])
        del hidden, gradient
        self.gate.check_timeout()
        self.channel.send("MICRO_DONE", microbatch=microbatch)
        self._microbatches_completed += 1

    def _apply(self):
        if self._hidden is not None or self._targets is not None:
            raise ProtocolError("WRONG_STATE", "optimizer apply with a live graph")
        self.optimizer.step()
        synchronize(self.device)
        self._check_parameters(gradients=True, optimizer=True)
        self.gate.check_timeout()
        self._notify("after_step")

    def _check_parameters(self, *, gradients=False, optimizer=False):
        for name, parameter in self.stage.named_parameters():
            if parameter.device.type != self.device.type or parameter.dtype != torch.float32:
                raise ProtocolError("DEVICE_ERROR", "parameter placement or precision mismatch")
            if not bool(torch.isfinite(parameter).all()):
                raise ProtocolError("NONFINITE", "nonfinite stage parameter")
            if gradients:
                gradient = parameter.grad
                if gradient is None or gradient.device != parameter.device or gradient.dtype != torch.float32:
                    raise ProtocolError("DEVICE_ERROR", "missing gradient or gradient placement mismatch")
                if not bool(torch.isfinite(gradient).all()):
                    raise ProtocolError("NONFINITE", "nonfinite stage gradient")
            if optimizer:
                state = self.optimizer.state[parameter]
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    value = state.get(key)
                    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                        raise ProtocolError("NONFINITE", "missing or nonfinite optimizer state")
                    if key != "step" and (value.device != parameter.device or value.dtype != torch.float32):
                        raise ProtocolError("DEVICE_ERROR", "optimizer moment placement mismatch")
                if state["step"].item() != self.gate.parameter_version + 1:
                    raise ProtocolError("VERSION_MISMATCH", "optimizer update counter differs from step")

    def _notify(self, event, *, microbatch=None, **details):
        if self.observer is None:
            return
        observed = {key: value.detach().to("cpu", copy=True) if isinstance(value, torch.Tensor) else value
                    for key, value in details.items()}
        observed.update(step_id=self.gate.step_id, microbatch_id=microbatch,
                        stage_id=self.gate.assignment["stage_id"], run_id=self.gate.run_id,
                        config_hash=self.gate.assignment["config_hash"])
        self.observer(self.stage, self.optimizer, event, observed)
