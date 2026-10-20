"""Bounded synchronous protocol-1 byte streams. No model or torch dependency.

Each stream has exactly one reader and one writer; no application frame queues.
The caller retains exclusive ownership of an outgoing payload until send returns.
All failures, including cancellation/interrupts, close the stream permanently.
"""

from dataclasses import dataclass
import hmac
import math
import secrets
import select
import socket
import threading
import time
from uuid import UUID

from .protocol import (Frame, MAX_TENSOR_BYTES, PREFIX, ProtocolError, decode_metadata,
                       encode_metadata, validate_prefix, validate_tensor_contract)


@dataclass(frozen=True)
class Timeouts:
    header: float = 10.0
    idle: float = 30.0
    handshake: float = 10.0
    no_progress: float = 30.0
    transfer_floor: float = 30.0
    minimum_bytes_per_second: float = 1024 * 1024
    multiplier: float = 5.0
    poll: float = 0.05

    def __post_init__(self):
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in self.__dict__.values()):
            raise ValueError("timeouts/rates must be finite positive numbers")

    def transfer(self, size):
        return max(self.transfer_floor, self.multiplier * size / self.minimum_bytes_per_second)


def _earlier(first, second):
    return first if second is None else min(first, second)


class SocketStream:
    def __init__(self, sock, *, timeouts=Timeouts(), max_tensor_bytes=MAX_TENSOR_BYTES):
        if type(max_tensor_bytes) is not int or not 0 < max_tensor_bytes <= MAX_TENSOR_BYTES:
            raise ValueError("invalid tensor cap")
        self.socket = sock
        self.timeouts = timeouts
        self.max_tensor_bytes = max_tensor_bytes
        self.cancelled = threading.Event()
        self._reader = threading.Lock()
        self._writer = threading.Lock()
        self.socket.setblocking(False)
        if self.socket.family in (socket.AF_INET, socket.AF_INET6):
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.bytes_received = 0
        self.bytes_sent = 0
        self.peak_received_payload = 0

    @property
    def closed(self):
        return self.cancelled.is_set()

    def close(self):
        self.cancelled.set()
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()

    cancel = close

    def _wait(self, *, writing, deadline):
        while True:
            if self.closed:
                raise ProtocolError("PEER_LOST", "stream cancelled/closed")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProtocolError("TIMEOUT", "absolute I/O deadline expired")
            try:
                readable, writable, _ = select.select(
                    [] if writing else [self.socket], [self.socket] if writing else [], [],
                    min(remaining, self.timeouts.poll))
            except (OSError, ValueError) as error:
                raise ProtocolError("PEER_LOST", "socket unavailable") from error
            if readable or writable:
                return

    def _read_into(self, buffer, deadline, *, idle=False):
        view = memoryview(buffer).cast("B")
        offset = 0
        progress_deadline = deadline if idle else time.monotonic() + self.timeouts.no_progress
        while offset < len(view):
            self._wait(writing=False, deadline=min(deadline, progress_deadline))
            try:
                received = self.socket.recv_into(view[offset:])
            except (BlockingIOError, InterruptedError):
                continue
            except OSError as error:
                raise ProtocolError("PEER_LOST", "receive failed") from error
            if received == 0:
                raise ProtocolError("PEER_LOST", "EOF before complete frame")
            offset += received
            self.bytes_received += received
            progress_deadline = time.monotonic() + self.timeouts.no_progress

    def _read(self, count, deadline, *, idle=False):
        data = bytearray(count)
        self._read_into(data, deadline, idle=idle)
        return data

    def _write(self, data, deadline):
        view = memoryview(data).cast("B")
        offset = 0
        progress_deadline = time.monotonic() + self.timeouts.no_progress
        while offset < len(view):
            self._wait(writing=True, deadline=min(deadline, progress_deadline))
            try:
                sent = self.socket.send(view[offset:])
            except (BlockingIOError, InterruptedError):
                continue
            except OSError as error:
                raise ProtocolError("PEER_LOST", "send failed") from error
            if sent == 0:
                raise ProtocolError("PEER_LOST", "zero-byte send")
            offset += sent
            self.bytes_sent += sent
            progress_deadline = time.monotonic() + self.timeouts.no_progress

    def receive_frame(self, *, key=None, handshake=False, deadline=None,
                      wait_timeout=None, metadata_validator=None):
        if not self._reader.acquire(blocking=False):
            self.close()
            raise ProtocolError("WRONG_STATE", "multiple stream readers")
        try:
            if self.closed:
                raise ProtocolError("PEER_LOST", "stream closed")
            if not handshake and (not isinstance(key, bytes) or len(key) != 32):
                raise ProtocolError("AUTH_FAILED", "missing session key")
            wait = self.timeouts.idle if wait_timeout is None else wait_timeout
            if not math.isfinite(wait) or wait <= 0:
                raise ProtocolError("TIMEOUT", "invalid wait budget")
            first = self._read(1, _earlier(time.monotonic() + wait, deadline), idle=True)
            started = time.monotonic()
            header_deadline = _earlier(started + self.timeouts.header, deadline)
            prefix = bytes(first + self._read(PREFIX.size - 1, header_deadline))
            metadata_size, payload_size = validate_prefix(prefix, self.max_tensor_bytes, handshake=handshake)
            raw = self._read(metadata_size, header_deadline)
            metadata = decode_metadata(bytes(raw), payload_size, self.max_tensor_bytes)
            if (metadata["type"] in {"CHALLENGE", "HELLO", "WELCOME"}) != handshake:
                raise ProtocolError("WRONG_STATE", "handshake/operational frame mismatch")
            if metadata_validator is not None:
                # Pure validation only! Metadata is not authenticated yet.
                metadata_validator(metadata)
            frame_deadline = _earlier(started + self.timeouts.transfer(payload_size), deadline)
            payload = self._read(payload_size, frame_deadline)
            self.peak_received_payload = max(self.peak_received_payload, payload_size)
            tag = self._read(32, frame_deadline)
            if handshake:
                expected = bytes(32)
            else:
                digest = hmac.new(key, digestmod="sha256")
                for part in (prefix, raw, payload):
                    digest.update(part)
                expected = digest.digest()
            if not hmac.compare_digest(tag, expected):
                raise ProtocolError("AUTH_FAILED", "frame authentication failed")
            return Frame(metadata, payload)
        except BaseException:
            self.close()
            raise
        finally:
            self._reader.release()

    def send_frame(self, frame, *, key=None, handshake=False, deadline=None):
        if not self._writer.acquire(blocking=False):
            self.close()
            raise ProtocolError("WRONG_STATE", "multiple stream writers")
        try:
            if self.closed:
                raise ProtocolError("PEER_LOST", "stream closed")
            if not handshake and (not isinstance(key, bytes) or len(key) != 32):
                raise ProtocolError("AUTH_FAILED", "missing session key")
            payload = memoryview(frame.payload).cast("B")
            raw = encode_metadata(frame.metadata, len(payload), self.max_tensor_bytes)
            prefix = PREFIX.pack(b"HTRN", 1, 0, len(raw), 0, len(payload))
            validate_prefix(prefix, self.max_tensor_bytes, handshake=handshake)
            if (frame.metadata["type"] in {"CHALLENGE", "HELLO", "WELCOME"}) != handshake:
                raise ProtocolError("WRONG_STATE", "handshake/operational frame mismatch")
            if handshake:
                tag = bytes(32)
            else:
                digest = hmac.new(key, digestmod="sha256")
                for part in (prefix, raw, payload):
                    digest.update(part)
                tag = digest.digest()
            started = time.monotonic()
            header_deadline = _earlier(started + self.timeouts.header, deadline)
            self._write(prefix, header_deadline)
            self._write(raw, header_deadline)
            frame_deadline = _earlier(started + self.timeouts.transfer(len(payload)), deadline)
            self._write(payload, frame_deadline)
            self._write(tag, frame_deadline)
        except BaseException:
            self.close()
            raise
        finally:
            self._writer.release()


def metadata(kind, run_id, sender, receiver, seq, *, step=None, microbatch=None, version=None,
             body=None, tensor=None):
    return {"type": kind, "run_id": run_id, "sender": sender, "receiver": receiver, "seq": seq,
            "step_id": step, "microbatch_id": microbatch, "parameter_version": version,
            "body": {} if body is None else body, "tensor": tensor}


class Session:
    """An authenticated stream with monotonic IDs; state gates are separate."""
    def __init__(self, stream, run_id, local, peer, key, *, send_seq, receive_seq):
        self.stream, self.run_id, self.local, self.peer = stream, run_id, local, peer
        self._key = key
        self.send_seq, self.receive_seq = send_seq, receive_seq
        self._send_guard, self._receive_guard = threading.Lock(), threading.Lock()

    def close(self):
        self.stream.close()

    def send(self, kind, *, step=None, microbatch=None, version=None, body=None, tensor=None,
             payload=b"", deadline=None):
        if not self._send_guard.acquire(blocking=False):
            self.close()
            raise ProtocolError("WRONG_STATE", "multiple session senders")
        try:
            meta = metadata(kind, self.run_id, self.local, self.peer, self.send_seq,
                            step=step, microbatch=microbatch, version=version, body=body, tensor=tensor)
            self.stream.send_frame(Frame(meta, payload), key=self._key, deadline=deadline)
            self.send_seq += 1
        finally:
            self._send_guard.release()

    def receive(self, *, config=None, expected_type=None, deadline=None, wait_timeout=None):
        if not self._receive_guard.acquire(blocking=False):
            self.close()
            raise ProtocolError("WRONG_STATE", "multiple session receivers")
        try:
            def validate(meta):
                if meta["run_id"] != self.run_id or meta["sender"] != self.peer or meta["receiver"] != self.local:
                    raise ProtocolError("WRONG_STATE", "run/peer identity mismatch")
                if meta["seq"] != self.receive_seq:
                    raise ProtocolError("DUPLICATE", "noncontiguous frame sequence")
                if expected_type is not None and meta["type"] != expected_type:
                    raise ProtocolError("WRONG_STATE", "unexpected message type")
                if config is not None and meta["tensor"] is not None:
                    validate_tensor_contract(meta, config)
            frame = self.stream.receive_frame(key=self._key, metadata_validator=validate,
                deadline=deadline, wait_timeout=wait_timeout)
            self.receive_seq += 1
            return frame
        finally:
            self._receive_guard.release()


def _secret_check(secret):
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise ValueError("run secret must contain exactly 32 bytes")


def _auth_bytes(run_id, server_nonce, worker_nonce, worker_id):
    return b"HTRN1\0" + UUID(run_id).bytes + server_nonce + worker_nonce + worker_id.encode("ascii")


def _mac(key, value):
    return hmac.digest(key, value, "sha256")


def server_handshake(stream, secret, run_id, *, admitted=None, expected_workers=("worker-0", "worker-1"), deadline=None):
    """Synchronous admission: caller serializes access to the admitted-ID set."""
    deadline = _earlier(time.monotonic() + stream.timeouts.handshake, deadline)
    try:
        _secret_check(secret)
        server_nonce = secrets.token_bytes(32)
        stream.send_frame(Frame(metadata("CHALLENGE", run_id, "coordinator", "pending", 0,
            body={"server_nonce": server_nonce.hex()}), bytearray()), handshake=True, deadline=deadline)
        hello = stream.receive_frame(handshake=True, deadline=deadline).metadata
        worker_id = hello["sender"]
        if (hello["type"] != "HELLO" or hello["run_id"] != run_id or hello["seq"] != 0
                or worker_id not in expected_workers or (admitted is not None and worker_id in admitted)):
            raise ProtocolError("AUTH_FAILED", "unexpected/duplicate worker admission")
        worker_nonce = bytes.fromhex(hello["body"]["worker_nonce"])
        auth = _auth_bytes(run_id, server_nonce, worker_nonce, worker_id)
        if not hmac.compare_digest(bytes.fromhex(hello["body"]["proof"]), _mac(secret, auth)):
            raise ProtocolError("AUTH_FAILED", "worker proof mismatch")
        stream.send_frame(Frame(metadata("WELCOME", run_id, "coordinator", worker_id, 1,
            body={"proof": _mac(secret, b"server\0" + auth).hex(),
                  "max_tensor_bytes": stream.max_tensor_bytes}), bytearray()), handshake=True, deadline=deadline)
        if admitted is not None:
            admitted.add(worker_id)
        return Session(stream, run_id, "coordinator", worker_id, _mac(secret, b"session\0" + auth),
                       send_seq=2, receive_seq=1)
    except BaseException:
        stream.close()
        raise


def client_handshake(stream, secret, worker_id):
    deadline = time.monotonic() + stream.timeouts.handshake
    try:
        _secret_check(secret)
        if worker_id not in ("worker-0", "worker-1"):
            raise ValueError("invalid worker identity")
        challenge = stream.receive_frame(handshake=True, deadline=deadline).metadata
        if challenge["type"] != "CHALLENGE" or challenge["seq"] != 0:
            raise ProtocolError("AUTH_FAILED", "expected initial challenge")
        run_id = challenge["run_id"]
        worker_nonce = secrets.token_bytes(32)
        auth = _auth_bytes(run_id, bytes.fromhex(challenge["body"]["server_nonce"]), worker_nonce, worker_id)
        stream.send_frame(Frame(metadata("HELLO", run_id, worker_id, "coordinator", 0,
            body={"worker_nonce": worker_nonce.hex(), "proof": _mac(secret, auth).hex()}), bytearray()),
            handshake=True, deadline=deadline)
        welcome = stream.receive_frame(handshake=True, deadline=deadline).metadata
        if (welcome["type"] != "WELCOME" or welcome["seq"] != 1 or welcome["run_id"] != run_id
                or welcome["receiver"] != worker_id):
            raise ProtocolError("AUTH_FAILED", "unexpected welcome")
        if not hmac.compare_digest(bytes.fromhex(welcome["body"]["proof"]), _mac(secret, b"server\0" + auth)):
            raise ProtocolError("AUTH_FAILED", "coordinator proof mismatch")
        # Version 1 currently negotiates only a smaller cap, never a larger allocation.
        stream.max_tensor_bytes = min(stream.max_tensor_bytes, welcome["body"]["max_tensor_bytes"])
        return Session(stream, run_id, worker_id, "coordinator", _mac(secret, b"session\0" + auth),
                       send_seq=1, receive_seq=2)
    except BaseException:
        stream.close()
        raise


def relay_one(source, destination, expected_type, *, config, state, deadline=None):
    """Receive/authenticate one bounded tensor, forward it before reading another.

    Consume authenticated receive and outgoing send events on the coordinator
    gate before forwarding. The caller must abort all other local work on error.
    Never retains a tensor backlog; transport failure closes both directions.
    """
    try:
        if expected_type not in ("ACTIVATION", "BOUNDARY_GRAD"):
            raise ProtocolError("WRONG_STATE", "only boundary tensors can be relayed")
        frame = source.receive(config=config, expected_type=expected_type, deadline=deadline,
            wait_timeout=None if deadline is None else max(0.001, deadline - time.monotonic()))
        meta = frame.metadata
        state.accept(meta)
        state.accept(metadata(expected_type, destination.run_id, destination.local,
            destination.peer, destination.send_seq, step=meta["step_id"],
            microbatch=meta["microbatch_id"], version=meta["parameter_version"],
            body=meta["body"], tensor=meta["tensor"]))
        destination.send(expected_type, step=meta["step_id"], microbatch=meta["microbatch_id"],
            version=meta["parameter_version"], body=meta["body"], tensor=meta["tensor"],
            payload=frame.payload, deadline=deadline)
        return dict(meta["body"])
    except BaseException:
        source.close()
        destination.close()
        try:
            state.disconnect()
        except ProtocolError:
            pass
        raise


def accept_members(listener, secret, run_id, *, timeouts=Timeouts(), timeout=30.0):
    """Admit exactly two workers, serial handshakes (one pending), then seal.

    Listener ownership transfers here. Close it on success/failure; established
    sockets remain alive on success. This helper never permits an in-run rejoin.
    """
    admitted, sessions = set(), {}
    deadline = time.monotonic() + timeout
    try:
        _secret_check(secret)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("invalid membership timeout")
        while len(sessions) < 2:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProtocolError("TIMEOUT", "membership admission timed out")
            listener.settimeout(remaining)
            sock, _ = listener.accept()
            stream = SocketStream(sock, timeouts=timeouts)
            session = server_handshake(stream, secret, run_id, admitted=admitted, deadline=deadline)
            sessions[session.peer] = session
        return sessions
    except BaseException:
        for session in sessions.values():
            session.close()
        raise
    finally:
        listener.close()
