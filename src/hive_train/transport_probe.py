"""Authenticated byte-transport diagnostic, not a training-protocol execution.

Both ends must use the same sizes/rounds/warmup options. Uses real protocol-1
frames and authentication, but no RUN_CONFIG/optimizer state machine or torch.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import secrets
import socket
import statistics
import struct
import time
from uuid import uuid4

from .protocol import MAX_TENSOR_BYTES, ProtocolError
from .transport import SocketStream, client_handshake, server_handshake

SIZES = (1024, 256 * 1024, 1024**2, 4 * 1024**2, 16 * 1024**2, 64 * 1024**2)


def _summary(samples):
    values = sorted(samples)
    return {"samples_seconds": samples, "p50_seconds": statistics.median(values),
            "p95_seconds": values[math.ceil(0.95 * len(values)) - 1],
            "p99_seconds": values[math.ceil(0.99 * len(values)) - 1], "max_seconds": max(values)}


def _settings(sizes, rounds, warmup, latency_rounds):
    if not sizes or any(type(n) is not int or not 4 <= n <= MAX_TENSOR_BYTES or n % 4 for n in sizes):
        raise ValueError("sizes must be positive multiples of four within 64 MiB")
    if type(rounds) is not int or rounds < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("invalid round/warmup count")
    if type(latency_rounds) is not int or latency_rounds < 1:
        raise ValueError("latency rounds must be positive")


def _payload(size):
    # All FP32 values finite; repeated pattern is uncompressed on the wire.
    chunk = struct.pack("<256f", *(i / 256 for i in range(256))) * 64
    data = bytearray(size)
    for start in range(0, size, len(chunk)):
        end = min(start + len(chunk), size)
        data[start:end] = chunk[:end - start]
    return data


def _descriptor(size, name):
    return {"name": name, "dtype": "float32", "shape": [1, 1, size // 4],
            "byte_order": "little", "nbytes": size}


def _ping(session, nonce):
    session.send("PING", version=0, body={"nonce": nonce})


def _receive_nonce(session, kind, nonce):
    received = session.receive(expected_type=kind)
    if received.metadata["body"]["nonce"] != nonce:
        session.close()
        raise ProtocolError("WRONG_STATE", "probe option/order mismatch")


def probe_server(session, *, sizes=SIZES, rounds=30, warmup=5, latency_rounds=1000):
    _settings(sizes, rounds, warmup, latency_rounds)
    times = []
    for i in range(latency_rounds + warmup):
        start = time.perf_counter()
        _ping(session, i)
        _receive_nonce(session, "PONG", i)
        if i >= warmup:
            times.append(time.perf_counter() - start)
    index = 0
    for size in sizes:
        for _ in range(rounds + warmup):
            frame = session.receive(expected_type="ACTIVATION")
            m = frame.metadata
            if (m["tensor"] != _descriptor(size, "hidden") or m["step_id"] != index
                    or m["microbatch_id"] != 0 or m["parameter_version"] != 0):
                raise ProtocolError("SHAPE_MISMATCH", "probe payload/options mismatch")
            _ping(session, index)  # upload acknowledgment
            _receive_nonce(session, "PONG", index)  # download request
            session.send("BOUNDARY_GRAD", step=index, microbatch=0, version=0,
                body={"loss_sum": 0.0, "valid_tokens": 1},
                tensor=_descriptor(size, "hidden_grad"), payload=frame.payload)
            del frame
            index += 1
    # A final request/response proves the client verified the last returned payload.
    _ping(session, index)
    _receive_nonce(session, "PONG", index)
    return {"status": "pass", "application_rtt": _summary(times), "exchanges": index}


def probe_client(session, *, sizes=SIZES, rounds=30, warmup=5, latency_rounds=1000):
    _settings(sizes, rounds, warmup, latency_rounds)
    for i in range(latency_rounds + warmup):
        _receive_nonce(session, "PING", i)
        session.send("PONG", version=0, body={"nonce": i})
    index, records = 0, []
    for size in sizes:
        payload = _payload(size)
        digest = hashlib.sha256(payload).digest()
        up, down = [], []
        for repetition in range(rounds + warmup):
            start = time.perf_counter()
            session.send("ACTIVATION", step=index, microbatch=0, version=0,
                         tensor=_descriptor(size, "hidden"), payload=payload)
            _receive_nonce(session, "PING", index)
            uploaded = time.perf_counter() - start
            start = time.perf_counter()
            session.send("PONG", version=0, body={"nonce": index})
            frame = session.receive(expected_type="BOUNDARY_GRAD")
            downloaded = time.perf_counter() - start
            m = frame.metadata
            if (m["tensor"] != _descriptor(size, "hidden_grad") or m["step_id"] != index
                    or m["microbatch_id"] != 0 or m["parameter_version"] != 0
                    or hashlib.sha256(frame.payload).digest() != digest):
                raise ProtocolError("BAD_FRAME", "probe echo differs")
            del frame
            if repetition >= warmup:
                up.append(uploaded)
                down.append(downloaded)
            index += 1
        records.append({"bytes": size, "payload_sha256": digest.hex(),
                        "client_to_server": _summary(up), "server_to_client": _summary(down),
                        "client_to_server_MiB_s": size / 2**20 / statistics.median(up),
                        "server_to_client_MiB_s": size / 2**20 / statistics.median(down)})
        del payload
    _receive_nonce(session, "PING", index)
    session.send("PONG", version=0, body={"nonce": index})
    return {"status": "pass", "directions": records, "exchanges": index,
            "timing": "application transfer including request/ack, framing, HMAC and allocation; not one-way latency"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    keygen = commands.add_parser("keygen", help="write an exclusive 32-byte secret file")
    keygen.add_argument("path", type=Path)
    for name in ("server", "client"):
        sub = commands.add_parser(name)
        if name == "server":
            sub.add_argument("--bind", default="127.0.0.1")
        else:
            sub.add_argument("--host", required=True)
        sub.add_argument("--port", type=int, default=29550)
        sub.add_argument("--secret-file", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--sizes", type=int, nargs="+", default=list(SIZES), help="bytes; identical on both ends")
        sub.add_argument("--rounds", type=int, default=30)
        sub.add_argument("--warmup", type=int, default=5)
        sub.add_argument("--latency-rounds", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.command == "keygen":
        fd = os.open(args.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as file:
            file.write(secrets.token_bytes(32))
        print("Created secret file; share privately, never commit it. Windows ACL must be checked separately.")
        return 0
    session = None
    report = {"schema": "m2_transport_probe_v1", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "role": args.command, "port": args.port,
              "sizes": args.sizes, "rounds": args.rounds, "warmup": args.warmup,
              "latency_rounds": args.latency_rounds,
              "scope": "transport only; no model, optimizer, training state gate or accelerator benchmark"}
    try:
        options = {k: getattr(args, k) for k in ("sizes", "rounds", "warmup", "latency_rounds")}
        _settings(**options)
        secret = args.secret_file.read_bytes()
        if len(secret) != 32:
            raise ValueError("secret file must contain exactly 32 bytes")
        if args.command == "server":
            # One diagnostic client only. The real runtime uses accept_members.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind((args.bind, args.port))
                listener.listen(1)
                listener.settimeout(60)
                print(f"Listening on {args.bind}:{args.port} for one diagnostic peer", flush=True)
                sock, _ = listener.accept()
            session = server_handshake(SocketStream(sock), secret, str(uuid4()), expected_workers=("worker-0",))
            report.update(probe_server(session, **options))
        else:
            sock = socket.create_connection((args.host, args.port), timeout=10)
            session = client_handshake(SocketStream(sock), secret, "worker-0")
            report.update(probe_client(session, **options))
        report["socket_send_buffer"] = session.stream.socket.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        report["socket_receive_buffer"] = session.stream.socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    except (OSError, ValueError, ProtocolError) as error:
        report.update(status="fail", error=str(error))
    finally:
        if session is not None:
            session.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output), "error": report.get("error")}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
