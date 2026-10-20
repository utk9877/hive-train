"""Socketpair tests exercise the real framing I/O without requiring TCP bind."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hmac
import json
import socket
import struct
import threading
import time
import unittest
from uuid import uuid4
from unittest.mock import Mock

from hive_train.protocol import Frame, PREFIX, ProtocolError, encode_metadata
from hive_train.transport import (Session, SocketStream, Timeouts, client_handshake,
                                  accept_members, metadata, relay_one, server_handshake)
from hive_train.transport_probe import probe_client, probe_server

RUN = "12345678-1234-4234-9234-123456789abc"
KEY = b"k" * 32
FAST = Timeouts(header=0.15, idle=0.3, handshake=0.5, no_progress=0.15,
                transfer_floor=0.5, poll=0.005)


def activation(seq=1, size=16):
    tensor = {"name": "hidden", "dtype": "float32", "shape": [1, 1, size // 4],
              "byte_order": "little", "nbytes": size}
    return Frame(metadata("ACTIVATION", RUN, "worker-0", "coordinator", seq,
                 step=0, microbatch=0, version=0, tensor=tensor), bytearray(size))


def wire(frame, key=KEY):
    raw = encode_metadata(frame.metadata, len(frame.payload))
    prefix = PREFIX.pack(b"HTRN", 1, 0, len(raw), 0, len(frame.payload))
    return prefix + raw + frame.payload + hmac.digest(key, prefix + raw + frame.payload, "sha256")


class ShortSocket:
    def __init__(self, sock, chunk=1):
        self.sock, self.chunk = sock, chunk

    def __getattr__(self, name):
        return getattr(self.sock, name)

    def recv_into(self, view):
        return self.sock.recv_into(view[:self.chunk])

    def send(self, view):
        return self.sock.send(view[:self.chunk])


class TransportTests(unittest.TestCase):
    def pair(self, *, limits=67108864, timeouts=FAST, short=False):
        a, b = socket.socketpair()
        if short:
            a, b = ShortSocket(a), ShortSocket(b)
        a, b = SocketStream(a, timeouts=timeouts, max_tensor_bytes=limits), SocketStream(b, timeouts=timeouts, max_tensor_bytes=limits)
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        return a, b

    def authenticated_pair(self, worker="worker-0", secret=KEY):
        a, b = self.pair(timeouts=replace(FAST, handshake=2))
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(server_handshake, a, KEY, RUN)
            client = client_handshake(b, secret, worker)
            server = future.result(timeout=2)
        return server, client

    def test_authenticated_handshake_sequences_and_ping(self):
        server, client = self.authenticated_pair()
        self.assertEqual((server.send_seq, client.send_seq), (2, 1))
        server.send("PING", version=0, body={"nonce": 9})
        self.assertEqual(client.receive().metadata["body"]["nonce"], 9)
        client.send("PONG", version=0, body={"nonce": 9})
        self.assertEqual(server.receive().metadata["seq"], 1)

    def test_partial_reads_and_writes(self):
        a, b = self.pair(short=True, timeouts=replace(FAST, header=2, transfer_floor=2))
        frame = activation()
        frame.payload[:] = struct.pack("<4f", 0, -0.0, 1.5, -2.0)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(a.send_frame, frame, key=KEY)
            result = b.receive_frame(key=KEY)
            future.result(timeout=2)
        self.assertEqual(result.metadata, frame.metadata)
        self.assertEqual(result.payload, frame.payload)
        self.assertIsInstance(result.payload, bytearray)

    def test_coalesced_frames_do_not_consume_next_frame(self):
        a, b = self.pair()
        a.socket.sendall(wire(activation(1)) + wire(activation(2)))
        self.assertEqual(b.receive_frame(key=KEY).metadata["seq"], 1)
        self.assertEqual(b.receive_frame(key=KEY).metadata["seq"], 2)

    def test_eof_at_each_byte_of_frame_is_terminal(self):
        data = wire(activation())
        for offset in range(len(data)):
            with self.subTest(offset=offset):
                a, b = socket.socketpair()
                stream = SocketStream(b, timeouts=FAST)
                try:
                    a.sendall(data[:offset])
                    a.shutdown(socket.SHUT_WR)
                    with self.assertRaises(ProtocolError) as caught:
                        stream.receive_frame(key=KEY)
                    self.assertEqual(caught.exception.code, "PEER_LOST")
                    self.assertTrue(stream.closed)
                finally:
                    a.close()
                    stream.close()

    def test_oversize_header_rejected_before_payload_allocation(self):
        a, b = self.pair(limits=16)
        a.socket.sendall(PREFIX.pack(b"HTRN", 1, 0, 12, 0, 17))
        with self.assertRaises(ProtocolError):
            b.receive_frame(key=KEY)
        self.assertEqual(b.peak_received_payload, 0)

    def test_wrong_config_shape_rejected_before_payload_read(self):
        a, b = self.pair()
        raw = wire(activation())
        header_end = PREFIX.size + PREFIX.unpack(raw[:PREFIX.size])[3]
        a.socket.sendall(raw[:header_end])
        session = Session(b, RUN, "coordinator", "worker-0", KEY, send_seq=2, receive_seq=1)
        config = {"data": {"max_microbatch_size": 2, "sequence_length": 32}, "model": {"width": 128}}
        with self.assertRaises(ProtocolError):
            session.receive(config=config)
        self.assertEqual(b.peak_received_payload, 0)

    def test_corrupt_tag_and_wrong_secret(self):
        a, b = self.pair()
        corrupted = bytearray(wire(activation()))
        corrupted[-1] ^= 1
        a.socket.sendall(corrupted)
        with self.assertRaises(ProtocolError) as caught:
            b.receive_frame(key=KEY)
        self.assertEqual(caught.exception.code, "AUTH_FAILED")
        a, b = self.pair()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(server_handshake, a, KEY, RUN)
            with self.assertRaises(ProtocolError):
                client_handshake(b, b"x" * 32, "worker-0")
            with self.assertRaises(ProtocolError):
                future.result(timeout=2)
        self.assertTrue(a.closed and b.closed)

    def test_duplicate_worker_admission_rejected(self):
        a, b = self.pair()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(server_handshake, a, KEY, RUN, admitted={"worker-0"})
            with self.assertRaises(ProtocolError):
                client_handshake(b, KEY, "worker-0")
            with self.assertRaises(ProtocolError):
                future.result(timeout=2)

    def test_bad_sequence_run_and_identity_rejected(self):
        for field, value in (("seq", 2), ("run_id", str(uuid4())), ("receiver", "worker-1")):
            a, b = self.pair()
            frame = activation()
            if field == "receiver":
                frame.metadata["sender"] = "coordinator"  # valid relay schema, wrong connection identity
            frame.metadata[field] = value
            a.socket.sendall(wire(frame))
            session = Session(b, RUN, "coordinator", "worker-0", KEY, send_seq=2, receive_seq=1)
            with self.assertRaises(ProtocolError):
                session.receive()
            self.assertTrue(b.closed)

    def test_idle_partial_header_and_stalled_payload_deadlines(self):
        for part in (b"", b"H", wire(activation())[:-40]):
            a, b = self.pair()
            if part:
                a.socket.sendall(part)
            started = time.monotonic()
            with self.assertRaises(ProtocolError) as caught:
                b.receive_frame(key=KEY)
            self.assertEqual(caught.exception.code, "TIMEOUT")
            self.assertLess(time.monotonic() - started, 1)

    def test_waiting_for_compute_does_not_use_transfer_no_progress_timeout(self):
        a, b = self.pair(timeouts=replace(FAST, no_progress=0.03, idle=0.04))
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(b.receive_frame, key=KEY, wait_timeout=0.5)
            time.sleep(0.08)
            a.send_frame(activation(), key=KEY)
            self.assertEqual(future.result(timeout=1).metadata["type"], "ACTIVATION")

    def test_backpressure_and_cancellation_are_bounded(self):
        a, b = self.pair()
        a.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        start = time.monotonic()
        with self.assertRaises(ProtocolError) as caught:
            a.send_frame(activation(size=4 * 1024 * 1024), key=KEY)
        self.assertEqual(caught.exception.code, "TIMEOUT")
        self.assertLess(time.monotonic() - start, 1)
        a, b = self.pair()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(b.receive_frame, key=KEY)
            b.cancel()
            with self.assertRaises(ProtocolError):
                future.result(timeout=1)

    def test_slow_trickle_cannot_extend_header_deadline(self):
        a, b = self.pair(timeouts=replace(FAST, header=0.08))
        def trickle():
            try:
                for byte in wire(activation()):
                    a.socket.send(bytes([byte]))
                    time.sleep(0.01)
            except OSError:
                pass
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(trickle)
            start = time.monotonic()
            with self.assertRaises(ProtocolError) as caught:
                b.receive_frame(key=KEY)
            self.assertEqual(caught.exception.code, "TIMEOUT")
            self.assertLess(time.monotonic() - start, 0.5)
            future.result(timeout=1)

    def test_bidirectional_payload_sizes_through_hard_limit(self):
        a, b = self.pair(timeouts=replace(FAST, header=2, no_progress=5, transfer_floor=5))
        for size in (1024, 256 * 1024, 1024**2, 4 * 1024**2, 16 * 1024**2, 64 * 1024**2):
            for sender, receiver in ((a, b), (b, a)):
                frame = activation(size=size)
                frame.payload[::4096] = b"\x80" * len(frame.payload[::4096])
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(sender.send_frame, frame, key=KEY)
                    result = receiver.receive_frame(key=KEY)
                    future.result(timeout=10)
                self.assertTrue(hmac.compare_digest(frame.payload, result.payload))
                del result, frame
        self.assertEqual(a.peak_received_payload, 67108864)
        self.assertEqual(b.peak_received_payload, 67108864)

    def test_serial_relay_rewrites_only_connection_envelope(self):
        c0, w0 = self.authenticated_pair("worker-0")
        c1, w1 = self.authenticated_pair("worker-1")
        frame = activation()
        config = {"data": {"max_microbatch_size": 1, "sequence_length": 1}, "model": {"width": 4}}
        w0.send("ACTIVATION", step=0, microbatch=0, version=0,
                tensor=frame.metadata["tensor"], payload=frame.payload)
        gate = Mock()
        relay_one(c0, c1, "ACTIVATION", config=config, state=gate)
        self.assertEqual(gate.accept.call_count, 2)
        received = w1.receive(config=config)
        self.assertEqual(received.payload, frame.payload)
        self.assertEqual(received.metadata["sender"], "coordinator")
        self.assertEqual(received.metadata["receiver"], "worker-1")
        self.assertEqual(received.metadata["seq"], 2)

    def test_relay_gate_failure_closes_both_links_before_forwarding(self):
        c0, w0 = self.authenticated_pair("worker-0")
        c1, w1 = self.authenticated_pair("worker-1")
        frame = activation()
        w0.send("ACTIVATION", step=0, microbatch=0, version=0,
                tensor=frame.metadata["tensor"], payload=frame.payload)
        gate = Mock()
        gate.accept.side_effect = ProtocolError("WRONG_STATE", "not ready")
        before = c1.stream.bytes_sent
        config = {"data": {"max_microbatch_size": 1, "sequence_length": 1}, "model": {"width": 4}}
        with self.assertRaises(ProtocolError):
            relay_one(c0, c1, "ACTIVATION", config=config, state=gate)
        self.assertTrue(c0.stream.closed and c1.stream.closed)
        self.assertEqual(c1.stream.bytes_sent, before)
        gate.disconnect.assert_called_once()

    def test_invalid_local_handshake_inputs_close_stream(self):
        for server_side in (True, False):
            a, b = self.pair()
            with self.assertRaises(ValueError):
                if server_side:
                    server_handshake(a, b"bad", RUN)
                else:
                    client_handshake(a, b"bad", "worker-0")
            self.assertTrue(a.closed)

    def test_membership_seals_after_two_distinct_workers(self):
        # A synthetic acceptor returns real socketpairs: no claim of TCP accept.
        sockets = []
        clients = []
        for _ in range(2):
            a, b = socket.socketpair()
            sockets.append((a, ("local-test", 0)))
            clients.append(SocketStream(b, timeouts=FAST))
            self.addCleanup(a.close)
            self.addCleanup(clients[-1].close)
        listener = Mock()
        listener.accept.side_effect = sockets
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(accept_members, listener, KEY, RUN, timeouts=FAST, timeout=2)
            for worker, stream in zip(("worker-1", "worker-0"), clients):
                client_handshake(stream, KEY, worker)
            sessions = future.result(timeout=2)
        self.assertEqual(set(sessions), {"worker-0", "worker-1"})
        self.assertEqual(listener.accept.call_count, 2)
        listener.close.assert_called_once()
        for session in sessions.values():
            self.addCleanup(session.close)

    def test_membership_deadline_also_bounds_pending_handshake(self):
        a, b = socket.socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        listener = Mock()
        listener.accept.return_value = a, ("local-test", 0)
        start = time.monotonic()
        with self.assertRaises(ProtocolError) as error:
            accept_members(listener, KEY, RUN, timeouts=replace(FAST, handshake=5), timeout=0.08)
        self.assertEqual(error.exception.code, "TIMEOUT")
        self.assertLess(time.monotonic() - start, 0.5)
        self.assertEqual(a.fileno(), -1)
        listener.close.assert_called_once()

    def test_bidirectional_probe_end_to_end_over_socketpair(self):
        server, client = self.authenticated_pair()
        options = dict(sizes=(16, 1024), rounds=3, warmup=1, latency_rounds=4)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(probe_server, server, **options)
            report = probe_client(client, **options)
            server_report = future.result(timeout=2)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(server_report["status"], "pass")
        self.assertEqual(report["exchanges"], 8)
        self.assertEqual(len(server_report["application_rtt"]["samples_seconds"]), 4)
        self.assertEqual(len(report["directions"]), 2)


if __name__ == "__main__":
    unittest.main()
