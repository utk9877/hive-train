"""Authenticated stream/state integration. Fake updates only, no training."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import socket
import unittest

from hive_train.protocol import Frame, ProtocolError
from hive_train.state import CoordinatorState, WorkerState
from hive_train.transport import SocketStream, client_handshake, server_handshake, relay_one
from test_state import Trace, RUN, HASH, WORKERS, config_body


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.addCleanup(self.pool.shutdown, wait=True)
        self.links = {}
        for worker in WORKERS:
            a, b = (SocketStream(s) for s in socket.socketpair())
            self.addCleanup(a.close)
            self.addCleanup(b.close)
            future = self.pool.submit(server_handshake, a, b"k" * 32, RUN)
            client = client_handshake(b, b"k" * 32, worker)
            self.links[worker] = future.result(timeout=2), client
        self.calls = []
        self.coordinator = CoordinatorState(RUN, parameter_hashes={w: HASH for w in WORKERS})
        self.workers = {w: WorkerState(RUN, w, expected_parameter_name_hash=HASH,
                         apply_update=lambda w=w: self.calls.append(w)) for w in WORKERS}
        self.config = config_body()["config"]

    @staticmethod
    def send(session, m):
        size = 0 if m["tensor"] is None else m["tensor"]["nbytes"]
        session.send(m["type"], step=m["step_id"], microbatch=m["microbatch_id"],
                     version=m["parameter_version"], body=m["body"], tensor=m["tensor"],
                     payload=bytearray(size))

    def exchange(self, m):
        outgoing = m["sender"] == "coordinator"
        worker = m["receiver"] if outgoing else m["sender"]
        server, client = self.links[worker]
        sender, receiver = (server, client) if outgoing else (client, server)
        send_gate, receive_gate = ((self.coordinator, self.workers[worker]) if outgoing
                                   else (self.workers[worker], self.coordinator))
        send_gate.accept(m)
        future = self.pool.submit(self.send, sender, m)
        frame = receiver.receive(config=self.config)
        future.result(timeout=2)
        self.assertEqual(frame.metadata, m)
        receive_gate.accept(frame.metadata)

    def replay(self, events):
        index = 0
        while index < len(events):
            m = events[index]
            if m["type"] in ("ACTIVATION", "BOUNDARY_GRAD"):
                following = events[index + 1]
                src, dst = m["sender"], following["receiver"]
                source, source_worker = self.links[src]
                destination, destination_worker = self.links[dst]
                self.workers[src].accept(m)
                sent = self.pool.submit(self.send, source_worker, m)
                relayed = self.pool.submit(relay_one, source, destination, m["type"],
                                          config=self.config, state=self.coordinator)
                frame = destination_worker.receive(config=self.config)
                sent.result(timeout=2)
                relayed.result(timeout=2)
                self.assertEqual(frame.metadata, following)
                self.workers[dst].accept(frame.metadata)
                index += 2
            else:
                self.exchange(m)
                index += 1

    def abort_all(self):
        for gate in (self.coordinator, *self.workers.values()):
            with self.assertRaises(ProtocolError):
                gate.disconnect()
        for pair in self.links.values():
            for session in pair:
                session.close()

    def test_authenticated_complete_step_with_real_relay_and_fake_updates(self):
        self.replay(Trace().complete().events)
        self.assertEqual(self.calls, list(WORKERS))
        for gate in (self.coordinator, *self.workers.values()):
            self.assertEqual((gate.phase, gate.step_id, gate.parameter_version), ("IDLE", 1, 1))

    def test_authenticated_duplicate_apply_new_sequence_never_updates_twice(self):
        trace = Trace().prepared()
        self.replay(trace.events)
        trace.apply()
        apply = next(m for m in trace.events if m["type"] == "STEP_APPLY")
        self.exchange(apply)
        server, worker = self.links["worker-0"]
        # Valid new frame sequence is insufficient: the application gate rejects
        # the duplicate apply ID even if the sender bypasses its own gate.
        self.send(server, apply)
        frame = worker.receive()
        with self.assertRaises(ProtocolError):
            self.workers["worker-0"].accept(frame.metadata)
        self.assertEqual(self.calls, ["worker-0"])
        self.abort_all()

    def test_bad_auth_before_apply_and_lost_ack_are_fail_stop(self):
        trace = Trace().prepared()
        self.replay(trace.events)
        trace.apply()
        apply = deepcopy(next(m for m in trace.events if m["type"] == "STEP_APPLY"))
        server, worker = self.links["worker-0"]
        apply["seq"] = server.send_seq
        server.stream.send_frame(Frame(apply, bytearray()), key=b"wrong".ljust(32, b"!"))
        with self.assertRaises(ProtocolError):
            worker.receive()
        self.assertEqual(self.calls, [])
        self.abort_all()

    def test_disconnect_after_first_update_cannot_continue_or_retry(self):
        trace = Trace().prepared()
        self.replay(trace.events)
        trace.apply()
        apply = next(m for m in trace.events if m["type"] == "STEP_APPLY")
        self.exchange(apply)
        self.abort_all()  # Update completed, but no STEP_APPLIED reached coordinator.
        for gate in (self.coordinator, self.workers["worker-0"]):
            with self.assertRaises(ProtocolError):
                gate.accept(apply)
        self.assertEqual(self.calls, ["worker-0"])


if __name__ == "__main__":
    unittest.main()
