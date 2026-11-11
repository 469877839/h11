# Dual-ended, reference-model-based tests for h11.
#
# These tests exercise two real h11.Connection objects through in-memory byte
# queues (see _pair_harness.Wire) and check every step against an independent
# ReferenceOracle whose rules are written out by hand instead of consulting
# h11's own transition tables.  The focus is on situations where the two ends
# temporarily disagree: split writes, delayed event consumption, early EOFs
# and the protocol corners around 100-continue, multiple 1xx, HEAD, 204/304,
# protocol switching and connection reuse.
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Type

import pytest

from .._connection import PAUSED
from .._events import (
    ConnectionClosed,
    Data,
    EndOfMessage,
    Event,
    InformationalResponse,
    Request,
    Response,
)
from .._state import (
    CLIENT,
    CLOSED,
    DONE,
    ERROR,
    IDLE,
    MIGHT_SWITCH_PROTOCOL,
    MUST_CLOSE,
    SEND_BODY,
    SEND_RESPONSE,
    SERVER,
    SWITCHED_PROTOCOL,
)
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from ._pair_harness import PairHarness
from .helpers import normalize_data_events

# ---------------------------------------------------------------------------
# Deterministic tiny RNG (so every run replays byte-for-byte without pytest
# plugin nondeterminism).  64-bit splitmix64.
# ---------------------------------------------------------------------------


class SplitMix64:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def randrange(self, stop: int) -> int:
        return self.next_u64() % stop

    def choice(self, options: Sequence[Any]) -> Any:
        return options[self.randrange(len(options))]


# ---------------------------------------------------------------------------
# Operation vocabulary
# ---------------------------------------------------------------------------
#
# An Op is one replayable step.  Ops are plain data so they survive the
# reducer and can be dumped into the event log.  ``run`` applies the op to a
# PairHarness.


@dataclass(frozen=True)
class Op:
    name: str
    role: Type[Sentinel]
    event_factory: Optional[Callable[[], Event]] = None
    split_bytes: Optional[int] = None

    def run(self, harness: PairHarness) -> None:
        if self.name == "send":
            assert self.event_factory is not None
            harness.send(self.role, self.event_factory())
        elif self.name == "deliver":
            harness.deliver(self.role, self.split_bytes)
        elif self.name == "drain":
            harness.drain(self.role)
        elif self.name == "wire_drop":
            harness.wire_drop(self.role)
        elif self.name == "deliver_eof":
            harness.deliver_eof(self.role)
        elif self.name == "cycle":
            harness.start_next_cycle()
        else:  # pragma: no cover
            raise ValueError(f"unknown op {self.name!r}")


def _request(kind: str) -> Request:
    if kind == "get":
        return Request(method="GET", target="/", headers=[("Host", "x")])
    if kind == "head":
        return Request(method="HEAD", target="/", headers=[("Host", "x")])
    if kind == "connect":
        return Request(
            method="CONNECT",
            target="example.com:443",
            headers=[("Host", "x"), ("Content-Length", "1")],
        )
    if kind == "upgrade":
        return Request(
            method="GET",
            target="/",
            headers=[
                ("Host", "x"),
                ("Upgrade", "a"),
                ("Connection", "upgrade"),
            ],
        )
    if kind == "expect":
        return Request(
            method="POST",
            target="/",
            headers=[
                ("Host", "x"),
                ("Content-Length", "4"),
                ("Expect", "100-continue"),
            ],
        )
    if kind == "close":
        return Request(
            method="GET",
            target="/",
            headers=[("Host", "x"), ("Connection", "close")],
        )
    return Request(
        method="PUT",
        target="/",
        headers=[("Host", "x"), ("Content-Length", "4")],
    )


def _response(kind: str) -> Event:
    if kind == "100":
        return InformationalResponse(status_code=100, headers=[])
    if kind == "103":
        return InformationalResponse(
            status_code=103, headers=[("Link", "</x>; rel=preload")]
        )
    if kind == "101":
        return InformationalResponse(
            status_code=101, headers=[("Upgrade", "a")]
        )
    if kind == "204":
        return Response(status_code=204, headers=[("Content-Length", "0")])
    if kind == "304":
        return Response(status_code=304, headers=[("Content-Length", "0")])
    if kind == "connect_200":
        return Response(status_code=200, headers=[])
    if kind == "deny":
        return Response(
            status_code=404, headers=[("Transfer-Encoding", "chunked")]
        )
    return Response(status_code=200, headers=[("Content-Length", "4")])


# State-dependent legal operations, restricted to ones that can actually make
# progress so generated sequences stay short.
def legal_ops(harness: PairHarness, rng: SplitMix64) -> List[Op]:
    """Enumerate operations that can make progress right now.

    Sends are preferred over pure bookkeeping so generated sequences stay
    short and dense; delivery/drain ops are added only when there is actually
    something to deliver or consume.
    """
    ops: List[Op] = []
    for role in (CLIENT, SERVER):
        conn = harness.conn[role]
        our_state = conn.our_state
        if our_state is ERROR:
            continue
        oracle = harness.oracle[role]

        def offered(event: Event) -> bool:
            return oracle.legal_to_send(role, event) is None

        if role is CLIENT and our_state is IDLE:
            for kind in ("get", "head", "connect", "upgrade", "expect", "close", "put"):
                event = _request(kind)
                if offered(event):
                    ops.append(Op("send", CLIENT, event_factory=lambda e=event: e))
        if role is CLIENT and our_state is SEND_BODY:
            for event in (Data(b"ab"), EndOfMessage()):
                if offered(event):
                    ops.append(Op("send", CLIENT, event_factory=lambda e=event: e))
        if role is SERVER and our_state is SEND_RESPONSE:
            for kind in ("100", "103", "200", "204", "304", "deny", "101"):
                event = _response(kind)
                if offered(event):
                    ops.append(Op("send", SERVER, event_factory=lambda e=event: e))
        if role is SERVER and our_state is SEND_BODY:
            for event in (Data(b"ab"), EndOfMessage()):
                if offered(event):
                    ops.append(Op("send", SERVER, event_factory=lambda e=event: e))
        if our_state in (DONE, MUST_CLOSE) and offered(ConnectionClosed()):
            ops.append(Op("send", role, event_factory=ConnectionClosed))

    if (
        harness.conn[CLIENT].states == {CLIENT: DONE, SERVER: DONE}
        and harness.conn[SERVER].states == {CLIENT: DONE, SERVER: DONE}
    ):
        ops.append(Op("cycle", CLIENT))

    # Delivery operations: let queued bytes be consumed whole or in tiny
    # pieces, sometimes deferring parsing entirely with a bare drain.
    for role in (CLIENT, SERVER):
        direction = harness.wire._queues[SERVER if role is CLIENT else CLIENT]
        if direction.segments:
            ops.append(Op("deliver", role, split_bytes=None))
            for size in (1, 2, 3):
                ops.append(Op("deliver", role, split_bytes=size))
            if direction.segments and direction.segments[0].eof:
                ops.append(Op("deliver_eof", role))
        if rng.randrange(3) == 0:
            ops.append(Op("drain", role))
    return ops


def generate_sequence(seed: int, length: int) -> List[Op]:
    """Build a random legal operation sequence with a fixed seed."""
    rng = SplitMix64(seed)
    harness = PairHarness()
    ops: List[Op] = []
    for _ in range(length):
        choices = legal_ops(harness, rng)
        if not choices:
            break
        # A completed, reusable cycle is always reused immediately so the
        # generated corpus reliably exercises start_next_cycle too.
        cycle_ops = [option for option in choices if option.name == "cycle"]
        op = cycle_ops[0] if cycle_ops else rng.choice(choices)
        op.run(harness)
        ops.append(op)
    return ops


def run_sequence(ops: Sequence[Op]) -> PairHarness:
    harness = PairHarness()
    for op in ops:
        op.run(harness)
    return harness


def shrink_sequence(
    ops: Sequence[Op], is_failure: Callable[[List[Op]], bool]
) -> List[Op]:
    """Greedy minimizer: repeatedly drop prefixes/suffixes/chunks that keep
    the failure, then single ops.  Deterministic and order-preserving."""
    current = list(ops)
    changed = True
    while changed:
        changed = False
        # Try removing contiguous chunks of decreasing size.
        size = len(current)
        while size >= 1:
            i = 0
            while i + size <= len(current):
                candidate = current[:i] + current[i + size :]
                if candidate and is_failure(candidate):
                    current = candidate
                    changed = True
                    break
                i += 1
            else:
                size //= 2
                continue
            break
    return current


def replayable_log(harness: PairHarness) -> str:
    """JSON dump sufficient to re-run the whole exchange."""
    return json.dumps(harness.replay_log(), indent=2, sort_keys=True)


def assert_replay(harness: PairHarness) -> None:
    replayed = PairHarness()
    replayed.replay(harness.replay_log())
    # The replayed run must land in exactly the same joint states, and every
    # endpoint view must still match its reference oracle.
    for role in (CLIENT, SERVER):
        assert replayed.conn[role].states == harness.conn[role].states
        assert replayed.conn[role].states == replayed.oracle[role].states()


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------


def test_100_continue_and_multiple_1xx() -> None:
    harness = PairHarness()
    harness.send(
        CLIENT,
        Request(
            method="POST",
            target="/",
            headers=[
                ("Host", "x"),
                ("Content-Length", "4"),
                ("Expect", "100-continue"),
            ],
        ),
    )
    # The client knows it is waiting; the server only learns after delivery.
    assert harness.conn[CLIENT].client_is_waiting_for_100_continue
    assert not harness.conn[SERVER].client_is_waiting_for_100_continue
    harness.deliver(SERVER)
    assert harness.conn[SERVER].they_are_waiting_for_100_continue

    # Several informational responses in a row, delivered lazily.
    harness.send(SERVER, InformationalResponse(status_code=100, headers=[]))
    harness.send(SERVER, InformationalResponse(status_code=103, headers=[]))
    assert harness.conn[SERVER].our_state is SEND_RESPONSE
    # Client has not consumed them yet: still believes the server is silent.
    assert harness.conn[CLIENT].their_state is SEND_RESPONSE
    events = harness.deliver(CLIENT)
    assert [type(event) for event in events] == [
        InformationalResponse,
        InformationalResponse,
    ]
    assert not harness.conn[CLIENT].client_is_waiting_for_100_continue

    # Client sends its body; server answers 200.
    harness.send(CLIENT, Data(data=b"abcd"))
    harness.deliver(SERVER)
    harness.send(CLIENT, EndOfMessage())
    harness.deliver(SERVER)
    harness.send(SERVER, Response(status_code=200, headers=[("Content-Length", "0")]))
    harness.deliver(CLIENT)
    harness.send(SERVER, EndOfMessage())
    harness.deliver(CLIENT)
    harness.assert_wire_quiet()
    assert harness.start_next_cycle() is None
    assert_replay(harness)


def test_head_204_304_have_no_body_despite_headers() -> None:
    for method, status in [(b"HEAD", 200), (b"GET", 204), (b"GET", 304)]:
        harness = PairHarness()
        harness.send(
            CLIENT,
            Request(method=method, target="/", headers=[("Host", "x")]),
        )
        harness.deliver(SERVER)
        harness.send(CLIENT, EndOfMessage())
        harness.deliver(SERVER)
        harness.send(
            SERVER,
            Response(status_code=status, headers=[("Content-Length", "9")]),
        )
        events = harness.deliver(CLIENT)
        # The client immediately sees EndOfMessage: no body is possible.
        assert [type(event) for event in events] == [Response, EndOfMessage]
        # Trying to send a body anyway is a local protocol error, and the
        # sender must end up in ERROR and stop accepting sends.
        error, _ = harness.send(SERVER, Data(data=b"not-allowed"))
        assert type(error) is LocalProtocolError
        assert harness.conn[SERVER].our_state is ERROR
        assert harness.conn[SERVER].their_state is not ERROR
        assert_replay(harness)


def test_connect_and_101_switch() -> None:
    # CONNECT accepted: both sides end up SWITCHED_PROTOCOL and pause.
    harness = PairHarness()
    harness.send(
        CLIENT,
        Request(
            method="CONNECT",
            target="example.com:443",
            headers=[("Host", "x"), ("Content-Length", "1")],
        ),
    )
    harness.deliver(SERVER)
    harness.send(CLIENT, Data(data=b"x"))
    harness.deliver(SERVER)
    harness.send(CLIENT, EndOfMessage())
    harness.deliver(SERVER)
    assert harness.conn[CLIENT].states[CLIENT] is MIGHT_SWITCH_PROTOCOL
    harness.send(SERVER, Response(status_code=200, headers=[]))
    harness.deliver(CLIENT)
    harness.assert_wire_quiet()
    assert harness.conn[CLIENT].states == {
        CLIENT: SWITCHED_PROTOCOL,
        SERVER: SWITCHED_PROTOCOL,
    }
    assert harness.conn[CLIENT].next_event() is PAUSED
    assert harness.conn[SERVER].next_event() is PAUSED
    assert_replay(harness)

    # Upgrade accepted with 101.
    harness = PairHarness()
    harness.send(
        CLIENT,
        Request(
            method="GET",
            target="/",
            headers=[
                ("Host", "x"),
                ("Upgrade", "a"),
                ("Connection", "upgrade"),
            ],
        ),
    )
    harness.deliver(SERVER)
    harness.send(CLIENT, EndOfMessage())
    harness.deliver(SERVER)
    harness.send(
        SERVER, InformationalResponse(status_code=101, headers=[("Upgrade", "a")])
    )
    harness.deliver(CLIENT)
    harness.assert_wire_quiet()
    assert harness.conn[CLIENT].states == {
        CLIENT: SWITCHED_PROTOCOL,
        SERVER: SWITCHED_PROTOCOL,
    }
    assert_replay(harness)

    # Upgrade denied: normal response resumes the cycle.
    harness = PairHarness()
    harness.send(
        CLIENT,
        Request(
            method="GET",
            target="/",
            headers=[
                ("Host", "x"),
                ("Upgrade", "a"),
                ("Connection", "upgrade"),
            ],
        ),
    )
    harness.deliver(SERVER)
    harness.send(CLIENT, EndOfMessage())
    harness.deliver(SERVER)
    harness.send(
        SERVER, Response(status_code=200, headers=[("Content-Length", "0")])
    )
    harness.deliver(CLIENT)
    harness.send(SERVER, EndOfMessage())
    harness.deliver(CLIENT)
    harness.assert_wire_quiet()
    assert harness.start_next_cycle() is None
    assert_replay(harness)


def test_non_keep_alive_blocks_reuse_and_closes() -> None:
    harness = PairHarness()
    harness.send(
        CLIENT,
        Request(
            method="GET",
            target="/",
            headers=[("Host", "x"), ("Connection", "close")],
        ),
    )
    harness.deliver(SERVER)
    harness.send(CLIENT, EndOfMessage())
    harness.deliver(SERVER)
    assert harness.conn[CLIENT].states[CLIENT] is MUST_CLOSE

    # A reuse attempt before the response arrives must fail on both sides
    # without entering ERROR: the connection stays usable.
    error = harness.start_next_cycle()
    assert type(error) is LocalProtocolError
    assert harness.conn[CLIENT].our_state is MUST_CLOSE
    assert harness.conn[SERVER].our_state is SEND_RESPONSE

    harness.send(SERVER, Response(status_code=200, headers=[]))
    harness.deliver(CLIENT)
    harness.send(SERVER, EndOfMessage())
    harness.deliver(CLIENT)
    harness.assert_wire_quiet()
    assert harness.conn[CLIENT].states == {CLIENT: MUST_CLOSE, SERVER: MUST_CLOSE}
    # Even with both cycles finished, keep-alive disabled forbids reuse.
    error = harness.start_next_cycle()
    assert type(error) is LocalProtocolError
    for role in (CLIENT, SERVER):
        assert harness.conn[role].our_state is MUST_CLOSE

    # Closing remains a legal local action.
    harness.close(CLIENT)
    harness.deliver_eof(SERVER)
    harness.close(SERVER)
    harness.deliver_eof(CLIENT)
    harness.assert_wire_quiet()
    assert harness.conn[CLIENT].states == {CLIENT: CLOSED, SERVER: CLOSED}
    assert_replay(harness)


def test_split_transfers_and_delayed_consumption() -> None:
    harness = PairHarness()
    harness.send(
        CLIENT,
        Request(
            method="POST",
            target="/",
            headers=[("Host", "x"), ("Transfer-Encoding", "chunked")],
        ),
    )
    # Deliver the request line one byte at a time; the server keeps asking
    # for more data until the headers complete.
    events = []
    while harness.has_pending_bytes(SERVER):
        events.extend(harness.deliver(SERVER, max_bytes=1))
    assert [type(event) for event in events] == [Request]
    assert harness.conn[SERVER].their_state is SEND_BODY

    # A chunk split across multiple reads must still parse as one Data event.
    harness.send(CLIENT, Data(data=b"hello", chunk_start=True, chunk_end=True))
    events = []
    while harness.has_pending_bytes(SERVER):
        events.extend(harness.deliver(SERVER, max_bytes=2))
    assert normalize_data_events(events) == [
        Data(data=b"hello", chunk_start=False, chunk_end=False)
    ]

    # Events stay unconsumed in the receive buffer until we drain.  Split the
    # terminating chunk so the server is forced to hold half of it in its
    # ReceiveBuffer while waiting for more bytes.
    harness.send(CLIENT, EndOfMessage())
    harness.deliver(SERVER, max_bytes=2)
    buffered = bytes(harness.conn[SERVER]._receive_buffer)
    assert buffered, "partial terminating chunk should be buffered, not lost"
    assert harness.conn[SERVER].their_state is SEND_BODY
    events = harness.deliver(SERVER)
    assert [type(event) for event in events] == [EndOfMessage]

    harness.send(
        SERVER, Response(status_code=200, headers=[("Content-Length", "0")])
    )
    harness.deliver(CLIENT)
    harness.send(SERVER, EndOfMessage())
    harness.deliver(CLIENT)
    harness.assert_wire_quiet()
    assert harness.start_next_cycle() is None
    assert_replay(harness)


def test_early_wire_close_during_request_body() -> None:
    harness = PairHarness()
    harness.send(
        CLIENT,
        Request(
            method="PUT",
            target="/",
            headers=[("Host", "x"), ("Content-Length", "5")],
        ),
    )
    harness.deliver(SERVER)
    harness.send(CLIENT, Data(data=b"ab"))
    harness.deliver(SERVER)

    # The client's transport disappears before the full body.
    harness.wire_drop(CLIENT)
    results = harness.deliver_eof(SERVER)
    error = results[0]
    assert type(error) is RemoteProtocolError
    assert "received 2 bytes" in str(error)
    assert "expected 5" in str(error)
    assert harness.conn[SERVER].their_state is ERROR
    # The server's own send path is still usable: it may answer with a 400.
    assert harness.conn[SERVER].our_state is SEND_RESPONSE
    local_error, _ = harness.send(
        SERVER, Response(status_code=400, headers=[("Connection", "close")])
    )
    assert local_error is None
    harness.send(SERVER, EndOfMessage())
    assert harness.conn[SERVER].our_state is MUST_CLOSE
    assert_replay(harness)


def test_early_connection_closed_is_rejected_while_body_open() -> None:
    harness = PairHarness()
    harness.send(
        CLIENT,
        Request(
            method="PUT",
            target="/",
            headers=[("Host", "x"), ("Content-Length", "5")],
        ),
    )
    harness.deliver(SERVER)
    # h11 must refuse to close from SEND_BODY and enter ERROR locally.
    error, _ = harness.close(CLIENT)
    assert type(error) is LocalProtocolError
    assert harness.conn[CLIENT].our_state is ERROR
    assert harness.conn[CLIENT].their_state is not ERROR
    assert_replay(harness)


# ---------------------------------------------------------------------------
# Illegal event sequences: error type, state, and continued-send behavior
# ---------------------------------------------------------------------------


def _expect_local_error(
    harness: PairHarness, role: Type[Sentinel], event: Event
) -> LocalProtocolError:
    error, _ = harness.send(role, event)
    assert type(error) is LocalProtocolError
    assert harness.conn[role].our_state is ERROR
    assert harness.conn[role].their_state is not ERROR
    return error


def test_illegal_second_request_while_body_open() -> None:
    harness = PairHarness()
    harness.send(
        CLIENT,
        Request(
            method="PUT",
            target="/",
            headers=[("Host", "x"), ("Content-Length", "1")],
        ),
    )
    harness.deliver(SERVER)
    error = _expect_local_error(
        harness,
        CLIENT,
        Request(method="GET", target="/", headers=[("Host", "x")]),
    )
    assert "Request" in str(error)
    # The server side is untouched and still accepts the body it was promised.
    assert harness.conn[SERVER].our_state is SEND_RESPONSE
    assert_replay(harness)


def test_illegal_data_from_idle_server() -> None:
    harness = PairHarness()
    error = _expect_local_error(harness, SERVER, Data(data=b"x"))
    assert "Data" in str(error)
    # The client is completely unaffected and can still make a request.
    assert harness.conn[CLIENT].our_state is IDLE
    harness.send(
        CLIENT,
        Request(method="GET", target="/", headers=[("Host", "x")]),
    )
    assert_replay(harness)


def test_illegal_eom_before_content_length_is_sent() -> None:
    harness = PairHarness()
    harness.send(
        CLIENT,
        Request(
            method="PUT",
            target="/",
            headers=[("Host", "x"), ("Content-Length", "5")],
        ),
    )
    harness.deliver(SERVER)
    error = _expect_local_error(harness, CLIENT, EndOfMessage())
    assert "Too little data" in str(error)
    assert harness.conn[CLIENT].our_state is ERROR
    assert_replay(harness)


def test_illegal_101_without_upgrade_proposal() -> None:
    harness = PairHarness()
    harness.send(
        CLIENT, Request(method="GET", target="/", headers=[("Host", "x")])
    )
    harness.deliver(SERVER)
    harness.send(CLIENT, EndOfMessage())
    harness.deliver(SERVER)
    error = _expect_local_error(
        harness,
        SERVER,
        InformationalResponse(status_code=101, headers=[("Upgrade", "a")]),
    )
    assert "pending proposal" in str(error)
    assert harness.conn[SERVER].our_state is ERROR
    assert_replay(harness)


def test_illegal_trailers_without_chunked() -> None:
    harness = PairHarness()
    harness.send(
        CLIENT,
        Request(
            method="PUT",
            target="/",
            headers=[("Host", "x"), ("Content-Length", "0")],
        ),
    )
    harness.deliver(SERVER)
    error = _expect_local_error(
        harness, CLIENT, EndOfMessage(headers=[("X-Trailer", "v")])
    )
    assert "trailers" in str(error)
    assert harness.conn[CLIENT].our_state is ERROR
    assert_replay(harness)


def test_remote_error_leaves_send_path_usable() -> None:
    # A truncated request body triggers RemoteProtocolError on the server;
    # its send side must keep working so it can emit a 400.
    harness = PairHarness()
    harness.send(
        CLIENT,
        Request(
            method="PUT",
            target="/",
            headers=[("Host", "x"), ("Content-Length", "5")],
        ),
    )
    harness.deliver(SERVER)
    harness.send(CLIENT, Data(data=b"ab"))
    harness.deliver(SERVER)
    harness.wire_drop(CLIENT)
    results = harness.deliver_eof(SERVER)
    assert any(type(item) is RemoteProtocolError for item in results)
    assert harness.conn[SERVER].their_state is ERROR
    assert harness.conn[SERVER].our_state is SEND_RESPONSE
    local_error, _ = harness.send(
        SERVER, Response(status_code=400, headers=[("Connection", "close")])
    )
    assert local_error is None
    assert harness.conn[SERVER].our_state is SEND_BODY
    assert_replay(harness)


# ---------------------------------------------------------------------------
# Fixed-seed randomized sequences with failure reduction
# ---------------------------------------------------------------------------

FIXED_SEEDS = (1, 3, 7, 13, 42, 101, 777, 2026, 31337)


def _oracle_mismatch(ops: Sequence[Op]) -> bool:
    try:
        run_sequence(list(ops))
    except AssertionError:
        # Every AssertionError raised by the harness is an oracle-vs-h11
        # disagreement; the exact wording does not matter to the reducer.
        return True
    return False


def _replayable_log_of_failure(ops: Sequence[Op]) -> str:
    # Run a known-bad sequence and capture the event log accumulated up to the
    # first failure, so a CI failure can be replayed from the test output.
    harness = PairHarness()
    try:
        for op in ops:
            op.run(harness)
    except AssertionError:
        pass
    return replayable_log(harness)


@pytest.mark.parametrize("seed", FIXED_SEEDS)
def test_fixed_seed_legal_sequences(seed: int) -> None:
    ops = generate_sequence(seed, length=40)
    assert len(ops) >= 5
    try:
        harness = run_sequence(ops)
    except AssertionError:
        # Failure reduction: keep the shortest subsequence that still fails.
        reduced = shrink_sequence(ops, _oracle_mismatch)
        pytest.fail(
            "reference/oracle mismatch after reduction\n"
            f"seed={seed} reduced from {len(ops)} to {len(reduced)} ops\n"
            f"replayable log:\n{_replayable_log_of_failure(reduced)}"
        )
    # Every oracle view must have stayed in lockstep with its Connection.
    for role in (CLIENT, SERVER):
        assert harness.conn[role].states == harness.oracle[role].states()
    assert_replay(harness)


def test_fixed_seeds_cover_all_required_features() -> None:
    # The fixed corpus must exercise every protocol feature this suite is
    # responsible for; if someone changes the RNG and the seeds stop
    # exploring, this test fails loudly instead of silently weakening.
    seen: set = set()
    for seed in FIXED_SEEDS:
        for op in generate_sequence(seed, length=40):
            if op.event_factory is None:
                seen.add(op.name)
                continue
            event = op.event_factory()
            seen.add(type(event).__name__)
            if isinstance(event, Request):
                seen.add(f"request:{event.method.decode()}")
                if b"100-continue" in [
                    value for _name, value in event.headers
                ]:
                    seen.add("expect-100")
                if b"close" in [value for _name, value in event.headers]:
                    seen.add("connection-close")
                if b"upgrade" in [name for name, _value in event.headers]:
                    seen.add("upgrade-header")
            if isinstance(event, (Response, InformationalResponse)):
                seen.add(f"status:{event.status_code}")
    required = {
        "request:GET",
        "request:HEAD",
        "request:POST",
        "request:CONNECT",
        "expect-100",
        "status:100",
        "status:204",
        "status:304",
        "connection-close",
        "Data",
        "EndOfMessage",
        "ConnectionClosed",
        "cycle",
        "deliver_eof",
        "deliver",
    }
    missing = required - seen
    assert not missing, f"fixed-seed corpus no longer covers: {sorted(missing)}"
    # Upgrade/101 is exercised explicitly by the dedicated scenario test.
    assert "upgrade-header" in seen


def test_shrinker_reduces_synthetic_failure() -> None:
    # Inject a known-bad op (a wire_drop with no following EOF drain is
    # harmless; instead construct a failing predicate) and confirm the
    # minimizer removes irrelevant ops while preserving the failure.
    ops = generate_sequence(seed=42, length=40)

    def predicate(candidate: Sequence[Op]) -> bool:
        # Fail only when the sequence includes the second op, giving the
        # shrinker something concrete to reduce toward.
        return len(candidate) >= 2 and candidate[1] is ops[1]

    reduced = shrink_sequence(ops, predicate)
    assert len(reduced) == 2
    assert reduced[0] is ops[0]
    assert reduced[1] is ops[1]
    assert predicate(reduced)


def test_failure_log_is_json_replayable() -> None:
    # Simulate the failure-reporting path used by the fuzzer: dump a scenario
    # log, round-trip it through JSON, and replay it on fresh objects.
    harness = PairHarness()
    harness.send(
        CLIENT,
        Request(
            method="PUT",
            target="/",
            headers=[("Host", "x"), ("Content-Length", "5")],
        ),
    )
    harness.deliver(SERVER)
    harness.wire_drop(CLIENT)
    harness.deliver_eof(SERVER)
    encoded = replayable_log(harness)
    decoded = json.loads(encoded)
    replayed = PairHarness()
    replayed.replay(decoded)
    assert replayed.conn[SERVER].their_state is ERROR


# ---------------------------------------------------------------------------
# Mutation self-check: deliberately reset state early and prove the model
# catches it.
# ---------------------------------------------------------------------------


def _eager_reset_start_next_cycle(self: Any) -> None:
    # Mutant: reset to IDLE even when the conversation is not complete.
    self._cstate.states = {CLIENT: IDLE, SERVER: IDLE}
    self._request_method = None


def test_early_reset_mutation_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    from .._connection import Connection

    monkeypatch.setattr(Connection, "start_next_cycle", _eager_reset_start_next_cycle)

    harness = PairHarness()
    harness.send(
        CLIENT, Request(method="GET", target="/", headers=[("Host", "x")])
    )
    harness.deliver(SERVER)
    harness.send(CLIENT, EndOfMessage())
    harness.deliver(SERVER)
    # The server has not answered yet, so reuse must be refused.
    with pytest.raises(AssertionError) as excinfo:
        harness.start_next_cycle()
    assert "not reusable" in str(excinfo.value)


def test_eager_reset_mutation_is_caught_by_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .._connection import Connection

    monkeypatch.setattr(
        Connection, "start_next_cycle", _eager_reset_start_next_cycle
    )

    # Running the fixed-seed corpus under the mutant must fail on at least one
    # seed.  Failures can surface either while building the sequence (the
    # mutant breaks h11's own invariants) or as an oracle mismatch during
    # replay; both mean the model has caught the premature reset.
    detected = 0
    for seed in FIXED_SEEDS:
        try:
            ops = generate_sequence(seed, length=40)
        except AssertionError:
            detected += 1
            continue
        if _oracle_mismatch(ops):
            reduced = shrink_sequence(ops, _oracle_mismatch)
            assert reduced
            assert _oracle_mismatch(reduced)
            assert len(reduced) <= len(ops)
            detected += 1
    assert detected >= 1, "the model failed to notice the early-reset mutant"


def test_unmutated_fuzz_corpus_matches_oracle() -> None:
    # Control experiment: without any mutant the same corpus is clean, so the
    # detection above cannot be explained by an over-eager oracle.
    for seed in FIXED_SEEDS:
        assert not _oracle_mismatch(generate_sequence(seed, length=40))


def test_reference_oracle_is_independent_of_h11_state_tables() -> None:
    # The oracle's rules must be written out by hand.  Guard against someone
    # shortcutting that by importing h11's transition tables into the
    # harness module at runtime.
    import h11.tests._pair_harness as pair_harness

    assert not hasattr(pair_harness, "EVENT_TRIGGERED_TRANSITIONS")
    assert not hasattr(pair_harness, "STATE_TRIGGERED_TRANSITIONS")
    import h11._state as h11_state

    # The local, hand-written legality table is what tests actually consult,
    # and it must not be the same object as h11's implementation table.
    assert pair_harness._LEGAL_SEND_EVENTS is not getattr(
        h11_state, "EVENT_TRIGGERED_TRANSITIONS", None
    )
    assert pair_harness._LEGAL_SEND_EVENTS[CLIENT][IDLE] == (Request,)
    assert pair_harness._LEGAL_SEND_EVENTS[SERVER][SEND_RESPONSE] == (
        InformationalResponse,
        Response,
    )
