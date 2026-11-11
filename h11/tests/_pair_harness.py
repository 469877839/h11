# Dual-ended test harness for h11.
#
# Two real h11.Connection objects (one CLIENT, one SERVER) talk to each other
# over in-memory byte queues.  The harness deliberately stays at the level of
# the public h11 API so that it exercises the same code paths a real network
# user would, while the ReferenceOracle below encodes what HTTP/1.1 (RFC
# 7230/7231) says *should* happen without ever reading h11's own transition
# tables.
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Type

from .._connection import Connection, NEED_DATA, PAUSED
from .._events import (
    ConnectionClosed,
    Data,
    EndOfMessage,
    Event,
    InformationalResponse,
    Request,
    Response,
)
from .._headers import get_comma_header
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
from .._util import LocalProtocolError, ProtocolError, RemoteProtocolError, Sentinel


# ---------------------------------------------------------------------------
# Replayable operation log
# ---------------------------------------------------------------------------
#
# Every step the harness takes is appended to a plain-data log so that any
# failing sequence can be dumped and replayed from scratch.  Entries are
# restricted to JSON-friendly builtins plus h11 event objects (which are
# frozen dataclasses and can themselves be reconstructed by the test code).

_SENTINEL_NAME = {
    CLIENT: "CLIENT",
    SERVER: "SERVER",
}


def _event_to_dict(event: Event) -> Dict[str, Any]:
    data: Dict[str, Any] = {"type": type(event).__name__}
    for name in type(event).__slots__:
        if name == "headers":
            data[name] = [
                (bytes(raw_name).decode("latin-1"), bytes(value).decode("latin-1"))
                for raw_name, _norm, value in event.headers._full_items
            ]
        else:
            value = getattr(event, name)
            if isinstance(value, (bytes, bytearray)):
                value = bytes(value).decode("latin-1")
            data[name] = value
    return data


def event_from_dict(data: Dict[str, Any]) -> Event:
    kind = data["type"]
    kwargs = {key: value for key, value in data.items() if key != "type"}
    for str_field in ("method", "target", "http_version", "reason"):
        if str_field in kwargs and isinstance(kwargs[str_field], str):
            kwargs[str_field] = kwargs[str_field].encode("latin-1")
    if "data" in kwargs and isinstance(kwargs["data"], str):
        kwargs["data"] = kwargs["data"].encode("latin-1")
    if kind == "Request":
        return Request(**kwargs)
    if kind == "InformationalResponse":
        return InformationalResponse(**kwargs)
    if kind == "Response":
        return Response(**kwargs)
    if kind == "Data":
        return Data(**kwargs)
    if kind == "EndOfMessage":
        return EndOfMessage(**kwargs)
    if kind == "ConnectionClosed":
        return ConnectionClosed()
    raise ValueError(f"unknown event type in log: {kind!r}")


@dataclass
class LogEntry:
    """One logged harness operation, plus what actually happened.

    ``outcome`` is one of:
      - ("ok",): the operation succeeded
      - ("local", message): LocalProtocolError while sending
      - ("remote", message): RemoteProtocolError while receiving
      - ("runtime", message): RuntimeError (e.g. data after EOF on the wire)
      - ("error", name): any other exception, by class name
    """

    op: str
    role: Optional[str]
    payload: Dict[str, Any]
    outcome: Tuple[Any, ...]
    states_after: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "role": self.role,
            "payload": self.payload,
            "outcome": list(self.outcome),
            "states_after": self.states_after,
        }


# ---------------------------------------------------------------------------
# In-memory byte queues standing in for the TCP connection
# ---------------------------------------------------------------------------
#
# There is one queue per direction.  Bytes are kept as immutable segments so
# the harness can simulate partial writes / segment splitting without losing
# the original boundaries chosen by the test.  An explicit EOF marker models
# one side shutting its half of the connection; any later write to that
# direction is rejected, mirroring the RuntimeError h11 raises when data
# arrives after receive_data(b"").


@dataclass
class _Segment:
    data: bytes
    eof: bool = False


@dataclass
class _Direction:
    segments: Deque[_Segment] = field(default_factory=deque)
    closed: bool = False


class Wire:
    """Pair of byte queues, one for each direction of the connection."""

    def __init__(self) -> None:
        self._queues: Dict[Type[Sentinel], _Direction] = {
            CLIENT: _Direction(),
            SERVER: _Direction(),
        }

    def push(self, sender: Type[Sentinel], data: bytes) -> None:
        direction = self._queues[sender]
        if direction.closed:
            raise RuntimeError("write after close on the in-memory wire")
        if data:
            direction.segments.append(_Segment(data))

    def close(self, sender: Type[Sentinel]) -> None:
        direction = self._queues[sender]
        if not direction.closed:
            direction.closed = True
            direction.segments.append(_Segment(b"", eof=True))

    # Pull up to ``max_bytes`` from the queue addressed to ``receiver``; when
    # max_bytes is None the whole next segment (and EOF marker) is delivered.
    def pull(
        self, receiver: Type[Sentinel], max_bytes: Optional[int]
    ) -> Optional[bytes]:
        direction = self._queues[
            SERVER if receiver is CLIENT else CLIENT
        ]
        if not direction.segments:
            return None
        segment = direction.segments[0]
        if segment.eof:
            direction.segments.popleft()
            return b""
        if max_bytes is None or max_bytes >= len(segment.data):
            direction.segments.popleft()
            return segment.data
        head, tail = segment.data[:max_bytes], segment.data[max_bytes:]
        segment.data = tail
        return head

    def is_quiet(self) -> bool:
        return all(not queue.segments for queue in self._queues.values())


# ---------------------------------------------------------------------------
# Independent reference oracle
# ---------------------------------------------------------------------------
#
# This class is the *spec* side of the tests.  It deliberately does not import
# or consult h11's EVENT_TRIGGERED_TRANSITIONS / STATE_TRIGGERED_TRANSITIONS
# tables; the rules below are written out by hand from the HTTP/1.1 state
# semantics documented in RFC 7230/7231 and h11's public docs.  The oracle is
# role symmetric: the PairHarness keeps one instance per endpoint, and each
# observed operation is applied once to the endpoint that observed it.
#
# Phases used here mirror the names of h11's public state constants, but the
# transition rules are declared locally below (see _LEGAL_SEND_EVENTS and
# _apply_*), so a change to h11's table cannot silently change the oracle.

# What an endpoint is allowed to *send* while playing each phase.  Connection
# closure has extra rules that depend on the peer's phase, so it is handled in
# legal_to_send() instead of appearing here.
_LEGAL_SEND_EVENTS: Dict[Type[Sentinel], Dict[Type[Sentinel], Tuple[type, ...]]] = {
    CLIENT: {
        IDLE: (Request,),
        SEND_BODY: (Data, EndOfMessage),
        DONE: (),
        MUST_CLOSE: (),
        CLOSED: (),
        MIGHT_SWITCH_PROTOCOL: (),
        SWITCHED_PROTOCOL: (),
        ERROR: (),
    },
    SERVER: {
        IDLE: (Response,),
        SEND_RESPONSE: (InformationalResponse, Response),
        SEND_BODY: (Data, EndOfMessage),
        DONE: (),
        MUST_CLOSE: (),
        CLOSED: (),
        SWITCHED_PROTOCOL: (),
        ERROR: (),
    },
}

# A ConnectionClosed is a legal local action only from a phase where the
# request/response side of the conversation has fully completed.
_CLOSEABLE_PHASES = (DONE, MUST_CLOSE, CLOSED)

_SWITCH_UPGRADE = "upgrade"
_SWITCH_CONNECT = "connect"


class ReferenceOracle:
    """Hand-written HTTP/1.1 legality and state oracle.

    One instance tracks one joint client/server conversation, including the
    keep-alive and protocol-switch sub-machines.  Its methods return either
    ``None`` (operation legal) or an ``(error_kind, reason)`` tuple describing
    the failure the real h11 endpoint must report.
    """

    def __init__(self) -> None:
        self.client_phase: Type[Sentinel] = IDLE
        self.server_phase: Type[Sentinel] = IDLE
        self.keep_alive = True
        self.pending_switches: Set[str] = set()
        self.waiting_for_100_continue = False
        # Framing bookkeeping for the message currently being written.
        self.request_method: Optional[bytes] = None
        self.client_body_remaining: Optional[int] = None
        self.client_body_chunked = False
        self.response_body_remaining: Optional[int] = None
        self.response_body_chunked = False
        self.errors: Dict[Type[Sentinel], bool] = {CLIENT: False, SERVER: False}

    # -- introspection ----------------------------------------------------

    def phase(self, role: Type[Sentinel]) -> Type[Sentinel]:
        if role is CLIENT:
            return self.client_phase
        return self.server_phase

    def states(self) -> Dict[Type[Sentinel], Type[Sentinel]]:
        return {CLIENT: self.client_phase, SERVER: self.server_phase}

    # -- wire-level helpers used by both request and response handling ----

    @staticmethod
    def _event_keeps_alive(event: Any) -> bool:
        connection = get_comma_header(event.headers, b"connection")
        if b"close" in connection:
            return False
        if getattr(event, "http_version", b"1.1") < b"1.1":
            return False
        return True

    def _fire_state_transitions(self) -> None:
        # Same fixed-point logic as the documented h11 state machine, written
        # out independently here.
        while True:
            before = (self.client_phase, self.server_phase, self.keep_alive)

            if self.pending_switches and self.client_phase is DONE:
                self.client_phase = MIGHT_SWITCH_PROTOCOL
            if not self.pending_switches and self.client_phase is MIGHT_SWITCH_PROTOCOL:
                self.client_phase = DONE

            if not self.keep_alive:
                if self.client_phase is DONE:
                    self.client_phase = MUST_CLOSE
                if self.server_phase is DONE:
                    self.server_phase = MUST_CLOSE

            joint = (self.client_phase, self.server_phase)
            if joint == (MIGHT_SWITCH_PROTOCOL, SWITCHED_PROTOCOL):
                self.client_phase = SWITCHED_PROTOCOL
            elif joint in ((CLOSED, DONE), (CLOSED, IDLE)):
                self.server_phase = MUST_CLOSE
            elif joint in ((DONE, CLOSED), (IDLE, CLOSED)):
                self.client_phase = MUST_CLOSE
            elif joint == (ERROR, DONE):
                self.server_phase = MUST_CLOSE
            elif joint == (DONE, ERROR):
                self.client_phase = MUST_CLOSE

            after = (self.client_phase, self.server_phase, self.keep_alive)
            if after == before:
                return

    # -- local send legality ----------------------------------------------

    def legal_to_send(
        self, role: Type[Sentinel], event: Event
    ) -> Optional[Tuple[str, str]]:
        if self.errors[role]:
            return ("local", "endpoint already in ERROR")
        phase = self.phase(role)
        if type(event) is ConnectionClosed:
            if phase not in _CLOSEABLE_PHASES:
                return ("local", f"cannot close while in {phase.__name__}")
            return None
        allowed = _LEGAL_SEND_EVENTS[role].get(phase, ())
        if type(event) not in allowed:
            return (
                "local",
                f"{type(event).__name__} not legal for "
                f"{role.__name__} in {phase.__name__}",
            )
        framing_error = self._framing_error(role, event)
        if framing_error is not None:
            return ("local", framing_error)
        if role is SERVER:
            switch_error = self._server_switch_error(event)
            if switch_error is not None:
                return ("local", switch_error)
        return None

    def _framing_error(self, role: Type[Sentinel], event: Event) -> Optional[str]:
        if type(event) is Data:
            if role is CLIENT:
                if self.client_body_remaining is not None:
                    if len(event.data) > self.client_body_remaining:
                        return "too much request body data for Content-Length"
                return None
            if self.response_body_remaining is not None:
                if len(event.data) > self.response_body_remaining:
                    return "too much response body data for Content-Length"
            return None
        if type(event) is EndOfMessage:
            if role is CLIENT and self.client_body_remaining not in (None, 0):
                return "request ended before declared Content-Length was sent"
            if role is SERVER and self.response_body_remaining not in (None, 0):
                return "response ended before declared Content-Length was sent"
            if event.headers:
                chunked = (
                    self.client_body_chunked
                    if role is CLIENT
                    else self.response_body_chunked
                )
                if not chunked:
                    return "trailers require Transfer-Encoding: chunked"
        return None

    def _server_switch_error(self, event: Event) -> Optional[str]:
        if type(event) is InformationalResponse and event.status_code == 101:
            if _SWITCH_UPGRADE not in self.pending_switches:
                return "101 without a pending Upgrade proposal"
        if type(event) is Response and 200 <= event.status_code < 300:
            if _SWITCH_CONNECT in self.pending_switches:
                return None
        return None

    # -- applying locally legal operations to the joint reference state ---

    def apply_local_send(self, role: Type[Sentinel], event: Event) -> None:
        if type(event) is ConnectionClosed:
            if role is CLIENT:
                self.client_phase = CLOSED
            else:
                self.server_phase = CLOSED
            self._fire_state_transitions()
            return

        if role is CLIENT:
            self._apply_client_event(event)
        else:
            self._apply_server_event(event)
        self._fire_state_transitions()

    def _apply_client_event(self, event: Event) -> None:
        if type(event) is Request:
            self.request_method = event.method
            if event.method == b"CONNECT":
                self.pending_switches.add(_SWITCH_CONNECT)
            if get_comma_header(event.headers, b"upgrade"):
                self.pending_switches.add(_SWITCH_UPGRADE)
            if b"100-continue" in get_comma_header(
                event.headers, b"expect"
            ):
                self.waiting_for_100_continue = True
            if not self._event_keeps_alive(event):
                self.keep_alive = False
            transfer = get_comma_header(event.headers, b"transfer-encoding")
            lengths = get_comma_header(event.headers, b"content-length")
            if transfer:
                self.client_body_chunked = True
                self.client_body_remaining = None
            elif lengths:
                self.client_body_remaining = int(lengths[0])
            else:
                self.client_body_remaining = 0
            self.client_phase = SEND_BODY
            self.server_phase = SEND_RESPONSE
        elif type(event) is Data:
            self.waiting_for_100_continue = False
            if self.client_body_remaining is not None:
                self.client_body_remaining -= len(event.data)
        elif type(event) is EndOfMessage:
            self.waiting_for_100_continue = False
            self.client_body_remaining = None
            self.client_body_chunked = False
            self.client_phase = DONE

    def _response_body_length(self, event: Response) -> Optional[int]:
        # RFC 7230: HEAD responses, 204/304 and successful CONNECT responses
        # always have an empty body regardless of their headers.
        if (
            event.status_code in (204, 304)
            or self.request_method == b"HEAD"
            or (
                self.request_method == b"CONNECT"
                and 200 <= event.status_code < 300
            )
        ):
            return 0
        transfer = get_comma_header(event.headers, b"transfer-encoding")
        if transfer:
            self.response_body_chunked = True
            return None
        lengths = get_comma_header(event.headers, b"content-length")
        if lengths:
            return int(lengths[0])
        return 0

    def _apply_server_event(self, event: Event) -> None:
        if type(event) is InformationalResponse:
            self.waiting_for_100_continue = False
            if event.status_code == 101 and _SWITCH_UPGRADE in self.pending_switches:
                self.server_phase = SWITCHED_PROTOCOL
            # Ordinary 1xx responses leave the server in SEND_RESPONSE.
            return

        if type(event) is Response:
            self.waiting_for_100_continue = False
            switch_connect = (
                _SWITCH_CONNECT in self.pending_switches
                and 200 <= event.status_code < 300
            )
            # A non-switch final response clears every outstanding proposal.
            if not switch_connect:
                self.pending_switches = set()
            if not self._event_keeps_alive(event):
                self.keep_alive = False
            if switch_connect:
                self.server_phase = SWITCHED_PROTOCOL
                self.response_body_remaining = 0
                return
            self.response_body_remaining = self._response_body_length(event)
            self.server_phase = SEND_BODY
            return

        if type(event) is Data:
            if self.response_body_remaining is not None:
                self.response_body_remaining -= len(event.data)
            return

        if type(event) is EndOfMessage:
            self.response_body_remaining = None
            self.response_body_chunked = False
            self.server_phase = DONE
            return

    def apply_remote_event(self, sender: Type[Sentinel], event: Event) -> None:
        # Apply an event that arrived from the peer.  The transition rules
        # are the same as for a local send -- the reference model is
        # role-symmetric, like the protocol itself.
        self.apply_local_send(sender, event)

    # -- remote failures ----------------------------------------------------

    def note_remote_error(self, receiver: Type[Sentinel]) -> None:
        # ``receiver`` is the endpoint whose *peer* misbehaved; in the joint
        # reference state the peer's side therefore enters ERROR.
        if receiver is CLIENT:
            self.errors[SERVER] = True
            self.server_phase = ERROR
        else:
            self.errors[CLIENT] = True
            self.client_phase = ERROR
        self._fire_state_transitions()

    def note_local_error(self, role: Type[Sentinel]) -> None:
        self.errors[role] = True
        if role is CLIENT:
            self.client_phase = ERROR
        else:
            self.server_phase = ERROR
        self._fire_state_transitions()

    def note_wire_eof(self, receiver: Type[Sentinel]) -> None:
        if receiver is CLIENT:
            self.server_phase = CLOSED
        else:
            self.client_phase = CLOSED
        self._fire_state_transitions()

    # -- cycle reuse --------------------------------------------------------

    def can_start_next_cycle(self) -> bool:
        return self.client_phase is DONE and self.server_phase is DONE

    def start_next_cycle(self) -> None:
        if not self.can_start_next_cycle():
            raise LocalProtocolError(
                f"reference model not reusable in {self.states()}"
            )
        assert self.keep_alive
        assert not self.pending_switches
        self.client_phase = IDLE
        self.server_phase = IDLE
        self.request_method = None


# ---------------------------------------------------------------------------
# The dual-ended harness
# ---------------------------------------------------------------------------
#
# One PairHarness owns two real h11.Connection objects, one Wire, and two
# ReferenceOracle instances.  Each oracle models what one endpoint should
# believe about the joint conversation; the two endpoints' views legitimately
# diverge while bytes are still sitting in the wire (for example, the server
# learns that a Content-Length: 0 request is finished as soon as it parses the
# headers, while the client only reaches DONE when it sends EndOfMessage).
# The harness asserts that each real Connection's view matches its own oracle
# after every operation, and that the two views converge once the wire has
# been drained.


class PairHarness:
    """Drive two real :class:`Connection` objects through one Wire."""

    def __init__(self) -> None:
        self.wire = Wire()
        self.conn: Dict[Type[Sentinel], Connection] = {
            CLIENT: Connection(CLIENT),
            SERVER: Connection(SERVER),
        }
        self.oracle: Dict[Type[Sentinel], ReferenceOracle] = {
            CLIENT: ReferenceOracle(),
            SERVER: ReferenceOracle(),
        }
        self.log: List[LogEntry] = []

    # -- small utilities ---------------------------------------------------

    @staticmethod
    def _role_name(role: Optional[Type[Sentinel]]) -> Optional[str]:
        if role is None:
            return None
        return _SENTINEL_NAME[role]

    @staticmethod
    def _peer(role: Type[Sentinel]) -> Type[Sentinel]:
        return SERVER if role is CLIENT else CLIENT

    def _snapshot_states(self) -> Dict[str, Any]:
        return {
            "client_view": {
                role.__name__: state.__name__
                for role, state in self.conn[CLIENT].states.items()
            },
            "server_view": {
                role.__name__: state.__name__
                for role, state in self.conn[SERVER].states.items()
            },
        }

    def _record(
        self,
        op: str,
        role: Optional[Type[Sentinel]],
        payload: Dict[str, Any],
        outcome: Tuple[Any, ...],
    ) -> None:
        self.log.append(
            LogEntry(
                op=op,
                role=self._role_name(role),
                payload=payload,
                outcome=outcome,
                states_after=self._snapshot_states(),
            )
        )

    # -- sending ------------------------------------------------------------

    def send(
        self, role: Type[Sentinel], event: Event
    ) -> Tuple[Optional[ProtocolError], Optional[bytes]]:
        """Send an event from one side, checking it against the oracle first.

        Returns ``(error, data)``: ``error`` is the LocalProtocolError when
        the send was illegal (data is then None), otherwise None and the
        serialized bytes (also None for a successful ConnectionClosed).
        """
        expected = self.oracle[role].legal_to_send(role, event)
        payload = {"event": _event_to_dict(event)}
        try:
            data = self.conn[role].send(event)
        except ProtocolError as exc:
            return self._record_send_failure(role, payload, expected, exc)
        if expected is not None:
            self._record(
                "send",
                role,
                payload,
                ("oracle_mismatch", f"h11 accepted: {expected[1]}"),
            )
            raise AssertionError(
                f"oracle rejected {type(event).__name__} for "
                f"{self._role_name(role)} ({expected[1]}) but h11 accepted it"
            )
        self.oracle[role].apply_local_send(role, event)
        if type(event) is ConnectionClosed:
            self.wire.close(role)
        else:
            assert data is not None
            self.wire.push(role, data)
        self._record("send", role, payload, ("ok",))
        self._assert_view_matches(role)
        return None, data

    def _record_send_failure(
        self,
        role: Type[Sentinel],
        payload: Dict[str, Any],
        expected: Optional[Tuple[str, str]],
        exc: ProtocolError,
    ) -> Tuple[ProtocolError, None]:
        if expected is None:
            self._record(
                "send",
                role,
                payload,
                ("oracle_mismatch", str(exc)),
            )
            raise AssertionError(
                f"oracle allowed {payload['event']['type']} but h11 raised: {exc}"
            )
        if type(exc) is not LocalProtocolError:
            raise AssertionError(
                f"expected LocalProtocolError for local send, got "
                f"{type(exc).__name__}: {exc}"
            )
        self.oracle[role].note_local_error(role)
        self._record("send", role, payload, ("local", str(exc)))
        self._assert_view_matches(role)
        self._assert_send_is_dead(role)
        return exc, None

    # -- receiving ----------------------------------------------------------

    def deliver(
        self, receiver: Type[Sentinel], max_bytes: Optional[int] = None
    ) -> List[Any]:
        """Move queued bytes into one endpoint and parse what we can.

        ``max_bytes`` limits each read so a single write can be split into
        arbitrarily small chunks.  Returned entries are h11 Event objects,
        with one possible RemoteProtocolError appended if parsing failed.
        """
        results: List[Any] = []
        while True:
            chunk = self.wire.pull(receiver, max_bytes)
            if chunk is None:
                break
            payload: Dict[str, Any] = {
                "max_bytes": max_bytes,
                "chunk": chunk.decode("latin-1"),
            }
            self.conn[receiver].receive_data(chunk)
            if not chunk:
                # An empty chunk is the wire's EOF marker, regardless of
                # whether h11 can turn it into a ConnectionClosed event right
                # now (it cannot while PAUSED).
                self.oracle[receiver].note_wire_eof(receiver)
            self._record("receive_data", receiver, payload, ("ok",))
            if max_bytes is not None:
                # Force the consumer to parse before handing over more bytes.
                break
        results.extend(self._drain_events(receiver))
        return results

    def has_pending_bytes(self, receiver: Type[Sentinel]) -> bool:
        """True if the wire still holds bytes addressed to ``receiver``."""
        return bool(self.wire._queues[self._peer(receiver)].segments)

    def wire_drop(self, sender: Type[Sentinel]) -> None:
        """Close one half of the transport without an h11 event.

        Models a peer that disappears mid-message (TCP FIN while a body is
        incomplete): the sender's own h11 object never sees the event, while
        the receiver gets a bare EOF from the wire.
        """
        self.wire.close(sender)
        self._record("wire_drop", sender, {}, ("ok",))

    def deliver_eof(self, receiver: Type[Sentinel]) -> List[Any]:
        """Deliver the EOF marker (``b""``) once it reaches the queue head."""
        direction = self.wire._queues[self._peer(receiver)]
        if not direction.segments or not direction.segments[0].eof:
            raise RuntimeError("EOF is not at the head of the queue yet")
        direction.segments.popleft()
        self.conn[receiver].receive_data(b"")
        self.oracle[receiver].note_wire_eof(receiver)
        self._record(
            "receive_data",
            receiver,
            {"max_bytes": None, "chunk": ""},
            ("ok", "eof"),
        )
        return self._drain_events(receiver)

    def drain(self, receiver: Type[Sentinel]) -> List[Any]:
        """Parse whatever is currently buffered without touching the wire."""
        return self._drain_events(receiver)

    def _drain_events(self, receiver: Type[Sentinel]) -> List[Any]:
        results: List[Any] = []
        while True:
            try:
                event = self.conn[receiver].next_event()
            except RemoteProtocolError as exc:
                self.oracle[receiver].note_remote_error(receiver)
                self._record("next_event", receiver, {}, ("remote", str(exc)))
                self._assert_view_matches(receiver)
                self._assert_receive_is_dead(receiver)
                results.append(exc)
                return results
            if event in (NEED_DATA, PAUSED):
                self._record(
                    "next_event",
                    receiver,
                    {},
                    ("ok", type(event).__name__),
                )
                break
            assert isinstance(event, Event)
            sender = self._peer(receiver)
            if type(event) is not ConnectionClosed:
                self.oracle[receiver].apply_remote_event(sender, event)
            else:
                self.oracle[receiver].note_wire_eof(receiver)
            self._record(
                "next_event",
                receiver,
                {"event": _event_to_dict(event)},
                ("ok",),
            )
            results.append(event)
            if type(event) is ConnectionClosed:
                break
        self._assert_view_matches(receiver)
        return results

    # -- connection lifecycle ----------------------------------------------

    def close(self, role: Type[Sentinel]) -> Tuple[Optional[ProtocolError], None]:
        """Send ConnectionClosed from one side; the wire carries the EOF."""
        error, _ = self.send(role, ConnectionClosed())
        return error, None

    def start_next_cycle(self) -> Optional[ProtocolError]:
        """Attempt to reuse the connection on both endpoints.

        A failed attempt must raise LocalProtocolError on *both* endpoints,
        leave neither endpoint in ERROR, and leave both usable for a legal
        send once the missing half-cycle arrives.  Returns the error, or None
        on success.
        """
        errors: List[Optional[LocalProtocolError]] = []
        for role in (CLIENT, SERVER):
            try:
                self.conn[role].start_next_cycle()
            except LocalProtocolError as exc:
                errors.append(exc)
            else:
                errors.append(None)
        client_error, server_error = errors
        expected_legal = all(
            oracle.states() == {CLIENT: DONE, SERVER: DONE}
            and oracle.keep_alive
            and not oracle.pending_switches
            for oracle in self.oracle.values()
        )
        if expected_legal:
            if client_error is not None or server_error is not None:
                raise AssertionError(
                    f"oracle says cycle is reusable but h11 raised: "
                    f"{client_error!r} {server_error!r}"
                )
            for oracle in self.oracle.values():
                oracle.start_next_cycle()
            self._record("start_next_cycle", None, {}, ("ok",))
        else:
            if client_error is None or server_error is None:
                raise AssertionError(
                    f"oracle says cycle is not reusable in "
                    f"{self.oracle[CLIENT].states()} but h11 accepted it"
                )
            self._record(
                "start_next_cycle", None, {}, ("local", str(client_error))
            )
            for role in (CLIENT, SERVER):
                assert self.conn[role].our_state is not ERROR
        self._assert_view_matches(CLIENT)
        self._assert_view_matches(SERVER)
        return client_error

    # -- consistency checks -------------------------------------------------

    def _assert_view_matches(self, role: Type[Sentinel]) -> None:
        observed = self.conn[role].states
        expected = self.oracle[role].states()
        if observed != expected:
            raise AssertionError(
                f"{self._role_name(role)} view {observed} does not match "
                f"its reference oracle {expected}"
            )

    def assert_wire_quiet(self) -> None:
        """Both endpoints have consumed everything and agree with each other."""
        assert self.wire.is_quiet(), "wire still has unread bytes"
        client_view = self.conn[CLIENT].states
        server_view = self.conn[SERVER].states
        if client_view != server_view:
            raise AssertionError(
                f"endpoints disagree after full delivery: client sees "
                f"{client_view}, server sees {server_view}"
            )
        for role in (CLIENT, SERVER):
            self._assert_view_matches(role)

    def _assert_send_is_dead(self, role: Type[Sentinel]) -> None:
        assert self.conn[role].our_state is ERROR
        try:
            self.conn[role].send(Data(data=b"x"))
        except LocalProtocolError:
            pass
        else:  # pragma: no cover - would indicate an h11 bug
            raise AssertionError("send still accepted after LocalProtocolError")
        assert self.conn[role].our_state is ERROR

    def _assert_receive_is_dead(self, receiver: Type[Sentinel]) -> None:
        assert self.conn[receiver].their_state is ERROR
        try:
            self.conn[receiver].next_event()
        except RemoteProtocolError:
            pass
        else:  # pragma: no cover - would indicate an h11 bug
            raise AssertionError("next_event still works after RemoteProtocolError")
        assert self.conn[receiver].their_state is ERROR

    # -- replay -------------------------------------------------------------

    def replay_log(self) -> List[Dict[str, Any]]:
        return [entry.to_dict() for entry in self.log]

    def replay(self, entries: List[Dict[str, Any]]) -> None:
        """Re-execute a dumped log on a fresh harness."""
        for entry in entries:
            role: Optional[Type[Sentinel]] = None
            if entry["role"] is not None:
                role = {"CLIENT": CLIENT, "SERVER": SERVER}[entry["role"]]
            op = entry["op"]
            if op == "send":
                assert role is not None
                self.send(role, event_from_dict(entry["payload"]["event"]))
            elif op == "receive_data":
                assert role is not None
                self.deliver(role, entry["payload"]["max_bytes"])
            elif op == "next_event":
                assert role is not None
                self.drain(role)
            elif op == "start_next_cycle":
                self.start_next_cycle()
            elif op == "wire_drop":
                assert role is not None
                self.wire_drop(role)
            else:  # pragma: no cover
                raise ValueError(f"unknown op in log: {op!r}")
