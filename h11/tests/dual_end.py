"""A two-end, byte-queue test model for h11.

The model deliberately keeps three things separate:

1. The two real ``Connection`` objects (one CLIENT, one SERVER).
2. An in-memory wire, so bytes can be queued, split, delivered late, or
   abandoned when one end closes early.
3. An event mailbox, so parsing an event and inspecting that event are
   separate operations.

This is useful because the two endpoints do not share a state table at
runtime.  After a client sends a request, the client immediately believes
``CLIENT: SEND_BODY`` / ``SERVER: SEND_RESPONSE``, while the server still
believes both sides are IDLE until it actually consumes the bytes.  The
same split exists while a response is in flight, when one end has reset for
the next cycle, and when one side has proposed or accepted a protocol
switch.

Reference constraints
=====================

``ReferenceView`` is a small, hand-written REST/lifecycle model used as the
oracle.  Its transition tables are literal, independently written maps;
they are not imported from, generated from, or patched into ``h11._state``,
and the model does not use h11's readers or writers.  The rules below are
written from the HTTP/1.1 message lifecycle (RFC 7230/7231/7232), not from
the implementation under test:

* A cycle has one client request head and one server final response head.
  The client may send ``Request`` only from ``IDLE``.  A request body is
  ``Data``* followed by ``EndOfMessage``.
* Once a request head has been seen by a server, that server is in
  ``SEND_RESPONSE``.  It may send zero or more informational responses
  (``1xx``) before exactly one final response.  A final response moves the
  server to its response body (``SEND_BODY``).
* A response body is zero-length for HEAD, for 204, and for 304.  A
  successful CONNECT response (2xx) also starts a tunnel, so its HTTP body
  is zero-length.
* ``101`` accepts a pending Upgrade proposal.  A 2xx response to CONNECT
  accepts a pending CONNECT proposal.  While a proposal is pending after
  the client finishes its request, the client is in
  ``MIGHT_SWITCH_PROTOCOL``.  Accepting a proposal puts both sides in
  ``SWITCHED_PROTOCOL``; after that no more h11 HTTP events are allowed on
  that connection.
* A denied proposal (any non-accepting final response) clears the
  proposal.  The client returns to ``DONE`` and a normal response body
  follows.
* A request or response carrying ``Connection: close``, or using
  HTTP/1.0, disables keep-alive permanently for that connection.  When
  each side reaches ``DONE`` it becomes ``MUST_CLOSE`` instead.
* ``Expect: 100-continue`` sets a flow-control flag on both views once the
  request is known.  Any informational or final response, or the client
  starting/finishing its body, clears it.  This is h11's documented
  behavior: a 1xx other than 100 still clears the wait.
* ``start_next_cycle`` is legal only when both roles are simultaneously
  ``DONE`` and the connection is still reusable (keep-alive is on and no
  switch proposal is active).  It resets both roles to ``IDLE``.  One end
  resetting is not enough: the other end will reject a new request until
  it resets too.
* Local illegal actions raise ``LocalProtocolError`` and put the acting
  role in ``ERROR``.  Illegal bytes from the peer raise
  ``RemoteProtocolError``, put the peer's role in ``ERROR``, and leave the
  local role's ability to send determined by the ordinary state rules
  (for example, a client that is still sending a request body may finish
  that body, but cannot start another request).
* Each error is sticky: issuing another local action while the local role
  is ``ERROR`` raises ``LocalProtocolError``; trying to parse another
  remote event while the peer role is ``ERROR`` raises
  ``RemoteProtocolError``.

The implementation is intentionally conservative: it models message
ordering, framing, keep-alive, switches, and error stickiness.  Scenario
tests in ``test_dual_end.py`` pin down exact wire bytes and event payloads;
the reference model pins down which ordering/state changes are legal.

Every operation and observation can be serialized as JSON.  A saved trace
is replayed without any Python object state::

    log = EventLog.from_json(Path("trace.json").read_text())
    log.replay()

``ModelMismatch`` includes the same JSON trace, so a failing test output can
be copied directly into a regression fixture.
"""

import copy
import dataclasses
import json
import random
from collections import deque
from typing import (
    Any,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
)

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
from .._state import CLIENT, DONE, IDLE, SERVER
from .._util import (
    LocalProtocolError,
    ProtocolError,
    RemoteProtocolError,
    Sentinel,
)

__all__ = [
    "ModelMismatch",
    "ReferenceViolation",
    "ReferenceView",
    "REFERENCE_TRANSITIONS",
    "REFERENCE_JOINT_TRANSITIONS",
    "Op",
    "Result",
    "EventLog",
    "DualEnd",
    "EarlyResetConnection",
    "run_sequence",
    "replay_log",
    "shrink_failure",
    "random_sequence",
]


# ---------------------------------------------------------------------------
# Event serialization.  Logs contain event constructors, not Python reprs,
# so a saved log can be replayed without relying on pickle or memory
# addresses.
# ---------------------------------------------------------------------------


def _bytes_to_text(value: bytes) -> str:
    return value.decode("latin-1")


def _text_to_bytes(value: str) -> bytes:
    return value.encode("latin-1")


def _headers_json(headers: Any) -> List[List[str]]:
    # Headers keeps the original spelling for wire replay; iterating it
    # yields normalized lowercase names.  Logs should reproduce the bytes.
    raw_items = getattr(headers, "_full_items", None)
    if raw_items is None:
        raw_items = [
            (name, name, value) for name, value in headers
        ]
    return [
        [_bytes_to_text(raw_name), _bytes_to_text(value)]
        for raw_name, _name, value in raw_items
    ]


def encode_event(event: Event) -> Dict[str, Any]:
    if type(event) is Request:
        request = event
        return {
            "kind": "Request",
            "method": _bytes_to_text(request.method),
            "target": _bytes_to_text(request.target),
            "headers": _headers_json(request.headers),
            "http_version": _bytes_to_text(request.http_version),
        }
    if type(event) is InformationalResponse:
        informational = event
        return {
            "kind": "InformationalResponse",
            "status_code": informational.status_code,
            "headers": _headers_json(informational.headers),
            "http_version": _bytes_to_text(informational.http_version),
            "reason": _bytes_to_text(informational.reason),
        }
    if type(event) is Response:
        response = event
        return {
            "kind": "Response",
            "status_code": response.status_code,
            "headers": _headers_json(response.headers),
            "http_version": _bytes_to_text(response.http_version),
            "reason": _bytes_to_text(response.reason),
        }
    if type(event) is Data:
        data = event
        return {
            "kind": "Data",
            "data": data.data.hex(),
            "chunk_start": data.chunk_start,
            "chunk_end": data.chunk_end,
        }
    if type(event) is EndOfMessage:
        end = event
        return {
            "kind": "EndOfMessage",
            "headers": _headers_json(end.headers),
        }
    if type(event) is ConnectionClosed:
        return {"kind": "ConnectionClosed"}
    raise TypeError(f"cannot serialize event {event!r}")


def decode_event(spec: Dict[str, Any]) -> Event:
    kind = spec["kind"]
    headers = [
        (_text_to_bytes(name), _text_to_bytes(value))
        for name, value in spec.get("headers", [])
    ]
    if kind == "Request":
        return Request(
            method=_text_to_bytes(spec["method"]),
            target=_text_to_bytes(spec["target"]),
            headers=headers,
            http_version=_text_to_bytes(spec.get("http_version", "1.1")),
        )
    if kind == "InformationalResponse":
        return InformationalResponse(
            status_code=spec["status_code"],
            headers=headers,
            http_version=_text_to_bytes(spec.get("http_version", "1.1")),
            reason=_text_to_bytes(spec.get("reason", "")),
        )
    if kind == "Response":
        return Response(
            status_code=spec["status_code"],
            headers=headers,
            http_version=_text_to_bytes(spec.get("http_version", "1.1")),
            reason=_text_to_bytes(spec.get("reason", "")),
        )
    if kind == "Data":
        return Data(
            bytes.fromhex(spec["data"]),
            chunk_start=spec.get("chunk_start", False),
            chunk_end=spec.get("chunk_end", False),
        )
    if kind == "EndOfMessage":
        return EndOfMessage(headers=headers)
    if kind == "ConnectionClosed":
        return ConnectionClosed()
    raise TypeError(f"unknown event kind {kind!r}")


def _role_name(role: Type[Sentinel]) -> str:
    return role.__name__


def _state_name(state: Type[Sentinel]) -> str:
    return state.__name__


def _event_name(event: Event) -> str:
    return type(event).__name__


# ---------------------------------------------------------------------------
# Independent reference model
# ---------------------------------------------------------------------------


class ReferenceViolation(Exception):
    """The reference model says this event is illegal for ``role``."""

    def __init__(self, role: str, reason: str) -> None:
        super().__init__(reason)
        self.role = role
        self.reason = reason


# The table is written independently of h11._state.  Keys are literal
# lifecycle names; event names ending in "/UPGRADE" and "/CONNECT" are the
# accepted switch variants, not ordinary informational/final responses.
REFERENCE_TRANSITIONS: Dict[Tuple[str, str], Dict[str, str]] = {
    ("CLIENT", "IDLE"): {"Request": "SEND_BODY", "ConnectionClosed": "CLOSED"},
    ("CLIENT", "SEND_BODY"): {"Data": "SEND_BODY", "EndOfMessage": "DONE"},
    ("CLIENT", "DONE"): {"ConnectionClosed": "CLOSED"},
    ("CLIENT", "MUST_CLOSE"): {"ConnectionClosed": "CLOSED"},
    ("CLIENT", "CLOSED"): {"ConnectionClosed": "CLOSED"},
    ("SERVER", "IDLE"): {
        "ConnectionClosed": "CLOSED",
        # An early error response may be sent before a request is complete.
        "Response": "SEND_BODY",
    },
    ("SERVER", "SEND_RESPONSE"): {
        "InformationalResponse": "SEND_RESPONSE",
        "InformationalResponse/UPGRADE": "SWITCHED_PROTOCOL",
        "Response": "SEND_BODY",
        "Response/CONNECT": "SWITCHED_PROTOCOL",
    },
    ("SERVER", "SEND_BODY"): {"Data": "SEND_BODY", "EndOfMessage": "DONE"},
    ("SERVER", "DONE"): {"ConnectionClosed": "CLOSED"},
    ("SERVER", "MUST_CLOSE"): {"ConnectionClosed": "CLOSED"},
    ("SERVER", "CLOSED"): {"ConnectionClosed": "CLOSED"},
}


_NO_EVENT_STATES = {"MIGHT_SWITCH_PROTOCOL", "SWITCHED_PROTOCOL", "ERROR"}

REFERENCE_JOINT_TRANSITIONS: Dict[Tuple[str, str], Dict[str, str]] = {
    ("MIGHT_SWITCH_PROTOCOL", "SWITCHED_PROTOCOL"): {"CLIENT": "SWITCHED_PROTOCOL"},
    ("CLOSED", "DONE"): {"SERVER": "MUST_CLOSE"},
    ("CLOSED", "IDLE"): {"SERVER": "MUST_CLOSE"},
    ("ERROR", "DONE"): {"SERVER": "MUST_CLOSE"},
    ("DONE", "CLOSED"): {"CLIENT": "MUST_CLOSE"},
    ("IDLE", "CLOSED"): {"CLIENT": "MUST_CLOSE"},
    ("DONE", "ERROR"): {"CLIENT": "MUST_CLOSE"},
}


def _comma_header(headers: Iterable[Tuple[bytes, bytes]], name: bytes) -> List[bytes]:
    values: List[bytes] = []
    for header_name, value in headers:
        if header_name.lower() == name:
            for item in value.split(b","):
                token = item.strip().lower()
                if token:
                    values.append(token)
    return values


def _first_int_header(
    headers: Iterable[Tuple[bytes, bytes]], name: bytes
) -> Optional[int]:
    for header_name, value in headers:
        if header_name.lower() == name:
            try:
                return int(value.strip())
            except ValueError:
                return None
    return None


def _reference_body_mode(
    role: str,
    event: Event,
    request_method: Optional[bytes],
    their_http_version: Optional[bytes],
) -> Optional[Tuple[str, Optional[int]]]:
    if type(event) is Request:
        transfer_encoding = _comma_header(event.headers, b"transfer-encoding")
        if transfer_encoding:
            return ("chunked", None)
        content_length = _first_int_header(event.headers, b"content-length")
        if content_length is not None:
            return ("length", content_length)
        return ("length", 0)
    if type(event) is Response:
        if (
            event.status_code in (204, 304)
            or request_method == b"HEAD"
            or (request_method == b"CONNECT" and 200 <= event.status_code < 300)
        ):
            return ("length", 0)
        transfer_encoding = _comma_header(event.headers, b"transfer-encoding")
        if transfer_encoding:
            return ("chunked", None)
        content_length = _first_int_header(event.headers, b"content-length")
        if content_length is not None:
            return ("length", content_length)
        if their_http_version is not None and their_http_version >= b"1.1":
            return ("chunked", None)
        return ("eof", None)
    return None


class ReferenceView:
    """One endpoint's independent view of the shared HTTP lifecycle.

    Each real ``Connection`` gets one ``ReferenceView``.  Sending an event
    updates only the sender's view; consuming an event updates only the
    receiver's view.  This is what lets the tests distinguish "I sent it"
    from "the peer has seen it".
    """

    def __init__(self, our_role: Type[Sentinel]) -> None:
        if our_role not in (CLIENT, SERVER):
            raise ValueError("our_role must be CLIENT or SERVER")
        self.our_role = _role_name(our_role)
        self.their_role = (
            "SERVER" if self.our_role == "CLIENT" else "CLIENT"
        )
        self.states = {"CLIENT": "IDLE", "SERVER": "IDLE"}
        self.keep_alive = True
        self.pending_switches: Set[str] = set()
        self.waiting_100_continue = False
        self.request_method: Optional[bytes] = None
        self.their_http_version: Optional[bytes] = None
        self.body_modes: Dict[str, Optional[Tuple[str, Optional[int]]]] = {
            "CLIENT": None,
            "SERVER": None,
        }

    def clone(self) -> "ReferenceView":
        return copy.deepcopy(self)

    @property
    def client_is_waiting_for_100_continue(self) -> bool:
        return self.waiting_100_continue

    def _raise(self, role: str, reason: str) -> None:
        raise ReferenceViolation(role, reason)

    def process_error(self, role: str) -> None:
        self.states[role] = "ERROR"
        self._fire_joint_transitions()

    def note_failed_event_side_effects(self, role: str, event: Event) -> None:
        # h11 performs a few bookkeeping steps before asking the state
        # machine whether an event is legal.  Those steps survive the
        # resulting LocalProtocolError.
        if role == "CLIENT" and type(event) is Request:
            if event.method == b"CONNECT":
                self.pending_switches.add("CONNECT")
            if _comma_header(event.headers, b"upgrade"):
                self.pending_switches.add("UPGRADE")
        if role == "SERVER" and type(event) is Response:
            accepts_connect = (
                "CONNECT" in self.pending_switches
                and 200 <= event.status_code < 300
            )
            if not accepts_connect:
                self.pending_switches.clear()

    def _event_key(
        self, role: str, event: Event, switch: Optional[str]
    ) -> str:
        key = _event_name(event)
        if switch is not None:
            key += "/" + switch
        return key

    def _check_event(self, role: str, event: Event) -> Optional[str]:
        if self.states[role] == "ERROR":
            self._raise(
                role, f"cannot process {_event_name(event)} after ERROR"
            )
        if self.states[role] in _NO_EVENT_STATES:
            self._raise(
                role,
                f"{_event_name(event)} is not allowed in {self.states[role]}",
            )

        switch: Optional[str] = None
        if role == "SERVER" and type(event) is InformationalResponse:
            if event.status_code == 101:
                switch = "UPGRADE"
                if "UPGRADE" not in self.pending_switches:
                    self._raise(
                        role,
                        "101 accepted without a pending Upgrade proposal",
                    )
        if role == "SERVER" and type(event) is Response:
            if (
                "CONNECT" in self.pending_switches
                and 200 <= event.status_code < 300
            ):
                switch = "CONNECT"

        key = self._event_key(role, event, switch)
        if key not in REFERENCE_TRANSITIONS.get((role, self.states[role]), {}):
            self._raise(
                role,
                f"{key} is not legal for {role} in {self.states[role]}",
            )

        if role == "CLIENT" and type(event) is Request:
            if self.states["SERVER"] != "IDLE":
                self._raise(
                    "SERVER",
                    "a Request arrived while the server was "
                    f"{self.states['SERVER']}",
                )
        return key

    def _check_body(self, role: str, event: Event) -> None:
        mode = self.body_modes[role]
        if type(event) is Data:
            if mode is None:
                self._raise(role, "Data without a message body")
            if mode[0] == "length":
                remaining = mode[1]
                assert remaining is not None
                if len(event.data) > remaining:
                    self._raise(role, "too much data for Content-Length")
            return
        if type(event) is EndOfMessage:
            if mode is not None and mode[0] == "length":
                remaining = mode[1]
                assert remaining is not None
                if remaining != 0:
                    self._raise(role, "too little data for Content-Length")
                if event.headers:
                    self._raise(role, "Content-Length and trailers do not mix")
            elif mode is not None and mode[0] == "eof" and event.headers:
                self._raise(role, "trailers require chunked framing")

    def _apply_body(self, role: str, event: Event) -> None:
        mode = self.body_modes[role]
        if type(event) is Request or type(event) is Response:
            self.body_modes[role] = _reference_body_mode(
                role, event, self.request_method, self.their_http_version
            )
        elif type(event) is Data:
            if mode is not None and mode[0] == "length":
                assert mode[1] is not None
                self.body_modes[role] = ("length", mode[1] - len(event.data))
        elif type(event) is EndOfMessage:
            self.body_modes[role] = None

    def process_event(self, role: Type[Sentinel], event: Event) -> None:
        role_name = _role_name(role)
        if (
            role_name == self.our_role
            and isinstance(
                event, (Request, Response, InformationalResponse)
            )
            and event.http_version != b"1.1"
        ):
            self._raise(
                role_name, "h11 only sends HTTP/1.1 message heads"
            )
        key = self._check_event(role_name, event)
        self._check_body(role_name, event)

        # Keep-alive is decided from the semantic headers, before h11's
        # response header normalization adds Connection: close.
        if type(event) in (Request, Response):
            connection_tokens = _comma_header(event.headers, b"connection")
            if b"close" in connection_tokens or event.http_version < b"1.1":
                self.keep_alive = False

        if type(event) is Request:
            if role_name == "CLIENT":
                self.pending_switches.discard("CONNECT")
                self.pending_switches.discard("UPGRADE")
                if event.method == b"CONNECT":
                    self.pending_switches.add("CONNECT")
                if _comma_header(event.headers, b"upgrade"):
                    self.pending_switches.add("UPGRADE")
            self.request_method = event.method

        if (
            role_name == "SERVER"
            and type(event) is Response
            and "/" not in key
        ):
            self.pending_switches.clear()

        # Apply the sender's transition.
        new_state = REFERENCE_TRANSITIONS[(role_name, self.states[role_name])][key]
        self.states[role_name] = new_state
        if role_name == "CLIENT" and type(event) is Request:
            self.states["SERVER"] = "SEND_RESPONSE"

        self._apply_body(role_name, event)
        if type(event) is Response:
            mode = self.body_modes[role_name]
            if (
                mode is not None
                and mode[0] == "eof"
                and self.request_method != b"HEAD"
            ):
                # An unknown-length body to an HTTP/1.0 or not-yet-known
                # peer is delimited by connection close.  h11 adds
                # Connection: close itself in this case.
                self.keep_alive = False

        if role_name != self.our_role and type(event) in (
            Request,
            Response,
            InformationalResponse,
        ):
            self.their_http_version = event.http_version

        if (
            type(event) is Request
            and event.http_version >= b"1.1"
            and b"100-continue" in _comma_header(event.headers, b"expect")
        ):
            self.waiting_100_continue = True
        if type(event) in (InformationalResponse, Response):
            self.waiting_100_continue = False
        if role_name == "CLIENT" and type(event) in (Data, EndOfMessage):
            self.waiting_100_continue = False

        self._fire_joint_transitions()

    def _fire_joint_transitions(self) -> None:
        while True:
            before = dict(self.states)

            if self.pending_switches and self.states["CLIENT"] == "DONE":
                self.states["CLIENT"] = "MIGHT_SWITCH_PROTOCOL"
            if (
                not self.pending_switches
                and self.states["CLIENT"] == "MIGHT_SWITCH_PROTOCOL"
            ):
                self.states["CLIENT"] = "DONE"

            if not self.keep_alive:
                for role in ("CLIENT", "SERVER"):
                    if self.states[role] == "DONE":
                        self.states[role] = "MUST_CLOSE"

            joint = (self.states["CLIENT"], self.states["SERVER"])
            self.states.update(REFERENCE_JOINT_TRANSITIONS.get(joint, {}))

            if self.states == before:
                return

    def start_next_cycle(self) -> None:
        if self.states != {"CLIENT": "DONE", "SERVER": "DONE"}:
            self._raise(
                "CLIENT",
                "start_next_cycle requires both roles in DONE, not "
                f"{self.states}",
            )
        if not self.keep_alive or self.pending_switches:
            self._raise("CLIENT", "connection is not reusable")
        self.states = {"CLIENT": "IDLE", "SERVER": "IDLE"}
        self.request_method = None
        self.body_modes = {"CLIENT": None, "SERVER": None}


# ---------------------------------------------------------------------------
# Operations and replayable log
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Op:
    action: str
    role: Optional[str] = None
    events: List[Event] = dataclasses.field(default_factory=list)
    chunk_size: Optional[int] = None
    limit: Optional[int] = None
    data: Optional[bytes] = None
    drop: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "role": self.role,
            "events": [encode_event(event) for event in self.events],
            "chunk_size": self.chunk_size,
            "limit": self.limit,
            "data": None if self.data is None else self.data.hex(),
            "drop": self.drop,
        }

    @classmethod
    def from_json(cls, spec: Dict[str, Any]) -> "Op":
        return cls(
            action=spec["action"],
            role=spec.get("role"),
            events=[decode_event(event) for event in spec.get("events", [])],
            chunk_size=spec.get("chunk_size"),
            limit=spec.get("limit"),
            data=(
                None
                if spec.get("data") is None
                else bytes.fromhex(spec["data"])
            ),
            drop=spec.get("drop", False),
        )


@dataclasses.dataclass
class Result:
    action: str
    role: Optional[str]
    status: str
    data: bytes = b""
    events: List[Event] = dataclasses.field(default_factory=list)
    terminal: Optional[str] = None
    error: Optional[str] = None
    error_message: Optional[str] = None
    states: Dict[str, Any] = dataclasses.field(default_factory=dict)
    delivered: Optional[int] = None
    dropped: Optional[int] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "role": self.role,
            "status": self.status,
            "data": self.data.hex(),
            "events": [encode_event(event) for event in self.events],
            "terminal": self.terminal,
            "error": self.error,
            "error_message": self.error_message,
            "states": self.states,
            "delivered": self.delivered,
            "dropped": self.dropped,
        }

    @classmethod
    def from_json(cls, spec: Dict[str, Any]) -> "Result":
        payload = dict(spec)
        payload["data"] = bytes.fromhex(payload.get("data") or "")
        payload["events"] = [
            decode_event(event) for event in payload.get("events", [])
        ]
        return cls(**payload)


@dataclasses.dataclass
class EventLog:
    ops: List[Op]
    observations: List[Result]

    def replay(
        self,
        connection_factory: Any = Connection,
        check: bool = True,
    ) -> "EventLog":
        return replay_log(
            self, connection_factory=connection_factory, check=check
        )

    def to_json(self, indent: Optional[int] = None) -> str:
        document = {
            "version": 1,
            "ops": [op.to_json() for op in self.ops],
            "observations": [
                observation.to_json() for observation in self.observations
            ],
        }
        return json.dumps(document, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "EventLog":
        document = json.loads(text)
        if document.get("version") != 1:
            raise ValueError("unsupported event log version")
        return cls(
            ops=[Op.from_json(spec) for spec in document["ops"]],
            observations=[
                Result.from_json(spec)
                for spec in document["observations"]
            ],
        )


class ModelMismatch(AssertionError):
    """The implementation disagrees with the independent reference model."""

    def __init__(
        self,
        message: str,
        log: Optional[EventLog] = None,
    ) -> None:
        self.log = log
        suffix = ""
        if log is not None:
            suffix = "\nreplayable log:\n" + log.to_json(indent=2)
        super().__init__(message + suffix)


@dataclasses.dataclass
class _WireEntry:
    data: bytes
    expected: Tuple[Event, ...] = ()
    eof: bool = False


class DualEnd:
    """Two Connections connected by in-memory, independently consumable wires."""

    def __init__(
        self,
        connection_factory: Any = Connection,
        check: bool = True,
    ) -> None:
        self.connection_factory = connection_factory
        self.conns: Dict[str, Connection] = {
            "CLIENT": connection_factory(CLIENT),
            "SERVER": connection_factory(SERVER),
        }
        self.refs: Dict[str, ReferenceView] = {
            "CLIENT": ReferenceView(CLIENT),
            "SERVER": ReferenceView(SERVER),
        }
        # wire[role] contains bytes/effects from role to other[role].
        self.wire: Dict[str, Deque[_WireEntry]] = {
            "CLIENT": deque(),
            "SERVER": deque(),
        }
        self.mailboxes: Dict[str, Deque[Event]] = {
            "CLIENT": deque(),
            "SERVER": deque(),
        }
        # expected[role] is the semantic event stream that has been
        # delivered to role, in wire order.  It is populated only when a
        # wire entry is actually delivered, so dropping an early close
        # cannot leave fake expectations behind.
        self.expected: Dict[str, Deque[Event]] = {
            "CLIENT": deque(),
            "SERVER": deque(),
        }
        # Raw bytes injected by tests may not have a known semantic stream.
        self.allow_unexpected: Dict[str, bool] = {
            "CLIENT": False,
            "SERVER": False,
        }
        self.ops: List[Op] = []
        self.observations: List[Result] = []
        self.check = check

    # -- public operations -------------------------------------------------

    def send(
        self,
        role: str,
        events: Sequence[Event],
        chunk_size: Optional[int] = None,
    ) -> Result:
        op = Op(
            action="send",
            role=role,
            events=list(events),
            chunk_size=chunk_size,
        )
        return self.execute(op)

    def deliver(self, role: str, limit: Optional[int] = None) -> Result:
        return self.execute(Op(action="deliver", role=role, limit=limit))

    def pump(self, role: str, limit: Optional[int] = None) -> Result:
        return self.execute(Op(action="pump", role=role, limit=limit))

    def take(self, role: str, limit: Optional[int] = None) -> Result:
        return self.execute(Op(action="take", role=role, limit=limit))

    def start_next_cycle(self, role: str) -> Result:
        return self.execute(Op(action="start_next_cycle", role=role))

    def abort(self, role: str, drop: bool = False) -> Result:
        return self.execute(Op(action="abort", role=role, drop=drop))

    def receive_raw(
        self,
        role: str,
        data: bytes,
        expected: Sequence[Event] = (),
    ) -> Result:
        return self.execute(
            Op(
                action="receive_raw",
                role=role,
                data=data,
                events=list(expected),
            )
        )

    def event_log(self) -> EventLog:
        return EventLog(list(self.ops), list(self.observations))

    def snapshot(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        for role_name in ("CLIENT", "SERVER"):
            conn = self.conns[role_name]
            pending = sorted(
                sentinel.__name__.replace("_SWITCH_", "")
                for sentinel in conn._cstate.pending_switch_proposals
            )
            snapshot[role_name] = {
                "states": {
                    client_or_server.__name__: state.__name__
                    for client_or_server, state in conn.states.items()
                },
                "waiting_100_continue": (
                    conn.client_is_waiting_for_100_continue
                ),
                "keep_alive": conn._cstate.keep_alive,
                "pending_switches": pending,
                "request_method": (
                    None
                    if conn._request_method is None
                    else _bytes_to_text(conn._request_method)
                ),
                "their_http_version": (
                    None
                    if conn.their_http_version is None
                    else _bytes_to_text(conn.their_http_version)
                ),
            }
        return snapshot

    # -- execution core ----------------------------------------------------

    def execute(self, op: Op) -> Result:
        self.ops.append(op)
        try:
            result = self._execute(op)
        except ModelMismatch as exc:
            mismatch = Result(
                action=op.action,
                role=op.role,
                status="mismatch",
                error="ModelMismatch",
                error_message=str(exc),
                states=self.snapshot(),
            )
            self.observations.append(mismatch)
            if exc.log is None:
                exc.log = self.event_log()
                exc.args = (exc.args[0] + "\nreplayable log:\n"
                            + exc.log.to_json(indent=2),)
            raise
        self.observations.append(result)
        return result

    def _execute(self, op: Op) -> Result:
        if op.action == "send":
            return self._send(op)
        if op.action == "deliver":
            return self._deliver(op)
        if op.action == "pump":
            return self._pump(op)
        if op.action == "take":
            return self._take(op)
        if op.action == "start_next_cycle":
            return self._start_next_cycle(op)
        if op.action == "abort":
            return self._abort(op)
        if op.action == "receive_raw":
            return self._receive_raw(op)
        raise ValueError(f"unknown action {op.action!r}")

    def _result(
        self,
        op: Op,
        *,
        status: str = "ok",
        data: bytes = b"",
        events: Optional[List[Event]] = None,
        terminal: Optional[str] = None,
        error: Optional[BaseException] = None,
        delivered: Optional[int] = None,
        dropped: Optional[int] = None,
    ) -> Result:
        return Result(
            action=op.action,
            role=op.role,
            status=status,
            data=data,
            events=list(events or []),
            terminal=terminal,
            error=None if error is None else type(error).__name__,
            error_message=None if error is None else str(error),
            states=self.snapshot(),
            delivered=delivered,
            dropped=dropped,
        )

    def _send(self, op: Op) -> Result:
        role_name = self._require_role(op)
        role = CLIENT if role_name == "CLIENT" else SERVER
        data_parts: List[bytes] = []
        for event in op.events:
            reference = self.refs[role_name].clone()
            mode_before = self.refs[role_name].body_modes[role_name]
            was_error = self.refs[role_name].states[role_name] == "ERROR"
            predicted_error = False
            try:
                reference.process_event(role, event)
            except ReferenceViolation:
                predicted_error = True

            try:
                data = self.conns[role_name].send(event)
            except ProtocolError as exc:
                if not isinstance(exc, LocalProtocolError):
                    raise ModelMismatch(
                        f"send raised {type(exc).__name__}; local sends "
                        "must raise LocalProtocolError"
                    )
                if not was_error:
                    self.refs[role_name].note_failed_event_side_effects(
                        role_name, event
                    )
                self.refs[role_name].process_error(role_name)
                self._assert_matches(
                    role_name, context=f"send {_event_name(event)}"
                )
                if not predicted_error:
                    raise ModelMismatch(
                        f"reference allowed send of {_event_name(event)} "
                        f"to {role_name}, but h11 raised "
                        f"{type(exc).__name__}: {exc}"
                    )
                return self._result(
                    op, status="error", data=b"".join(data_parts), error=exc
                )

            if predicted_error:
                raise ModelMismatch(
                    f"reference rejected send of {_event_name(event)} "
                    f"to {role_name}, but h11 accepted it"
                )

            self.refs[role_name] = reference
            self._assert_matches(
                role_name, context=f"send {_event_name(event)}"
            )
            expected_events = self._expected_for_send(
                role_name, event, reference, mode_before
            )
            if data is None:
                self._enqueue(role_name, b"", expected_events, eof=True, op=op)
            else:
                data_parts.append(data)
                self._enqueue(
                    role_name, data, expected_events, eof=False, op=op
                )

        return self._result(op, status="ok", data=b"".join(data_parts))

    def _expected_for_send(
        self,
        role_name: str,
        event: Event,
        reference: ReferenceView,
        mode_before: Optional[Tuple[str, Optional[int]]],
    ) -> Tuple[Event, ...]:
        mode_after = reference.body_modes[role_name]
        if type(event) is Request or type(event) is Response:
            if reference.states[role_name] == "SWITCHED_PROTOCOL":
                # Acceptance switches protocols before any body reader can
                # produce EndOfMessage.
                return (event,)
            if mode_after is not None and mode_after[0] == "length" \
                    and mode_after[1] == 0:
                # A zero-length body is complete as soon as the head is
                # parsed; the peer emits EndOfMessage without waiting for
                # the sender's empty-byte EOM event.
                return (event, EndOfMessage())
            return (event,)
        if type(event) is InformationalResponse:
            return (event,)
        if type(event) is Data:
            if mode_after is None:
                return ()
            if mode_after[0] == "length":
                expected = [
                    Data(
                        bytes(event.data),
                        chunk_start=False,
                        chunk_end=False,
                    )
                ]
                if mode_after[1] == 0:
                    expected.append(EndOfMessage())
                return tuple(expected)
            if mode_after[0] == "chunked":
                if not event.data:
                    return ()
                return (
                    Data(
                        bytes(event.data),
                        chunk_start=True,
                        chunk_end=True,
                    ),
                )
            if event.data:
                return (
                    Data(
                        bytes(event.data),
                        chunk_start=False,
                        chunk_end=False,
                    ),
                )
            return ()
        if type(event) is EndOfMessage:
            if mode_before is not None and mode_before[0] == "chunked":
                return (EndOfMessage(headers=event.headers),)
            # Content-Length EOM has no bytes and the reader has already
            # produced EOM at the end of the body.  HTTP/1.0 EOM is implied
            # by EOF.
            return ()
        if type(event) is ConnectionClosed:
            return (ConnectionClosed(),)
        return ()

    def _enqueue(
        self,
        source: str,
        data: bytes,
        expected: Tuple[Event, ...],
        *,
        eof: bool,
        op: Op,
    ) -> None:
        if eof:
            self.wire[source].append(_WireEntry(b"", expected, eof=True))
            return
        if not data:
            if expected:
                self.wire[source].append(_WireEntry(b"", expected))
            return
        chunk_size = op.chunk_size
        if chunk_size is None or chunk_size <= 0:
            chunks = [data]
        else:
            chunks = [
                data[index : index + chunk_size]
                for index in range(0, len(data), chunk_size)
            ]
        for index, chunk in enumerate(chunks):
            self.wire[source].append(
                _WireEntry(chunk, expected if index == 0 else ())
            )

    def _deliver(self, op: Op) -> Result:
        source = self._require_role(op)
        destination = "SERVER" if source == "CLIENT" else "CLIENT"
        entries = self.wire[source]
        delivered = 0
        error: Optional[BaseException] = None
        while entries and (op.limit is None or delivered < op.limit):
            entry = entries.popleft()
            delivered += 1
            try:
                if entry.eof:
                    self.conns[destination].receive_data(b"")
                elif entry.data:
                    self.conns[destination].receive_data(entry.data)
            except BaseException as exc:
                self.expected[destination].clear()
                error = exc
                break
            if entry.expected:
                self.expected[destination].extend(entry.expected)
        result = self._result(op, delivered=delivered)
        if error is not None:
            result.status = "error"
            result.error = type(error).__name__
            result.error_message = str(error)
        return result

    def _pump(self, op: Op) -> Result:
        role_name = self._require_role(op)
        their_role = CLIENT if role_name == "SERVER" else SERVER
        events: List[Event] = []
        terminal: Optional[str] = None
        error: Optional[BaseException] = None
        calls = 0
        while op.limit is None or calls < op.limit:
            calls += 1
            try:
                event = self.conns[role_name].next_event()
            except ProtocolError as exc:
                if not isinstance(exc, RemoteProtocolError):
                    raise ModelMismatch(
                        f"pump raised {type(exc).__name__}; remote parse "
                        "failures must raise RemoteProtocolError"
                    )
                error = exc
                self.refs[role_name].process_error(_role_name(their_role))
                self.expected[role_name].clear()
                self._assert_matches(role_name, context="pump error")
                break
            if event is NEED_DATA or event is PAUSED:
                terminal = _state_name(event)
                break
            assert isinstance(event, Event)
            expected_events = self.expected[role_name]
            closed_repeat = (
                type(event) is ConnectionClosed
                and self.refs[role_name].states[_role_name(their_role)]
                == "CLOSED"
            )
            if expected_events and not closed_repeat:
                wanted = expected_events.popleft()
                self._assert_expected_event(wanted, event)
            elif (
                not expected_events
                and not closed_repeat
                and not self.allow_unexpected[role_name]
            ):
                raise ModelMismatch(
                    f"{role_name} produced unexpected {_event_name(event)} "
                    "with no delivered bytes/events expecting it"
                )

            reference = self.refs[role_name].clone()
            try:
                reference.process_event(their_role, event)
            except ReferenceViolation as exc:
                raise ModelMismatch(
                    f"{role_name} accepted {_event_name(event)}, but the "
                    f"reference says it is illegal: {exc.reason}"
                )
            self.refs[role_name] = reference
            self._assert_matches(role_name, context=f"pump {_event_name(event)}")
            events.append(event)
            self.mailboxes[role_name].append(event)
            if type(event) is ConnectionClosed:
                break
        result = self._result(
            op, events=events, terminal=terminal, error=error
        )
        if error is not None:
            result.status = "error"
        elif terminal is not None:
            result.status = "terminal"
        return result

    def _take(self, op: Op) -> Result:
        role_name = self._require_role(op)
        events: List[Event] = []
        mailbox = self.mailboxes[role_name]
        while mailbox and (op.limit is None or len(events) < op.limit):
            events.append(mailbox.popleft())
        return self._result(op, events=events)

    def _start_next_cycle(self, op: Op) -> Result:
        role_name = self._require_role(op)
        reference = self.refs[role_name].clone()
        try:
            reference.start_next_cycle()
        except ReferenceViolation as exc:
            expected_error = exc
        else:
            expected_error = None

        try:
            self.conns[role_name].start_next_cycle()
        except ProtocolError as exc:
            self._assert_matches(role_name, context="failed start_next_cycle")
            if expected_error is None:
                raise ModelMismatch(
                    f"reference allowed {role_name}.start_next_cycle(), "
                    f"but h11 raised {type(exc).__name__}: {exc}"
                )
            return self._result(op, status="error", error=exc)

        if expected_error is not None:
            raise ModelMismatch(
                f"reference rejected {role_name}.start_next_cycle(): "
                f"{expected_error.reason}; h11 accepted it"
            )
        self.refs[role_name] = reference
        self._assert_matches(role_name, context="start_next_cycle")
        return self._result(op)

    def _abort(self, op: Op) -> Result:
        role_name = self._require_role(op)
        destination = "SERVER" if role_name == "CLIENT" else "CLIENT"
        dropped = 0
        if op.drop:
            dropped = len(self.wire[role_name])
            self.wire[role_name].clear()
        expected: Tuple[Event, ...] = (ConnectionClosed(),)
        mode = self.refs[destination].body_modes[role_name]
        if mode is not None and mode[0] == "eof":
            expected = (EndOfMessage(), ConnectionClosed())
        self.wire[role_name].append(_WireEntry(b"", expected, eof=True))
        return self._result(op, dropped=dropped)

    def _receive_raw(self, op: Op) -> Result:
        role_name = self._require_role(op)
        assert op.data is not None
        try:
            self.conns[role_name].receive_data(op.data)
        except BaseException as exc:
            return self._result(op, status="error", error=exc)
        if op.events:
            self.expected[role_name].extend(op.events)
        else:
            self.allow_unexpected[role_name] = True
        return self._result(op, data=op.data)

    # -- oracle comparison -------------------------------------------------

    def _require_role(self, op: Op) -> str:
        if op.role not in ("CLIENT", "SERVER"):
            raise ValueError(f"operation {op.action!r} needs a role")
        return op.role

    def _actual_view(self, role_name: str) -> Dict[str, Any]:
        conn = self.conns[role_name]
        pending = {
            sentinel.__name__.replace("_SWITCH_", "")
            for sentinel in conn._cstate.pending_switch_proposals
        }
        return {
            "states": {
                client_or_server.__name__: state.__name__
                for client_or_server, state in conn.states.items()
            },
            "keep_alive": conn._cstate.keep_alive,
            "pending": pending,
            "waiting": conn.client_is_waiting_for_100_continue,
            "request_method": conn._request_method,
            "their_http_version": conn.their_http_version,
        }

    def _reference_view(self, role_name: str) -> Dict[str, Any]:
        reference = self.refs[role_name]
        return {
            "states": dict(reference.states),
            "keep_alive": reference.keep_alive,
            "pending": set(reference.pending_switches),
            "waiting": reference.waiting_100_continue,
            "request_method": reference.request_method,
            "their_http_version": reference.their_http_version,
        }

    def _assert_matches(self, role_name: str, context: str) -> None:
        if not self.check:
            return
        actual = self._actual_view(role_name)
        expected = self._reference_view(role_name)
        if actual != expected:
            raise ModelMismatch(
                f"{role_name} view mismatch after {context}:\n"
                f"  actual:    {actual}\n"
                f"  reference: {expected}"
            )

    def _assert_expected_event(self, wanted: Event, actual: Event) -> None:
        if self._event_matches(wanted, actual):
            return
        raise ModelMismatch(
            "delivered-event mismatch: wire/state model expected "
            f"{encode_event(wanted)}, h11 produced {encode_event(actual)}"
        )

    def _event_matches(self, wanted: Event, actual: Event) -> bool:
        if type(wanted) is not type(actual):
            return False
        if type(wanted) is Request:
            assert type(actual) is Request
            return (
                wanted.method == actual.method
                and wanted.target == actual.target
                and wanted.http_version == actual.http_version
                and list(wanted.headers) == list(actual.headers)
            )
        if type(wanted) is InformationalResponse:
            assert type(actual) is InformationalResponse
            return (
                wanted.status_code == actual.status_code
                and wanted.http_version == actual.http_version
                and list(wanted.headers) == list(actual.headers)
            )
        if type(wanted) is Response:
            assert type(actual) is Response
            # h11 may add framing/Connection headers.  Scenarios assert the
            # exact bytes for those; here compare the semantic response and
            # any non-managed headers.
            ignored = {b"connection", b"transfer-encoding", b"content-length"}
            wanted_headers = [
                (name, value)
                for name, value in wanted.headers
                if name.lower() not in ignored
            ]
            actual_headers = [
                (name, value)
                for name, value in actual.headers
                if name.lower() not in ignored
            ]
            return (
                wanted.status_code == actual.status_code
                and wanted.http_version == actual.http_version
                and wanted_headers == actual_headers
            )
        if type(wanted) is Data:
            assert type(actual) is Data
            return bytes(wanted.data) == bytes(actual.data)
        if type(wanted) is EndOfMessage:
            assert type(actual) is EndOfMessage
            return list(wanted.headers) == list(actual.headers)
        return type(wanted) is ConnectionClosed


# ---------------------------------------------------------------------------
# Mutant used to prove that the model catches premature state resets.
# ---------------------------------------------------------------------------


class EarlyResetConnection(Connection):
    """A deliberately broken Connection that reuses a half-finished cycle.

    Correct h11 requires both CLIENT and SERVER to be DONE before a cycle
    reset.  This mutant resets as soon as either one is DONE, which is
    exactly the kind of "the sender thinks the cycle is over but the peer
    has not answered yet" bug that the two-end model is supposed to catch.
    """

    def start_next_cycle(self) -> None:
        states = self._cstate.states
        if states[CLIENT] is DONE or states[SERVER] is DONE:
            old_states = dict(states)
            self._cstate.states = {CLIENT: IDLE, SERVER: IDLE}
            self._request_method = None
            self.client_is_waiting_for_100_continue = False
            self._respond_to_state_changes(old_states)
            return
        super().start_next_cycle()


# ---------------------------------------------------------------------------
# Seeded short sequences, replay, and shrinking
# ---------------------------------------------------------------------------


_HOST_HEADER = [(b"host", b"example.com")]


def _safe_event_actions() -> List[Op]:
    """A compact alphabet whose byte-level event stream is deterministic.

    All bodies are zero length (explicit Content-Length: 0 plus the special
    HEAD/204/304/CONNECT cases).  This keeps the semantic expected-event
    stream one-for-one with h11's emitted events, while still exercising
    early replies, multiple 1xx, switches, close, reset, and error paths.
    """

    request_get = Request(
        method="GET", target="/a", headers=_HOST_HEADER
    )
    request_head = Request(
        method="HEAD", target="/a", headers=_HOST_HEADER
    )
    request_connect = Request(
        method="CONNECT", target="example.com:443", headers=_HOST_HEADER
    )
    request_upgrade = Request(
        method="GET",
        target="/a",
        headers=_HOST_HEADER + [(b"upgrade", b"websocket")],
    )
    request_close = Request(
        method="GET",
        target="/a",
        headers=_HOST_HEADER + [(b"connection", b"close")],
    )

    return [
        Op("send", "CLIENT", [request_get]),
        Op("send", "CLIENT", [request_head]),
        Op("send", "CLIENT", [request_connect]),
        Op("send", "CLIENT", [request_upgrade]),
        Op("send", "CLIENT", [request_close]),
        Op("send", "CLIENT", [EndOfMessage()]),
        Op("send", "CLIENT", [Data(b"x")]),
        Op("send", "CLIENT", [ConnectionClosed()]),
        Op(
            "send",
            "SERVER",
            [InformationalResponse(status_code=100, headers=[])],
        ),
        Op(
            "send",
            "SERVER",
            [InformationalResponse(status_code=103, headers=[])],
        ),
        Op(
            "send",
            "SERVER",
            [
                InformationalResponse(
                    status_code=101, headers=[(b"upgrade", b"websocket")]
                )
            ],
        ),
        Op(
            "send",
            "SERVER",
            [Response(status_code=204, headers=[])],
        ),
        Op(
            "send",
            "SERVER",
            [Response(status_code=304, headers=[])],
        ),
        Op(
            "send",
            "SERVER",
            [Response(status_code=200, headers=[(b"content-length", b"0")])],
        ),
        Op(
            "send",
            "SERVER",
            [Response(status_code=404, headers=[(b"content-length", b"0")])],
        ),
        Op("send", "SERVER", [EndOfMessage()]),
        Op("send", "SERVER", [Data(b"x")]),
        Op("send", "SERVER", [ConnectionClosed()]),
        Op("deliver", "CLIENT"),
        Op("deliver", "SERVER"),
        Op("pump", "CLIENT"),
        Op("pump", "SERVER"),
        Op("take", "CLIENT"),
        Op("take", "SERVER"),
        Op("start_next_cycle", "CLIENT"),
        Op("start_next_cycle", "SERVER"),
        Op("abort", "CLIENT"),
        Op("abort", "SERVER"),
    ]


def _copy_op(op: Op) -> Op:
    # h11 events are immutable frozen dataclasses; deep-copying them is not
    # supported by their slots/init implementation.  The Op container and
    # its list are all that need copying.
    return Op(
        action=op.action,
        role=op.role,
        events=list(op.events),
        chunk_size=op.chunk_size,
        limit=op.limit,
        data=op.data,
        drop=op.drop,
    )


def random_sequence(
    seed: int,
    length: int,
    actions: Optional[Sequence[Op]] = None,
) -> List[Op]:
    """Return a deterministic short operation sequence."""
    if actions is None:
        actions = _safe_event_actions()
    rng = random.Random(seed)
    return [_copy_op(rng.choice(actions)) for _ in range(length)]


@dataclasses.dataclass
class RunReport:
    ops: List[Op]
    log: EventLog
    mismatch: Optional[ModelMismatch] = None


def run_sequence(
    ops: Sequence[Op],
    connection_factory: Any = Connection,
    check: bool = True,
) -> RunReport:
    model = DualEnd(connection_factory=connection_factory, check=check)
    mismatch: Optional[ModelMismatch] = None
    for op in ops:
        try:
            model.execute(_copy_op(op))
        except ModelMismatch as exc:
            mismatch = exc
            break
    return RunReport(
        ops=list(ops), log=model.event_log(), mismatch=mismatch
    )


def replay_log(
    log: EventLog,
    connection_factory: Any = Connection,
    check: bool = True,
) -> EventLog:
    replayed = run_sequence(
        log.ops,
        connection_factory=connection_factory,
        check=check,
    )
    if replayed.mismatch is not None:
        raise replayed.mismatch
    expected = [observation.to_json() for observation in log.observations]
    actual = [
        observation.to_json() for observation in replayed.log.observations
    ]
    if expected != actual:
        raise ModelMismatch(
            "replayed observations differ from the original log"
        )
    return replayed.log


def shrink_failure(
    ops: Sequence[Op],
    fails: Any,
    max_rounds: int = 100,
) -> List[Op]:
    """Greedily remove operations while ``fails`` still returns True.

    This is the deliberately simple kind used for short event traces: it
    preserves the first failing predicate at every step, so the result
    changes deterministically from run to run.
    """
    current = list(ops)
    if not fails(current):
        raise ValueError("the supplied sequence does not fail")
    for _ in range(max_rounds):
        changed = False
        for index in range(len(current)):
            candidate = current[:index] + current[index + 1 :]
            if candidate and fails(candidate):
                current = candidate
                changed = True
                break
        if not changed:
            return current
    return current
