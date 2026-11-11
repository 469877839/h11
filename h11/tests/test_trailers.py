# Tests for the pluggable trailer policy. These all use two in-memory
# Connections wired back to back (a client and a server), so every rule is
# exercised through the real send -> wire -> receive path rather than against
# the readers/writers in isolation.

from typing import cast, Dict, List, Optional, Tuple, Type

import pytest

from .._connection import Connection, NEED_DATA, PAUSED
from .._events import Data, EndOfMessage, Event, Request, Response
from .._state import CLIENT, DONE, ERROR, SERVER
from .._trailers import (
    FORBIDDEN_TRAILER_FIELDS,
    STRICT_TRAILER_POLICY,
    TrailerPolicy,
)
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel

CHUNKED_REQUEST_HEADERS = [
    ("Host", "example.com"),
    ("Transfer-Encoding", "chunked"),
]


class PolicyConnectionPair:
    """Two Connections talking over an in-memory byte pipe.

    Bytes produced by send() are buffered per role and only handed to the
    peer when deliver() is called, so tests can assert exactly what would
    have gone over the wire (notably, that a rejected event produced
    nothing).
    """

    def __init__(
        self,
        client_policy: TrailerPolicy = TrailerPolicy(),
        server_policy: TrailerPolicy = TrailerPolicy(),
    ) -> None:
        self.client = Connection(CLIENT, trailer_policy=client_policy)
        self.server = Connection(SERVER, trailer_policy=server_policy)
        self._conns = {CLIENT: self.client, SERVER: self.server}
        self._peer = {CLIENT: self.server, SERVER: self.client}
        self.wire: Dict[Type[Sentinel], bytes] = {CLIENT: b"", SERVER: b""}

    def send(self, role: Type[Sentinel], event: Event) -> Optional[bytes]:
        data = self._conns[role].send(event)
        if data:
            self.wire[role] += data
        return data

    def deliver(self, role: Type[Sentinel]) -> None:
        data = self.wire[role]
        self.wire[role] = b""
        if data:
            self._peer[role].receive_data(data)

    def next_events(self, role: Type[Sentinel]) -> List[Event]:
        """Drain all events currently available to the peer of ``role``."""
        conn = self._peer[role]
        events: List[Event] = []
        while True:
            event = conn.next_event()
            if event is NEED_DATA or event is PAUSED:
                break
            events.append(cast(Event, event))
        return events


def start_chunked_request(
    pair: PolicyConnectionPair, extra_headers: Optional[List[Tuple[str, str]]] = None
) -> None:
    headers = CHUNKED_REQUEST_HEADERS + list(extra_headers or [])
    pair.send(CLIENT, Request(method="POST", target="/", headers=headers))
    pair.send(CLIENT, Data(data=b"hello"))
    pair.deliver(CLIENT)
    assert [type(event) for event in pair.next_events(CLIENT)] == [Request, Data]


def test_send_rejects_undeclared_trailer_before_any_bytes() -> None:
    pair = PolicyConnectionPair(client_policy=STRICT_TRAILER_POLICY)
    start_chunked_request(pair)
    # Nothing has been written yet, so the peer still thinks the body is open.
    with pytest.raises(LocalProtocolError) as excinfo:
        pair.send(CLIENT, EndOfMessage(headers=[("Content-MD5", "abc")]))
    assert "not declared" in str(excinfo.value)
    # The rejection happened before any framing bytes were produced, so the
    # peer can never observe an apparently successful end-of-message.
    assert pair.wire[CLIENT] == b""
    assert pair.client.our_state is ERROR
    assert pair.server.their_state is not ERROR


def test_send_rejects_forbidden_field_at_declaration_time() -> None:
    pair = PolicyConnectionPair(client_policy=STRICT_TRAILER_POLICY)
    with pytest.raises(LocalProtocolError) as excinfo:
        pair.send(
            CLIENT,
            Request(
                method="POST",
                target="/",
                headers=CHUNKED_REQUEST_HEADERS + [("Trailer", "Authorization")],
            ),
        )
    assert "forbidden" in str(excinfo.value)
    assert pair.wire[CLIENT] == b""
    assert pair.client.our_state is ERROR


def test_send_rejects_forbidden_trailer_field_even_if_declared() -> None:
    # A policy without declaration checking still rejects forbidden fields.
    policy = TrailerPolicy(forbidden_fields=FORBIDDEN_TRAILER_FIELDS)
    pair = PolicyConnectionPair(client_policy=policy)
    start_chunked_request(pair)
    with pytest.raises(LocalProtocolError) as excinfo:
        pair.send(CLIENT, EndOfMessage(headers=[("Host", "evil.example")]))
    assert "forbidden" in str(excinfo.value)
    assert pair.wire[CLIENT] == b""


def test_receive_rejects_undeclared_trailer_as_protocol_error() -> None:
    pair = PolicyConnectionPair(server_policy=STRICT_TRAILER_POLICY)
    start_chunked_request(pair)
    pair.send(CLIENT, EndOfMessage(headers=[("Content-MD5", "abc")]))
    pair.deliver(CLIENT)
    with pytest.raises(RemoteProtocolError) as excinfo:
        pair.next_events(CLIENT)
    assert "not declared" in str(excinfo.value)
    # The violation is surfaced before EndOfMessage is delivered, and the
    # peer state is poisoned so nothing can silently complete the message.
    assert pair.server.their_state is ERROR
    with pytest.raises(RemoteProtocolError):
        pair.server.next_event()
    # Once in ERROR the receiver can still send a response.
    pair.server.send(Response(status_code=400, headers=[]))
    assert pair.server.our_state is not ERROR


def test_receive_rejects_forbidden_trailer_as_protocol_error() -> None:
    # The sender declares and emits a routing header as a trailer; the
    # receiver's policy must still refuse it.
    pair = PolicyConnectionPair(server_policy=STRICT_TRAILER_POLICY)
    start_chunked_request(pair, extra_headers=[("Trailer", "host")])
    pair.send(CLIENT, EndOfMessage(headers=[("Host", "evil.example")]))
    pair.deliver(CLIENT)
    with pytest.raises(RemoteProtocolError) as excinfo:
        pair.next_events(CLIENT)
    assert "forbidden" in str(excinfo.value)
    assert pair.server.their_state is ERROR


def test_legal_trailers_preserve_order_case_and_multiple_values() -> None:
    # Both directions enforce the policy, and the receivers keep trailers
    # exactly as the sender emitted them: original casing, original order,
    # and repeated field names as separate pairs.
    pair = PolicyConnectionPair(
        client_policy=STRICT_TRAILER_POLICY,
        server_policy=STRICT_TRAILER_POLICY,
    )
    start_chunked_request(pair, extra_headers=[("Trailer", "X-Checksum, X-Note")])
    trailers = [
        ("X-Checksum", "sha-256=abc"),
        ("x-note", "one"),
        ("X-Note", "two"),
    ]
    wire = pair.send(CLIENT, EndOfMessage(headers=trailers))
    assert wire == (
        b"0\r\n"
        b"X-Checksum: sha-256=abc\r\n"
        b"x-note: one\r\n"
        b"X-Note: two\r\n"
        b"\r\n"
    )
    pair.deliver(CLIENT)
    received = pair.next_events(CLIENT)[-1]
    assert type(received) is EndOfMessage
    assert list(received.headers) == [
        (b"x-checksum", b"sha-256=abc"),
        (b"x-note", b"one"),
        (b"x-note", b"two"),
    ]
    assert [raw for raw, _ in received.headers.raw_items()] == [
        b"X-Checksum",
        b"x-note",
        b"X-Note",
    ]
    assert pair.server.their_state is DONE


def test_repeated_trailer_declarations_count_once() -> None:
    pair = PolicyConnectionPair(client_policy=STRICT_TRAILER_POLICY)
    # Repeated declarations, odd casing, and comma splitting must all fold
    # to the same declared set.
    start_chunked_request(
        pair, extra_headers=[("Trailer", "X-Checksum, x-checksum, X-NOTE")]
    )
    pair.send(
        CLIENT,
        EndOfMessage(
            headers=[("x-checksum", "1"), ("X-Note", "2"), ("X-Checksum", "3")]
        ),
    )
    assert pair.wire[CLIENT] == (
        b"0\r\nx-checksum: 1\r\nX-Note: 2\r\nX-Checksum: 3\r\n\r\n"
    )


def test_same_field_name_in_regular_response_headers_is_not_trailers() -> None:
    # Regression: declaration/forbidden checks must only look at the
    # EndOfMessage trailers, never at fields with the same name that
    # legitimately appear in ordinary response headers.
    pair = PolicyConnectionPair(
        client_policy=STRICT_TRAILER_POLICY,
        server_policy=STRICT_TRAILER_POLICY,
    )
    pair.send(
        CLIENT,
        Request(
            method="GET",
            target="/",
            headers=[("Host", "example.com"), ("TE", "trailers")],
        ),
    )
    pair.deliver(CLIENT)
    # A GET with no framing headers has an empty body, so the server sees
    # the message complete immediately.
    assert [type(event) for event in pair.next_events(CLIENT)] == [
        Request,
        EndOfMessage,
    ]

    pair.send(
        SERVER,
        Response(
            status_code=200,
            reason=b"OK",
            headers=[
                ("Transfer-Encoding", "chunked"),
                # Same names that the trailers below will use, but here they
                # are regular headers and are not trailers at all.
                ("X-Checksum", "regular"),
                ("X-Note", "regular"),
                ("Trailer", "X-Checksum, X-Note"),
            ],
        ),
    )
    pair.send(SERVER, Data(data=b"body"))
    pair.send(SERVER, EndOfMessage(headers=[("X-Checksum", "trailer")]))
    pair.deliver(SERVER)
    events = pair.next_events(SERVER)
    assert [type(event) for event in events] == [Response, Data, EndOfMessage]
    response = events[0]
    # h11's response header cleanup rewrites framing headers, so compare
    # contents rather than exact order.
    assert set(response.headers) == {
        (b"transfer-encoding", b"chunked"),
        (b"x-checksum", b"regular"),
        (b"x-note", b"regular"),
        (b"trailer", b"X-Checksum, X-Note"),
    }
    assert list(events[-1].headers) == [(b"x-checksum", b"trailer")]
    assert pair.client.their_state is DONE


def test_non_chunked_message_cannot_carry_trailers() -> None:
    # Default policy: framing rule still applies, matching historical errors.
    pair = PolicyConnectionPair()
    pair.send(
        CLIENT,
        Request(
            method="POST",
            target="/",
            headers=[("Host", "example.com"), ("Content-Length", "5")],
        ),
    )
    pair.send(CLIENT, Data(data=b"hello"))
    wire_before = pair.wire[CLIENT]
    with pytest.raises(LocalProtocolError):
        pair.send(CLIENT, EndOfMessage(headers=[("X-Checksum", "abc")]))
    assert pair.wire[CLIENT] == wire_before

    # A Trailer: declaration on a non-chunked message is rejected up front.
    pair = PolicyConnectionPair(client_policy=STRICT_TRAILER_POLICY)
    with pytest.raises(LocalProtocolError) as excinfo:
        pair.send(
            CLIENT,
            Request(
                method="POST",
                target="/",
                headers=[
                    ("Host", "example.com"),
                    ("Content-Length", "0"),
                    ("Trailer", "X-Checksum"),
                ],
            ),
        )
    assert "chunked" in str(excinfo.value)
    assert pair.wire[CLIENT] == b""


def test_default_policy_remains_permissive() -> None:
    # Backwards compatibility: undeclared, oddly-cased, multi-valued trailers
    # pass in both directions with no configuration.
    pair = PolicyConnectionPair()
    start_chunked_request(pair)
    pair.send(
        CLIENT,
        EndOfMessage(
            headers=[("X-Checksum", "1"), ("x-checksum", "2"), ("X-Note", "3")]
        ),
    )
    pair.deliver(CLIENT)
    received = pair.next_events(CLIENT)[-1]
    assert type(received) is EndOfMessage
    assert list(received.headers) == [
        (b"x-checksum", b"1"),
        (b"x-checksum", b"2"),
        (b"x-note", b"3"),
    ]

    # A stray Trailer: declaration on a non-chunked message used to be
    # ignored, and the default policy must keep ignoring it.
    pair = PolicyConnectionPair()
    pair.send(
        CLIENT,
        Request(
            method="POST",
            target="/",
            headers=[
                ("Host", "example.com"),
                ("Content-Length", "0"),
                ("Trailer", "X-Checksum"),
            ],
        ),
    )
    pair.send(CLIENT, EndOfMessage())
    pair.deliver(CLIENT)
    assert [type(event) for event in pair.next_events(CLIENT)] == [
        Request,
        EndOfMessage,
    ]


def test_custom_forbidden_and_undeclared_handling() -> None:
    policy = TrailerPolicy(
        forbidden_fields=frozenset({b"x-secret"}),
        require_declaration=True,
        allow_undeclared=True,
    )
    pair = PolicyConnectionPair(client_policy=policy)
    start_chunked_request(pair)
    # Undeclared but allowed.
    pair.send(CLIENT, EndOfMessage(headers=[("X-Note", "ok")]))
    assert pair.wire[CLIENT] == b"0\r\nX-Note: ok\r\n\r\n"

    pair = PolicyConnectionPair(client_policy=policy)
    start_chunked_request(pair)
    with pytest.raises(LocalProtocolError) as excinfo:
        pair.send(CLIENT, EndOfMessage(headers=[("X-Secret", "leak")]))
    assert "forbidden" in str(excinfo.value)


def test_send_failure_does_not_reach_peer_even_when_retried() -> None:
    # Once the policy rejects an EndOfMessage we are in ERROR and can't
    # subsequently sneak a "clean" ending through.
    pair = PolicyConnectionPair(client_policy=STRICT_TRAILER_POLICY)
    start_chunked_request(pair)
    with pytest.raises(LocalProtocolError):
        pair.send(CLIENT, EndOfMessage(headers=[("Content-MD5", "abc")]))
    with pytest.raises(LocalProtocolError):
        pair.send(CLIENT, EndOfMessage())
    assert pair.wire[CLIENT] == b""
    pair.deliver(CLIENT)
    # Nothing new arrives, and the peer's message never completes.
    assert pair.next_events(CLIENT) == []
    assert pair.server.their_state is not DONE
