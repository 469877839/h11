from typing import List, Optional, Tuple

import pytest

from .._connection import Connection, NEED_DATA
from .._events import Data, EndOfMessage, Event, Request, Response
from .._state import CLIENT, DONE, ERROR, SERVER
from .._trailers import DEFAULT_FORBIDDEN_TRAILER_FIELDS, TrailerPolicy
from .._util import LocalProtocolError, RemoteProtocolError
from .helpers import get_all_events, normalize_data_events


def make_pair(
    client_policy: Optional[TrailerPolicy] = None,
    server_policy: Optional[TrailerPolicy] = None,
) -> Tuple[Connection, Connection]:
    client = Connection(CLIENT, trailer_policy=client_policy)
    server = Connection(SERVER, trailer_policy=server_policy)
    return client, server


def pump(sender: Connection, receiver: Connection, events: List[Event]) -> List[Event]:
    """Send events through one Connection and collect what the peer parses."""
    data = b""
    for event in events:
        chunk = sender.send(event)
        assert chunk is not None
        data += chunk
    receiver.receive_data(data)
    return get_all_events(receiver)


def receive_until_error(conn: Connection, data: bytes) -> List[Event]:
    """Feed raw bytes; return events parsed before the protocol error."""
    conn.receive_data(data)
    events = []
    with pytest.raises(RemoteProtocolError):
        while True:
            event = conn.next_event()
            if event is NEED_DATA:
                break
            events.append(event)
    return events


CHUNKED_REQUEST_HEAD = Request(
    method="POST",
    target="/upload",
    headers=[
        ("Host", "example.com"),
        ("Transfer-Encoding", "chunked"),
        ("Trailer", "X-Checksum, X-Note"),
    ],
)


def test_default_policy_is_compatible() -> None:
    # With no policy, h11 keeps its historical behavior: undeclared and even
    # forbidden-ish trailer fields pass through unchecked.
    client, server = make_pair()
    events = normalize_data_events(
        pump(
            client,
            server,
            [
                CHUNKED_REQUEST_HEAD,
                Data(b"abcd"),
                EndOfMessage(headers=[("Content-Length", "4"), ("X-Undeclared", "1")]),
            ],
        )
    )
    assert events == [
        CHUNKED_REQUEST_HEAD,
        Data(b"abcd"),
        EndOfMessage(headers=[("Content-Length", "4"), ("X-Undeclared", "1")]),
    ]


def test_send_rejects_forbidden_trailer_before_emitting_bytes() -> None:
    client, server = make_pair(client_policy=TrailerPolicy())
    client.send(CHUNKED_REQUEST_HEAD)
    client.send(Data(b"abcd"))
    with pytest.raises(LocalProtocolError, match="forbidden field"):
        client.send(EndOfMessage(headers=[("Content-Length", "4")]))
    # The failure happened before any bytes were produced, so nothing reached
    # the peer and the message never appeared to end successfully.
    assert server.next_event() is NEED_DATA
    assert client.our_state is ERROR


def test_send_rejects_undeclared_trailer() -> None:
    policy = TrailerPolicy(require_declaration=True)
    client, _ = make_pair(client_policy=policy)
    client.send(CHUNKED_REQUEST_HEAD)
    client.send(Data(b"abcd"))
    with pytest.raises(LocalProtocolError, match="not announced"):
        client.send(EndOfMessage(headers=[("X-Other", "1")]))


def test_send_rejects_forbidden_declaration() -> None:
    policy = TrailerPolicy()
    client, _ = make_pair(client_policy=policy)
    client.send(
        Request(
            method="POST",
            target="/upload",
            headers=[
                ("Host", "example.com"),
                ("Transfer-Encoding", "chunked"),
                ("Trailer", "Content-Length"),
            ],
        )
    )
    client.send(Data(b"abcd"))
    with pytest.raises(LocalProtocolError, match="announces forbidden"):
        client.send(EndOfMessage(headers=[("X-Note", "hi")]))


def test_send_rejects_trailers_on_non_chunked_message() -> None:
    policy = TrailerPolicy()
    client, _ = make_pair(client_policy=policy)
    client.send(
        Request(
            method="POST",
            target="/upload",
            headers=[("Host", "example.com"), ("Content-Length", "4")],
        )
    )
    client.send(Data(b"abcd"))
    with pytest.raises(LocalProtocolError, match="chunked"):
        client.send(EndOfMessage(headers=[("X-Checksum", "abc")]))


def test_send_forbidden_check_is_case_insensitive() -> None:
    policy = TrailerPolicy()
    client, _ = make_pair(client_policy=policy)
    client.send(CHUNKED_REQUEST_HEAD)
    client.send(Data(b"abcd"))
    with pytest.raises(LocalProtocolError, match="forbidden field"):
        client.send(EndOfMessage(headers=[("HOST", "evil.example")]))


def test_custom_forbidden_fields_replace_default() -> None:
    policy = TrailerPolicy(forbidden_fields=["X-Secret"])
    assert b"content-length" not in policy.forbidden_fields
    assert DEFAULT_FORBIDDEN_TRAILER_FIELDS.isdisjoint(policy.forbidden_fields)
    client, _ = make_pair(client_policy=policy)
    client.send(CHUNKED_REQUEST_HEAD)
    client.send(Data(b"abcd"))
    with pytest.raises(LocalProtocolError, match="forbidden field"):
        client.send(EndOfMessage(headers=[("x-secret", "1")]))


def test_receive_rejects_forbidden_trailer() -> None:
    _, server = make_pair(server_policy=TrailerPolicy())
    events = receive_until_error(
        server,
        b"POST /upload HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"Trailer: X-Checksum\r\n"
        b"\r\n"
        b"4\r\nabcd\r\n"
        b"0\r\n"
        b"Authorization: Bearer stolen\r\n"
        b"\r\n",
    )
    # The Request and Data events arrived, but the offending EndOfMessage was
    # never delivered to the application.
    assert [type(e) for e in events] == [Request, Data]
    assert server.their_state is ERROR


def test_receive_rejects_undeclared_trailer() -> None:
    _, server = make_pair(server_policy=TrailerPolicy(require_declaration=True))
    events = receive_until_error(
        server,
        b"POST /upload HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"Trailer: X-Checksum\r\n"
        b"\r\n"
        b"4\r\nabcd\r\n"
        b"0\r\n"
        b"X-Checksum: abc\r\n"
        b"X-Surprise: 1\r\n"
        b"\r\n",
    )
    assert [type(e) for e in events] == [Request, Data]
    assert server.their_state is ERROR


def test_legal_trailers_survive_roundtrip_in_order() -> None:
    # Strict policy on both ends; declarations exercise case-insensitivity,
    # comma-separated multi-values, and duplicate Trailer headers. The trailer
    # section repeats a field name, and the received event must preserve the
    # exact field order.
    policy = TrailerPolicy(require_declaration=True)
    client, server = make_pair(client_policy=policy, server_policy=policy)
    request = Request(
        method="POST",
        target="/upload",
        headers=[
            ("Host", "example.com"),
            ("Transfer-Encoding", "chunked"),
            ("Trailer", "X-Checksum, X-Note"),
            ("Trailer", "x-note"),
        ],
    )
    eom = EndOfMessage(
        headers=[
            ("X-Note", "first"),
            ("X-Checksum", "abc123"),
            ("x-note", "second"),
        ]
    )
    events = pump(client, server, [request, Data(b"abcd"), eom])
    received_eom = events[-1]
    assert isinstance(received_eom, EndOfMessage)
    assert list(received_eom.headers) == [
        (b"x-note", b"first"),
        (b"x-checksum", b"abc123"),
        (b"x-note", b"second"),
    ]
    # Raw casing is preserved on the wire, in order.
    assert received_eom.headers.raw_items() == [
        (b"X-Note", b"first"),
        (b"X-Checksum", b"abc123"),
        (b"x-note", b"second"),
    ]
    assert server.their_state is DONE


def test_declaration_matching_is_case_insensitive() -> None:
    policy = TrailerPolicy(require_declaration=True)
    client, server = make_pair(client_policy=policy, server_policy=policy)
    request = Request(
        method="POST",
        target="/upload",
        headers=[
            ("Host", "example.com"),
            ("Transfer-Encoding", "chunked"),
            ("Trailer", "X-CHECKSUM"),
        ],
    )
    eom = EndOfMessage(headers=[("x-checksum", "abc")])
    events = pump(client, server, [request, Data(b"abcd"), eom])
    assert events[-1] == eom


def test_same_name_field_in_regular_headers_is_not_a_trailer() -> None:
    # Regression: a field name that is forbidden *in the trailer section* is
    # still perfectly fine in the regular header section, and a field may
    # appear both as a regular header and as a declared trailer.
    policy = TrailerPolicy(require_declaration=True)
    client, server = make_pair(client_policy=policy, server_policy=policy)
    request = Request(
        method="POST",
        target="/upload",
        headers=[
            ("Host", "example.com"),
            ("Authorization", "Bearer legit"),
            ("X-Checksum", "header-value"),
            ("Transfer-Encoding", "chunked"),
            ("Trailer", "X-Checksum"),
        ],
    )
    events = pump(
        client,
        server,
        [
            request,
            Data(b"abcd"),
            EndOfMessage(headers=[("X-Checksum", "trailer-value")]),
        ],
    )
    received_request = events[0]
    assert isinstance(received_request, Request)
    assert (b"authorization", b"Bearer legit") in list(received_request.headers)
    assert (b"x-checksum", b"header-value") in list(received_request.headers)
    received_eom = events[-1]
    assert isinstance(received_eom, EndOfMessage)
    assert list(received_eom.headers) == [(b"x-checksum", b"trailer-value")]


def test_response_trailers_roundtrip() -> None:
    # The policy applies to responses too, on both directions.
    policy = TrailerPolicy(require_declaration=True)
    client, server = make_pair(client_policy=policy, server_policy=policy)
    pump(
        client,
        server,
        [Request(method="GET", target="/", headers=[("Host", "example.com")])],
    )
    response = Response(
        status_code=200,
        headers=[("Trailer", "X-Checksum"), ("Transfer-Encoding", "chunked")],
    )
    eom = EndOfMessage(headers=[("X-Checksum", "abc")])
    events = pump(server, client, [response, Data(b"body"), eom])
    assert events[-1] == eom

    # An undeclared response trailer is rejected before any bytes are emitted.
    client3, server3 = make_pair(server_policy=policy)
    pump(
        client3,
        server3,
        [Request(method="GET", target="/", headers=[("Host", "example.com")])],
    )
    server3.send(Response(status_code=200, headers=[]))
    server3.send(Data(b"body"))
    with pytest.raises(LocalProtocolError, match="not announced"):
        server3.send(EndOfMessage(headers=[("X-Other", "1")]))
