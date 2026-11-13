from typing import cast, List, Optional, Tuple, Type, Union

import pytest

from .._connection import Connection, NEED_DATA, PAUSED
from .._events import ConnectionClosed, Data, EndOfMessage, Event, Request
from .._receivebuffer import ReceivePolicy
from .._state import CLIENT, ERROR, SERVER
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import normalize_data_events


def _run(
    chunks: List[bytes], role: Type[Sentinel], policy: Union[str, ReceivePolicy]
) -> Tuple[List[Event], Optional[RemoteProtocolError], Connection]:
    conn = Connection(role, receive_policy=policy)
    events: List[Event] = []
    error: Optional[RemoteProtocolError] = None
    for chunk in chunks:
        conn.receive_data(chunk)
        while True:
            try:
                event = conn.next_event()
            except RemoteProtocolError as exc:
                error = exc
                break
            if event in (NEED_DATA, PAUSED):
                break
            event = cast(Event, event)
            events.append(event)
            if type(event) is ConnectionClosed:
                break
        if error is not None:
            break
    return normalize_data_events(events), error, conn


TOLERATED_LF_MESSAGES = [
    pytest.param(
        SERVER,
        b"GET / HTTP/1.1\nHost: example.com\r\n\r\n",
        id="start-line",
    ),
    pytest.param(
        SERVER,
        b"GET / HTTP/1.1\r\nHost: example.com\n\r\n",
        id="header",
    ),
    pytest.param(
        SERVER,
        b"GET / HTTP/1.1\r\nHost: example.com\r\n\n",
        id="end-of-headers",
    ),
    pytest.param(
        CLIENT,
        b"HTTP/1.1 200 OK\nContent-Length: 0\r\n\r\n",
        id="status-line",
    ),
    pytest.param(
        SERVER,
        b"POST / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
        b"0\r\nX-Trailer: value\n\r\n",
        id="trailer",
    ),
    pytest.param(
        SERVER,
        b"POST / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
        b"0\r\nX-Trailer: value\r\n\n",
        id="end-of-trailers",
    ),
]


@pytest.mark.parametrize("role,data", TOLERATED_LF_MESSAGES)
def test_compatible_and_strict_at_every_single_cut(
    role: Type[Sentinel], data: bytes
) -> None:
    compatible_events, compatible_error, _ = _run([data], role, "compatible")
    strict_events, strict_error, _ = _run([data], role, "strict")

    assert compatible_error is None
    assert strict_error is not None

    for cut in range(1, len(data)):
        events, error, _ = _run([data[:cut], data[cut:]], role, "compatible")
        assert error is None
        assert events == compatible_events

        events, error, _ = _run([data[:cut], data[cut:]], role, "strict")
        assert error is not None
        assert type(error) is RemoteProtocolError
        assert events == strict_events


ALWAYS_REJECTED_LF_MESSAGES = [
    pytest.param(
        b"POST / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
        b"1\nx\r\n0\r\n\r\n",
        id="chunk-size",
    ),
    pytest.param(
        b"POST / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
        b"1\r\nx\n0\r\n\r\n",
        id="chunk-footer",
    ),
]


@pytest.mark.parametrize("data", ALWAYS_REJECTED_LF_MESSAGES)
def test_chunk_lf_is_rejected_in_both_modes_at_every_single_cut(data: bytes) -> None:
    compatible_events, compatible_error, _ = _run([data], SERVER, "compatible")
    strict_events, strict_error, _ = _run([data], SERVER, "strict")

    assert compatible_error is not None
    assert strict_error is not None

    for cut in range(1, len(data)):
        events, error, _ = _run([data[:cut], data[cut:]], SERVER, "compatible")
        assert type(error) is RemoteProtocolError
        assert events == compatible_events

        events, error, _ = _run([data[:cut], data[cut:]], SERVER, "strict")
        assert type(error) is RemoteProtocolError
        assert events == strict_events


def test_strict_rejects_bare_lf_before_line_is_complete() -> None:
    conn = Connection(SERVER, receive_policy=ReceivePolicy.STRICT_CRLF)
    conn.receive_data(b"GET / HTTP/1.1\n")
    with pytest.raises(RemoteProtocolError, match="bare LF"):
        conn.next_event()

    conn = Connection(SERVER, receive_policy="strict")
    conn.receive_data(
        b"POST / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
        b"1\n"
    )
    assert type(conn.next_event()) is Request
    with pytest.raises(RemoteProtocolError, match="bare LF"):
        conn.next_event()


def test_strict_allows_lf_in_content_length_body() -> None:
    data = (
        b"POST / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Content-Length: 5\r\n\r\n"
        b"\n\r\nX\n"
    )
    compatible_events, compatible_error, _ = _run([data], SERVER, "compatible")
    strict_events, strict_error, _ = _run([data], SERVER, "strict")

    assert compatible_error is None
    assert strict_error is None
    assert strict_events == compatible_events
    assert strict_events == [
        Request(
            method=b"POST",
            target=b"/",
            headers=[
                (b"host", b"example.com"),
                (b"content-length", b"5"),
            ],
        ),
        Data(data=b"\n\r\nX\n"),
        EndOfMessage(headers=[]),
    ]

    for cut in range(1, len(data)):
        events, error, _ = _run([data[:cut], data[cut:]], SERVER, "strict")
        assert error is None
        assert events == strict_events


def test_strict_allows_lf_in_chunked_body() -> None:
    data = (
        b"POST / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
        b"4\r\n\n\r\nx\r\n"
        b"0\r\nT: v\r\n\r\n"
    )
    events, error, _ = _run([data], SERVER, "strict")
    assert error is None
    assert events == [
        Request(
            method=b"POST",
            target=b"/",
            headers=[
                (b"host", b"example.com"),
                (b"transfer-encoding", b"chunked"),
            ],
        ),
        Data(data=b"\n\r\nx"),
        EndOfMessage(headers=[(b"t", b"v")]),
    ]

    for cut in range(1, len(data)):
        got_events, got_error, _ = _run([data[:cut], data[cut:]], SERVER, "strict")
        assert got_error is None
        assert got_events == events


def test_strict_error_prevents_reusing_a_partial_message() -> None:
    conn = Connection(SERVER, receive_policy="strict")
    conn.receive_data(
        b"POST / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
        b"1\r\nx\r\n0\r\nX-Trailer: value\n\r\n"
    )
    assert type(conn.next_event()) is Request
    assert conn.next_event() == Data(data=b"x", chunk_start=True, chunk_end=True)

    with pytest.raises(RemoteProtocolError):
        conn.next_event()
    assert conn.their_state is ERROR
    with pytest.raises(LocalProtocolError):
        conn.start_next_cycle()


def test_receive_policy_does_not_change_sending() -> None:
    event = Request(method="GET", target="/", headers=[("Host", "example.com")])
    compatible = Connection(CLIENT, receive_policy="compatible")
    strict = Connection(CLIENT, receive_policy="strict")
    assert strict.send(event) == compatible.send(event)


def test_receive_policy_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        Connection(SERVER, receive_policy="sometimes")
