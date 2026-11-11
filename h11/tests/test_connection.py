from typing import Any, cast, Dict, List, Optional, Tuple, Type

import pytest

from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
    ConnectionClosed,
    Data,
    EndOfMessage,
    InformationalResponse,
    Request,
    Response,
)
from .._state import (
    CLIENT,
    CLOSED,
    DONE,
    ERROR,
    MIGHT_SWITCH_PROTOCOL,
    MUST_CLOSE,
    SEND_BODY,
    SEND_RESPONSE,
    SERVER,
    SWITCHED_PROTOCOL,
)
from .._util import (
    LENIENT,
    LineEndingPolicy,
    LocalProtocolError,
    RemoteProtocolError,
    Sentinel,
    STRICT,
)
from .helpers import ConnectionPair, get_all_events, receive_and_get


def test__keep_alive() -> None:
    assert _keep_alive(
        Request(method="GET", target="/", headers=[("Host", "Example.com")])
    )
    assert not _keep_alive(
        Request(
            method="GET",
            target="/",
            headers=[("Host", "Example.com"), ("Connection", "close")],
        )
    )
    assert not _keep_alive(
        Request(
            method="GET",
            target="/",
            headers=[("Host", "Example.com"), ("Connection", "a, b, cLOse, foo")],
        )
    )
    assert not _keep_alive(
        Request(method="GET", target="/", headers=[], http_version="1.0")
    )

    assert _keep_alive(Response(status_code=200, headers=[]))
    assert not _keep_alive(Response(status_code=200, headers=[("Connection", "close")]))
    assert not _keep_alive(
        Response(status_code=200, headers=[("Connection", "a, b, cLOse, foo")])
    )
    assert not _keep_alive(Response(status_code=200, headers=[], http_version="1.0"))


def test__body_framing() -> None:
    def headers(cl: Optional[int], te: bool) -> List[Tuple[str, str]]:
        headers = []
        if cl is not None:
            headers.append(("Content-Length", str(cl)))
        if te:
            headers.append(("Transfer-Encoding", "chunked"))
        return headers

    def resp(
        status_code: int = 200, cl: Optional[int] = None, te: bool = False
    ) -> Response:
        return Response(status_code=status_code, headers=headers(cl, te))

    def req(cl: Optional[int] = None, te: bool = False) -> Request:
        h = headers(cl, te)
        h += [("Host", "example.com")]
        return Request(method="GET", target="/", headers=h)

    # Special cases where the headers are ignored:
    for kwargs in [{}, {"cl": 100}, {"te": True}, {"cl": 100, "te": True}]:
        kwargs = cast(Dict[str, Any], kwargs)
        for meth, r in [
            (b"HEAD", resp(**kwargs)),
            (b"GET", resp(status_code=204, **kwargs)),
            (b"GET", resp(status_code=304, **kwargs)),
        ]:
            assert _body_framing(meth, r) == ("content-length", (0,))

    # Transfer-encoding
    for kwargs in [{"te": True}, {"cl": 100, "te": True}]:
        kwargs = cast(Dict[str, Any], kwargs)
        for meth, r in [(None, req(**kwargs)), (b"GET", resp(**kwargs))]:  # type: ignore
            assert _body_framing(meth, r) == ("chunked", ())

    # Content-Length
    for meth, r in [(None, req(cl=100)), (b"GET", resp(cl=100))]:  # type: ignore
        assert _body_framing(meth, r) == ("content-length", (100,))

    # No headers
    assert _body_framing(None, req()) == ("content-length", (0,))  # type: ignore
    assert _body_framing(b"GET", resp()) == ("http/1.0", ())


def test_Connection_basics_and_content_length() -> None:
    with pytest.raises(ValueError):
        Connection("CLIENT")  # type: ignore

    p = ConnectionPair()
    assert p.conn[CLIENT].our_role is CLIENT
    assert p.conn[CLIENT].their_role is SERVER
    assert p.conn[SERVER].our_role is SERVER
    assert p.conn[SERVER].their_role is CLIENT

    data = p.send(
        CLIENT,
        Request(
            method="GET",
            target="/",
            headers=[("Host", "example.com"), ("Content-Length", "10")],
        ),
    )
    assert data == (
        b"GET / HTTP/1.1\r\n" b"Host: example.com\r\n" b"Content-Length: 10\r\n\r\n"
    )

    for conn in p.conns:
        assert conn.states == {CLIENT: SEND_BODY, SERVER: SEND_RESPONSE}
    assert p.conn[CLIENT].our_state is SEND_BODY
    assert p.conn[CLIENT].their_state is SEND_RESPONSE
    assert p.conn[SERVER].our_state is SEND_RESPONSE
    assert p.conn[SERVER].their_state is SEND_BODY

    assert p.conn[CLIENT].their_http_version is None
    assert p.conn[SERVER].their_http_version == b"1.1"

    data = p.send(SERVER, InformationalResponse(status_code=100, headers=[]))
    assert data == b"HTTP/1.1 100 \r\n\r\n"

    data = p.send(SERVER, Response(status_code=200, headers=[("Content-Length", "11")]))
    assert data == b"HTTP/1.1 200 \r\nContent-Length: 11\r\n\r\n"

    for conn in p.conns:
        assert conn.states == {CLIENT: SEND_BODY, SERVER: SEND_BODY}

    assert p.conn[CLIENT].their_http_version == b"1.1"
    assert p.conn[SERVER].their_http_version == b"1.1"

    data = p.send(CLIENT, Data(data=b"12345"))
    assert data == b"12345"
    data = p.send(
        CLIENT, Data(data=b"67890"), expect=[Data(data=b"67890"), EndOfMessage()]
    )
    assert data == b"67890"
    data = p.send(CLIENT, EndOfMessage(), expect=[])
    assert data == b""

    for conn in p.conns:
        assert conn.states == {CLIENT: DONE, SERVER: SEND_BODY}

    data = p.send(SERVER, Data(data=b"1234567890"))
    assert data == b"1234567890"
    data = p.send(SERVER, Data(data=b"1"), expect=[Data(data=b"1"), EndOfMessage()])
    assert data == b"1"
    data = p.send(SERVER, EndOfMessage(), expect=[])
    assert data == b""

    for conn in p.conns:
        assert conn.states == {CLIENT: DONE, SERVER: DONE}


def test_chunked() -> None:
    p = ConnectionPair()

    p.send(
        CLIENT,
        Request(
            method="GET",
            target="/",
            headers=[("Host", "example.com"), ("Transfer-Encoding", "chunked")],
        ),
    )
    data = p.send(CLIENT, Data(data=b"1234567890", chunk_start=True, chunk_end=True))
    assert data == b"a\r\n1234567890\r\n"
    data = p.send(CLIENT, Data(data=b"abcde", chunk_start=True, chunk_end=True))
    assert data == b"5\r\nabcde\r\n"
    data = p.send(CLIENT, Data(data=b""), expect=[])
    assert data == b""
    data = p.send(CLIENT, EndOfMessage(headers=[("hello", "there")]))
    assert data == b"0\r\nhello: there\r\n\r\n"

    p.send(
        SERVER, Response(status_code=200, headers=[("Transfer-Encoding", "chunked")])
    )
    p.send(SERVER, Data(data=b"54321", chunk_start=True, chunk_end=True))
    p.send(SERVER, Data(data=b"12345", chunk_start=True, chunk_end=True))
    p.send(SERVER, EndOfMessage())

    for conn in p.conns:
        assert conn.states == {CLIENT: DONE, SERVER: DONE}


def test_chunk_boundaries() -> None:
    conn = Connection(our_role=SERVER)

    request = (
        b"POST / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
    )
    conn.receive_data(request)
    assert conn.next_event() == Request(
        method="POST",
        target="/",
        headers=[("Host", "example.com"), ("Transfer-Encoding", "chunked")],
    )
    assert conn.next_event() is NEED_DATA

    conn.receive_data(b"5\r\nhello\r\n")
    assert conn.next_event() == Data(data=b"hello", chunk_start=True, chunk_end=True)

    conn.receive_data(b"5\r\nhel")
    assert conn.next_event() == Data(data=b"hel", chunk_start=True, chunk_end=False)

    conn.receive_data(b"l")
    assert conn.next_event() == Data(data=b"l", chunk_start=False, chunk_end=False)

    conn.receive_data(b"o\r\n")
    assert conn.next_event() == Data(data=b"o", chunk_start=False, chunk_end=True)

    conn.receive_data(b"5\r\nhello")
    assert conn.next_event() == Data(data=b"hello", chunk_start=True, chunk_end=True)

    conn.receive_data(b"\r\n")
    assert conn.next_event() == NEED_DATA

    conn.receive_data(b"0\r\n\r\n")
    assert conn.next_event() == EndOfMessage()


def test_client_talking_to_http10_server() -> None:
    c = Connection(CLIENT)
    c.send(Request(method="GET", target="/", headers=[("Host", "example.com")]))
    c.send(EndOfMessage())
    assert c.our_state is DONE
    # No content-length, so Http10 framing for body
    assert receive_and_get(c, b"HTTP/1.0 200 OK\r\n\r\n") == [
        Response(status_code=200, headers=[], http_version="1.0", reason=b"OK")
    ]
    assert c.our_state is MUST_CLOSE
    assert receive_and_get(c, b"12345") == [Data(data=b"12345")]
    assert receive_and_get(c, b"67890") == [Data(data=b"67890")]
    assert receive_and_get(c, b"") == [EndOfMessage(), ConnectionClosed()]
    assert c.their_state is CLOSED


def test_server_talking_to_http10_client() -> None:
    c = Connection(SERVER)
    # No content-length, so no body
    # NB: no host header
    assert receive_and_get(c, b"GET / HTTP/1.0\r\n\r\n") == [
        Request(method="GET", target="/", headers=[], http_version="1.0"),
        EndOfMessage(),
    ]
    assert c.their_state is MUST_CLOSE

    # We automatically Connection: close back at them
    assert (
        c.send(Response(status_code=200, headers=[]))
        == b"HTTP/1.1 200 \r\nConnection: close\r\n\r\n"
    )

    assert c.send(Data(data=b"12345")) == b"12345"
    assert c.send(EndOfMessage()) == b""
    assert c.our_state is MUST_CLOSE

    # Check that it works if they do send Content-Length
    c = Connection(SERVER)
    # NB: no host header
    assert receive_and_get(c, b"POST / HTTP/1.0\r\nContent-Length: 10\r\n\r\n1") == [
        Request(
            method="POST",
            target="/",
            headers=[("Content-Length", "10")],
            http_version="1.0",
        ),
        Data(data=b"1"),
    ]
    assert receive_and_get(c, b"234567890") == [Data(data=b"234567890"), EndOfMessage()]
    assert c.their_state is MUST_CLOSE
    assert receive_and_get(c, b"") == [ConnectionClosed()]


def test_automatic_transfer_encoding_in_response() -> None:
    # Check that in responses, the user can specify either Transfer-Encoding:
    # chunked or no framing at all, and in both cases we automatically select
    # the right option depending on whether the peer speaks HTTP/1.0 or
    # HTTP/1.1
    for user_headers in [
        [("Transfer-Encoding", "chunked")],
        [],
        # In fact, this even works if Content-Length is set,
        # because if both are set then Transfer-Encoding wins
        [("Transfer-Encoding", "chunked"), ("Content-Length", "100")],
    ]:
        user_headers = cast(List[Tuple[str, str]], user_headers)
        p = ConnectionPair()
        p.send(
            CLIENT,
            [
                Request(method="GET", target="/", headers=[("Host", "example.com")]),
                EndOfMessage(),
            ],
        )
        # When speaking to HTTP/1.1 client, all of the above cases get
        # normalized to Transfer-Encoding: chunked
        p.send(
            SERVER,
            Response(status_code=200, headers=user_headers),
            expect=Response(
                status_code=200, headers=[("Transfer-Encoding", "chunked")]
            ),
        )

        # When speaking to HTTP/1.0 client, all of the above cases get
        # normalized to no-framing-headers
        c = Connection(SERVER)
        receive_and_get(c, b"GET / HTTP/1.0\r\n\r\n")
        assert (
            c.send(Response(status_code=200, headers=user_headers))
            == b"HTTP/1.1 200 \r\nConnection: close\r\n\r\n"
        )
        assert c.send(Data(data=b"12345")) == b"12345"


def test_automagic_connection_close_handling() -> None:
    p = ConnectionPair()
    # If the user explicitly sets Connection: close, then we notice and
    # respect it
    p.send(
        CLIENT,
        [
            Request(
                method="GET",
                target="/",
                headers=[("Host", "example.com"), ("Connection", "close")],
            ),
            EndOfMessage(),
        ],
    )
    for conn in p.conns:
        assert conn.states[CLIENT] is MUST_CLOSE
    # And if the client sets it, the server automatically echoes it back
    p.send(
        SERVER,
        # no header here...
        [Response(status_code=204, headers=[]), EndOfMessage()],
        # ...but oh look, it arrived anyway
        expect=[
            Response(status_code=204, headers=[("connection", "close")]),
            EndOfMessage(),
        ],
    )
    for conn in p.conns:
        assert conn.states == {CLIENT: MUST_CLOSE, SERVER: MUST_CLOSE}


def test_100_continue() -> None:
    def setup() -> ConnectionPair:
        p = ConnectionPair()
        p.send(
            CLIENT,
            Request(
                method="GET",
                target="/",
                headers=[
                    ("Host", "example.com"),
                    ("Content-Length", "100"),
                    ("Expect", "100-continue"),
                ],
            ),
        )
        for conn in p.conns:
            assert conn.client_is_waiting_for_100_continue
        assert not p.conn[CLIENT].they_are_waiting_for_100_continue
        assert p.conn[SERVER].they_are_waiting_for_100_continue
        return p

    # Disabled by 100 Continue
    p = setup()
    p.send(SERVER, InformationalResponse(status_code=100, headers=[]))
    for conn in p.conns:
        assert not conn.client_is_waiting_for_100_continue
        assert not conn.they_are_waiting_for_100_continue

    # Disabled by a real response
    p = setup()
    p.send(
        SERVER, Response(status_code=200, headers=[("Transfer-Encoding", "chunked")])
    )
    for conn in p.conns:
        assert not conn.client_is_waiting_for_100_continue
        assert not conn.they_are_waiting_for_100_continue

    # Disabled by the client going ahead and sending stuff anyway
    p = setup()
    p.send(CLIENT, Data(data=b"12345"))
    for conn in p.conns:
        assert not conn.client_is_waiting_for_100_continue
        assert not conn.they_are_waiting_for_100_continue


def test_max_incomplete_event_size_countermeasure() -> None:
    # Infinitely long headers are definitely not okay
    c = Connection(SERVER)
    c.receive_data(b"GET / HTTP/1.0\r\nEndless: ")
    assert c.next_event() is NEED_DATA
    with pytest.raises(RemoteProtocolError):
        while True:
            c.receive_data(b"a" * 1024)
            c.next_event()

    # Checking that the same header is accepted / rejected depending on the
    # max_incomplete_event_size setting:
    c = Connection(SERVER, max_incomplete_event_size=5000)
    c.receive_data(b"GET / HTTP/1.0\r\nBig: ")
    c.receive_data(b"a" * 4000)
    c.receive_data(b"\r\n\r\n")
    assert get_all_events(c) == [
        Request(
            method="GET", target="/", http_version="1.0", headers=[("big", "a" * 4000)]
        ),
        EndOfMessage(),
    ]

    c = Connection(SERVER, max_incomplete_event_size=4000)
    c.receive_data(b"GET / HTTP/1.0\r\nBig: ")
    c.receive_data(b"a" * 4000)
    with pytest.raises(RemoteProtocolError):
        c.next_event()

    # Temporarily exceeding the size limit is fine, as long as its done with
    # complete events:
    c = Connection(SERVER, max_incomplete_event_size=5000)
    c.receive_data(b"GET / HTTP/1.0\r\nContent-Length: 10000")
    c.receive_data(b"\r\n\r\n" + b"a" * 10000)
    assert get_all_events(c) == [
        Request(
            method="GET",
            target="/",
            http_version="1.0",
            headers=[("Content-Length", "10000")],
        ),
        Data(data=b"a" * 10000),
        EndOfMessage(),
    ]

    c = Connection(SERVER, max_incomplete_event_size=100)
    # Two pipelined requests to create a way-too-big receive buffer... but
    # it's fine because we're not checking
    c.receive_data(
        b"GET /1 HTTP/1.1\r\nHost: a\r\n\r\n"
        b"GET /2 HTTP/1.1\r\nHost: b\r\n\r\n" + b"X" * 1000
    )
    assert get_all_events(c) == [
        Request(method="GET", target="/1", headers=[("host", "a")]),
        EndOfMessage(),
    ]
    # Even more data comes in, still no problem
    c.receive_data(b"X" * 1000)
    # We can respond and reuse to get the second pipelined request
    c.send(Response(status_code=200, headers=[]))
    c.send(EndOfMessage())
    c.start_next_cycle()
    assert get_all_events(c) == [
        Request(method="GET", target="/2", headers=[("host", "b")]),
        EndOfMessage(),
    ]
    # But once we unpause and try to read the next message, and find that it's
    # incomplete and the buffer is *still* way too large, then *that's* a
    # problem:
    c.send(Response(status_code=200, headers=[]))
    c.send(EndOfMessage())
    c.start_next_cycle()
    with pytest.raises(RemoteProtocolError):
        c.next_event()


def test_reuse_simple() -> None:
    p = ConnectionPair()
    p.send(
        CLIENT,
        [Request(method="GET", target="/", headers=[("Host", "a")]), EndOfMessage()],
    )
    p.send(
        SERVER,
        [
            Response(status_code=200, headers=[(b"transfer-encoding", b"chunked")]),
            EndOfMessage(),
        ],
    )
    for conn in p.conns:
        assert conn.states == {CLIENT: DONE, SERVER: DONE}
        conn.start_next_cycle()

    p.send(
        CLIENT,
        [
            Request(method="DELETE", target="/foo", headers=[("Host", "a")]),
            EndOfMessage(),
        ],
    )
    p.send(
        SERVER,
        [
            Response(status_code=404, headers=[(b"transfer-encoding", b"chunked")]),
            EndOfMessage(),
        ],
    )


def test_pipelining() -> None:
    # Client doesn't support pipelining, so we have to do this by hand
    c = Connection(SERVER)
    assert c.next_event() is NEED_DATA
    # 3 requests all bunched up
    c.receive_data(
        b"GET /1 HTTP/1.1\r\nHost: a.com\r\nContent-Length: 5\r\n\r\n"
        b"12345"
        b"GET /2 HTTP/1.1\r\nHost: a.com\r\nContent-Length: 5\r\n\r\n"
        b"67890"
        b"GET /3 HTTP/1.1\r\nHost: a.com\r\n\r\n"
    )
    assert get_all_events(c) == [
        Request(
            method="GET",
            target="/1",
            headers=[("Host", "a.com"), ("Content-Length", "5")],
        ),
        Data(data=b"12345"),
        EndOfMessage(),
    ]
    assert c.their_state is DONE
    assert c.our_state is SEND_RESPONSE

    assert c.next_event() is PAUSED

    c.send(Response(status_code=200, headers=[]))
    c.send(EndOfMessage())
    assert c.their_state is DONE
    assert c.our_state is DONE

    c.start_next_cycle()

    assert get_all_events(c) == [
        Request(
            method="GET",
            target="/2",
            headers=[("Host", "a.com"), ("Content-Length", "5")],
        ),
        Data(data=b"67890"),
        EndOfMessage(),
    ]
    assert c.next_event() is PAUSED
    c.send(Response(status_code=200, headers=[]))
    c.send(EndOfMessage())
    c.start_next_cycle()

    assert get_all_events(c) == [
        Request(method="GET", target="/3", headers=[("Host", "a.com")]),
        EndOfMessage(),
    ]
    # Doesn't pause this time, no trailing data
    assert c.next_event() is NEED_DATA
    c.send(Response(status_code=200, headers=[]))
    c.send(EndOfMessage())

    # Arrival of more data triggers pause
    assert c.next_event() is NEED_DATA
    c.receive_data(b"SADF")
    assert c.next_event() is PAUSED
    assert c.trailing_data == (b"SADF", False)
    # If EOF arrives while paused, we don't see that either:
    c.receive_data(b"")
    assert c.trailing_data == (b"SADF", True)
    assert c.next_event() is PAUSED
    c.receive_data(b"")
    assert c.next_event() is PAUSED
    # Can't call receive_data with non-empty buf after closing it
    with pytest.raises(RuntimeError):
        c.receive_data(b"FDSA")


def test_protocol_switch() -> None:
    for req, deny, accept in [
        (
            Request(
                method="CONNECT",
                target="example.com:443",
                headers=[("Host", "foo"), ("Content-Length", "1")],
            ),
            Response(status_code=404, headers=[(b"transfer-encoding", b"chunked")]),
            Response(status_code=200, headers=[(b"transfer-encoding", b"chunked")]),
        ),
        (
            Request(
                method="GET",
                target="/",
                headers=[("Host", "foo"), ("Content-Length", "1"), ("Upgrade", "a, b")],
            ),
            Response(status_code=200, headers=[(b"transfer-encoding", b"chunked")]),
            InformationalResponse(status_code=101, headers=[("Upgrade", "a")]),
        ),
        (
            Request(
                method="CONNECT",
                target="example.com:443",
                headers=[("Host", "foo"), ("Content-Length", "1"), ("Upgrade", "a, b")],
            ),
            Response(status_code=404, headers=[(b"transfer-encoding", b"chunked")]),
            # Accept CONNECT, not upgrade
            Response(status_code=200, headers=[(b"transfer-encoding", b"chunked")]),
        ),
        (
            Request(
                method="CONNECT",
                target="example.com:443",
                headers=[("Host", "foo"), ("Content-Length", "1"), ("Upgrade", "a, b")],
            ),
            Response(status_code=404, headers=[(b"transfer-encoding", b"chunked")]),
            # Accept Upgrade, not CONNECT
            InformationalResponse(status_code=101, headers=[("Upgrade", "b")]),
        ),
    ]:

        def setup() -> ConnectionPair:
            p = ConnectionPair()
            p.send(CLIENT, req)
            # No switch-related state change stuff yet; the client has to
            # finish the request before that kicks in
            for conn in p.conns:
                assert conn.states[CLIENT] is SEND_BODY
            p.send(CLIENT, [Data(data=b"1"), EndOfMessage()])
            for conn in p.conns:
                assert conn.states[CLIENT] is MIGHT_SWITCH_PROTOCOL
            assert p.conn[SERVER].next_event() is PAUSED
            return p

        # Test deny case
        p = setup()
        p.send(SERVER, deny)
        for conn in p.conns:
            assert conn.states == {CLIENT: DONE, SERVER: SEND_BODY}
        p.send(SERVER, EndOfMessage())
        # Check that re-use is still allowed after a denial
        for conn in p.conns:
            conn.start_next_cycle()

        # Test accept case
        p = setup()
        p.send(SERVER, accept)
        for conn in p.conns:
            assert conn.states == {CLIENT: SWITCHED_PROTOCOL, SERVER: SWITCHED_PROTOCOL}
            conn.receive_data(b"123")
            assert conn.next_event() is PAUSED
            conn.receive_data(b"456")
            assert conn.next_event() is PAUSED
            assert conn.trailing_data == (b"123456", False)

        # Pausing in might-switch, then recovery
        # (weird artificial case where the trailing data actually is valid
        # HTTP for some reason, because this makes it easier to test the state
        # logic)
        p = setup()
        sc = p.conn[SERVER]
        sc.receive_data(b"GET / HTTP/1.0\r\n\r\n")
        assert sc.next_event() is PAUSED
        assert sc.trailing_data == (b"GET / HTTP/1.0\r\n\r\n", False)
        sc.send(deny)
        assert sc.next_event() is PAUSED
        sc.send(EndOfMessage())
        sc.start_next_cycle()
        assert get_all_events(sc) == [
            Request(method="GET", target="/", headers=[], http_version="1.0"),
            EndOfMessage(),
        ]

        # When we're DONE, have no trailing data, and the connection gets
        # closed, we report ConnectionClosed(). When we're in might-switch or
        # switched, we don't.
        p = setup()
        sc = p.conn[SERVER]
        sc.receive_data(b"")
        assert sc.next_event() is PAUSED
        assert sc.trailing_data == (b"", True)
        p.send(SERVER, accept)
        assert sc.next_event() is PAUSED

        p = setup()
        sc = p.conn[SERVER]
        sc.receive_data(b"")
        assert sc.next_event() is PAUSED
        sc.send(deny)
        assert sc.next_event() == ConnectionClosed()

        # You can't send after switching protocols, or while waiting for a
        # protocol switch
        p = setup()
        with pytest.raises(LocalProtocolError):
            p.conn[CLIENT].send(
                Request(method="GET", target="/", headers=[("Host", "a")])
            )
        p = setup()
        p.send(SERVER, accept)
        with pytest.raises(LocalProtocolError):
            p.conn[SERVER].send(Data(data=b"123"))


def test_close_simple() -> None:
    # Just immediately closing a new connection without anything having
    # happened yet.
    for who_shot_first, who_shot_second in [(CLIENT, SERVER), (SERVER, CLIENT)]:

        def setup() -> ConnectionPair:
            p = ConnectionPair()
            p.send(who_shot_first, ConnectionClosed())
            for conn in p.conns:
                assert conn.states == {
                    who_shot_first: CLOSED,
                    who_shot_second: MUST_CLOSE,
                }
            return p

        # You can keep putting b"" into a closed connection, and you keep
        # getting ConnectionClosed() out:
        p = setup()
        assert p.conn[who_shot_second].next_event() == ConnectionClosed()
        assert p.conn[who_shot_second].next_event() == ConnectionClosed()
        p.conn[who_shot_second].receive_data(b"")
        assert p.conn[who_shot_second].next_event() == ConnectionClosed()
        # Second party can close...
        p = setup()
        p.send(who_shot_second, ConnectionClosed())
        for conn in p.conns:
            assert conn.our_state is CLOSED
            assert conn.their_state is CLOSED
        # But trying to receive new data on a closed connection is a
        # RuntimeError (not ProtocolError, because the problem here isn't
        # violation of HTTP, it's violation of physics)
        p = setup()
        with pytest.raises(RuntimeError):
            p.conn[who_shot_second].receive_data(b"123")
        # And receiving new data on a MUST_CLOSE connection is a ProtocolError
        p = setup()
        p.conn[who_shot_first].receive_data(b"GET")
        with pytest.raises(RemoteProtocolError):
            p.conn[who_shot_first].next_event()


def test_close_different_states() -> None:
    req = [
        Request(method="GET", target="/foo", headers=[("Host", "a")]),
        EndOfMessage(),
    ]
    resp = [
        Response(status_code=200, headers=[(b"transfer-encoding", b"chunked")]),
        EndOfMessage(),
    ]

    # Client before request
    p = ConnectionPair()
    p.send(CLIENT, ConnectionClosed())
    for conn in p.conns:
        assert conn.states == {CLIENT: CLOSED, SERVER: MUST_CLOSE}

    # Client after request
    p = ConnectionPair()
    p.send(CLIENT, req)
    p.send(CLIENT, ConnectionClosed())
    for conn in p.conns:
        assert conn.states == {CLIENT: CLOSED, SERVER: SEND_RESPONSE}

    # Server after request -> not allowed
    p = ConnectionPair()
    p.send(CLIENT, req)
    with pytest.raises(LocalProtocolError):
        p.conn[SERVER].send(ConnectionClosed())
    p.conn[CLIENT].receive_data(b"")
    with pytest.raises(RemoteProtocolError):
        p.conn[CLIENT].next_event()

    # Server after response
    p = ConnectionPair()
    p.send(CLIENT, req)
    p.send(SERVER, resp)
    p.send(SERVER, ConnectionClosed())
    for conn in p.conns:
        assert conn.states == {CLIENT: MUST_CLOSE, SERVER: CLOSED}

    # Both after closing (ConnectionClosed() is idempotent)
    p = ConnectionPair()
    p.send(CLIENT, req)
    p.send(SERVER, resp)
    p.send(CLIENT, ConnectionClosed())
    p.send(SERVER, ConnectionClosed())
    p.send(CLIENT, ConnectionClosed())
    p.send(SERVER, ConnectionClosed())

    # In the middle of sending -> not allowed
    p = ConnectionPair()
    p.send(
        CLIENT,
        Request(
            method="GET", target="/", headers=[("Host", "a"), ("Content-Length", "10")]
        ),
    )
    with pytest.raises(LocalProtocolError):
        p.conn[CLIENT].send(ConnectionClosed())
    p.conn[SERVER].receive_data(b"")
    with pytest.raises(RemoteProtocolError):
        p.conn[SERVER].next_event()


# Receive several requests and then client shuts down their side of the
# connection; we can respond to each
def test_pipelined_close() -> None:
    c = Connection(SERVER)
    # 2 requests then a close
    c.receive_data(
        b"GET /1 HTTP/1.1\r\nHost: a.com\r\nContent-Length: 5\r\n\r\n"
        b"12345"
        b"GET /2 HTTP/1.1\r\nHost: a.com\r\nContent-Length: 5\r\n\r\n"
        b"67890"
    )
    c.receive_data(b"")
    assert get_all_events(c) == [
        Request(
            method="GET",
            target="/1",
            headers=[("host", "a.com"), ("content-length", "5")],
        ),
        Data(data=b"12345"),
        EndOfMessage(),
    ]
    assert c.states[CLIENT] is DONE
    c.send(Response(status_code=200, headers=[]))
    c.send(EndOfMessage())
    assert c.states[SERVER] is DONE
    c.start_next_cycle()
    assert get_all_events(c) == [
        Request(
            method="GET",
            target="/2",
            headers=[("host", "a.com"), ("content-length", "5")],
        ),
        Data(data=b"67890"),
        EndOfMessage(),
        ConnectionClosed(),
    ]
    assert c.states == {CLIENT: CLOSED, SERVER: SEND_RESPONSE}
    c.send(Response(status_code=200, headers=[]))
    c.send(EndOfMessage())
    assert c.states == {CLIENT: CLOSED, SERVER: MUST_CLOSE}
    c.send(ConnectionClosed())
    assert c.states == {CLIENT: CLOSED, SERVER: CLOSED}


def test_sendfile() -> None:
    class SendfilePlaceholder:
        def __len__(self) -> int:
            return 10

    placeholder = SendfilePlaceholder()

    def setup(
        header: Tuple[str, str], http_version: str
    ) -> Tuple[Connection, Optional[List[bytes]]]:
        c = Connection(SERVER)
        receive_and_get(
            c, f"GET / HTTP/{http_version}\r\nHost: a\r\n\r\n".encode("ascii")
        )
        headers = []
        if header:
            headers.append(header)
        c.send(Response(status_code=200, headers=headers))
        return c, c.send_with_data_passthrough(Data(data=placeholder))  # type: ignore

    c, data = setup(("Content-Length", "10"), "1.1")
    assert data == [placeholder]  # type: ignore
    # Raises an error if the connection object doesn't think we've sent
    # exactly 10 bytes
    c.send(EndOfMessage())

    _, data = setup(("Transfer-Encoding", "chunked"), "1.1")
    assert placeholder in data  # type: ignore
    data[data.index(placeholder)] = b"x" * 10  # type: ignore
    assert b"".join(data) == b"a\r\nxxxxxxxxxx\r\n"  # type: ignore

    c, data = setup(None, "1.0")  # type: ignore
    assert data == [placeholder]  # type: ignore
    assert c.our_state is SEND_BODY


def test_errors() -> None:
    # After a receive error, you can't receive
    for role in [CLIENT, SERVER]:
        c = Connection(our_role=role)
        c.receive_data(b"gibberish\r\n\r\n")
        with pytest.raises(RemoteProtocolError):
            c.next_event()
        # Now any attempt to receive continues to raise
        assert c.their_state is ERROR
        assert c.our_state is not ERROR
        print(c._cstate.states)
        with pytest.raises(RemoteProtocolError):
            c.next_event()
        # But we can still yell at the client for sending us gibberish
        if role is SERVER:
            assert (
                c.send(Response(status_code=400, headers=[]))
                == b"HTTP/1.1 400 \r\nConnection: close\r\n\r\n"
            )

    # After an error sending, you can no longer send
    # (This is especially important for things like content-length errors,
    # where there's complex internal state being modified)
    def conn(role: Type[Sentinel]) -> Connection:
        c = Connection(our_role=role)
        if role is SERVER:
            # Put it into the state where it *could* send a response...
            receive_and_get(c, b"GET / HTTP/1.0\r\n\r\n")
            assert c.our_state is SEND_RESPONSE
        return c

    for role in [CLIENT, SERVER]:
        if role is CLIENT:
            # This HTTP/1.0 request won't be detected as bad until after we go
            # through the state machine and hit the writing code
            good = Request(method="GET", target="/", headers=[("Host", "example.com")])
            bad = Request(
                method="GET",
                target="/",
                headers=[("Host", "example.com")],
                http_version="1.0",
            )
        elif role is SERVER:
            good = Response(status_code=200, headers=[])  # type: ignore[assignment]
            bad = Response(status_code=200, headers=[], http_version="1.0")  # type: ignore[assignment]
        # Make sure 'good' actually is good
        c = conn(role)
        c.send(good)
        assert c.our_state is not ERROR
        # Do that again, but this time sending 'bad' first
        c = conn(role)
        with pytest.raises(LocalProtocolError):
            c.send(bad)
        assert c.our_state is ERROR
        assert c.their_state is not ERROR
        # Now 'good' is not so good
        with pytest.raises(LocalProtocolError):
            c.send(good)

        # And check send_failed() too
        c = conn(role)
        c.send_failed()
        assert c.our_state is ERROR
        assert c.their_state is not ERROR
        # This is idempotent
        c.send_failed()
        assert c.our_state is ERROR
        assert c.their_state is not ERROR


def test_idle_receive_nothing() -> None:
    # At one point this incorrectly raised an error
    for role in [CLIENT, SERVER]:
        c = Connection(role)
        assert c.next_event() is NEED_DATA


def test_connection_drop() -> None:
    c = Connection(SERVER)
    c.receive_data(b"GET /")
    assert c.next_event() is NEED_DATA
    c.receive_data(b"")
    with pytest.raises(RemoteProtocolError):
        c.next_event()


def test_408_request_timeout() -> None:
    # Should be able to send this spontaneously as a server without seeing
    # anything from client
    p = ConnectionPair()
    p.send(SERVER, Response(status_code=408, headers=[(b"connection", b"close")]))


# This used to raise IndexError
def test_empty_request() -> None:
    c = Connection(SERVER)
    c.receive_data(b"\r\n")
    with pytest.raises(RemoteProtocolError):
        c.next_event()


# This used to raise IndexError
def test_empty_response() -> None:
    c = Connection(CLIENT)
    c.send(Request(method="GET", target="/", headers=[("Host", "a")]))
    c.receive_data(b"\r\n")
    with pytest.raises(RemoteProtocolError):
        c.next_event()


@pytest.mark.parametrize(
    "data",
    [
        b"\x00",
        b"\x20",
        b"\x16\x03\x01\x00\xa5",  # Typical start of a TLS Client Hello
    ],
)
def test_early_detection_of_invalid_request(data: bytes) -> None:
    c = Connection(SERVER)
    # Early detection should occur before even receiving a `\r\n`
    c.receive_data(data)
    with pytest.raises(RemoteProtocolError):
        c.next_event()


@pytest.mark.parametrize(
    "data",
    [
        b"\x00",
        b"\x20",
        b"\x16\x03\x03\x00\x31",  # Typical start of a TLS Server Hello
    ],
)
def test_early_detection_of_invalid_response(data: bytes) -> None:
    c = Connection(CLIENT)
    # Early detection should occur before even receiving a `\r\n`
    c.receive_data(data)
    with pytest.raises(RemoteProtocolError):
        c.next_event()


# This used to give different headers for HEAD and GET.
# The correct way to handle HEAD is to put whatever headers we *would* have
# put if it were a GET -- even though we know that for HEAD, those headers
# will be ignored.
def test_HEAD_framing_headers() -> None:
    def setup(method: bytes, http_version: bytes) -> Connection:
        c = Connection(SERVER)
        c.receive_data(
            method + b" / HTTP/" + http_version + b"\r\n" + b"Host: example.com\r\n\r\n"
        )
        assert type(c.next_event()) is Request
        assert type(c.next_event()) is EndOfMessage
        return c

    for method in [b"GET", b"HEAD"]:
        # No Content-Length, HTTP/1.1 peer, should use chunked
        c = setup(method, b"1.1")
        assert (
            c.send(Response(status_code=200, headers=[])) == b"HTTP/1.1 200 \r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
        )

        # No Content-Length, HTTP/1.0 peer, frame with connection: close
        c = setup(method, b"1.0")
        assert (
            c.send(Response(status_code=200, headers=[])) == b"HTTP/1.1 200 \r\n"
            b"Connection: close\r\n\r\n"
        )

        # Content-Length + Transfer-Encoding, TE wins
        c = setup(method, b"1.1")
        assert (
            c.send(
                Response(
                    status_code=200,
                    headers=[
                        ("Content-Length", "100"),
                        ("Transfer-Encoding", "chunked"),
                    ],
                )
            )
            == b"HTTP/1.1 200 \r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
        )


def test_special_exceptions_for_lost_connection_in_message_body() -> None:
    c = Connection(SERVER)
    c.receive_data(
        b"POST / HTTP/1.1\r\n" b"Host: example.com\r\n" b"Content-Length: 100\r\n\r\n"
    )
    assert type(c.next_event()) is Request
    assert c.next_event() is NEED_DATA
    c.receive_data(b"12345")
    assert c.next_event() == Data(data=b"12345")
    c.receive_data(b"")
    with pytest.raises(RemoteProtocolError) as excinfo:
        c.next_event()
    assert "received 5 bytes" in str(excinfo.value)
    assert "expected 100" in str(excinfo.value)

    c = Connection(SERVER)
    c.receive_data(
        b"POST / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
    )
    assert type(c.next_event()) is Request
    assert c.next_event() is NEED_DATA
    c.receive_data(b"8\r\n012345")
    assert c.next_event().data == b"012345"  # type: ignore
    c.receive_data(b"")
    with pytest.raises(RemoteProtocolError) as excinfo:
        c.next_event()
    assert "incomplete chunked read" in str(excinfo.value)


################################################################
# Line-ending policies
################################################################

# Each entry is (id, wire bytes). The CRLF variant establishes the expected
# event sequence; the others differ only in line terminators.
LINE_ENDING_MESSAGES = {
    "crlf_request": (b"GET / HTTP/1.1\r\nHost: a\r\nConnection: close\r\n\r\n"),
    "lf_request_line": (b"GET / HTTP/1.1\nHost: a\r\nConnection: close\r\n\r\n"),
    "lf_header": (b"GET / HTTP/1.1\r\nHost: a\nConnection: close\r\n\r\n"),
    "lf_last_header": (b"GET / HTTP/1.1\r\nHost: a\r\nConnection: close\n\r\n"),
    "lf_blank_line": (b"GET / HTTP/1.1\r\nHost: a\r\nConnection: close\r\n\n"),
    "all_lf": (b"GET / HTTP/1.1\nHost: a\nConnection: close\n\n"),
}

GOOD_CRLF_REQUEST = LINE_ENDING_MESSAGES["crlf_request"]

LINE_ENDING_POLICIES: Tuple[LineEndingPolicy, LineEndingPolicy] = (LENIENT, STRICT)


def _consume_all(c: Connection) -> List[Any]:
    """Return every event until next_event() needs more data, letting any
    protocol error propagate to the caller.
    """
    events: List[Any] = []
    while True:
        event = c.next_event()
        if event is NEED_DATA or event is PAUSED:
            return events
        events.append(event)
        if type(event) is ConnectionClosed:
            return events


def _feed_in_two_parts(c: Connection, data: bytes, cut: int) -> None:
    """Feed *data* split at *cut* bytes. Empty halves are skipped because
    receive_data(b"") means EOF.
    """
    if cut > 0:
        c.receive_data(data[:cut])
    if cut < len(data):
        c.receive_data(data[cut:])


def _result_at_cut(data: bytes, policy: Any, cut: int) -> Any:
    c = Connection(SERVER, line_ending_policy=policy)
    _feed_in_two_parts(c, data, cut)
    try:
        return _consume_all(c)
    except RemoteProtocolError as exc:
        return exc


def test_line_ending_policy_defaults_to_lenient() -> None:
    c = Connection(SERVER)
    assert c.line_ending_policy is LENIENT
    assert Connection(SERVER, line_ending_policy=LENIENT).line_ending_policy is LENIENT
    assert Connection(SERVER, line_ending_policy=STRICT).line_ending_policy is STRICT


def test_invalid_line_ending_policy_rejected() -> None:
    with pytest.raises(ValueError):
        Connection(SERVER, line_ending_policy="lenient")  # type: ignore[arg-type]


def test_lenient_still_accepts_bare_lf() -> None:
    # Backwards compatibility: the default policy is unchanged, and every
    # bare-LF variant parses to exactly the same events as all-CRLF input.
    expected = receive_and_get(Connection(SERVER), GOOD_CRLF_REQUEST)
    assert len(expected) == 2
    for name, data in LINE_ENDING_MESSAGES.items():
        got = receive_and_get(Connection(SERVER, line_ending_policy=LENIENT), data)
        assert got == expected, name


def test_strict_accepts_crlf() -> None:
    c = Connection(SERVER, line_ending_policy=STRICT)
    c.receive_data(GOOD_CRLF_REQUEST)
    assert _consume_all(c) == [
        Request(
            method="GET",
            target="/",
            headers=[("Host", "a"), ("Connection", "close")],
        ),
        EndOfMessage(),
    ]


@pytest.mark.parametrize(
    "name",
    [name for name in LINE_ENDING_MESSAGES if name != "crlf_request"],
)
def test_strict_rejects_bare_lf(name: str) -> None:
    c = Connection(SERVER, line_ending_policy=STRICT)
    c.receive_data(LINE_ENDING_MESSAGES[name])
    with pytest.raises(RemoteProtocolError):
        c.next_event()

    # A rejected message must not leave a half-open cycle that can be
    # recycled: the peer side is ERROR, so it can't be reused.
    assert c.their_state is ERROR
    with pytest.raises(LocalProtocolError):
        c.start_next_cycle()
    # And further reads keep failing rather than resuming a half-message.
    with pytest.raises(RemoteProtocolError):
        c.next_event()


def test_strict_rejects_bare_lf_in_response() -> None:
    # The policy applies symmetrically to a client parsing responses.
    c = Connection(CLIENT, line_ending_policy=STRICT)
    c.send(Request(method="GET", target="/", headers=[("Host", "a")]))
    c.receive_data(b"HTTP/1.1 200 OK\nContent-Length: 0\n\n")
    with pytest.raises(RemoteProtocolError):
        c.next_event()
    assert c.their_state is ERROR

    # Same bytes with CRLF parse fine.
    c = Connection(CLIENT, line_ending_policy=STRICT)
    c.send(Request(method="GET", target="/", headers=[("Host", "a")]))
    c.receive_data(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
    assert _consume_all(c) == [
        Response(status_code=200, headers=[("Content-Length", "0")], reason=b"OK"),
        EndOfMessage(),
    ]


@pytest.mark.parametrize("name", list(LINE_ENDING_MESSAGES))
def test_line_ending_result_is_stable_at_every_split(name: str) -> None:
    """Feeding the same short message with every possible single cut point
    must not change the outcome: LENIENT always yields the CRLF event
    sequence; STRICT yields it only for CRLF input and a protocol error
    otherwise.
    """
    data = LINE_ENDING_MESSAGES[name]
    lenient_expected = _result_at_cut(data, LENIENT, 0)
    assert lenient_expected == receive_and_get(Connection(SERVER), GOOD_CRLF_REQUEST)
    is_crlf = name == "crlf_request"
    strict_expected = _result_at_cut(data, STRICT, 0)

    for cut in range(len(data) + 1):
        lenient_result = _result_at_cut(data, LENIENT, cut)
        assert lenient_result == lenient_expected, (name, cut)

        strict_result = _result_at_cut(data, STRICT, cut)
        assert type(strict_result) is type(strict_expected), (name, cut)
        if isinstance(strict_result, RemoteProtocolError):
            assert strict_result.error_status_hint == 400
        else:
            assert strict_result == strict_expected, (name, cut)
            assert is_crlf, (name, cut)


def test_strict_crlf_survives_byte_at_a_time_feeding() -> None:
    # A CR at the end of one receive_data() call is completed by the LF in
    # the next; the strict check must look at complete lines only.
    c = Connection(SERVER, line_ending_policy=STRICT)
    events: List[Any] = []
    for byte in GOOD_CRLF_REQUEST:
        c.receive_data(bytes([byte]))
        events += _consume_all(c)
    assert events == [
        Request(
            method="GET",
            target="/",
            headers=[("Host", "a"), ("Connection", "close")],
        ),
        EndOfMessage(),
    ]


# Chunked framing: chunk size lines, chunk terminators, and trailers all go
# through the same policy.
CHUNKED_MESSAGES = {
    "crlf": (
        b"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n"
        b"Connection: close\r\n\r\n"
        b"5\r\nhello\r\n0\r\nX-T: v\r\n\r\n"
    ),
    "lf_chunk_size": (
        b"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n"
        b"Connection: close\r\n\r\n"
        b"5\nhello\r\n0\r\n\r\n"
    ),
    "lf_chunk_footer": (
        b"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n"
        b"Connection: close\r\n\r\n"
        b"5\r\nhello\n0\r\n\r\n"
    ),
    "lf_trailer": (
        b"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n"
        b"Connection: close\r\n\r\n"
        b"5\r\nhello\r\n0\r\nX-T: v\n\r\n"
    ),
    "lf_zero_chunk": (
        b"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n"
        b"Connection: close\r\n\r\n"
        b"0\n\r\n"
    ),
}


def test_chunked_crlf_events_for_reference() -> None:
    c = Connection(SERVER)
    c.receive_data(CHUNKED_MESSAGES["crlf"])
    assert _consume_all(c) == [
        Request(
            method="POST",
            target="/",
            headers=[
                ("Host", "a"),
                ("Transfer-Encoding", "chunked"),
                ("Connection", "close"),
            ],
        ),
        Data(data=b"hello", chunk_start=True, chunk_end=True),
        EndOfMessage(headers=[("X-T", "v")]),
    ]


@pytest.mark.parametrize("name", [name for name in CHUNKED_MESSAGES if name != "crlf"])
def test_chunked_bare_lf_rejected_by_strict(name: str) -> None:
    c = Connection(SERVER, line_ending_policy=STRICT)
    c.receive_data(CHUNKED_MESSAGES[name])
    with pytest.raises(RemoteProtocolError):
        for _ in range(10):
            c.next_event()
    assert c.their_state is ERROR
    with pytest.raises(LocalProtocolError):
        c.start_next_cycle()


# For each chunked message, whether each policy accepts it (events) or
# rejects it (RemoteProtocolError). LENIENT has always required CRLF for
# chunk size/terminator framing, but does accept bare LF in trailers.
CHUNKED_EXPECTATIONS = {
    "crlf": {LENIENT: "events", STRICT: "events"},
    "lf_chunk_size": {LENIENT: "error", STRICT: "error"},
    "lf_chunk_footer": {LENIENT: "error", STRICT: "error"},
    "lf_trailer": {LENIENT: "events", STRICT: "error"},
    "lf_zero_chunk": {LENIENT: "error", STRICT: "error"},
}


@pytest.mark.parametrize("name", list(CHUNKED_MESSAGES))
def test_chunked_outcome_is_stable_at_every_split(name: str) -> None:
    # Same matrix as the start-line/header messages: splitting the
    # payload at every single byte boundary must not change whether a
    # given policy accepts it or rejects it.
    data = CHUNKED_MESSAGES[name]
    reference = _result_at_cut(data, LENIENT, 0)
    for policy in LINE_ENDING_POLICIES:
        expected_kind = CHUNKED_EXPECTATIONS[name][policy]
        for cut in range(len(data) + 1):
            result = _result_at_cut(data, policy, cut)
            if expected_kind == "error":
                assert isinstance(result, RemoteProtocolError), (policy, name, cut)
                assert result.error_status_hint == 400
            else:
                assert result == reference, (policy, name, cut)


@pytest.mark.parametrize(
    "name",
    ["lf_chunk_size", "lf_chunk_footer", "lf_zero_chunk"],
)
def test_lenient_chunk_framing_was_already_rejected(name: str) -> None:
    # h11 has historically required CRLF for chunk framing even in lenient
    # mode (the chunk ABNF demands it), so these stay errors either way.
    c = Connection(SERVER, line_ending_policy=LENIENT)
    c.receive_data(CHUNKED_MESSAGES[name])
    with pytest.raises(RemoteProtocolError):
        for _ in range(10):
            c.next_event()


def test_lenient_allows_bare_lf_trailer() -> None:
    # ...but a bare LF in a trailer block is accepted by LENIENT, because
    # trailers are parsed as ordinary header lines.
    c = Connection(SERVER, line_ending_policy=LENIENT)
    c.receive_data(CHUNKED_MESSAGES["lf_trailer"])
    events = _consume_all(c)
    assert events[-1] == EndOfMessage(headers=[("X-T", "v")])


@pytest.mark.parametrize(
    "data,body",
    [
        (
            b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\nx\nyz\n",
            b"x\nyz\n",
        ),
        (
            b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 3\r\n\r\n\r\n\n",
            b"\r\n\n",
        ),
        (
            b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\n\r\n\r\nx",
            b"\r\n\r\nx",
        ),
    ],
)
def test_strict_never_applies_to_content_length_body(data: bytes, body: bytes) -> None:
    # Message bodies are opaque: bare LF (and bare CR) inside a
    # Content-Length body must not be mistaken for bad framing.
    for policy in LINE_ENDING_POLICIES:
        c = Connection(SERVER, line_ending_policy=policy)
        c.receive_data(data)
        events = _consume_all(c)
        assert type(events[0]) is Request
        assert events[1] == Data(data=body)
        assert events[2] == EndOfMessage()
        assert len(events) == 3


def test_strict_never_applies_to_chunk_body() -> None:
    # Same for the payload bytes inside a chunk.
    body = b"a\nb\r\nc\rd"
    data = (
        b"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n"
        b"Connection: close\r\n\r\n"
        + str(len(body)).encode()
        + b"\r\n"
        + body
        + b"\r\n0\r\n\r\n"
    )
    for policy in LINE_ENDING_POLICIES:
        c = Connection(SERVER, line_ending_policy=policy)
        c.receive_data(data)
        events = _consume_all(c)
        assert events[1] == Data(data=body, chunk_start=True, chunk_end=True)


def test_sending_is_unaffected_by_strict_policy() -> None:
    # The policy only governs parsing; our own output is always CRLF and
    # identical to what a LENIENT connection would produce.
    def client_request_bytes(policy: Any) -> bytes:
        client = Connection(CLIENT, line_ending_policy=policy)
        out = client.send(Request(method="GET", target="/", headers=[("Host", "a")]))
        assert out is not None
        return out

    def server_response_bytes(policy: Any) -> bytes:
        server = Connection(SERVER, line_ending_policy=policy)
        server.receive_data(GOOD_CRLF_REQUEST)
        _consume_all(server)
        out = server.send(Response(status_code=200, headers=[]))
        assert out is not None
        return out

    strict_request = client_request_bytes(STRICT)
    assert strict_request == client_request_bytes(LENIENT)
    assert strict_request == b"GET / HTTP/1.1\r\nHost: a\r\n\r\n"
    assert b"\n" not in strict_request.replace(b"\r\n", b"")

    strict_response = server_response_bytes(STRICT)
    assert strict_response == server_response_bytes(LENIENT)
    assert strict_response.startswith(b"HTTP/1.1 200 \r\n")
    assert b"\n" not in strict_response.replace(b"\r\n", b"")
