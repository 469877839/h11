import zlib
from typing import Any, Dict, List, Optional, Tuple

import pytest

from .._checkpoint import (
    _FIELD,
    _FORMAT_VERSION,
    _HEADER,
    _MAGIC,
    _parse_fields,
    CheckpointError,
    restore_checkpoint,
)
from .._connection import Connection, NEED_DATA, PAUSED
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
    DONE,
    IDLE,
    MIGHT_SWITCH_PROTOCOL,
    MUST_CLOSE,
    SEND_BODY,
    SERVER,
    SWITCHED_PROTOCOL,
)
from .helpers import get_all_events


def _connection_signature(conn: Connection) -> Tuple[Any, ...]:
    """All externally observable, replay-relevant Connection state."""
    reader = conn._reader
    reader_state: Any = None
    if reader is not None and conn.their_state is SEND_BODY:
        reader_state = (
            type(reader).__name__,
            getattr(reader, "_length", None),
            getattr(reader, "_remaining", None),
            getattr(reader, "_bytes_in_chunk", None),
            getattr(reader, "_bytes_to_discard", None),
            getattr(reader, "_reading_trailer", None),
        )
    writer = conn._writer
    writer_state: Any = None
    if writer is not None and conn.our_state is SEND_BODY:
        writer_state = (type(writer).__name__, getattr(writer, "_length", None))
    return (
        conn.our_role,
        conn.their_role,
        dict(conn._cstate.states),
        conn._cstate.keep_alive,
        frozenset(conn._cstate.pending_switch_proposals),
        conn.trailing_data,
        conn.their_http_version,
        conn._request_method,
        conn.client_is_waiting_for_100_continue,
        conn._max_incomplete_event_size,
        reader_state,
        writer_state,
    )


def _check_restart_equivalence(
    conn: Connection, remaining: bytes, *, eof: bool = False
) -> List[Any]:
    """Restart from a checkpoint and replay the remaining stream.

    Returns the events produced by the original connection after the replay,
    so callers can assert on them.
    """
    checkpoint = conn.checkpoint()
    restored = restore_checkpoint(checkpoint)

    # Restoring must be a pure, independent copy: state matches but the buffer
    # objects are not shared.
    assert _connection_signature(restored) == _connection_signature(conn)
    assert restored is not conn
    assert restored._receive_buffer is not conn._receive_buffer
    # The bytes survive a copy, and the snapshot contains no pickle artifacts.
    assert isinstance(checkpoint, (bytes, bytearray))
    assert b"__reduce__" not in checkpoint
    assert b"cos\nsystem" not in checkpoint

    if remaining:
        conn.receive_data(remaining)
        restored.receive_data(remaining)
    if eof:
        conn.receive_data(b"")
        restored.receive_data(b"")

    original_events: List[Any] = []
    restored_events: List[Any] = []
    while True:
        original = conn.next_event()
        restarted = restored.next_event()
        assert restarted == original
        assert type(restarted) is type(original)
        if original in (NEED_DATA, PAUSED):
            break
        original_events.append(original)
        restored_events.append(restarted)
        if type(original) is ConnectionClosed:
            break
    assert restored_events == original_events
    assert _connection_signature(restored) == _connection_signature(conn)
    assert restored.trailing_data == conn.trailing_data
    assert restored.states == conn.states
    return original_events


def test_checkpoint_while_headers_are_partial() -> None:
    conn = Connection(SERVER)
    full_request = (
        b"POST /thing HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Content-Length: 5\r\n"
        b"\r\n"
        b"hello"
    )
    split = 21  # inside the header block, before the blank line
    conn.receive_data(full_request[:split])
    assert conn.next_event() is NEED_DATA
    assert _check_restart_equivalence(conn, full_request[split:]) == [
        Request(
            method="POST",
            target="/thing",
            headers=[("Host", "example.com"), ("Content-Length", "5")],
        ),
        Data(data=b"hello"),
        EndOfMessage(),
    ]


def test_checkpoint_inside_fixed_length_body() -> None:
    conn = Connection(SERVER)
    head = (
        b"POST /thing HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Content-Length: 10\r\n"
        b"\r\n"
    )
    conn.receive_data(head + b"0123")
    assert conn.next_event() == Request(
        method="POST",
        target="/thing",
        headers=[("Host", "example.com"), ("Content-Length", "10")],
    )
    # First Data event consumed 4 bytes; the reader still owes six.
    assert conn.next_event() == Data(data=b"0123")
    assert conn.their_state is SEND_BODY
    assert conn._reader._remaining == 6
    assert conn._reader._length == 10
    # The remaining six bytes arrive in a later, post-restore feed.
    assert _check_restart_equivalence(conn, b"456789") == [
        Data(data=b"456789"),
        EndOfMessage(),
    ]
    assert conn.their_state is DONE


def test_checkpoint_with_unconsumed_bytes_in_buffer() -> None:
    # The checkpoint must preserve already-received but unparsed input too.
    conn = Connection(SERVER)
    head = (
        b"POST /thing HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Content-Length: 10\r\n"
        b"\r\n"
    )
    conn.receive_data(head + b"0123456789")
    assert conn.next_event() == Request(
        method="POST",
        target="/thing",
        headers=[("Host", "example.com"), ("Content-Length", "10")],
    )
    # No Data event consumed yet; the whole body sits in the receive buffer
    # while the reader still expects all ten bytes.
    assert conn.trailing_data == (b"0123456789", False)
    assert conn._reader._remaining == 10
    assert _check_restart_equivalence(conn, b"") == [
        Data(data=b"0123456789"),
        EndOfMessage(),
    ]


def test_checkpoint_inside_fixed_length_body_then_more_arrives() -> None:
    conn = Connection(SERVER)
    head = (
        b"POST /thing HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Content-Length: 10\r\n"
        b"\r\n"
    )
    conn.receive_data(head + b"abc")
    assert conn.next_event() == Request(
        method="POST",
        target="/thing",
        headers=[("Host", "example.com"), ("Content-Length", "10")],
    )
    assert conn.next_event() == Data(data=b"abc")
    # The remaining seven bytes arrive in two later, post-restore feeds.
    assert _check_restart_equivalence(conn, b"defg", eof=False) == [
        Data(data=b"defg"),
    ]
    conn.receive_data(b"hij")
    assert get_all_events(conn) == [
        Data(data=b"hij"),
        EndOfMessage(),
    ]


def test_checkpoint_records_eof_after_partial_body() -> None:
    conn = Connection(CLIENT, max_incomplete_event_size=4096)
    conn.send(
        Request(
            method="POST",
            target="/",
            headers=[("Host", "example.com"), ("Content-Length", "4")],
        )
    )
    response = b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nab"
    conn.receive_data(response)
    assert conn.next_event() == Response(
        status_code=200,
        headers=[("Content-Length", "4")],
        reason="OK",
    )
    assert conn.next_event() == Data(data=b"ab")
    # The closed flag must survive even though the final two bytes never come.
    conn.receive_data(b"")
    assert conn.trailing_data == (b"", True)
    checkpoint = conn.checkpoint()
    restored = restore_checkpoint(checkpoint)
    assert restored.trailing_data == (b"", True)
    assert _connection_signature(restored) == _connection_signature(conn)
    with pytest.raises(Exception):
        restored.next_event()
    with pytest.raises(Exception):
        conn.next_event()
    assert conn.states == restored.states


def test_checkpoint_inside_chunked_data() -> None:
    conn = Connection(SERVER)
    head = (
        b"POST / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
    )
    conn.receive_data(head)
    assert conn.next_event() == Request(
        method="POST",
        target="/",
        headers=[("Host", "example.com"), ("Transfer-Encoding", "chunked")],
    )
    # A chunk whose payload is only partially available: 3 of 5 bytes.
    conn.receive_data(b"5\r\nhel")
    assert conn.next_event() == Data(data=b"hel", chunk_start=True, chunk_end=False)
    assert conn._reader._bytes_in_chunk == 2
    assert _check_restart_equivalence(conn, b"lo\r\n") == [
        Data(data=b"lo", chunk_start=False, chunk_end=True),
    ]
    # The trailing CRLF is still pending; a fresh chunk then completes it.
    assert _check_restart_equivalence(conn, b"5\r\nhello\r\n0\r\n\r\n") == [
        Data(data=b"hello", chunk_start=True, chunk_end=True),
        EndOfMessage(),
    ]


def test_checkpoint_between_chunk_footer_and_next_chunk() -> None:
    conn = Connection(SERVER)
    head = (
        b"POST / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
    )
    conn.receive_data(head)
    conn.next_event()
    conn.receive_data(b"5\r\nhello")
    assert conn.next_event() == Data(
        data=b"hello", chunk_start=True, chunk_end=True
    )
    # The reader is now discarding the chunk terminating CRLF.
    assert conn._reader._bytes_to_discard == b"\r\n"
    conn.receive_data(b"\r")
    assert conn.next_event() is NEED_DATA
    assert conn._reader._bytes_to_discard == b"\n"
    assert _check_restart_equivalence(conn, b"\n0\r\nX-trailer: yes\r\n\r\n") == [
        EndOfMessage(headers=[("X-trailer", "yes")]),
    ]


def test_checkpoint_inside_chunked_trailer() -> None:
    conn = Connection(SERVER)
    head = (
        b"POST / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
    )
    conn.receive_data(head)
    conn.next_event()
    conn.receive_data(b"0\r\nX-trailer: yes\r\n")
    assert conn.next_event() is NEED_DATA
    assert conn._reader._reading_trailer is True
    assert conn._reader._bytes_in_chunk == 0
    assert _check_restart_equivalence(conn, b"\r\n") == [
        EndOfMessage(headers=[("X-trailer", "yes")])
    ]


def test_checkpoint_before_and_after_upgrade() -> None:
    # Server-side connection watching a client upgrade to a new protocol.
    request = (
        b"GET / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Connection: upgrade\r\n"
        b"Upgrade: websocket\r\n"
        b"\r\n"
    )
    conn = Connection(SERVER)
    conn.receive_data(request)
    assert conn.next_event() == Request(
        method="GET",
        target="/",
        headers=[
            ("Host", "example.com"),
            ("Connection", "upgrade"),
            ("Upgrade", "websocket"),
        ],
    )
    # Checkpoint 1: the request is parsed but the body is still pending and a
    # switch proposal is live.
    assert conn.their_state is SEND_BODY
    assert conn._cstate.pending_switch_proposals
    # The request has no declared body, so the next event is its EndOfMessage;
    # the checkpoint still captures the SEND_BODY reader state.
    assert conn._reader._remaining == 0
    assert _check_restart_equivalence(conn, b"") == [EndOfMessage()]
    # Checkpoint 2: the client is MIGHT_SWITCH_PROTOCOL and the server has not
    # answered yet.
    assert conn.states[CLIENT] is MIGHT_SWITCH_PROTOCOL
    _check_restart_equivalence(conn, b"")
    conn.send(
        InformationalResponse(status_code=101, headers=[("Upgrade", "websocket")])
    )
    # Checkpoint 3: both sides have switched; trailing bytes are opaque.
    assert conn.states == {CLIENT: SWITCHED_PROTOCOL, SERVER: SWITCHED_PROTOCOL}
    conn.receive_data(b"\x81\x82binary")
    assert conn.next_event() is PAUSED
    _check_restart_equivalence(conn, b"\x00\x01")
    assert conn.trailing_data == (b"\x81\x82binary\x00\x01", False)


def test_checkpoint_after_connect_switch() -> None:
    conn = Connection(CLIENT)
    conn.send(
        Request(
            method="CONNECT",
            target="example.com:443",
            headers=[("Host", "example.com")],
        )
    )
    conn.send(EndOfMessage())
    assert conn.our_state is MIGHT_SWITCH_PROTOCOL
    # Checkpoint before the server accepts.
    _check_restart_equivalence(conn, b"")
    conn.receive_data(b"HTTP/1.1 200 OK\r\n\r\n")
    assert conn.next_event() == Response(
        status_code=200, headers=[], reason="OK"
    )
    # Checkpoint after the tunnel is established.
    assert conn.states == {CLIENT: SWITCHED_PROTOCOL, SERVER: SWITCHED_PROTOCOL}
    conn.receive_data(b"\x16\x03\x01")
    assert conn.next_event() is PAUSED
    _check_restart_equivalence(conn, b"\x02\x00")
    assert conn.trailing_data == (b"\x16\x03\x01\x02\x00", False)


def test_checkpoint_after_denied_upgrade() -> None:
    conn = Connection(SERVER)
    conn.receive_data(
        b"GET / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Connection: upgrade\r\n"
        b"Upgrade: websocket\r\n"
        b"\r\n"
    )
    conn.next_event()
    assert conn.next_event() == EndOfMessage()
    assert conn.states[CLIENT] is MIGHT_SWITCH_PROTOCOL
    conn.send(Response(status_code=200, headers=[("Content-Length", "5")]))
    # The proposal is cleared; the connection continues as plain HTTP.
    assert not conn._cstate.pending_switch_proposals
    assert conn.our_state is SEND_BODY
    _check_restart_equivalence(conn, b"")
    conn.send(Data(data=b"hello"))
    conn.send(EndOfMessage())
    conn.start_next_cycle()
    assert conn.states == {CLIENT: IDLE, SERVER: IDLE}
    _check_restart_equivalence(conn, b"")


def _frame(
    payload: bytes,
    *,
    magic: bytes = _MAGIC,
    version: int = _FORMAT_VERSION,
) -> bytes:
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    return _HEADER.pack(magic, version, len(payload), checksum) + payload


def _rewrite(
    checkpoint: bytes,
    *,
    changes: Optional[Dict[int, bytes]] = None,
    drop: Tuple[int, ...] = (),
    extra: Tuple[Tuple[int, bytes], ...] = (),
) -> bytes:
    payload = checkpoint[_HEADER.size :]
    parsed = _parse_fields(payload)
    changes = {} if changes is None else changes
    fields = []
    for tag, value in parsed.items():
        if tag in drop:
            continue
        fields.append((tag, changes.get(tag, value)))
    fields.extend(extra)
    new_payload = b"".join(
        _FIELD.pack(tag, len(value)) + value for tag, value in fields
    )
    return _frame(new_payload)


def _body_checkpoint() -> bytes:
    conn = Connection(SERVER)
    conn.receive_data(
        b"POST / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Content-Length: 10\r\n"
        b"\r\n"
        b"abcd"
    )
    assert conn.next_event() == Request(
        method="POST",
        target="/",
        headers=[("Host", "example.com"), ("Content-Length", "10")],
    )
    assert conn.next_event() == Data(data=b"abcd")
    return conn.checkpoint()


def test_corrupt_and_unsupported_checkpoints_are_rejected() -> None:
    checkpoint = _body_checkpoint()
    valid = restore_checkpoint(checkpoint)
    assert valid.their_state is SEND_BODY

    # Not an h11 checkpoint at all.
    with pytest.raises(CheckpointError, match="not an h11 checkpoint"):
        restore_checkpoint(b"not a checkpoint at all")
    with pytest.raises(CheckpointError):
        restore_checkpoint(b"")
    with pytest.raises(CheckpointError):
        restore_checkpoint(checkpoint[: _HEADER.size - 1])

    # Truncation and length mismatches.
    with pytest.raises(CheckpointError, match="length mismatch"):
        restore_checkpoint(checkpoint[:-1])
    truncated_payload = checkpoint[_HEADER.size :][:-2]
    with pytest.raises(CheckpointError):
        restore_checkpoint(
            _HEADER.pack(
                _MAGIC,
                _FORMAT_VERSION,
                len(truncated_payload),
                zlib.crc32(truncated_payload) & 0xFFFFFFFF,
            )
            + truncated_payload
        )

    # Corrupted bytes must fail the checksum.
    flipped = bytearray(checkpoint)
    flipped[-1] ^= 0xFF
    with pytest.raises(CheckpointError, match="checksum"):
        restore_checkpoint(bytes(flipped))

    # Unknown versions are rejected before any field is used.
    payload = checkpoint[_HEADER.size :]
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    future = _HEADER.pack(_MAGIC, _FORMAT_VERSION + 1, len(payload), checksum) + payload
    with pytest.raises(CheckpointError, match="unsupported checkpoint version"):
        restore_checkpoint(future)
    with pytest.raises(CheckpointError, match="unsupported"):
        restore_checkpoint(_rewrite(checkpoint, changes={2: b"2"}))

    # Unknown / duplicated / missing fields.
    with pytest.raises(CheckpointError, match="unknown checkpoint field"):
        restore_checkpoint(_rewrite(checkpoint, extra=((200, b"x"),)))
    parsed = _parse_fields(payload)
    duplicate_payload = payload + _FIELD.pack(11, len(parsed[11])) + parsed[11]
    with pytest.raises(CheckpointError, match="duplicate"):
        restore_checkpoint(_frame(duplicate_payload))
    with pytest.raises(CheckpointError, match="missing"):
        restore_checkpoint(_rewrite(checkpoint, drop=(13,)))

    # Wrong roles and bad names.
    with pytest.raises(CheckpointError, match="inconsistent"):
        restore_checkpoint(_rewrite(checkpoint, changes={4: b"server"}))
    with pytest.raises(CheckpointError, match="unknown value"):
        restore_checkpoint(_rewrite(checkpoint, changes={3: b"proxy"}))
    with pytest.raises(CheckpointError, match="invalid boolean"):
        restore_checkpoint(_rewrite(checkpoint, changes={7: b"yes"}))
    with pytest.raises(CheckpointError, match="invalid integer"):
        restore_checkpoint(_rewrite(checkpoint, changes={5: b"lots"}))

    # Inconsistent framing fields cannot reach a Connection.
    with pytest.raises(CheckpointError, match="remaining bytes exceed"):
        restore_checkpoint(_rewrite(checkpoint, changes={17: b"11"}))
    with pytest.raises(CheckpointError, match="framing fields disagree"):
        restore_checkpoint(_rewrite(checkpoint, changes={16: b"chunked"}))
    with pytest.raises(CheckpointError, match="missing reader framing state"):
        restore_checkpoint(_rewrite(checkpoint, drop=(15, 16, 17, 18)))

    # Rejection leaves a fresh restore from the same bytes working normally.
    restored_again = restore_checkpoint(checkpoint)
    assert restored_again._reader._remaining == 6
    assert restored_again._reader._length == 10


def _chunked_checkpoint() -> bytes:
    conn = Connection(SERVER)
    conn.receive_data(
        b"POST / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\nhel"
    )
    conn.next_event()
    assert conn.next_event() == Data(data=b"hel", chunk_start=True)
    return conn.checkpoint()


def test_corrupt_chunked_checkpoints_are_rejected() -> None:
    checkpoint = _chunked_checkpoint()
    assert restore_checkpoint(checkpoint)._reader._bytes_in_chunk == 2

    # Trailer flag cannot coexist with an unfinished chunk.
    with pytest.raises(CheckpointError, match="trailers"):
        restore_checkpoint(_rewrite(checkpoint, changes={21: b"1"}))
    # The chunk footer must be a (possibly partial) CRLF.
    with pytest.raises(CheckpointError, match="chunk footer"):
        restore_checkpoint(_rewrite(checkpoint, changes={20: b"xx"}))
    # A recorded footer implies the chunk itself is complete.
    with pytest.raises(CheckpointError, match="before chunk was complete"):
        restore_checkpoint(_rewrite(checkpoint, changes={20: b"\r\n"}))
    # Chunked readers carry no content-length fields.
    with pytest.raises(CheckpointError, match="unexpected checkpoint fields"):
        restore_checkpoint(_rewrite(checkpoint, extra=((17, b"0"), (18, b"5"))))
    # Unknown framing types are unsupported.
    with pytest.raises(CheckpointError, match="unsupported framing"):
        restore_checkpoint(
            _rewrite(checkpoint, changes={15: b"deflate", 16: b"deflate"})
        )


def test_writer_side_checkpoint() -> None:
    conn = Connection(CLIENT)
    conn.send(
        Request(
            method="POST",
            target="/",
            headers=[("Host", "example.com"), ("Content-Length", "10")],
        )
    )
    conn.send(Data(data=b"abc"))
    assert conn._writer._length == 7
    checkpoint = conn.checkpoint()
    restored = restore_checkpoint(checkpoint)
    assert _connection_signature(restored) == _connection_signature(conn)
    # Both copies must emit the same bytes for the rest of the body.
    assert restored.send(Data(data=b"defg")) == conn.send(Data(data=b"defg"))
    assert restored.send(Data(data=b"hij")) == conn.send(Data(data=b"hij"))
    assert restored.send(EndOfMessage()) == conn.send(EndOfMessage())
    assert restored.states == conn.states


def test_checkpoint_does_not_mutate_and_is_deterministic() -> None:
    conn = Connection(SERVER)
    conn.receive_data(
        b"POST / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Content-Length: 5\r\n"
        b"\r\n"
        b"ab"
    )
    before = _connection_signature(conn)
    first = conn.checkpoint()
    second = conn.checkpoint()
    assert first == second
    assert _connection_signature(conn) == before
    # Exporting does not consume or copy the buffered input out of the conn.
    assert conn.trailing_data[0].endswith(b"ab")


def test_default_connection_carries_no_persistence_state() -> None:
    # A plain Connection must not grow extra attributes for checkpointing.
    assert set(Connection(SERVER).__dict__) == {
        "_max_incomplete_event_size",
        "our_role",
        "their_role",
        "_cstate",
        "_writer",
        "_reader",
        "_receive_buffer",
        "_receive_buffer_closed",
        "their_http_version",
        "_request_method",
        "client_is_waiting_for_100_continue",
    }


def test_checkpoint_public_api_aliases() -> None:
    conn = Connection(SERVER)
    conn.receive_data(b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
    checkpoint = conn.export_checkpoint()
    assert checkpoint == conn.checkpoint()
    for restore in (
        Connection.from_checkpoint,
        Connection.restore_checkpoint,
        restore_checkpoint,
    ):
        restored = restore(checkpoint)
        assert _connection_signature(restored) == _connection_signature(conn)
        assert restored.next_event() == Request(
            method="GET", target="/", headers=[("Host", "example.com")]
        )
    with pytest.raises(TypeError):
        restore_checkpoint("not bytes")  # type: ignore[arg-type]


def test_checkpoint_after_connection_close_and_keep_alive_disabled() -> None:
    conn = Connection(SERVER)
    conn.receive_data(
        b"GET / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Connection: close\r\n"
        b"\r\n"
    )
    assert conn.next_event() == Request(
        method="GET",
        target="/",
        headers=[("Host", "example.com"), ("Connection", "close")],
    )
    assert conn.next_event() == EndOfMessage()
    # keep_alive is now False and the client side is MUST_CLOSE.
    assert conn._cstate.keep_alive is False
    _check_restart_equivalence(conn, b"")
    conn.send(Response(status_code=200, headers=[("Content-Length", "2")]))
    conn.send(Data(data=b"hi"))
    conn.send(EndOfMessage())
    assert conn.states[SERVER] is MUST_CLOSE
    _check_restart_equivalence(conn, b"")
    # Peer EOF produces ConnectionClosed on both copies.
    _check_restart_equivalence(conn, b"", eof=True)
    assert conn.next_event() == ConnectionClosed()


def test_checkpoint_preserves_max_incomplete_event_size() -> None:
    conn = Connection(SERVER, max_incomplete_event_size=8)
    conn.receive_data(b"GET /very/long/target")
    checkpoint = conn.checkpoint()
    restored = restore_checkpoint(checkpoint)
    assert restored._max_incomplete_event_size == 8
    with pytest.raises(Exception) as original_error:
        conn.next_event()
    with pytest.raises(Exception) as restored_error:
        restored.next_event()
    assert type(restored_error.value) is type(original_error.value)
    assert restored.states == conn.states


def test_checkpoint_with_pipelined_request_buffered() -> None:
    first = b"GET /first HTTP/1.1\r\nHost: example.com\r\n\r\n"
    second = b"GET /second HTTP/1.1\r\nHost: example.com\r\n\r\n"
    conn = Connection(SERVER)
    conn.receive_data(first + second)
    assert conn.next_event() == Request(
        method="GET", target="/first", headers=[("Host", "example.com")]
    )
    assert conn.next_event() == EndOfMessage()
    # A second request is already buffered while the first cycle finishes.
    assert conn.next_event() is PAUSED
    assert conn.trailing_data == (second, False)
    checkpoint = conn.checkpoint()
    restored = restore_checkpoint(checkpoint)
    assert _connection_signature(restored) == _connection_signature(conn)
    for copy in (conn, restored):
        copy.send(Response(status_code=200, headers=[("Content-Length", "0")]))
        copy.send(EndOfMessage())
        copy.start_next_cycle()
        assert copy.next_event() == Request(
            method="GET", target="/second", headers=[("Host", "example.com")]
        )
        assert copy.next_event() == EndOfMessage()
    assert _connection_signature(restored) == _connection_signature(conn)


def test_checkpoint_with_eof_and_unparsed_buffer() -> None:
    # EOF can arrive while bytes are still waiting to be parsed.
    conn = Connection(SERVER)
    request = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
    conn.receive_data(request)
    conn.receive_data(b"")
    assert conn.trailing_data == (request, True)
    restored = restore_checkpoint(conn.checkpoint())
    assert restored.trailing_data == (request, True)
    assert restored.next_event() == Request(
        method="GET", target="/", headers=[("Host", "example.com")]
    )
    assert restored.next_event() == EndOfMessage()
    assert restored.next_event() == ConnectionClosed()








