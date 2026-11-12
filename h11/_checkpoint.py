# Versioned, pickle-free checkpoints of Connection state.
#
# A checkpoint is intended for replaying a recorded, immutable byte stream:
# take one, throw the Connection away, restore it later, and keep feeding the
# remaining bytes. It deliberately contains no socket and no way to reattach
# to a live connection.
from __future__ import annotations

import re
import struct
import zlib
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Type, Union

from ._abnf import method as _method_pattern
from ._readers import (
    ChunkedReader,
    ContentLengthReader,
    Http10Reader,
    READERS,
)
from ._receivebuffer import ReceiveBuffer
from ._state import (
    _SWITCH_CONNECT,
    _SWITCH_UPGRADE,
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
from ._util import Sentinel
from ._version import __version__ as _H11_VERSION
from ._writers import (
    ChunkedWriter,
    ContentLengthWriter,
    Http10Writer,
    WRITERS,
)

__all__ = ["CheckpointError", "restore_checkpoint", "export_checkpoint"]


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot be exported or restored.

    Bad input is rejected before any Connection is created or mutated.
    """


# Format overview (version 1):
#
#   magic     8 bytes   b"h11CKPT\x00"
#   version   1 byte    struct format "B"
#   length    4 bytes   big-endian payload length ("!I")
#   checksum  4 bytes   big-endian CRC32 of the payload ("!I")
#   payload   length bytes, a sequence of (tag, length, value) fields
#
# Each payload field is:
#
#   tag       1 byte
#   length    4 bytes  big-endian value length ("!I")
#   value     length bytes
#
# Lengths are always byte lengths, so arbitrary binary payloads (including
# NUL) are safe. Values are integers, booleans, ASCII names of h11 sentinels,
# or opaque bytes. There is no pickle, eval, or reconstruction of arbitrary
# Python objects.
_MAGIC = b"h11CKPT\x00"
_FORMAT_VERSION = 1
_HEADER = struct.Struct("!8sBII")
_FIELD = struct.Struct("!BI")

# Version metadata is stored in every snapshot so that a future version can be
# detected before any of its fields are interpreted.
_TAG_H11_VERSION = 1

# Required connection metadata.
_TAG_FORMAT_VERSION = 2
_TAG_OUR_ROLE = 3
_TAG_THEIR_ROLE = 4
_TAG_MAX_INCOMPLETE_EVENT_SIZE = 5

# Receive state.
_TAG_RECEIVE_BUFFER = 6
_TAG_RECEIVE_BUFFER_CLOSED = 7
_TAG_THEIR_HTTP_VERSION = 8
_TAG_REQUEST_METHOD = 9
_TAG_CLIENT_IS_WAITING_FOR_100_CONTINUE = 10

# State machine.
_TAG_CLIENT_STATE = 11
_TAG_SERVER_STATE = 12
_TAG_KEEP_ALIVE = 13
_TAG_PENDING_SWITCHES = 14

# Message framing and the reader/writer objects that carry residual framing
# progress. Parse and send state are kept separate so they can never be
# confused with each other.
_TAG_READER_TYPE = 15
_TAG_READER_FRAMING = 16
_TAG_READER_REMAINING = 17
_TAG_READER_LENGTH = 18
_TAG_READER_BYTES_IN_CHUNK = 19
_TAG_READER_BYTES_TO_DISCARD = 20
_TAG_READER_READING_TRAILER = 21
_TAG_WRITER_TYPE = 22
_TAG_WRITER_FRAMING = 23
_TAG_WRITER_REMAINING = 24

_KNOWN_TAGS = frozenset(range(1, 25))

_REQUIRED_TAGS = frozenset(
    (
        _TAG_H11_VERSION,
        _TAG_FORMAT_VERSION,
        _TAG_OUR_ROLE,
        _TAG_THEIR_ROLE,
        _TAG_MAX_INCOMPLETE_EVENT_SIZE,
        _TAG_RECEIVE_BUFFER,
        _TAG_RECEIVE_BUFFER_CLOSED,
        _TAG_THEIR_HTTP_VERSION,
        _TAG_REQUEST_METHOD,
        _TAG_CLIENT_IS_WAITING_FOR_100_CONTINUE,
        _TAG_CLIENT_STATE,
        _TAG_SERVER_STATE,
        _TAG_KEEP_ALIVE,
        _TAG_PENDING_SWITCHES,
    )
)

_ROLE_NAMES: Dict[Type[Sentinel], str] = {CLIENT: "client", SERVER: "server"}
_ROLES_BY_NAME = {name: role for role, name in _ROLE_NAMES.items()}

_STATE_NAMES: Dict[Type[Sentinel], str] = {
    IDLE: "idle",
    SEND_RESPONSE: "send_response",
    SEND_BODY: "send_body",
    DONE: "done",
    MUST_CLOSE: "must_close",
    CLOSED: "closed",
    MIGHT_SWITCH_PROTOCOL: "might_switch_protocol",
    SWITCHED_PROTOCOL: "switched_protocol",
    ERROR: "error",
}
_STATES_BY_NAME = {name: state for state, name in _STATE_NAMES.items()}

_SWITCH_NAMES: Dict[Type[Sentinel], str] = {
    _SWITCH_CONNECT: "connect",
    _SWITCH_UPGRADE: "upgrade",
}
_SWITCHES_BY_NAME = {name: switch for switch, name in _SWITCH_NAMES.items()}

_FRAMINGS = (b"content-length", b"chunked", b"http/1.0")

_method_re = re.compile(_method_pattern.encode("ascii"))
_http_version_re = re.compile(rb"[0-9]\.[0-9]")


def _encode_int(value: int) -> bytes:
    return str(int(value)).encode("ascii")


def _decode_int(value: bytes, field: str, *, minimum: int = 0) -> int:
    try:
        text = value.decode("ascii")
    except UnicodeDecodeError:
        raise CheckpointError(f"{field}: invalid integer encoding") from None
    if not text:
        raise CheckpointError(f"{field}: invalid integer")
    if text[0] == "-":
        digits = text[1:]
    else:
        digits = text
    if not digits.isdigit():
        raise CheckpointError(f"{field}: invalid integer")
    parsed = int(text)
    if parsed < minimum:
        raise CheckpointError(f"{field}: must be >= {minimum}")
    return parsed


def _encode_bool(value: bool) -> bytes:
    return b"1" if value else b"0"


def _decode_bool(value: bytes, field: str) -> bool:
    if value == b"1":
        return True
    if value == b"0":
        return False
    raise CheckpointError(f"{field}: invalid boolean")


def _encode_name(table: Dict[Type[Sentinel], str], value: Type[Sentinel]) -> bytes:
    try:
        return table[value].encode("ascii")
    except KeyError:
        raise CheckpointError(f"cannot serialize unknown sentinel {value!r}") from None


def _decode_name(
    table: Dict[str, Type[Sentinel]], value: bytes, field: str
) -> Type[Sentinel]:
    try:
        name = value.decode("ascii")
    except UnicodeDecodeError:
        raise CheckpointError(f"{field}: invalid name encoding") from None
    try:
        return table[name]
    except KeyError:
        raise CheckpointError(f"{field}: unknown value {name!r}") from None


def _encode_switches(switches: Set[Type[Sentinel]]) -> bytes:
    names: List[str] = []
    for switch in switches:
        try:
            names.append(_SWITCH_NAMES[switch])
        except KeyError:
            raise CheckpointError(
                f"cannot serialize unknown switch proposal {switch!r}"
            ) from None
    return ",".join(sorted(names)).encode("ascii")


def _decode_switches(value: bytes, field: str) -> Set[Type[Sentinel]]:
    if not value:
        return set()
    try:
        text = value.decode("ascii")
    except UnicodeDecodeError:
        raise CheckpointError(f"{field}: invalid switch encoding") from None
    names = text.split(",")
    if names != sorted(names):
        raise CheckpointError(f"{field}: switch proposals must be sorted")
    if len(names) != len(set(names)):
        raise CheckpointError(f"{field}: duplicate switch proposal")
    result: Set[Type[Sentinel]] = set()
    for name in names:
        try:
            result.add(_SWITCHES_BY_NAME[name])
        except KeyError:
            raise CheckpointError(
                f"{field}: unknown switch proposal {name!r}"
            ) from None
    return result


def _decode_optional_bytes(
    value: bytes, field: str, pattern: Pattern[bytes]
) -> Optional[bytes]:
    # The empty byte-string is reserved as the None marker; a real value can
    # never be empty, so this is unambiguous.
    if value == b"":
        return None
    if not pattern.fullmatch(value):
        raise CheckpointError(f"{field}: invalid value {value!r}")
    return value


def _decode_ascii(value: bytes, field: str) -> str:
    try:
        text = value.decode("ascii")
    except UnicodeDecodeError:
        raise CheckpointError(f"{field}: invalid string encoding") from None
    if not text:
        raise CheckpointError(f"{field}: must not be empty")
    return text


################################################################
#
# Export
#
################################################################


def _add_reader_fields(add: Callable[[int, bytes], None], conn: Any) -> None:
    role = conn.their_role
    state = conn._cstate.states[role]
    reader = conn._reader
    if state is not SEND_BODY:
        # The reader is fully determined by the state machine; there is no
        # framing progress to preserve.
        expected = READERS.get((role, state))
        if reader is not expected:
            raise CheckpointError(
                "internally inconsistent connection: unexpected reader for "
                f"state {state!r}"
            )
        return
    if isinstance(reader, ContentLengthReader):
        add(_TAG_READER_TYPE, b"content-length")
        add(_TAG_READER_FRAMING, b"content-length")
        add(_TAG_READER_LENGTH, _encode_int(reader._length))
        add(_TAG_READER_REMAINING, _encode_int(reader._remaining))
    elif isinstance(reader, ChunkedReader):
        add(_TAG_READER_TYPE, b"chunked")
        add(_TAG_READER_FRAMING, b"chunked")
        add(_TAG_READER_BYTES_IN_CHUNK, _encode_int(reader._bytes_in_chunk))
        add(_TAG_READER_BYTES_TO_DISCARD, bytes(reader._bytes_to_discard))
        add(_TAG_READER_READING_TRAILER, _encode_bool(reader._reading_trailer))
    elif isinstance(reader, Http10Reader):
        add(_TAG_READER_TYPE, b"http/1.0")
        add(_TAG_READER_FRAMING, b"http/1.0")
    else:
        raise CheckpointError(f"cannot serialize unknown reader {reader!r}")


def _add_writer_fields(add: Callable[[int, bytes], None], conn: Any) -> None:
    role = conn.our_role
    state = conn._cstate.states[role]
    writer = conn._writer
    if state is not SEND_BODY:
        expected = WRITERS.get((role, state))
        if writer is not expected:
            raise CheckpointError(
                "internally inconsistent connection: unexpected writer for "
                f"state {state!r}"
            )
        return
    if isinstance(writer, ContentLengthWriter):
        add(_TAG_WRITER_TYPE, b"content-length")
        add(_TAG_WRITER_FRAMING, b"content-length")
        add(_TAG_WRITER_REMAINING, _encode_int(writer._length))
    elif isinstance(writer, ChunkedWriter):
        add(_TAG_WRITER_TYPE, b"chunked")
        add(_TAG_WRITER_FRAMING, b"chunked")
    elif isinstance(writer, Http10Writer):
        add(_TAG_WRITER_TYPE, b"http/1.0")
        add(_TAG_WRITER_FRAMING, b"http/1.0")
    else:
        raise CheckpointError(f"cannot serialize unknown writer {writer!r}")


def _build_payload(conn: Any) -> bytes:
    """Serialize the mutable state of a Connection into TLV fields."""

    fields: List[bytes] = []

    def add(tag: int, value: bytes) -> None:
        fields.append(_FIELD.pack(tag, len(value)))
        fields.append(value)

    if not isinstance(conn._max_incomplete_event_size, int):
        raise CheckpointError("max_incomplete_event_size must be an int")
    if conn._max_incomplete_event_size < 0:
        raise CheckpointError("max_incomplete_event_size must be >= 0")

    add(_TAG_H11_VERSION, _H11_VERSION.encode("ascii"))
    add(_TAG_FORMAT_VERSION, _encode_int(_FORMAT_VERSION))
    add(_TAG_OUR_ROLE, _encode_name(_ROLE_NAMES, conn.our_role))
    add(_TAG_THEIR_ROLE, _encode_name(_ROLE_NAMES, conn.their_role))
    add(_TAG_MAX_INCOMPLETE_EVENT_SIZE, _encode_int(conn._max_incomplete_event_size))

    add(_TAG_RECEIVE_BUFFER, bytes(conn._receive_buffer))
    add(_TAG_RECEIVE_BUFFER_CLOSED, _encode_bool(conn._receive_buffer_closed))
    if conn.their_http_version is None:
        add(_TAG_THEIR_HTTP_VERSION, b"")
    elif isinstance(conn.their_http_version, bytes):
        add(_TAG_THEIR_HTTP_VERSION, conn.their_http_version)
    else:
        raise CheckpointError("their_http_version must be bytes or None")
    if conn._request_method is None:
        add(_TAG_REQUEST_METHOD, b"")
    elif isinstance(conn._request_method, bytes):
        add(_TAG_REQUEST_METHOD, conn._request_method)
    else:
        raise CheckpointError("_request_method must be bytes or None")
    add(
        _TAG_CLIENT_IS_WAITING_FOR_100_CONTINUE,
        _encode_bool(conn.client_is_waiting_for_100_continue),
    )

    add(
        _TAG_CLIENT_STATE,
        _encode_name(_STATE_NAMES, conn._cstate.states[CLIENT]),
    )
    add(
        _TAG_SERVER_STATE,
        _encode_name(_STATE_NAMES, conn._cstate.states[SERVER]),
    )
    add(_TAG_KEEP_ALIVE, _encode_bool(conn._cstate.keep_alive))
    add(
        _TAG_PENDING_SWITCHES,
        _encode_switches(conn._cstate.pending_switch_proposals),
    )

    _add_reader_fields(add, conn)
    _add_writer_fields(add, conn)

    return b"".join(fields)


def export_checkpoint(conn: Any) -> bytes:
    """Export the replayable state of *conn* as versioned bytes.

    This is the function behind :meth:`Connection.checkpoint`; see its
    documentation for details. The result can be stored anywhere and later
    passed to :func:`restore_checkpoint` or :meth:`Connection.from_checkpoint`.
    """

    payload = _build_payload(conn)
    if len(payload) > 0xFFFFFFFF:
        raise CheckpointError("checkpoint payload too large")
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    return _HEADER.pack(_MAGIC, _FORMAT_VERSION, len(payload), checksum) + payload


################################################################
#
# Parse and validate
#
################################################################


def _unpack_header(data: bytes) -> bytes:
    if len(data) < _HEADER.size:
        raise CheckpointError("truncated checkpoint header")
    magic, version, length, checksum = _HEADER.unpack_from(data)
    if magic != _MAGIC:
        raise CheckpointError("not an h11 checkpoint")
    if version != _FORMAT_VERSION:
        raise CheckpointError(f"unsupported checkpoint version {version}")
    payload = data[_HEADER.size :]
    if len(payload) != length:
        raise CheckpointError("checkpoint length mismatch")
    actual_checksum = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_checksum != checksum:
        raise CheckpointError("checkpoint checksum mismatch")
    return payload


def _parse_fields(payload: bytes) -> Dict[int, bytes]:
    fields: Dict[int, bytes] = {}
    offset = 0
    total = len(payload)
    while offset < total:
        if total - offset < _FIELD.size:
            raise CheckpointError("truncated checkpoint field header")
        tag, length = _FIELD.unpack_from(payload, offset)
        offset += _FIELD.size
        end = offset + length
        if end > total:
            raise CheckpointError("truncated checkpoint field value")
        if tag not in _KNOWN_TAGS:
            raise CheckpointError(f"unknown checkpoint field tag {tag}")
        if tag in fields:
            raise CheckpointError(f"duplicate checkpoint field tag {tag}")
        fields[tag] = payload[offset:end]
        offset = end
    return fields


def _field(fields: Dict[int, bytes], tag: int, name: str) -> bytes:
    try:
        return fields[tag]
    except KeyError:
        raise CheckpointError(f"missing checkpoint field {name}") from None


def _require_state(
    state: Type[Sentinel],
    allowed: Set[Type[Sentinel]],
    *,
    role: str,
    field: str,
) -> None:
    if state not in allowed:
        raise CheckpointError(
            f"{field}: {_STATE_NAMES[state]!r} is not a valid {role} state"
        )


def _parse_reader(
    fields: Dict[int, bytes],
    their_role: Type[Sentinel],
    their_state: Type[Sentinel],
) -> Any:
    reader_type = fields.get(_TAG_READER_TYPE)
    if reader_type is None:
        # No reader tags may be present, and the role must not be in the middle
        # of an HTTP body.
        used: Set[int] = set()
        if their_state is SEND_BODY:
            raise CheckpointError(
                "missing reader framing state while peer is in SEND_BODY"
            )
        return READERS.get((their_role, their_state)), used

    if their_state is not SEND_BODY:
        raise CheckpointError("reader framing state outside SEND_BODY")
    if reader_type not in _FRAMINGS:
        raise CheckpointError(f"reader: unsupported framing {reader_type!r}")
    framing = _field(fields, _TAG_READER_FRAMING, "reader framing")
    if framing != reader_type:
        raise CheckpointError("reader: framing fields disagree")
    used = {_TAG_READER_TYPE, _TAG_READER_FRAMING}

    if reader_type == b"content-length":
        length = _decode_int(
            _field(fields, _TAG_READER_LENGTH, "reader length"), "reader length"
        )
        remaining = _decode_int(
            _field(fields, _TAG_READER_REMAINING, "reader remaining"),
            "reader remaining",
        )
        if remaining > length:
            raise CheckpointError(
                "reader: remaining bytes exceed declared content length"
            )
        used.update((_TAG_READER_LENGTH, _TAG_READER_REMAINING))
        reader = ContentLengthReader(length)
        reader._remaining = remaining
        return reader, used

    if reader_type == b"chunked":
        bytes_in_chunk = _decode_int(
            _field(fields, _TAG_READER_BYTES_IN_CHUNK, "reader chunk remaining"),
            "reader chunk remaining",
        )
        bytes_to_discard = _field(
            fields, _TAG_READER_BYTES_TO_DISCARD, "reader chunk footer"
        )
        reading_trailer = _decode_bool(
            _field(fields, _TAG_READER_READING_TRAILER, "reader trailer flag"),
            "reader trailer flag",
        )
        if reading_trailer and bytes_in_chunk != 0:
            raise CheckpointError(
                "reader: chunk remaining must be 0 while reading trailers"
            )
        # The footer is always a (possibly partially consumed) CRLF, so its
        # remaining value must be one of its suffixes.
        if bytes_to_discard not in (b"", b"\r", b"\n", b"\r\n"):
            raise CheckpointError("reader: invalid chunk footer state")
        if bytes_to_discard and bytes_in_chunk != 0:
            raise CheckpointError(
                "reader: chunk footer recorded before chunk was complete"
            )
        used.update(
            (
                _TAG_READER_BYTES_IN_CHUNK,
                _TAG_READER_BYTES_TO_DISCARD,
                _TAG_READER_READING_TRAILER,
            )
        )
        reader = ChunkedReader()
        reader._bytes_in_chunk = bytes_in_chunk
        reader._bytes_to_discard = bytes_to_discard
        reader._reading_trailer = reading_trailer
        return reader, used

    return Http10Reader(), used


def _parse_writer(
    fields: Dict[int, bytes],
    our_role: Type[Sentinel],
    our_state: Type[Sentinel],
) -> Any:
    writer_type = fields.get(_TAG_WRITER_TYPE)
    if writer_type is None:
        used: Set[int] = set()
        if our_state is SEND_BODY:
            raise CheckpointError(
                "missing writer framing state while our side is in SEND_BODY"
            )
        return WRITERS.get((our_role, our_state)), used

    if our_state is not SEND_BODY:
        raise CheckpointError("writer framing state outside SEND_BODY")
    if writer_type not in _FRAMINGS:
        raise CheckpointError(f"writer: unsupported framing {writer_type!r}")
    framing = _field(fields, _TAG_WRITER_FRAMING, "writer framing")
    if framing != writer_type:
        raise CheckpointError("writer: framing fields disagree")
    used = {_TAG_WRITER_TYPE, _TAG_WRITER_FRAMING}

    if writer_type == b"content-length":
        length = _decode_int(
            _field(fields, _TAG_WRITER_REMAINING, "writer remaining"),
            "writer remaining",
        )
        used.add(_TAG_WRITER_REMAINING)
        writer = ContentLengthWriter(length)
        return writer, used

    if writer_type == b"chunked":
        return ChunkedWriter(), used

    return Http10Writer(), used


def _restore(fields: Dict[int, bytes], connection_class: Type[Any]) -> Any:
    missing = _REQUIRED_TAGS - set(fields)
    if missing:
        raise CheckpointError(f"missing checkpoint fields: {sorted(missing)}")

    format_version = _decode_int(
        _field(fields, _TAG_FORMAT_VERSION, "format version"), "format version"
    )
    if format_version != _FORMAT_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint payload version {format_version}"
        )
    _decode_ascii(_field(fields, _TAG_H11_VERSION, "h11 version"), "h11 version")

    our_role = _decode_name(
        _ROLES_BY_NAME, _field(fields, _TAG_OUR_ROLE, "our role"), "our role"
    )
    their_role = _decode_name(
        _ROLES_BY_NAME, _field(fields, _TAG_THEIR_ROLE, "their role"), "their role"
    )
    expected_their_role = SERVER if our_role is CLIENT else CLIENT
    if their_role is not expected_their_role:
        raise CheckpointError("our role and their role are inconsistent")

    max_incomplete_event_size = _decode_int(
        _field(
            fields, _TAG_MAX_INCOMPLETE_EVENT_SIZE, "max_incomplete_event_size"
        ),
        "max_incomplete_event_size",
    )

    receive_buffer = _field(fields, _TAG_RECEIVE_BUFFER, "receive buffer")
    receive_buffer_closed = _decode_bool(
        _field(fields, _TAG_RECEIVE_BUFFER_CLOSED, "receive buffer closed"),
        "receive buffer closed",
    )
    their_http_version = _decode_optional_bytes(
        _field(fields, _TAG_THEIR_HTTP_VERSION, "their http version"),
        "their http version",
        _http_version_re,
    )
    request_method = _decode_optional_bytes(
        _field(fields, _TAG_REQUEST_METHOD, "request method"),
        "request method",
        _method_re,
    )
    waiting_for_100_continue = _decode_bool(
        _field(
            fields,
            _TAG_CLIENT_IS_WAITING_FOR_100_CONTINUE,
            "client_is_waiting_for_100_continue",
        ),
        "client_is_waiting_for_100_continue",
    )

    client_state = _decode_name(
        _STATES_BY_NAME,
        _field(fields, _TAG_CLIENT_STATE, "client state"),
        "client state",
    )
    server_state = _decode_name(
        _STATES_BY_NAME,
        _field(fields, _TAG_SERVER_STATE, "server state"),
        "server state",
    )
    keep_alive = _decode_bool(
        _field(fields, _TAG_KEEP_ALIVE, "keep alive"), "keep alive"
    )
    pending_switches = _decode_switches(
        _field(fields, _TAG_PENDING_SWITCHES, "pending switch proposals"),
        "pending switch proposals",
    )

    _require_state(
        client_state,
        {
            IDLE,
            SEND_BODY,
            DONE,
            MUST_CLOSE,
            CLOSED,
            MIGHT_SWITCH_PROTOCOL,
            SWITCHED_PROTOCOL,
            ERROR,
        },
        role="client",
        field="client state",
    )
    _require_state(
        server_state,
        {
            IDLE,
            SEND_RESPONSE,
            SEND_BODY,
            DONE,
            MUST_CLOSE,
            CLOSED,
            SWITCHED_PROTOCOL,
            ERROR,
        },
        role="server",
        field="server state",
    )
    if pending_switches and client_state not in (
        SEND_BODY,
        MIGHT_SWITCH_PROTOCOL,
        SWITCHED_PROTOCOL,
        ERROR,
    ):
        raise CheckpointError(
            "pending switch proposals are inconsistent with client state"
        )
    if not pending_switches and (
        client_state in (MIGHT_SWITCH_PROTOCOL, SWITCHED_PROTOCOL)
        or server_state is SWITCHED_PROTOCOL
    ):
        raise CheckpointError(
            "a protocol switch state requires a pending switch proposal"
        )
    if not keep_alive and (client_state is DONE or server_state is DONE):
        raise CheckpointError(
            "keep-alive disabled but a side is still marked DONE"
        )

    reader, reader_tags = _parse_reader(
        fields,
        their_role,
        server_state if their_role is SERVER else client_state,
    )
    writer, writer_tags = _parse_writer(
        fields,
        our_role,
        client_state if our_role is CLIENT else server_state,
    )

    used_tags = set(_REQUIRED_TAGS) | reader_tags | writer_tags
    unexpected = set(fields) - used_tags
    if unexpected:
        raise CheckpointError(
            f"unexpected checkpoint fields: {sorted(unexpected)}"
        )

    conn = connection_class(
        our_role,
        max_incomplete_event_size=max_incomplete_event_size,
    )
    conn._cstate.states = {CLIENT: client_state, SERVER: server_state}
    conn._cstate.keep_alive = keep_alive
    conn._cstate.pending_switch_proposals = pending_switches
    conn._reader = reader
    conn._writer = writer
    conn._receive_buffer = ReceiveBuffer()
    conn._receive_buffer += receive_buffer
    conn._receive_buffer_closed = receive_buffer_closed
    conn.their_http_version = their_http_version
    conn._request_method = request_method
    conn.client_is_waiting_for_100_continue = waiting_for_100_continue
    return conn


def restore_checkpoint(
    checkpoint: Union[bytes, bytearray, memoryview],
    *,
    connection_class: Optional[Type[Any]] = None,
) -> Any:
    """Restore a Connection from bytes produced by :func:`export_checkpoint`.

    All validation, including the version, length and checksum, happens before
    a Connection is constructed. Corrupt or unsupported checkpoints therefore
    cannot leave a half-restored object behind.

    The returned object is a fresh offline replay object. It is not connected
    to any live socket and does not share mutable state with the exporter.
    """
    if not isinstance(checkpoint, (bytes, bytearray, memoryview)):
        raise TypeError("checkpoint must be a bytes-like object")
    payload = _unpack_header(bytes(checkpoint))
    fields = _parse_fields(payload)
    if connection_class is None:
        from ._connection import Connection

        connection_class = Connection
    return _restore(fields, connection_class)
