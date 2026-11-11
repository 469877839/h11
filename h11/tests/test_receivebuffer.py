import re
from typing import Tuple

import pytest

from .._receivebuffer import ReceiveBuffer
from .._util import LENIENT, LineEndingPolicy, LocalProtocolError, STRICT


def test_receivebuffer() -> None:
    b = ReceiveBuffer()
    assert not b
    assert len(b) == 0
    assert bytes(b) == b""

    b += b"123"
    assert b
    assert len(b) == 3
    assert bytes(b) == b"123"

    assert bytes(b) == b"123"

    assert b.maybe_extract_at_most(2) == b"12"
    assert b
    assert len(b) == 1
    assert bytes(b) == b"3"

    assert bytes(b) == b"3"

    assert b.maybe_extract_at_most(10) == b"3"
    assert bytes(b) == b""

    assert b.maybe_extract_at_most(10) is None
    assert not b

    ################################################################
    # maybe_extract_until_next
    ################################################################

    b += b"123\n456\r\n789\r\n"

    assert b.maybe_extract_next_line() == b"123\n456\r\n"
    assert bytes(b) == b"789\r\n"

    assert b.maybe_extract_next_line() == b"789\r\n"
    assert bytes(b) == b""

    b += b"12\r"
    assert b.maybe_extract_next_line() is None
    assert bytes(b) == b"12\r"

    b += b"345\n\r"
    assert b.maybe_extract_next_line() is None
    assert bytes(b) == b"12\r345\n\r"

    # here we stopped at the middle of b"\r\n" delimiter

    b += b"\n6789aaa123\r\n"
    assert b.maybe_extract_next_line() == b"12\r345\n\r\n"
    assert b.maybe_extract_next_line() == b"6789aaa123\r\n"
    assert b.maybe_extract_next_line() is None
    assert bytes(b) == b""

    ################################################################
    # maybe_extract_lines
    ################################################################

    b += b"123\r\na: b\r\nfoo:bar\r\n\r\ntrailing"
    lines = b.maybe_extract_lines()
    assert lines == [b"123", b"a: b", b"foo:bar"]
    assert bytes(b) == b"trailing"

    assert b.maybe_extract_lines() is None

    b += b"\r\n\r"
    assert b.maybe_extract_lines() is None

    assert b.maybe_extract_at_most(100) == b"trailing\r\n\r"
    assert not b

    # Empty body case (as happens at the end of chunked encoding if there are
    # no trailing headers, e.g.)
    b += b"\r\ntrailing"
    assert b.maybe_extract_lines() == []
    assert bytes(b) == b"trailing"


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            (
                b"HTTP/1.1 200 OK\r\n",
                b"Content-type: text/plain\r\n",
                b"Connection: close\r\n",
                b"\r\n",
                b"Some body",
            ),
            id="with_crlf_delimiter",
        ),
        pytest.param(
            (
                b"HTTP/1.1 200 OK\n",
                b"Content-type: text/plain\n",
                b"Connection: close\n",
                b"\n",
                b"Some body",
            ),
            id="with_lf_only_delimiter",
        ),
        pytest.param(
            (
                b"HTTP/1.1 200 OK\n",
                b"Content-type: text/plain\r\n",
                b"Connection: close\n",
                b"\n",
                b"Some body",
            ),
            id="with_mixed_crlf_and_lf",
        ),
    ],
)
def test_receivebuffer_for_invalid_delimiter(data: Tuple[bytes]) -> None:
    b = ReceiveBuffer()

    for line in data:
        b += line

    lines = b.maybe_extract_lines()

    assert lines == [
        b"HTTP/1.1 200 OK",
        b"Content-type: text/plain",
        b"Connection: close",
    ]
    assert bytes(b) == b"Some body"


# A complete set of framing lines, each terminator written with CRLF:
# start line + two header lines + blank line.
CRLF_FRAME = b"GET / HTTP/1.1\r\nHost: a\r\nConnection: close\r\n\r\n"

# Every way to break one or more terminators with a bare LF.
BARE_LF_FRAMES = [
    b"GET / HTTP/1.1\nHost: a\r\nConnection: close\r\n\r\n",
    b"GET / HTTP/1.1\r\nHost: a\nConnection: close\r\n\r\n",
    b"GET / HTTP/1.1\r\nHost: a\r\nConnection: close\n\r\n",
    b"GET / HTTP/1.1\r\nHost: a\r\nConnection: close\r\n\n",
    b"GET / HTTP/1.1\nHost: a\nConnection: close\n\n",
]


def test_default_line_ending_policy_is_lenient() -> None:
    assert ReceiveBuffer()._line_ending_policy is LENIENT


def test_invalid_line_ending_policy_rejected() -> None:
    with pytest.raises(ValueError):
        ReceiveBuffer(line_ending_policy="strict")  # type: ignore[arg-type]


@pytest.mark.parametrize("data", BARE_LF_FRAMES)
def test_lenient_accepts_bare_lf_lines(data: bytes) -> None:
    b = ReceiveBuffer(line_ending_policy=LENIENT)
    b += data
    assert b.maybe_extract_lines() == [
        b"GET / HTTP/1.1",
        b"Host: a",
        b"Connection: close",
    ]
    assert bytes(b) == b""


def test_strict_accepts_crlf_lines() -> None:
    b = ReceiveBuffer(line_ending_policy=STRICT)
    b += CRLF_FRAME
    assert b.maybe_extract_lines() == [
        b"GET / HTTP/1.1",
        b"Host: a",
        b"Connection: close",
    ]
    assert bytes(b) == b""


@pytest.mark.parametrize("data", BARE_LF_FRAMES)
def test_strict_rejects_bare_lf_lines(data: bytes) -> None:
    b = ReceiveBuffer(line_ending_policy=STRICT)
    b += data
    with pytest.raises(LocalProtocolError):
        b.maybe_extract_lines()
    # Nothing gets consumed on failure, but the caller is responsible for
    # shutting the connection down.
    assert bytes(b) == data


def test_strict_rejects_immediate_bare_lf_blank_line() -> None:
    b = ReceiveBuffer(line_ending_policy=STRICT)
    b += b"\n"
    with pytest.raises(LocalProtocolError):
        b.maybe_extract_lines()


def test_strict_accepts_immediate_crlf_blank_line() -> None:
    b = ReceiveBuffer(line_ending_policy=STRICT)
    b += b"\r\n"
    assert b.maybe_extract_lines() == []


@pytest.mark.parametrize("data", BARE_LF_FRAMES)
def test_strict_check_works_at_every_buffer_split(data: bytes) -> None:
    # The bare LF must be detected even when the offending bytes arrive in
    # different receive_data() calls, and a CR waiting for its LF must not
    # be rejected prematurely.
    for cut in range(len(data) + 1):
        b = ReceiveBuffer(line_ending_policy=STRICT)
        if cut > 0:
            b += data[:cut]
        try:
            result = b.maybe_extract_lines()
        except LocalProtocolError:
            # An error can surface before the split point; it must never
            # have consumed any bytes.
            assert bytes(b) == data[:cut]
        else:
            assert result is None
        if cut < len(data):
            b += data[cut:]
        with pytest.raises(LocalProtocolError):
            b.maybe_extract_lines()


def test_strict_crlf_frame_survives_every_buffer_split() -> None:
    for cut in range(len(CRLF_FRAME) + 1):
        b = ReceiveBuffer(line_ending_policy=STRICT)
        if cut > 0:
            b += CRLF_FRAME[:cut]
        if cut < len(CRLF_FRAME):
            assert b.maybe_extract_lines() is None
            b += CRLF_FRAME[cut:]
        assert b.maybe_extract_lines() == [
            b"GET / HTTP/1.1",
            b"Host: a",
            b"Connection: close",
        ]


def test_strict_next_line_rejects_bare_lf() -> None:
    b = ReceiveBuffer(line_ending_policy=STRICT)
    b += b"5\nhello\r\n"
    with pytest.raises(LocalProtocolError):
        b.maybe_extract_next_line()

    b = ReceiveBuffer(line_ending_policy=STRICT)
    b += b"5\r\nhello\r\n"
    assert b.maybe_extract_next_line() == b"5\r\n"
    assert bytes(b) == b"hello\r\n"


def test_line_ending_policy_does_not_touch_body_bytes() -> None:
    # maybe_extract_at_most is what pulls opaque message-body bytes, and a
    # body is allowed to contain anything -- bare LF included.
    body = b"line1\nline2\r\nline3"
    policies: Tuple[LineEndingPolicy, LineEndingPolicy] = (LENIENT, STRICT)
    for policy in policies:
        b = ReceiveBuffer(line_ending_policy=policy)
        b += body
        assert b.maybe_extract_at_most(len(body)) == bytearray(body)
        assert not b
