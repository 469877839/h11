import re
import sys
from enum import Enum
from typing import Any, List, Optional, Union

from ._util import LocalProtocolError

__all__ = [
    "ReceiveBuffer",
    "ReceivePolicy",
    "LineEndingPolicy",
    "ReceiveLinePolicy",
    "NewlinePolicy",
]


class ReceivePolicy(str, Enum):
    """How strictly line endings in the HTTP message framing are accepted."""

    COMPATIBLE = "compatible"
    DEFAULT = "compatible"
    LENIENT = "compatible"
    TOLERANT = "compatible"
    LF = "compatible"
    STRICT = "strict"
    STRICT_MODE = "strict"
    STRICT_CRLF = "strict_crlf"
    CRLF = "strict_crlf"


LineEndingPolicy = ReceivePolicy
ReceiveLinePolicy = ReceivePolicy
NewlinePolicy = ReceivePolicy


def _normalize_receive_policy(policy: Any) -> ReceivePolicy:
    if isinstance(policy, ReceivePolicy):
        return policy
    if isinstance(policy, (bytes, bytearray)):
        policy = bytes(policy).decode("ascii")
    if policy is None or policy is False:
        return ReceivePolicy.COMPATIBLE
    if policy is True:
        return ReceivePolicy.STRICT_CRLF
    if isinstance(policy, str):
        normalized = policy.strip().lower().replace("_", "-").replace(" ", "-")
        if normalized in {
            "compatible",
            "compat",
            "default",
            "lenient",
            "tolerant",
            "either",
            "auto",
            "both",
            "permissive",
            "relaxed",
            "crlf-or-lf",
            "lf-or-crlf",
        }:
            return ReceivePolicy.COMPATIBLE
        if normalized == "strict":
            return ReceivePolicy.STRICT
        if normalized in {
            "strict-crlf",
            "crlf",
            "strict-line-endings",
            "strict-newlines",
            "crlf-only",
        }:
            return ReceivePolicy.STRICT_CRLF
    raise ValueError(f"unknown receive policy: {policy!r}")


def _policy_is_strict(policy: ReceivePolicy) -> bool:
    return policy in (ReceivePolicy.STRICT, ReceivePolicy.STRICT_CRLF)


# Operations we want to support:
# - find next \r\n or \r\n\r\n (\n or \n\n are also acceptable),
#   or wait until there is one
# - read at-most-N bytes
# Goals:
# - on average, do this fast
# - worst case, do this in O(n) where n is the number of bytes processed
# Plan:
# - store bytearray, offset, how far we've searched for a separator token
# - use the how-far-we've-searched data to avoid rescanning
# - while doing a stream of uninterrupted processing, advance offset instead
#   of constantly copying
# WARNING:
# - I haven't benchmarked or profiled any of this yet.
#
# Note that starting in Python 3.4, deleting the initial n bytes from a
# bytearray is amortized O(n), thanks to some excellent work by Antoine
# Martin:
#
#     https://bugs.python.org/issue19087
#
# This means that if we only supported 3.4+, we could get rid of the code here
# involving self._start and self.compress, because it's doing exactly the same
# thing that bytearray now does internally.
#
# BUT unfortunately, we still support 2.7, and reading short segments out of a
# long buffer MUST be O(bytes read) to avoid DoS issues, so we can't actually
# delete this code. Yet:
#
#     https://pythonclock.org/
#
# (Two things to double-check first though: make sure PyPy also has the
# optimization, and benchmark to make sure it's a win, since we do have a
# slightly clever thing where we delay calling compress() until we've
# processed a whole event, which could in theory be slightly more efficient
# than the internal bytearray support.)
blank_line_regex = re.compile(b"\n\r?\n", re.MULTILINE)


class ReceiveBuffer:
    def __init__(
        self,
        strict_crlf: Any = False,
        receive_policy: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            aliases = {
                key: kwargs.pop(key)
                for key in (
                    "line_ending",
                    "line_endings",
                    "line_ending_policy",
                    "line_terminator",
                    "line_terminators",
                    "receive_line_ending",
                    "receive_line_endings",
                    "receive_line_terminator",
                    "receive_line_terminators",
                    "newline_policy",
                    "newline_mode",
                    "receive_newline_policy",
                    "receive_mode",
                    "line_mode",
                    "crlf_mode",
                    "receive_line_ending_policy",
                    "crlf_policy",
                    "policy",
                    "strict",
                )
                if key in kwargs
            }
            if kwargs or len(aliases) > 1:
                unexpected = sorted(kwargs) if kwargs else sorted(aliases)
                raise TypeError(f"unexpected receive policy arguments: {unexpected}")
            if aliases:
                receive_policy = next(iter(aliases.values()))
        if receive_policy is None:
            receive_policy = strict_crlf
        self._receive_policy = _normalize_receive_policy(receive_policy)
        self._strict_crlf = _policy_is_strict(self._receive_policy)
        self._data = bytearray()
        self._next_line_search = 0
        self._multiple_lines_search = 0

    @property
    def receive_policy(self) -> ReceivePolicy:
        return self._receive_policy

    @property
    def strict_crlf(self) -> bool:
        return self._strict_crlf

    def __iadd__(self, byteslike: Union[bytes, bytearray]) -> "ReceiveBuffer":
        self._data += byteslike
        return self

    def __bool__(self) -> bool:
        return bool(len(self))

    def __len__(self) -> int:
        return len(self._data)

    # for @property unprocessed_data
    def __bytes__(self) -> bytes:
        return bytes(self._data)

    def _extract(self, count: int) -> bytearray:
        # extracting an initial slice of the data buffer and return it
        out = self._data[:count]
        del self._data[:count]

        self._next_line_search = 0
        self._multiple_lines_search = 0

        return out

    def maybe_extract_at_most(self, count: int) -> Optional[bytearray]:
        """
        Extract a fixed number of bytes from the buffer.
        """
        out = self._data[:count]
        if not out:
            return None

        return self._extract(count)

    def maybe_extract_next_line(self) -> Optional[bytearray]:
        """
        Extract the first line, if it is completed in the buffer.
        """
        if self._strict_crlf:
            return self._maybe_extract_next_line_strict()
        # Only search in buffer space that we've not already looked at.
        search_start_index = max(0, self._next_line_search - 1)
        partial_idx = self._data.find(b"\r\n", search_start_index)

        if partial_idx == -1:
            self._next_line_search = len(self._data)
            return None

        # + 2 is to compensate len(b"\r\n")
        idx = partial_idx + 2

        return self._extract(idx)

    def _maybe_extract_next_line_strict(self) -> Optional[bytearray]:
        search_start_index = max(0, self._next_line_search - 1)
        partial_idx = self._data.find(b"\n", search_start_index)
        if partial_idx == -1:
            self._validate_strict_line_bytes(search_start_index, len(self._data))
            self._next_line_search = len(self._data)
            return None

        self._validate_strict_line_bytes(search_start_index, partial_idx + 1)
        return self._extract(partial_idx + 1)

    def maybe_extract_lines(self) -> Optional[List[bytearray]]:
        """
        Extract everything up to the first blank line, and return a list of lines.
        """
        if self._strict_crlf:
            return self._maybe_extract_lines_strict()
        # Handle the case where we have an immediate empty line.
        if self._data[:1] == b"\n":
            self._extract(1)
            return []

        if self._data[:2] == b"\r\n":
            self._extract(2)
            return []

        # Only search in buffer space that we've not already looked at.
        match = blank_line_regex.search(self._data, self._multiple_lines_search)
        if match is None:
            self._multiple_lines_search = max(0, len(self._data) - 2)
            return None

        # Truncate the buffer and return it.
        idx = match.span(0)[-1]
        out = self._extract(idx)
        lines = out.split(b"\n")

        for line in lines:
            if line.endswith(b"\r"):
                del line[-1]

        assert lines[-2] == lines[-1] == b""

        del lines[-2:]

        return lines

    def _maybe_extract_lines_strict(self) -> Optional[List[bytearray]]:
        # Handle the case where we have an immediate empty line.
        if self._data[:2] == b"\r\n":
            self._extract(2)
            return []

        data = self._data
        search_start = max(0, self._multiple_lines_search - 1)
        while True:
            line_end = data.find(b"\n", search_start)
            if line_end == -1:
                self._validate_strict_line_bytes(search_start, len(data))
                self._multiple_lines_search = len(data)
                return None

            self._validate_strict_line_bytes(search_start, line_end + 1)
            if line_end >= 3 and data[line_end - 3 : line_end + 1] == b"\r\n\r\n":
                out = self._extract(line_end + 1)
                lines = out.split(b"\r\n")
                del lines[-2:]
                return lines

            search_start = line_end + 1
            self._multiple_lines_search = search_start

    def _validate_strict_line_bytes(self, start: int, end: int) -> None:
        data = self._data
        for index in range(start, end):
            if data[index] == 0x0A and (index == 0 or data[index - 1] != 0x0D):
                raise LocalProtocolError(
                    f"line ending must be CRLF, found bare LF at offset {index}"
                )
            if data[index] == 0x0D and (
                index + 1 < len(data) and data[index + 1] != 0x0A
            ):
                raise LocalProtocolError(
                    f"line ending must be CRLF, found bare CR at offset {index}"
                )

    # In theory we should wait until `\r\n` before starting to validate
    # incoming data. However it's interesting to detect (very) invalid data
    # early given they might not even contain `\r\n` at all (hence only
    # timeout will get rid of them).
    # This is not a 100% effective detection but more of a cheap sanity check
    # allowing for early abort in some useful cases.
    # This is especially interesting when peer is messing up with HTTPS and
    # sent us a TLS stream where we were expecting plain HTTP given all
    # versions of TLS so far start handshake with a 0x16 message type code.
    def is_next_line_obviously_invalid_request_line(self) -> bool:
        try:
            # HTTP header line must not contain non-printable characters
            # and should not start with a space
            return self._data[0] < 0x21
        except IndexError:
            return False
