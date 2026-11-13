# Optional trailer policy enforcement.
#
# RFC 9110 section 6.5 (and RFC 7230 section 4.1.2 before it) place two kinds
# of constraints on trailer sections:
#
# - Some fields must never appear in a trailer, because processing them as if
#   they were present in the header section might bypass security filters
#   (framing, routing, authentication, ...).
#
# - A sender SHOULD announce the trailer fields it intends to send using the
#   Trailer header field, so recipients can prepare (or reject) ahead of
#   time. Whether undeclared trailer fields are acceptable is a local policy
#   decision.
#
# This module implements a TrailerPolicy class that Connection can enforce in
# both the send and receive directions. It is entirely opt-in: passing no
# policy preserves h11's historical behavior.

from typing import Iterable, Optional, Sequence, Set, Tuple, Union

from ._util import LocalProtocolError

__all__ = ["TrailerPolicy"]

# Fields that must never appear in a trailer section, per RFC 9110 section
# 6.5.1 / RFC 7230 section 4.1.2: message framing, routing, connection
# management, request modifiers, authentication, and content metadata.
DEFAULT_FORBIDDEN_TRAILER_FIELDS = frozenset(
    [
        # Message framing
        b"transfer-encoding",
        b"content-length",
        b"trailer",
        # Routing / connection management
        b"host",
        b"connection",
        b"keep-alive",
        b"upgrade",
        b"proxy-connection",
        # Request modifiers
        b"expect",
        b"max-forwards",
        b"te",
        # Authentication
        b"authorization",
        b"proxy-authenticate",
        b"proxy-authorization",
        b"www-authenticate",
        # Content metadata
        b"content-encoding",
        b"content-type",
        b"content-range",
    ]
)


def _normalize_field_name(name: Union[bytes, str]) -> bytes:
    if isinstance(name, str):
        name = name.encode("ascii")
    return name.strip().lower()


class TrailerPolicy:
    """Policy describing which trailer fields are acceptable.

    Args:
        forbidden_fields: Field names that may never appear in a trailer
            section (nor be announced for one via the ``Trailer`` header).
            Names are case-insensitive and may be given as ``str`` or
            ``bytes``. If not given, defaults to
            ``DEFAULT_FORBIDDEN_TRAILER_FIELDS``. Pass an explicit iterable
            to fully control the set, e.g. to add deployment-specific fields.

        require_declaration: If True, every trailer field must have been
            announced in the message's ``Trailer`` header field; an
            undeclared trailer field is a protocol error. If False (the
            default), undeclared fields are allowed (but forbidden fields are
            still rejected).

    A policy is enforced by passing it to :class:`h11.Connection` as the
    ``trailer_policy`` argument. On send, a violating
    :class:`h11.EndOfMessage` raises :exc:`h11.LocalProtocolError` before any
    bytes are produced, so the peer never sees a message that appeared to end
    successfully. On receive, a violating trailer section raises
    :exc:`h11.RemoteProtocolError` instead of returning the
    :class:`h11.EndOfMessage` event.

    """

    def __init__(
        self,
        *,
        forbidden_fields: Optional[Iterable[Union[bytes, str]]] = None,
        require_declaration: bool = False,
    ) -> None:
        if forbidden_fields is None:
            forbidden_fields = DEFAULT_FORBIDDEN_TRAILER_FIELDS
        self.forbidden_fields = frozenset(
            _normalize_field_name(name) for name in forbidden_fields
        )
        self.require_declaration = require_declaration

    def validate(
        self,
        *,
        declared_fields: Sequence[bytes],
        headers: Sequence[Tuple[bytes, bytes]],
        is_chunked: bool,
    ) -> None:
        """Check a trailer section against this policy.

        ``declared_fields`` are the (already lowercased) names collected from
        the message's ``Trailer`` header; ``headers`` are the trailer fields
        themselves (with lowercased names, as h11 normalizes them);
        ``is_chunked`` says whether the message body used chunked transfer
        coding. Raises :exc:`LocalProtocolError` on any violation.

        """
        if not headers:
            return
        if not is_chunked:
            raise LocalProtocolError(
                "trailers are only allowed with Transfer-Encoding: chunked"
            )
        declared: Set[bytes] = set(declared_fields)
        for name in declared:
            if name in self.forbidden_fields:
                raise LocalProtocolError(
                    "Trailer: header announces forbidden trailer field "
                    "{!r}".format(name)
                )
        for name, _value in headers:
            if name in self.forbidden_fields:
                raise LocalProtocolError(
                    "forbidden field {!r} in trailer section".format(name)
                )
            if self.require_declaration and name not in declared:
                raise LocalProtocolError(
                    "trailer field {!r} was not announced in the Trailer: "
                    "header".format(name)
                )
