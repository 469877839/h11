# Pluggable policy for validating HTTP/1.1 chunked trailer fields.
#
# RFC 7230 section 4.1.2:
# - A sender MUST NOT generate trailer fields until it has advertised them in
#   the Trailer header field of the message's header section.
# - A server SHOULD omit the Trailer header from responses that use a
#   "Content-Length" framing (trailers are only legal on chunked messages).
# - A recipient MUST ignore (or consider as an error) any fields that are
#   forbidden to be sent in a trailer, since processing them as if they were
#   present in the header section might bypass external security filters.
#
# The exact set of forbidden fields and the handling of fields that were not
# declared ahead of time are deployment-specific policy decisions, so h11
# keeps them behind an explicit policy object. The default policy preserves
# the historical, fully permissive behavior.

from dataclasses import dataclass
from typing import FrozenSet, Optional, Set

from ._headers import Headers, get_comma_header
from ._util import LocalProtocolError, bytesify

__all__ = [
    "TrailerPolicy",
    "DEFAULT_TRAILER_POLICY",
    "STRICT_TRAILER_POLICY",
    "FORBIDDEN_TRAILER_FIELDS",
]


# RFC 7230 section 4.1.2: fields that define message framing, routing,
# authentication, or response control data, and so cannot be generated as
# trailer fields.
FORBIDDEN_TRAILER_FIELDS: FrozenSet[bytes] = frozenset(
    {
        b"authorization",
        b"content-encoding",
        b"content-length",
        b"content-range",
        b"content-type",
        b"cookie",
        b"date",
        b"expect",
        b"host",
        b"max-forwards",
        b"proxy-authenticate",
        b"proxy-authorization",
        b"range",
        b"set-cookie",
        b"te",
        b"trailer",
        b"transfer-encoding",
        b"www-authenticate",
    }
)


def _coerce_field_names(
    names: Optional[FrozenSet[bytes]],
) -> Optional[FrozenSet[bytes]]:
    if names is None:
        return None
    return frozenset(bytesify(name).lower() for name in names)


def is_chunked(headers: Headers) -> bool:
    return get_comma_header(headers, b"transfer-encoding") == [b"chunked"]


@dataclass(frozen=True)
class TrailerPolicy:
    """Policy controlling which fields may appear in chunked trailers.

    The same policy is enforced on both directions of a
    :class:`~h11.Connection`: violations while sending raise
    :class:`~h11.LocalProtocolError` before any framing bytes are produced,
    and violations while receiving raise
    :class:`~h11.RemoteProtocolError` before the offending
    :class:`~h11.EndOfMessage` is delivered.

    All field names are matched case-insensitively; h11 stores header names
    in lower case, so policy configuration uses lower-case byte strings.

    Args:
        forbidden_fields:
            Field names that may never appear in a trailer. Pass
            ``None`` (the default) to disable this check and accept any
            field; otherwise pass a frozenset of lower-case byte strings,
            e.g. :data:`FORBIDDEN_TRAILER_FIELDS` for the fields forbidden
            by RFC 7230 section 4.1.2 (framing, routing and authentication
            controls).

        require_declaration:
            If true, every trailer field must have been advertised in the
            ``Trailer:`` header of the message's header section. A field
            repeated in ``Trailer:`` (e.g. ``Trailer: x, x``) counts as a
            single declaration.

        allow_undeclared:
            What to do with trailer fields that were not declared in
            ``Trailer:``. Only meaningful when ``require_declaration`` is
            true: leave it false (the default) to treat undeclared fields
            as a protocol error, or set it true to keep them on the
            :class:`~h11.EndOfMessage` anyway. When declarations are not
            required, all fields are accepted regardless.

    A bare ``TrailerPolicy()`` accepts any legal chunked trailer and is
    fully backwards compatible. For a strict policy that enforces
    declaration and the RFC's forbidden-field rules, see
    :data:`STRICT_TRAILER_POLICY`.
    """

    forbidden_fields: Optional[FrozenSet[bytes]] = None
    require_declaration: bool = False
    allow_undeclared: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "forbidden_fields", _coerce_field_names(self.forbidden_fields)
        )

    def declared_fields(self, message_headers: Headers) -> Set[bytes]:
        # The Trailer header itself uses comma-separated field names, which
        # is exactly what get_comma_header knows how to split.
        return set(get_comma_header(message_headers, b"trailer"))

    def check(
        self,
        trailers: Headers,
        message_headers: Headers,
        chunked: bool,
    ) -> None:
        """Validate the trailers of one message.

        ``message_headers`` are the regular (non-trailer) headers of the
        :class:`~h11.Request` or :class:`~h11.Response` that started this
        message; ``chunked`` says whether that message uses
        ``Transfer-Encoding: chunked`` framing. Raises
        :class:`~h11.LocalProtocolError` on any violation.
        """
        if not trailers:
            return
        if not chunked:
            raise LocalProtocolError(
                "trailer fields are only allowed on Transfer-Encoding: "
                "chunked messages"
            )
        forbidden = self.forbidden_fields
        if forbidden is not None:
            for name, _value in trailers:
                if name in forbidden:
                    raise LocalProtocolError(f"field {name!r} is forbidden in trailers")
        if self.require_declaration and not self.allow_undeclared:
            declared = self.declared_fields(message_headers)
            for name, _value in trailers:
                if name not in declared:
                    raise LocalProtocolError(
                        f"trailer field {name!r} was not declared in a "
                        "Trailer: header"
                    )

    def check_declarations(self, message_headers: Headers) -> None:
        """Validate the ``Trailer:`` declaration of a message being sent.

        Only applied on the send side, where the local application controls
        the message headers: a ``Trailer:`` header cannot promise fields
        that this policy forbids. When ``require_declaration`` is set, a
        declaration is also meaningless -- and therefore rejected -- unless
        the message actually uses chunked framing. (Without that setting a
        stray ``Trailer:`` header is left alone, preserving historical
        behavior.)
        """
        declared = self.declared_fields(message_headers)
        if not declared:
            return
        if self.require_declaration and not is_chunked(message_headers):
            raise LocalProtocolError(
                "Trailer: header is only valid on Transfer-Encoding: chunked "
                "messages"
            )
        forbidden = self.forbidden_fields
        if forbidden is not None:
            for name in declared:
                if name in forbidden:
                    raise LocalProtocolError(f"field {name!r} is forbidden in trailers")


# Accepts anything that h11 historically accepted: no declaration required,
# no forbidden-field filtering.
DEFAULT_TRAILER_POLICY = TrailerPolicy()

# Strict, spec-aligned policy: trailers must be declared up front, may only
# contain fields that are safe to process outside the header section, and
# anything else is a protocol error.
STRICT_TRAILER_POLICY = TrailerPolicy(
    forbidden_fields=FORBIDDEN_TRAILER_FIELDS,
    require_declaration=True,
)
