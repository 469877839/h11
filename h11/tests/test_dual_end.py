from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from .._events import (
    ConnectionClosed,
    Data,
    EndOfMessage,
    InformationalResponse,
    Request,
    Response,
)
from .._state import CLIENT, SERVER
from .dual_end import (
    DualEnd,
    EarlyResetConnection,
    EventLog,
    ModelMismatch,
    Op,
    ReferenceView,
    ReferenceViolation,
    Result,
    random_sequence,
    replay_log,
    run_sequence,
    shrink_failure,
)


FIXED_SEEDS = (0, 1, 2, 3, 7, 11, 19, 42)
MUTANT_SEED = 187


def host_headers(*extra: Tuple[bytes, bytes]) -> List[Tuple[bytes, bytes]]:
    return [(b"host", b"example.com")] + list(extra)


def request(
    method: str = "GET",
    target: str = "/",
    headers: Optional[Sequence[Tuple[bytes, bytes]]] = None,
    http_version: str = "1.1",
) -> Request:
    return Request(
        method=method,
        target=target,
        headers=host_headers() if headers is None else list(headers),
        http_version=http_version,
    )


def response(status_code: int = 204, headers: Any = None) -> Response:
    return Response(
        status_code=status_code,
        headers=[] if headers is None else headers,
    )


def informational(status_code: int = 100, headers: Any = None) -> InformationalResponse:
    return InformationalResponse(
        status_code=status_code,
        headers=[] if headers is None else headers,
    )


def endpoint_states(result: Result, endpoint: str) -> Dict[str, str]:
    return result.states[endpoint]["states"]


def role_state(result: Result, endpoint: str, role: str) -> str:
    return result.states[endpoint]["states"][role]


def assert_ok(result: Result) -> None:
    # A pump normally ends at NEED_DATA or PAUSED; that is successful
    # operation, not an error.
    assert result.status in ("ok", "terminal"), result.to_json()
    assert result.error is None, result.to_json()


def assert_error(result: Result, exception_name: str) -> None:
    assert result.status == "error", result.to_json()
    assert result.error == exception_name, result.to_json()


def complete_request(
    model: DualEnd,
    *,
    method: str = "GET",
    headers: Optional[Sequence[Tuple[bytes, bytes]]] = None,
    body: Optional[bytes] = None,
) -> None:
    request_headers = list(headers or host_headers())
    if body is not None:
        request_headers.append((b"content-length", str(len(body)).encode("ascii")))
    sent = model.send("CLIENT", [request(method=method, headers=request_headers)])
    assert_ok(sent)
    assert_ok(model.deliver("CLIENT"))
    pumped = model.pump("SERVER")
    assert_ok(pumped)
    if body is None:
        assert [type(event) for event in pumped.events] == [
            Request,
            EndOfMessage,
        ]
    else:
        assert [type(event) for event in pumped.events] == [Request]
        assert_ok(model.send("CLIENT", [Data(body)]))
        assert_ok(model.deliver("CLIENT"))
        pumped = model.pump("SERVER")
        assert [type(event) for event in pumped.events] == [Data, EndOfMessage]
    assert_ok(model.send("CLIENT", [EndOfMessage()]))


def test_wire_splits_and_deferred_consumption() -> None:
    model = DualEnd()
    sent = model.send(
        "CLIENT",
        [request(target="/split")],
        chunk_size=3,
    )
    assert_ok(sent)
    assert endpoint_states(sent, "CLIENT") == {
        "CLIENT": "SEND_BODY",
        "SERVER": "SEND_RESPONSE",
    }
    assert endpoint_states(sent, "SERVER") == {"CLIENT": "IDLE", "SERVER": "IDLE"}

    # Bytes can arrive without the server choosing to parse them.
    assert_ok(model.deliver("CLIENT"))
    assert endpoint_states(model.send("CLIENT", [EndOfMessage()]), "SERVER") == {
        "CLIENT": "IDLE",
        "SERVER": "IDLE",
    }
    assert endpoint_states(model.pump("SERVER"), "SERVER") == {
        "CLIENT": "DONE",
        "SERVER": "SEND_RESPONSE",
    }

    # A second copy, consumed byte-chunk by byte-chunk, shows NEED_DATA until
    # the request line and headers are complete.
    model = DualEnd()
    assert_ok(model.send("CLIENT", [request(target="/slow")], chunk_size=4))
    assert model.pump("SERVER").events == []
    for _ in range(3):
        assert_ok(model.deliver("CLIENT", limit=1))
        pumped = model.pump("SERVER")
        assert pumped.terminal == "NEED_DATA"
        assert pumped.events == []
    assert_ok(model.deliver("CLIENT"))
    pumped = model.pump("SERVER")
    assert [type(event) for event in pumped.events] == [Request, EndOfMessage]
    assert pumped.terminal == "NEED_DATA"

    taken = model.take("SERVER")
    assert [type(event) for event in taken.events] == [Request, EndOfMessage]
    assert model.take("SERVER").events == []
    assert role_state(pumped, "SERVER", "CLIENT") == "DONE"
    assert role_state(pumped, "SERVER", "SERVER") == "SEND_RESPONSE"


def test_100_continue_multiple_1xx_and_wait_clearing() -> None:
    model = DualEnd()
    headers = host_headers((b"content-length", b"4"), (b"expect", b"100-continue"))
    assert_ok(model.send("CLIENT", [request(method="POST", headers=headers)]))
    assert model.conns["CLIENT"].client_is_waiting_for_100_continue
    assert_ok(model.deliver("CLIENT"))
    assert_ok(model.pump("SERVER"))
    assert model.conns["CLIENT"].client_is_waiting_for_100_continue
    assert model.conns["SERVER"].client_is_waiting_for_100_continue

    assert_ok(model.deliver("SERVER"))  # nothing queued yet
    sent = model.send("SERVER", [informational(100)])
    assert_ok(sent)
    assert not model.conns["SERVER"].client_is_waiting_for_100_continue
    assert_ok(model.deliver("SERVER"))
    pumped = model.pump("CLIENT")
    statuses = [getattr(event, "status_code") for event in pumped.events]
    assert statuses == [100]
    assert not model.conns["CLIENT"].client_is_waiting_for_100_continue

    # Several informational responses are legal and do not end the response
    # phase.  103 is enough to clear h11's wait flag too.
    assert_ok(model.send("SERVER", [informational(103)]))
    assert_ok(model.send("SERVER", [informational(102)]))
    assert_ok(model.deliver("SERVER"))
    pumped = model.pump("CLIENT")
    statuses = [getattr(event, "status_code") for event in pumped.events]
    assert statuses == [103, 102]
    assert endpoint_states(pumped, "CLIENT") == {
        "CLIENT": "SEND_BODY",
        "SERVER": "SEND_RESPONSE",
    }

    assert_ok(model.send("SERVER", [response(204)]))
    assert_ok(model.send("SERVER", [EndOfMessage()]))
    assert_ok(model.deliver("SERVER"))
    pumped = model.pump("CLIENT")
    assert [type(event) for event in pumped.events] == [Response, EndOfMessage]
    assert endpoint_states(pumped, "CLIENT") == {
        "CLIENT": "SEND_BODY",
        "SERVER": "DONE",
    }

    # The client may finish a body that was already in progress even after a
    # final response arrived.  This also clears the 100-continue wait.
    assert_ok(model.send("CLIENT", [Data(b"abcd")]))
    assert_ok(model.send("CLIENT", [EndOfMessage()]))
    assert_ok(model.deliver("CLIENT"))
    pumped = model.pump("SERVER")
    assert [type(event) for event in pumped.events] == [Data, EndOfMessage]
    assert endpoint_states(pumped, "SERVER") == {
        "CLIENT": "DONE",
        "SERVER": "DONE",
    }

    # A non-100 1xx also clears the wait, matching h11's documented behavior.
    model = DualEnd()
    headers = host_headers((b"content-length", b"4"), (b"expect", b"100-continue"))
    assert_ok(model.send("CLIENT", [request(method="POST", headers=headers)]))
    assert_ok(model.deliver("CLIENT"))
    assert_ok(model.pump("SERVER"))
    assert_ok(model.send("SERVER", [informational(103)]))
    assert not model.conns["SERVER"].client_is_waiting_for_100_continue
    assert_ok(model.deliver("SERVER"))
    assert_ok(model.pump("CLIENT"))
    assert not model.conns["CLIENT"].client_is_waiting_for_100_continue

    # So does the client starting its body on its own.
    model = DualEnd()
    headers = host_headers((b"content-length", b"4"), (b"expect", b"100-continue"))
    assert_ok(model.send("CLIENT", [request(method="POST", headers=headers)]))
    assert_ok(model.send("CLIENT", [Data(b"abcd")]))
    assert not model.conns["CLIENT"].client_is_waiting_for_100_continue
    assert_ok(model.deliver("CLIENT"))
    assert_ok(model.pump("SERVER"))
    assert not model.conns["SERVER"].client_is_waiting_for_100_continue


@pytest.mark.parametrize(
    "method,response_event",
    [
        ("HEAD", response(200, [(b"content-length", b"100")])),
        ("GET", response(204, [(b"content-length", b"10")])),
        ("GET", response(304, [(b"content-length", b"10")])),
    ],
)
def test_head_204_304_have_no_body(
    method: str, response_event: Response
) -> None:
    model = DualEnd()
    complete_request(model, method=method)
    assert_ok(model.send("SERVER", [response_event]))
    assert_ok(model.send("SERVER", [EndOfMessage()]))
    assert_ok(model.deliver("SERVER"))
    pumped = model.pump("CLIENT")
    assert [type(event) for event in pumped.events] == [Response, EndOfMessage]
    assert endpoint_states(pumped, "CLIENT") == {
        "CLIENT": "DONE",
        "SERVER": "DONE",
    }

    # Extra body bytes are not consumed: they stay in trailing_data.  A reset
    # then tries to parse them as the next response and reports a remote
    # protocol error.
    assert_ok(model.receive_raw("CLIENT", b"EXTRA\r\n\r\n"))
    assert_ok(model.pump("CLIENT"))
    assert model.conns["CLIENT"].trailing_data == (b"EXTRA\r\n\r\n", False)
    assert_ok(model.start_next_cycle("CLIENT"))
    assert_ok(model.start_next_cycle("SERVER"))
    parsed = model.pump("CLIENT")
    assert_error(parsed, "RemoteProtocolError")
    assert role_state(parsed, "CLIENT", "SERVER") == "ERROR"

    # A local sender of such a response must not be allowed to emit body
    # data, and after the error it cannot send the EOM either.
    model = DualEnd()
    complete_request(model, method=method)
    assert_ok(model.send("SERVER", [response_event]))
    bad_data = model.send("SERVER", [Data(b"x")])
    assert_error(bad_data, "LocalProtocolError")
    assert "Too much data" in (bad_data.error_message or "")
    assert role_state(bad_data, "SERVER", "SERVER") == "ERROR"
    assert role_state(bad_data, "SERVER", "CLIENT") == "MUST_CLOSE"

    after_error = model.send("SERVER", [EndOfMessage()])
    assert_error(after_error, "LocalProtocolError")
    assert "Can't send data when our state is ERROR" in (
        after_error.error_message or ""
    )
    assert endpoint_states(after_error, "SERVER") == {
        "CLIENT": "MUST_CLOSE",
        "SERVER": "ERROR",
    }

    # A fresh connection can legally finish the zero-length body.
    model = DualEnd()
    complete_request(model, method=method)
    assert_ok(model.send("SERVER", [response_event]))
    done = model.send("SERVER", [EndOfMessage()])
    assert endpoint_states(done, "SERVER") == {
        "CLIENT": "DONE",
        "SERVER": "DONE",
    }


def test_connect_accept_and_deny() -> None:
    model = DualEnd()
    assert_ok(
        model.send(
            "CLIENT",
            [request(method="CONNECT", target="example.com:443")],
        )
    )
    client_done = model.send("CLIENT", [EndOfMessage()])
    assert endpoint_states(client_done, "CLIENT") == {
        "CLIENT": "MIGHT_SWITCH_PROTOCOL",
        "SERVER": "SEND_RESPONSE",
    }
    # The server has not consumed anything yet, so its view is still IDLE.
    assert endpoint_states(client_done, "SERVER") == {
        "CLIENT": "IDLE",
        "SERVER": "IDLE",
    }
    assert_ok(model.deliver("CLIENT"))
    server_view = model.pump("SERVER")
    assert [type(event) for event in server_view.events] == [Request, EndOfMessage]
    assert endpoint_states(server_view, "SERVER") == {
        "CLIENT": "MIGHT_SWITCH_PROTOCOL",
        "SERVER": "SEND_RESPONSE",
    }
    accepted = model.send("SERVER", [response(200)])
    assert_ok(accepted)
    assert endpoint_states(accepted, "SERVER") == {
        "CLIENT": "SWITCHED_PROTOCOL",
        "SERVER": "SWITCHED_PROTOCOL",
    }
    assert_ok(model.deliver("SERVER"))
    client_view = model.pump("CLIENT")
    assert [type(event) for event in client_view.events] == [Response]
    assert endpoint_states(client_view, "CLIENT") == {
        "CLIENT": "SWITCHED_PROTOCOL",
        "SERVER": "SWITCHED_PROTOCOL",
    }
    cycle = model.start_next_cycle("CLIENT")
    assert_error(cycle, "LocalProtocolError")
    assert endpoint_states(cycle, "CLIENT") == {
        "CLIENT": "SWITCHED_PROTOCOL",
        "SERVER": "SWITCHED_PROTOCOL",
    }

    # No more HTTP events are legal, but opaque trailing bytes are simply
    # left for the caller.
    assert_ok(model.receive_raw("CLIENT", b"opaque"))
    paused = model.pump("CLIENT")
    assert paused.status == "terminal"
    assert paused.terminal == "PAUSED"
    assert paused.events == []
    assert model.conns["CLIENT"].trailing_data == (b"opaque", False)
    switched_send = model.send("CLIENT", [Data(b"x")])
    assert_error(switched_send, "LocalProtocolError")
    assert role_state(switched_send, "CLIENT", "CLIENT") == "ERROR"

    # Denied CONNECT clears the proposal and permits a normal next cycle.
    model = DualEnd()
    assert_ok(
        model.send(
            "CLIENT",
            [request(method="CONNECT", target="example.com:443")],
        )
    )
    assert_ok(model.send("CLIENT", [EndOfMessage()]))
    assert_ok(model.deliver("CLIENT"))
    assert_ok(model.pump("SERVER"))
    assert_ok(model.send("SERVER", [response(404, [(b"content-length", b"0")])]))
    assert_ok(model.send("SERVER", [EndOfMessage()]))
    assert_ok(model.deliver("SERVER"))
    pumped = model.pump("CLIENT")
    assert [type(event) for event in pumped.events] == [Response, EndOfMessage]
    assert endpoint_states(pumped, "CLIENT") == {
        "CLIENT": "DONE",
        "SERVER": "DONE",
    }
    assert model.conns["CLIENT"]._cstate.pending_switch_proposals == set()
    assert_ok(model.start_next_cycle("CLIENT"))
    assert_ok(model.start_next_cycle("SERVER"))
    assert_ok(model.send("CLIENT", [request(target="/next")]))


def test_101_upgrade_and_illegal_101() -> None:
    model = DualEnd()
    headers = host_headers((b"upgrade", b"websocket"))
    assert_ok(model.send("CLIENT", [request(target="/ws", headers=headers)]))
    assert_ok(model.send("CLIENT", [EndOfMessage()]))
    assert_ok(model.deliver("CLIENT"))
    pumped = model.pump("SERVER")
    assert [type(event) for event in pumped.events] == [Request, EndOfMessage]
    assert endpoint_states(pumped, "SERVER") == {
        "CLIENT": "MIGHT_SWITCH_PROTOCOL",
        "SERVER": "SEND_RESPONSE",
    }
    assert_ok(
        model.send(
            "SERVER",
            [informational(101, [(b"upgrade", b"websocket")])],
        )
    )
    assert_ok(model.deliver("SERVER"))
    switched = model.pump("CLIENT")
    assert [type(event) for event in switched.events] == [InformationalResponse]
    assert endpoint_states(switched, "CLIENT") == {
        "CLIENT": "SWITCHED_PROTOCOL",
        "SERVER": "SWITCHED_PROTOCOL",
    }
    assert endpoint_states(
        model.send("SERVER", [informational(102)]), "SERVER"
    ) == {
        "CLIENT": "SWITCHED_PROTOCOL",
        "SERVER": "ERROR",
    }

    model = DualEnd()
    complete_request(model)
    illegal = model.send("SERVER", [informational(101, [(b"upgrade", b"websocket")])])
    assert_error(illegal, "LocalProtocolError")
    assert "without a pending proposal" in (illegal.error_message or "")
    assert endpoint_states(illegal, "SERVER") == {
        "CLIENT": "MUST_CLOSE",
        "SERVER": "ERROR",
    }


def test_non_keep_alive_and_next_cycle_conditions() -> None:
    # Connection: close is permanent.  Both sides reach MUST_CLOSE and no
    # reset is legal.
    model = DualEnd()
    close_request = request(headers=host_headers((b"connection", b"close")))
    assert_ok(model.send("CLIENT", [close_request]))
    client_done = model.send("CLIENT", [EndOfMessage()])
    assert endpoint_states(client_done, "CLIENT") == {
        "CLIENT": "MUST_CLOSE",
        "SERVER": "SEND_RESPONSE",
    }
    assert_ok(model.deliver("CLIENT"))
    assert_ok(model.pump("SERVER"))
    assert_ok(model.send("SERVER", [response(204)]))
    done = model.send("SERVER", [EndOfMessage()])
    assert endpoint_states(done, "SERVER") == {
        "CLIENT": "MUST_CLOSE",
        "SERVER": "MUST_CLOSE",
    }

    model = DualEnd()
    assert_ok(model.send("CLIENT", [close_request]))
    assert_ok(model.send("CLIENT", [EndOfMessage()]))
    assert_ok(model.deliver("CLIENT"))
    assert_ok(model.pump("SERVER"))
    assert_ok(model.send("SERVER", [response(204)]))
    assert_ok(model.send("SERVER", [EndOfMessage()]))
    assert_ok(model.deliver("SERVER"))
    assert_ok(model.pump("CLIENT"))
    client_cycle = model.start_next_cycle("CLIENT")
    assert_error(client_cycle, "LocalProtocolError")
    assert "not in a reusable state" in (client_cycle.error_message or "")
    server_cycle = model.start_next_cycle("SERVER")
    assert_error(server_cycle, "LocalProtocolError")
    assert endpoint_states(server_cycle, "SERVER") == {
        "CLIENT": "MUST_CLOSE",
        "SERVER": "MUST_CLOSE",
    }
    new_request = model.send("CLIENT", [request(target="/again")])
    assert_error(new_request, "LocalProtocolError")
    assert role_state(new_request, "CLIENT", "CLIENT") == "ERROR"

    # A server can also make the connection non-reusable.
    model = DualEnd()
    complete_request(model)
    assert_ok(model.send("SERVER", [response(204, [(b"connection", b"close")])]))
    server_done = model.send("SERVER", [EndOfMessage()])
    assert endpoint_states(server_done, "SERVER") == {
        "CLIENT": "MUST_CLOSE",
        "SERVER": "MUST_CLOSE",
    }
    assert_ok(model.deliver("SERVER"))
    assert_ok(model.pump("CLIENT"))
    assert_error(model.start_next_cycle("CLIENT"), "LocalProtocolError")
    assert_error(model.start_next_cycle("SERVER"), "LocalProtocolError")

    # HTTP/1.0 requests are also one-shot, and the response is framed by
    # EOF rather than chunked encoding.
    model = DualEnd()
    assert_ok(model.receive_raw("SERVER", b"GET / HTTP/1.0\r\n\r\n"))
    pumped = model.pump("SERVER")
    assert [type(event) for event in pumped.events] == [Request, EndOfMessage]
    assert endpoint_states(pumped, "SERVER") == {
        "CLIENT": "MUST_CLOSE",
        "SERVER": "SEND_RESPONSE",
    }
    sent = model.send(
        "SERVER",
        [response(200, [(b"connection", b"close")])],
    )
    assert_ok(sent)
    assert b"Connection: close" in sent.data
    assert_ok(model.send("SERVER", [EndOfMessage()]))
    assert endpoint_states(
        model.start_next_cycle("SERVER"), "SERVER"
    ) == {
        "CLIENT": "MUST_CLOSE",
        "SERVER": "MUST_CLOSE",
    }

    # A reset is legal only when both ends are DONE.  One end resetting is
    # not enough; the peer pauses until it resets too.
    model = DualEnd()
    complete_request(model)
    assert_ok(model.send("SERVER", [response(204)]))
    assert_ok(model.send("SERVER", [EndOfMessage()]))
    assert_ok(model.deliver("SERVER"))
    done = model.pump("CLIENT")
    assert endpoint_states(done, "CLIENT") == {
        "CLIENT": "DONE",
        "SERVER": "DONE",
    }

    model = DualEnd()
    complete_request(model)
    assert_ok(model.send("SERVER", [response(204)]))
    assert_ok(model.send("SERVER", [EndOfMessage()]))
    assert_ok(model.deliver("SERVER"))
    assert_ok(model.pump("CLIENT"))
    assert_ok(model.start_next_cycle("CLIENT"))
    assert_ok(model.send("CLIENT", [request(target="/second")]))
    assert_ok(model.deliver("CLIENT"))
    paused = model.pump("SERVER")
    assert paused.status == "terminal"
    assert paused.terminal == "PAUSED"
    assert paused.events == []
    assert_ok(model.start_next_cycle("SERVER"))
    resumed = model.pump("SERVER")
    assert [type(event) for event in resumed.events] == [Request, EndOfMessage]

    # Calling start_next_cycle before DONE fails without damaging the
    # connection: the pending EOM can still be sent afterwards.
    model = DualEnd()
    assert_ok(model.send("CLIENT", [request()]))
    early = model.start_next_cycle("CLIENT")
    assert_error(early, "LocalProtocolError")
    assert endpoint_states(early, "CLIENT") == {
        "CLIENT": "SEND_BODY",
        "SERVER": "SEND_RESPONSE",
    }
    client_done = model.send("CLIENT", [EndOfMessage()])
    assert endpoint_states(client_done, "CLIENT") == {
        "CLIENT": "DONE",
        "SERVER": "SEND_RESPONSE",
    }


def test_illegal_sequences_are_classified_and_stateful() -> None:
    # Local errors: exact exception type, ERROR state, and a second send
    # proves whether the endpoint can continue.
    model = DualEnd()
    first = model.send("SERVER", [Data(b"x")])
    assert_error(first, "LocalProtocolError")
    assert endpoint_states(first, "SERVER") == {
        "CLIENT": "IDLE",
        "SERVER": "ERROR",
    }
    second = model.send("SERVER", [response(204)])
    assert_error(second, "LocalProtocolError")
    assert "Can't send data when our state is ERROR" in (second.error_message or "")
    assert endpoint_states(second, "SERVER") == {
        "CLIENT": "IDLE",
        "SERVER": "ERROR",
    }

    model = DualEnd()
    wrong_role = model.send("CLIENT", [response(204)])
    assert_error(wrong_role, "LocalProtocolError")
    assert role_state(wrong_role, "CLIENT", "CLIENT") == "ERROR"

    model = DualEnd()
    complete_request(model)
    first_response = model.send("SERVER", [response(204)])
    assert_ok(first_response)
    second_response = model.send("SERVER", [response(200)])
    assert_error(second_response, "LocalProtocolError")
    assert endpoint_states(second_response, "SERVER") == {
        "CLIENT": "MUST_CLOSE",
        "SERVER": "ERROR",
    }

    model = DualEnd()
    complete_request(model)
    assert_ok(model.send("SERVER", [response(204)]))
    late_info = model.send("SERVER", [informational(103)])
    assert_error(late_info, "LocalProtocolError")
    assert role_state(late_info, "SERVER", "SERVER") == "ERROR"

    model = DualEnd()
    complete_request(model)
    late_data = model.send("CLIENT", [Data(b"x")])
    assert_error(late_data, "LocalProtocolError")
    assert endpoint_states(late_data, "CLIENT") == {
        "CLIENT": "ERROR",
        "SERVER": "SEND_RESPONSE",
    }

    model = DualEnd()
    headers = host_headers((b"content-length", b"5"))
    assert_ok(model.send("CLIENT", [request(method="POST", headers=headers)]))
    closed_too_early = model.send("CLIENT", [ConnectionClosed()])
    assert_error(closed_too_early, "LocalProtocolError")
    assert endpoint_states(closed_too_early, "CLIENT") == {
        "CLIENT": "ERROR",
        "SERVER": "SEND_RESPONSE",
    }

    # Remote error while the server is still IDLE: the server may send an
    # error response, but its peer role is ERROR and the error is sticky.
    model = DualEnd()
    assert_ok(
        model.receive_raw(
            "SERVER", b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
        )
    )
    remote = model.pump("SERVER")
    assert_error(remote, "RemoteProtocolError")
    assert "illegal request line" in (remote.error_message or "")
    assert endpoint_states(remote, "SERVER") == {
        "CLIENT": "ERROR",
        "SERVER": "IDLE",
    }
    error_response = model.send("SERVER", [response(400)])
    assert_ok(error_response)
    assert endpoint_states(error_response, "SERVER") == {
        "CLIENT": "ERROR",
        "SERVER": "SEND_BODY",
    }
    assert_ok(model.send("SERVER", [EndOfMessage()]))
    assert model.pump("SERVER").error == "RemoteProtocolError"

    # Remote error while the client still has a request body to send: it can
    # finish that body, but after DONE the ERROR peer makes it MUST_CLOSE.
    model = DualEnd()
    headers = host_headers((b"content-length", b"4"))
    assert_ok(model.send("CLIENT", [request(method="POST", headers=headers)]))
    assert_ok(model.receive_raw("CLIENT", b"garbage\r\n\r\n"))
    remote = model.pump("CLIENT")
    assert_error(remote, "RemoteProtocolError")
    assert endpoint_states(remote, "CLIENT") == {
        "CLIENT": "SEND_BODY",
        "SERVER": "ERROR",
    }
    body = model.send("CLIENT", [Data(b"abcd")])
    assert_ok(body)
    eom = model.send("CLIENT", [EndOfMessage()])
    assert_ok(eom)
    assert endpoint_states(eom, "CLIENT") == {
        "CLIENT": "MUST_CLOSE",
        "SERVER": "ERROR",
    }
    stuck = model.send("CLIENT", [request(target="/another")])
    assert_error(stuck, "LocalProtocolError")
    assert endpoint_states(stuck, "CLIENT") == {
        "CLIENT": "ERROR",
        "SERVER": "ERROR",
    }

    # Remote error after the client request is done: its local role becomes
    # MUST_CLOSE through the joint ERROR/DONE transition and cannot send.
    model = DualEnd()
    complete_request(model)
    assert_ok(model.receive_raw("CLIENT", b"GET / HTTP/1.1\r\nHost: x\r\n\r\n"))
    remote = model.pump("CLIENT")
    assert_error(remote, "RemoteProtocolError")
    assert endpoint_states(remote, "CLIENT") == {
        "CLIENT": "MUST_CLOSE",
        "SERVER": "ERROR",
    }
    repeated = model.pump("CLIENT")
    assert_error(repeated, "RemoteProtocolError")
    assert "Can't receive data when peer state is ERROR" in (
        repeated.error_message or ""
    )
    assert endpoint_states(repeated, "CLIENT") == {
        "CLIENT": "MUST_CLOSE",
        "SERVER": "ERROR",
    }


def test_early_transport_close_is_visible_to_the_other_end() -> None:
    # A close before the request body is complete is a remote protocol
    # error, not a normal ConnectionClosed event.
    model = DualEnd()
    headers = host_headers((b"content-length", b"5"))
    assert_ok(model.send("CLIENT", [request(method="POST", headers=headers)]))
    assert_ok(model.deliver("CLIENT"))
    assert [type(event) for event in model.pump("SERVER").events] == [Request]
    assert_ok(model.abort("CLIENT"))
    assert_ok(model.deliver("CLIENT"))
    truncated = model.pump("SERVER")
    assert_error(truncated, "RemoteProtocolError")
    assert "expected 5" in (truncated.error_message or "")
    assert endpoint_states(truncated, "SERVER") == {
        "CLIENT": "ERROR",
        "SERVER": "SEND_RESPONSE",
    }

    # The server may still send an error response after that remote error,
    # but the cycle can never be reused.
    assert_ok(model.send("SERVER", [response(400)]))
    closed = model.send("SERVER", [EndOfMessage()])
    assert endpoint_states(closed, "SERVER") == {
        "CLIENT": "ERROR",
        "SERVER": "MUST_CLOSE",
    }
    assert_error(model.start_next_cycle("SERVER"), "LocalProtocolError")

    # Closing the server side while it still owes a response is likewise a
    # remote error at the client, and drives the client to MUST_CLOSE.
    model = DualEnd()
    complete_request(model)
    assert_ok(model.abort("SERVER"))
    assert_ok(model.deliver("SERVER"))
    closed_before_response = model.pump("CLIENT")
    assert_error(closed_before_response, "RemoteProtocolError")
    assert endpoint_states(closed_before_response, "CLIENT") == {
        "CLIENT": "MUST_CLOSE",
        "SERVER": "ERROR",
    }

    # Closing before any message at all is the normal end-of-stream case.
    model = DualEnd()
    assert_ok(model.abort("CLIENT"))
    assert_ok(model.deliver("CLIENT"))
    normal_close = model.pump("SERVER")
    assert [type(event) for event in normal_close.events] == [ConnectionClosed]
    assert endpoint_states(normal_close, "SERVER") == {
        "CLIENT": "CLOSED",
        "SERVER": "MUST_CLOSE",
    }


def test_reference_model_states_constraints_without_h11_state_tables() -> None:
    client = ReferenceView(CLIENT)
    client.process_event(CLIENT, request())
    assert client.states == {"CLIENT": "SEND_BODY", "SERVER": "SEND_RESPONSE"}
    client.process_event(CLIENT, EndOfMessage())
    assert client.states == {"CLIENT": "DONE", "SERVER": "SEND_RESPONSE"}
    with pytest.raises(ReferenceViolation):
        client.start_next_cycle()
    client.process_event(SERVER, response(204))
    client.process_event(SERVER, EndOfMessage())
    assert client.states == {"CLIENT": "DONE", "SERVER": "DONE"}
    client.start_next_cycle()
    assert client.states == {"CLIENT": "IDLE", "SERVER": "IDLE"}

    server = ReferenceView(SERVER)
    server.process_event(CLIENT, request())
    server.process_event(SERVER, informational(100))
    server.process_event(SERVER, informational(103))
    assert server.states["SERVER"] == "SEND_RESPONSE"
    with pytest.raises(ReferenceViolation):
        server.process_event(SERVER, informational(101))
    server.process_event(SERVER, response(204))
    with pytest.raises(ReferenceViolation):
        server.process_event(SERVER, informational(102))

    close_view = ReferenceView(CLIENT)
    close_view.process_event(
        CLIENT, request(headers=host_headers((b"connection", b"close")))
    )
    close_view.process_event(CLIENT, EndOfMessage())
    assert close_view.states == {
        "CLIENT": "MUST_CLOSE",
        "SERVER": "SEND_RESPONSE",
    }


def test_fixed_seed_sequences_are_deterministic_and_replayable() -> None:
    for seed in FIXED_SEEDS:
        first = random_sequence(seed, 8)
        second = random_sequence(seed, 8)
        assert [op.to_json() for op in first] == [
            op.to_json() for op in second
        ]
        report = run_sequence(first)
        assert report.mismatch is None, report.mismatch
        assert len(report.log.ops) == 8

        text = report.log.to_json()
        loaded = EventLog.from_json(text)
        assert [op.to_json() for op in loaded.ops] == [
            op.to_json() for op in report.log.ops
        ]
        # Replaying through the JSON form must reproduce every observation,
        # including exact wire bytes, states, terminals, and exception text.
        replay_log(loaded)


def test_early_reset_mutant_is_caught_and_failure_shrinks() -> None:
    seeded = random_sequence(MUTANT_SEED, 8)
    assert run_sequence(seeded).mismatch is None
    mutant = run_sequence(seeded, connection_factory=EarlyResetConnection)
    assert mutant.mismatch is not None
    assert "start_next_cycle" in str(mutant.mismatch)

    def fails(ops: Sequence[Op]) -> bool:
        return (
            run_sequence(
                ops, connection_factory=EarlyResetConnection
            ).mismatch
            is not None
        )

    minimal = shrink_failure(seeded, fails)
    assert len(minimal) <= 4
    assert any(op.action == "start_next_cycle" for op in minimal)
    assert run_sequence(minimal).mismatch is None

    bad = run_sequence(minimal, connection_factory=EarlyResetConnection)
    assert bad.mismatch is not None
    assert "start_next_cycle" in str(bad.mismatch)

    # The failing trace is itself a replayable JSON event log.
    text = bad.log.to_json()
    loaded = EventLog.from_json(text)
    with pytest.raises(ModelMismatch):
        replay_log(loaded, connection_factory=EarlyResetConnection)

    # The same operations are legal and model-clean on real h11.
    clean = run_sequence(loaded.ops)
    assert clean.mismatch is None
    replay_log(clean.log)
