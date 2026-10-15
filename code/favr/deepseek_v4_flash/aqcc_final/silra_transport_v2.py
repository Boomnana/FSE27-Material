"""One Silra request with bounded phases, a wall-clock deadline, and safe events.

No retries, redirects, custom proxies, or disabled TLS verification. A daemon
I/O worker lets the caller stop even during an uninterruptible DNS lookup;
cancellation prevents that worker from subsequently sending the HTTP request.
Event callbacks run serially on the calling thread and should not block.
"""
from __future__ import annotations

import http.client
import json
import math
import queue
import socket
import ssl
import threading
import time
from typing import Callable


ENDPOINT = "https://api.silra.cn/v1/chat/completions"
MODEL = "qwen3.5-flash"
ALLOWED_MODELS = frozenset(("qwen3.5-flash", "deepseek-v4-flash"))
MAX_BODY_BYTES = 4 * 1024 * 1024


class TransportError(RuntimeError):
    def __init__(self, phase: str, error_class: str, http_status: int | None):
        self.phase, self.error_class, self.http_status = phase, error_class, http_status
        super().__init__(f"transport failure: {phase}/{error_class}")


class TransportTimeoutError(TransportError, TimeoutError):
    pass


class RedirectRejected(Exception):
    pass


class IncompleteBody(Exception):
    pass


class InvalidContentLength(Exception):
    pass


class BodyTooLarge(Exception):
    pass


def _connection_factory(timeout: float):
    return http.client.HTTPSConnection(
        "api.silra.cn", port=443, timeout=timeout, context=ssl.create_default_context())


def _close(connection, retained_socket=None) -> None:
    if connection is None:
        return
    
    sock = retained_socket if retained_socket is not None else getattr(connection, "sock", None)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
    try:
        connection.close()
    except Exception:
        pass


class _State:
    def __init__(self, connect_timeout: float, read_timeout: float, total_timeout: float):
        self.started = time.monotonic()
        self.total_deadline = self.started + total_timeout
        self.limits = {"connect": connect_timeout, "send": connect_timeout,
                       "headers": read_timeout, "body": read_timeout}
        self.phase = "connect"
        self.phase_started = self.started
        self.status = None
        self.connection = None
        self.socket = None
        self.response = None
        self.cancelled = threading.Event()
        self.lock = threading.Lock()

    def remaining(self) -> float:
        with self.lock:
            deadline = min(self.total_deadline, self.phase_started + self.limits[self.phase])
        return deadline - time.monotonic()

    def check(self) -> float:
        left = self.remaining()
        if self.cancelled.is_set() or left <= 0:
            raise TimeoutError()
        return left

    def event(self, name: str, *, error_class: str | None = None) -> dict:
        now = time.monotonic()
        with self.lock:
            return {"event": name, "phase": self.phase,
                    "elapsed_seconds": max(0.0, now - self.started),
                    "phase_elapsed_seconds": max(0.0, now - self.phase_started),
                    "http_status": self.status, "error_class": error_class}

    def start_phase(self, phase: str, output: queue.Queue) -> None:
        self.check()
        with self.lock:
            self.phase = phase
            self.phase_started = time.monotonic()
        self.check()
        output.put(("event", self.event("start")))

    def socket_timeout(self) -> None:
        left = self.check()
        if self.socket is not None:
            self.socket.settimeout(left)

    def abort(self) -> None:
        self.cancelled.set()
        _close(self.connection, self.socket)


def send(wire: dict, key: str, endpoint: str, *, connect_timeout: float = 20,
         read_timeout: float = 300, total_timeout: float = 330,
         on_event: Callable[[dict], None] = lambda _: None) -> tuple[int, str]:
    """Send once; return an HTTP status and complete UTF-8 body.

    ``connect_timeout`` bounds each of connect/TLS and sending; ``read_timeout``
    bounds each complete headers/body phase (not an indefinitely reset timeout).
    ``total_timeout`` bounds the caller's entire network operation. Callbacks
    must be nonblocking: their own execution cannot be preempted by this API.
    Ordinary HTTP errors return their status/body; redirects raise immediately.
    Raised transport exceptions expose only phase, class, and observed status.
    """
    if (endpoint != ENDPOINT or type(wire) is not dict
            or type(wire.get("model")) is not str or wire["model"] not in ALLOWED_MODELS):
        raise ValueError("authorized endpoint/model required")
    if type(key) is not str or not key or any(c.isspace() for c in key):
        raise ValueError("invalid credential shape")
    for value in (connect_timeout, read_timeout, total_timeout):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("finite positive timeouts required")
    body = json.dumps(wire, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    state = _State(connect_timeout, read_timeout, total_timeout)
    output = queue.Queue()

    def finish_phase() -> None:
        state.check()
        output.put(("event", state.event("end")))

    def worker() -> None:
        result = None
        try:
            output.put(("event", state.event("start")))
            connection = _connection_factory(state.check())
            state.connection = connection
            
            
            connection.auto_open = 0
            state.check()
            connection.connect()
            state.socket = connection.sock
            state.check()
            finish_phase()

            state.start_phase("send", output)
            state.socket_timeout()
            state.check()
            connection.request("POST", "/v1/chat/completions", body=body, headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json", "Connection": "close",
            })
            finish_phase()

            state.start_phase("headers", output)
            state.socket_timeout()
            response = connection.getresponse()
            state.response = response
            state.status = response.status
            finish_phase()  
            if 300 <= response.status < 400:
                raise RedirectRejected()

            state.start_phase("body", output)
            length_header = response.getheader("Content-Length")
            expected = None
            if length_header is not None and not response.chunked:
                if not length_header.isascii() or not length_header.isdigit():
                    raise InvalidContentLength()
                expected = int(length_header)
                if expected > MAX_BODY_BYTES:
                    raise BodyTooLarge()
            chunks, size = [], 0
            while True:
                
                
                
                if (expected is not None and size >= expected) or response.isclosed():
                    break
                state.socket_timeout()
                chunk = response.read1(65536)
                state.check()
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_BODY_BYTES:
                    raise BodyTooLarge()
                chunks.append(chunk)
            if expected is not None and size != expected:
                raise IncompleteBody()
            raw_body = b"".join(chunks).decode("utf-8")
            finish_phase()
            result = (response.status, raw_body)
        except BaseException as error:
            
            output.put(("error", state.event("error", error_class=type(error).__name__)))
        finally:
            _close(state.connection, state.socket)
            if state.response is not None:
                try:
                    state.response.close()
                except Exception:
                    pass
        if result is not None:
            output.put(("result", result))

    thread = threading.Thread(target=worker, name="silra-single-request", daemon=True)
    thread.start()
    try:
        while True:
            left = state.remaining()
            if left <= 0:
                item = ("error", state.event("error", error_class="TimeoutError"))
            else:
                try:
                    item = output.get(timeout=left)
                except queue.Empty:
                    item = ("error", state.event("error", error_class="TimeoutError"))
            kind, value = item
            if kind == "event":
                on_event(value)
            elif kind == "error":
                state.abort()
                on_event(value)
                error_type = TransportTimeoutError if value["error_class"] == "TimeoutError" else TransportError
                raise error_type(value["phase"], value["error_class"], value["http_status"]) from None
            else:
                if state.cancelled.is_set() or state.remaining() <= 0:
                    error = state.event("error", error_class="TimeoutError")
                    state.abort()
                    on_event(error)
                    raise TransportTimeoutError(error["phase"], "TimeoutError", error["http_status"]) from None
                elapsed = time.monotonic() - state.started
                on_event({"event": "complete", "phase": "complete", "elapsed_seconds": elapsed,
                          "phase_elapsed_seconds": 0.0, "http_status": value[0], "error_class": None})
                return value
    finally:
        state.abort()
