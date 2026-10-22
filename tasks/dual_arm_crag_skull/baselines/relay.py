"""The billing relay between ASPIRE's coding agent (Claude Code, in a bwrap sandbox) and the Claude API.

Claude Code gets a dummy key and ANTHROPIC_BASE_URL pointing here. The relay holds the real key: it reads
ANTHROPIC_API_KEY once at start, removes it from os.environ and makes its process non-dumpable. As DexAgent's
transport does, each POST /v1/messages first reserves its worst case in the pool every live baseline run shares
(budget.py): every request-body byte an input token (plus allowances for text the API adds, images and PDFs) at the
1 h cache-write rate, plus max_tokens of output. A call whose cost its body cannot bound (content the API fetches or
runs itself, fallbacks, fast or regional inference) is refused with a 400, and one that does not fit in the pool
with a 403 permission_error. Otherwise it goes upstream with the real key, the answer streams back as it arrives,
and the reservation is settled at the usage the answer reports (SSE message_start and message_delta, or the JSON
body). A non-200 answer costs 0; a 200 that breaks off costs its worst case; a call dearer than its reservation is
charged in full and stops the relay. count_tokens is forwarded free; anything else is a 404. The ledger gets one
JSON line per request and never a body, a header or the key.

  python -m tasks.dual_arm_crag_skull.baselines.relay serve --who aspire --socket PATH --ledger PATH [--pool DIR]
      [--upstream URL]
  python -m tasks.dual_arm_crag_skull.baselines.relay forward --port 8787 --socket PATH

`serve` runs outside the sandbox, with the key in its environment, and listens on the Unix socket PATH; --pool
defaults to baselines.llm_budget.pool_dir of configs/jing.json. `forward` runs inside the sandbox's network namespace,
where that socket file is bound in, and pipes 127.0.0.1:PORT to it; it needs only this file and budget.py. Claude
Code then uses ANTHROPIC_BASE_URL=http://127.0.0.1:8787. Both stop on SIGINT or SIGTERM; a call cut off then is
settled at its worst case by the pool (a dead process's reservation).
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import socket
import socketserver
import ssl
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .budget import BudgetExhausted, BudgetPool, configured_pool_dir

DEFAULT_UPSTREAM = 'https://api.anthropic.com'
DEFAULT_MODELS = ('claude-opus-5-5',)
MESSAGES, COUNT_TOKENS = '/v1/messages', '/v1/messages/count_tokens'
REQUEST_HEADERS = ('anthropic-version', 'anthropic-beta', 'content-type', 'accept')  # all the API gets of the client's
RESPONSE_HEADERS = ('content-type', 'request-id', 'retry-after', 'retry-after-ms', 'x-should-retry')  # and anthropic-*
HOP_BY_HOP = frozenset({'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'proxy-connection',
                        'te', 'trailer', 'trailers', 'transfer-encoding', 'upgrade'})
LOOPBACK = ('127.0.0.1', 'localhost', '::1')
ENDINGS = {'upstream_broke': 'the upstream connection broke', 'client_gone': 'the client went away',
           'relaying': 'the relay failed while passing the answer on'}
OVERHEAD_TOKENS = 1_000        # input the API adds to a call itself, such as the tool-use system prompt (~350 tokens)
IMAGE_TOKENS = 6_000           # more than one image can cost however few its bytes (up to ~4,784 tokens at 2576 px)
CONTEXT_TOKENS = 1_000_000     # claude-opus-5-5's context window: the most input one call can read
MAX_BODY_BYTES = 64 * 2 ** 20  # twice the Messages API's own limit
MAX_LINE_BYTES = 16 * 2 ** 20  # an SSE line longer than this is not parsed (the call is charged its worst case)
UPSTREAM_TIMEOUT_S = 600.0     # the SDKs' default; a stream sends ping events while the model thinks
CLIENT_TIMEOUT_S = 600.0       # a client connection idle this long is closed


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _dict(value):
    return value if isinstance(value, dict) else {}


def _named(value):
    """The header names a Connection header lists, which are hop-by-hop as well."""
    return {name.strip().lower() for name in (value or '').split(',') if name.strip()}


def _object(data):
    """`data` parsed as a JSON object, or None."""
    try:
        value = json.loads(data)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _model(request):
    """The model a request names, as ledger text (None if it names none)."""
    model = _dict(request).get('model')
    return None if model is None else str(model)[:200]


def _record(path):
    """A new ledger line: the fields every line has."""
    return {'path': path, 'model': None, 'stream': None, 'status': None, 'request_id': None, 'usage': None, 'usd': 0.0,
            'reserved_usd': 0.0, 'outcome': 'error', 'detail': None}


def _unique(pairs):
    """A JSON object's pairs as a dict, refusing a repeated key: the relay and the API could each read another value."""
    keys = [key for key, _ in pairs]
    if len(set(keys)) < len(keys):
        raise ValueError('a JSON object repeats a key')
    return dict(pairs)


def _input_bound(request, body_bytes):
    """(the most input tokens the call can be billed for, None), or (None, what in it the relay cannot bound). Every
    body byte counts as a token, plus OVERHEAD_TOKENS, plus IMAGE_TOKENS per image; a PDF (its pages are read as text
    and images) raises the bound to a whole context window. Refused: what the API fetches or runs itself (an image or
    document from a url or a file, a container upload, server or Anthropic-defined tools, MCP servers, a container),
    fallback models, and fast or regional inference, which are priced higher."""
    for key, standard in (('speed', 'standard'), ('inference_geo', 'global')):
        if request.get(key) not in (None, standard):
            return None, f'{key} {request.get(key)!r}, which is priced higher'
    for key in ('fallbacks', 'mcp_servers', 'container'):
        if request.get(key):
            return None, key
    tools = request.get('tools')
    if isinstance(tools, list) and any(_dict(tool).get('type') not in (None, 'custom') for tool in tools):
        return None, 'a server or Anthropic-defined tool'
    images, pdf, pending = 0, False, [request.get('system'), request.get('messages')]
    while pending:
        item = pending.pop()
        if isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, dict):
            kind, source = item.get('type'), _dict(item.get('source')).get('type')
            if kind == 'container_upload' or (kind == 'image' and source != 'base64') or (
                    kind == 'document' and source not in ('base64', 'text', 'content')):
                return None, f'a {kind} the API fetches itself ({source or "no"} source)'
            images, pdf = images + (kind == 'image'), pdf or (kind == 'document' and source == 'base64')
            pending.extend(item.values())
    tokens = body_bytes + OVERHEAD_TOKENS + IMAGE_TOKENS * images
    return (max(tokens, CONTEXT_TOKENS) if pdf else tokens), None


def _cache_ttl(request):
    """'1h' if any cache_control in the request asks for the 1 h TTL, else '5m': the TTL usage_record gives cache
    writes the API reports without a breakdown."""
    pending = [request]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if _dict(item.get('cache_control')).get('ttl') == '1h':
                return '1h'
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return '5m'


class _Meter:
    """The usage of one answer, read as it passes: an SSE stream event by event (message_start's usage, then the
    fields each message_delta carries, which replace the earlier ones: output_tokens there is cumulative), a JSON
    answer from its whole body. SSE lines end with LF or CRLF."""

    def __init__(self, sse):
        self.sse, self.usage, self.stopped, self.error, self.malformed = sse, None, False, None, False
        self._tail, self._event, self._data, self._body = b'', None, [], []

    def feed(self, data):
        """Take the next bytes of the answer, which may end anywhere, even inside a line."""
        if not self.sse:
            self._body.append(data)
            return
        lines = (self._tail + data).split(b'\n')
        self._tail = lines.pop()
        if len(self._tail) > MAX_LINE_BYTES:
            self._tail, self.malformed = b'', True
        for line in lines:
            line = line.removesuffix(b'\r')
            if not line:
                self._dispatch()
            elif not line.startswith(b':'):  # a comment otherwise
                name, _, value = line.partition(b':')
                if name == b'event':
                    self._event = value.removeprefix(b' ').decode('utf-8', 'replace')
                elif name == b'data':
                    self._data.append(value.removeprefix(b' '))

    def _dispatch(self):
        event, data, self._event, self._data = self._event, self._data, None, []
        if not data:
            return
        payload = _object(b'\n'.join(data))
        if payload is None:
            self.malformed = True
            return
        kind = payload.get('type', event)
        if kind == 'message_start':
            usage = _dict(payload.get('message')).get('usage')
            self.usage = dict(usage) if isinstance(usage, dict) else None
        elif kind == 'message_delta' and self.usage is not None and isinstance(payload.get('usage'), dict):
            self.usage = {**self.usage, **{k: v for k, v in payload['usage'].items() if v is not None}}
        elif kind == 'message_stop':
            self.stopped = True
        elif kind == 'error':
            self.error = str(_dict(payload.get('error')).get('type') or 'error')

    def result(self):
        """(usage, None) for a whole answer, else (None, why it cannot be priced)."""
        if self.sse:
            usage = self.usage
            problem = (f'error event ({self.error})' if self.error else 'malformed event stream' if self.malformed
                       else 'the stream ended before message_stop' if not self.stopped else None)
        else:
            answer = _object(b''.join(self._body))
            usage, problem = _dict(answer).get('usage'), None if answer is not None else 'the answer is not JSON'
        if problem is None and (_dict(usage).get('input_tokens') is None or _dict(usage).get('output_tokens') is None):
            problem = 'the answer carries no usage'
        return (None, problem) if problem else (usage, None)


class _Handler(BaseHTTPRequestHandler):
    """One client connection (HTTP/1.1, kept alive); the module docstring says what each request gets."""

    protocol_version = 'HTTP/1.1'
    server_version = 'baseline-relay'
    sys_version = ''
    timeout = CLIENT_TIMEOUT_S

    def setup(self):
        self.disable_nagle_algorithm = self.server.address_family != socket.AF_UNIX  # TCP_NODELAY is TCP's only
        super().setup()

    def __getattr__(self, name):
        if name.startswith('do_'):  # http.server dispatches to do_<METHOD>: every method but POST is a 404
            return self._not_found
        raise AttributeError(name)

    def log_message(self, format, *args):  # the ledger records the requests; nothing is printed
        pass

    # ------------------------------------------------------------------ requests
    def do_POST(self):
        """/v1/messages is metered, /v1/messages/count_tokens forwarded free, any other path a 404."""
        self._started = time.monotonic()
        call = self._call()
        if call['path'] not in (MESSAGES, COUNT_TOKENS):
            return self._not_found()
        body, problem = self._body(required=True)
        if problem:
            return self._refuse(call, *problem, close=True)
        if not (self.path.isascii() and self.path.isprintable()):  # http.client could not send it on
            return self._refuse(call, 400, 'invalid_request_error', 'the request target must be printable ASCII')
        if call['path'] == COUNT_TOKENS:
            call['model'] = _model(_object(body))
            return self._forward(call, body)  # free: nothing is reserved or charged
        self._message(call, body)

    def _not_found(self):
        self._started = time.monotonic()
        call = self._call()
        _, problem = self._body(required=False)
        self._refuse(call, 404, 'not_found_error', f'{self.command} {call["path"]} is not served by this relay',
                     close=problem is not None)

    def _call(self):
        """A new ledger line for this request."""
        return _record(urllib.parse.urlsplit(self.path).path[:200])

    def _body(self, *, required):
        """(the request body, None), or (None, (status, error type, message)) when the relay will not read it."""
        length = self.headers.get('Content-Length')
        if self.headers.get('Transfer-Encoding') or (length is None and required):
            return None, (411, 'invalid_request_error', 'the request body needs a Content-Length')
        if length is None:
            return b'', None
        if not (length.isascii() and length.isdigit()):
            return None, (400, 'invalid_request_error', 'the Content-Length is not a number')
        if int(length) > MAX_BODY_BYTES:
            return None, (413, 'request_too_large', f'the request body is over {MAX_BODY_BYTES} bytes')
        body = self.rfile.read(int(length))
        if len(body) < int(length):
            return None, (400, 'invalid_request_error', 'the request body ended early')
        return body, None

    def _message(self, call, body):
        """POST /v1/messages: check it, reserve its worst case, forward it, settle."""
        try:
            request = json.loads(body, object_pairs_hook=_unique)
        except (ValueError, RecursionError):
            request = None
        if not isinstance(request, dict):
            return self._refuse(call, 400, 'invalid_request_error',
                                'the request body must be a JSON object that repeats no key')
        model, max_tokens = request.get('model'), request.get('max_tokens')
        call.update(model=_model(request), stream=request.get('stream') is True)
        price = self.server.prices.get(model) if isinstance(model, str) else None
        if price is None:
            return self._refuse(call, 400, 'invalid_request_error',
                                f'model: this relay serves {", ".join(self.server.prices)} only')
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 0 <= max_tokens < 2 ** 31:
            return self._refuse(call, 400, 'invalid_request_error', 'max_tokens: a whole number of tokens is required')
        tokens, unbounded = _input_bound(request, len(body))
        if unbounded:  # its cost could exceed any reservation, and calls in flight would overdraw the pool
            return self._refuse(call, 400, 'invalid_request_error', f'this relay cannot bound the cost of {unbounded}; '
                                'send the content in the request instead')
        worst, ttl = price.worst_case(tokens, max_tokens, self.server.dearest_ttl), _cache_ttl(request)
        call.update(request_bytes=len(body), input_tokens_bound=tokens, max_tokens=max_tokens, reserved_usd=worst)
        if self.server.halted:
            return self._refuse(call, 403, 'permission_error', self.server.halted, outcome='refused')
        try:
            what = f'{model}: {tokens} input tokens at most, max_tokens {max_tokens}'
            reservation = self.server.pool.reserve(worst, who=self.server.who, what=what)
        except BudgetExhausted as exc:
            return self._refuse(call, 403, 'permission_error', f'The shared model budget is exhausted: {exc}. The '
                                'call was not made; stop, since a retry cannot fit either.', outcome='refused')
        self._forward(call, body, (reservation, worst, ttl, price))

    # ------------------------------------------------------------------ answers
    def _refuse(self, call, status, kind, message, *, outcome='error', close=False):
        """Answer with an error of the relay's own (nothing was forwarded) and log it."""
        call.update(status=status, outcome=outcome, detail=message)
        self.server.log(call, self._started)
        self._send_error(status, kind, message, close=close)

    def _send_error(self, status, kind, message, *, close=False):
        """An Anthropic-style error body; below 500 with x-should-retry: false, which the Anthropic SDKs honour."""
        data = json.dumps({'type': 'error', 'error': {'type': kind, 'message': message}}).encode()
        try:
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            if status < 500:
                self.send_header('x-should-retry', 'false')
            if close:
                self.send_header('Connection', 'close')
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(data)
        except OSError:
            self.close_connection = True

    def _write(self, data):
        try:
            self.wfile.write(data)
            return True
        except OSError:
            self.close_connection = True
            return False

    # ------------------------------------------------------------------ forwarding
    def _upstream_headers(self):
        """The client's anthropic-version, anthropic-beta, content-type and accept, and the real key: never the
        client's own key or authorization, nor a hop-by-hop header."""
        hop = HOP_BY_HOP | _named(self.headers.get('Connection'))
        sent = {name: ', '.join(self.headers.get_all(name)) for name in REQUEST_HEADERS
                if name not in hop and self.headers.get_all(name)}
        return {**sent, 'x-api-key': self.server.key}

    def _forward(self, call, body, bill=None):
        """Send the request upstream and stream the answer back. `bill` (reservation, worst case, cache TTL, Price) is
        settled, and the call logged, before the answer's last bytes go out: a client that has read a whole answer
        finds it settled."""
        ended, meter, connection = 'unsent', None, None
        try:
            connection = self.server.connect()
            connection.request('POST', self.server.target(self.path), body=body, headers=self._upstream_headers())
            ended = 'unanswered'
            response = connection.getresponse()
            call.update(status=response.status, request_id=response.getheader('request-id'))
            meter = _Meter(sse=(response.getheader('Content-Type') or '').startswith('text/event-stream'))
            ended = 'relaying'  # stays so only if _relay fails unexpectedly: priced as a broken answer
            ended = self._relay(response, meter)
        except (OSError, ValueError, http.client.HTTPException) as exc:  # ValueError: a target http.client refuses
            call['detail'] = self.server.scrub(f'{type(exc).__name__}: {exc}')
        finally:
            try:
                self._settle(call, ended, meter, bill)
            finally:
                if connection is not None:
                    connection.close()
        if ended == 'complete':
            self._write(b'0\r\n\r\n')
        elif ended in ('unsent', 'unanswered'):
            self._send_error(502, 'api_error', f'the relay got no answer from the API ({call["detail"]})')
        else:  # a cut answer stays cut: the client sees the connection end before the last chunk
            self.close_connection = True

    def _relay(self, response, meter):
        """Pass the answer on in HTTP/1.1 chunks as it arrives, one per read, all but the last; returns 'complete',
        'upstream_broke' or 'client_gone'."""
        hop = HOP_BY_HOP | _named(response.getheader('Connection'))
        try:
            self.send_response(response.status, response.reason)
            for name, value in response.getheaders():
                lower = name.lower()
                if lower not in hop and (lower in RESPONSE_HEADERS or lower.startswith('anthropic-')):
                    self.send_header(name, value)
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
        except OSError:
            return 'client_gone'
        while True:
            try:
                piece = response.read1(65536)
            except (OSError, http.client.HTTPException):
                return 'upstream_broke'
            if not piece:  # the end, or a body shorter than its Content-Length
                return 'upstream_broke' if response.length else 'complete'
            meter.feed(piece)
            if not self._write(b'%x\r\n%b\r\n' % (len(piece), piece)):
                return 'client_gone'

    def _settle(self, call, ended, meter, bill):
        """Price the call by how it ended, settle its reservation and log it."""
        try:
            usd, outcome = self._price(call, ended, meter, bill)
        except Exception as exc:  # noqa: BLE001 - an answer that cannot be priced is charged its worst case
            usd, outcome = (bill[1] if bill else 0.0), 'broken_stream'
            call['detail'] = self.server.scrub(f'unpriced: {type(exc).__name__}: {exc}')
        call.update(usd=usd, outcome=outcome)
        if bill is not None:
            try:
                details = {'request_id': call['request_id'], 'model': call['model']}
                exceeded = self.server.pool.settle(bill[0], usd, outcome=outcome, details=details)['exceeded']
                call['exceeded_reservation'] = exceeded
                if exceeded:
                    self.server.halted = (f'This relay has stopped: a call cost ${usd:.4f}, more than its reserved '
                                          f'worst case ${bill[1]:.4f}, so the bound is wrong and no call is made now.')
            except Exception as exc:  # noqa: BLE001 - the reservation stays open (the pool charges its worst case
                # once this process ends), and a relay that cannot account for its calls makes no more
                call['detail'] = self.server.scrub(f'settling failed: {type(exc).__name__}: {exc}')
                self.server.halted = 'This relay has stopped: it could not settle a call in the budget pool.'
        self.server.log(call, self._started)

    def _price(self, call, ended, meter, bill):
        """(USD, outcome) of a forwarded call; see the module docstring."""
        from .dexagent.transport import usage_record
        worst = bill[1] if bill else 0.0
        if ended in ('unsent', 'unanswered'):
            call['status'] = 502
            return (0.0 if ended == 'unsent' else worst), 'error'  # a request that went out may have been billed
        if call['status'] != 200:
            return 0.0, 'error'
        if ended != 'complete':
            usage, problem = None, ENDINGS[ended]
        elif bill is None:
            return 0.0, 'call'
        else:
            usage, problem = meter.result()
        if problem:
            call['detail'] = problem
            return worst, 'broken_stream'
        call['usage'] = usage_record(usage, bill[2])
        usd = bill[3].cost(call['usage'])
        if not 0 <= usd < float('inf'):  # the pool would refuse it and leave the reservation open
            raise ValueError(f'the usage prices at {usd!r} USD')
        return usd, 'call'


class _Running:
    """What serve() and forward() return: a server already serving in a daemon thread. `address` is where it listens,
    `wait()` blocks until it is closed, `close()` stops it without waiting for calls in flight."""

    def start(self):
        self._thread = threading.Thread(target=self.serve_forever, kwargs={'poll_interval': 0.1}, daemon=True)
        self._thread.start()
        return self

    @property
    def address(self):
        return self.server_address

    def wait(self):
        self._thread.join()

    def close(self):
        self.shutdown()
        self.server_close()
        if self.address_family == socket.AF_UNIX:
            Path(self.server_address).unlink(missing_ok=True)
        self._thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


class _RelayServer(_Running, ThreadingHTTPServer):
    """The relay on host:port, which every local user can reach (tests); live runs use _UnixRelayServer."""

    block_on_close = False

    def server_bind(self):  # HTTPServer.server_bind would also look the address's host name up
        socketserver.TCPServer.server_bind(self)

    def handle_error(self, request, client_address):
        """A request the relay itself failed on is logged (scrubbed), not printed; a client hanging up is neither."""
        error = sys.exc_info()[1]
        if not isinstance(error, ConnectionError):
            self.log({**_record(None), 'detail': self.scrub(f'relay failure: {type(error).__name__}: {error}')})

    def connect(self):
        """A new connection to the API (one per call)."""
        upstream = self.upstream
        if upstream.scheme == 'https':
            return http.client.HTTPSConnection(upstream.hostname, upstream.port, timeout=UPSTREAM_TIMEOUT_S,
                                               context=self.tls)
        return http.client.HTTPConnection(upstream.hostname, upstream.port, timeout=UPSTREAM_TIMEOUT_S)

    def target(self, path):
        """The upstream request target of a client's: the same path and query string, under the upstream's path."""
        split = urllib.parse.urlsplit(path)
        return self.upstream.path.rstrip('/') + split.path + (f'?{split.query}' if split.query else '')

    def scrub(self, text, limit=500):
        """`text` (exception text, say) with the key cut out, at most `limit` characters."""
        return str(text).replace(self.key, '[key]')[:limit]

    def log(self, call, started=None):
        """Append one line to the ledger: never a body, a header or the key."""
        seconds = None if started is None else round(time.monotonic() - started, 3)
        line = json.dumps({'time_utc': _now(), **call, 'seconds': seconds}, allow_nan=False)
        with self.ledger_lock, open(self.ledger_path, 'a') as handle:
            handle.write(line.replace(self.key, '[key]') + '\n')


class _UnixRelayServer(_RelayServer):
    """The relay on a Unix socket (mode 0600): only this user's processes, and a sandbox it is bound into, reach it."""

    address_family = socket.AF_UNIX


def _upstream(url):
    """The API's URL, parsed: https, or plain http to this host only (tests), since the key travels in a header."""
    target = urllib.parse.urlsplit(url)
    if target.scheme not in ('http', 'https') or not target.hostname or target.port == 0 or (
            target.scheme == 'http' and target.hostname not in LOOPBACK):
        raise ValueError(f'upstream must be an https URL (plain http to this host only), not {url!r}')
    return target


def serve(pool_dir, who, *, socket_path=None, host='127.0.0.1', port=0, ledger_path, upstream=DEFAULT_UPSTREAM,
          models=DEFAULT_MODELS):
    """Start the relay and return it serving in a daemon thread (`address`, `wait()`, `close()`): on the Unix socket
    `socket_path`, which must not exist yet (made mode 0600), else on host:port. It makes this process non-dumpable,
    takes the key out of ANTHROPIC_API_KEY and refuses to start without one; every model it serves needs a price."""
    from .dexagent.dexenv import protect_process  # imported here: forward() needs only the standard library
    from .dexagent.transport import CACHE_WRITE, Price
    protect_process()
    key = os.environ.pop('ANTHROPIC_API_KEY', '').strip()
    if not key:
        raise RuntimeError('ANTHROPIC_API_KEY is not set: the relay holds the real key and makes no call without it')
    if not all('!' <= character <= '~' for character in key):
        raise RuntimeError('ANTHROPIC_API_KEY holds characters an HTTP header cannot carry')
    if not isinstance(who, str) or not who:
        raise ValueError('who must name the run that spends from the pool')
    target, prices = _upstream(upstream), {model: Price.for_model(model) for model in models}
    if not prices:
        raise ValueError('models must name at least one model')
    pool, ledger = BudgetPool(pool_dir), Path(ledger_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    tls = ssl.create_default_context()
    tls.keylog_filename = None  # SSLKEYLOGFILE would write TLS secrets, and with them the key, to a file
    if socket_path is None:
        server = _RelayServer((host, int(port)), _Handler)
    else:
        if os.path.lexists(socket_path):
            raise FileExistsError(f'{socket_path} exists; the relay makes its socket itself')
        server = _UnixRelayServer(str(socket_path), _Handler, bind_and_activate=False)
        try:  # 0600 before it listens: nobody else can connect while its mode is wider
            server.server_bind()
            os.chmod(socket_path, 0o600)
            server.server_activate()
        except BaseException:
            server.server_close()
            Path(socket_path).unlink(missing_ok=True)
            raise
    server.key, server.pool, server.who, server.prices, server.upstream = key, pool, who, prices, target
    server.tls, server.dearest_ttl = tls, max(CACHE_WRITE, key=CACHE_WRITE.get)
    server.ledger_path, server.ledger_lock, server.halted = ledger, threading.Lock(), None
    return server.start()


# ---------------------------------------------------------------------- the forwarder, inside the sandbox
def _copy(source, sink):
    """Copy bytes from source to sink until source ends, then end sink's sending side; on an error end both."""
    try:
        while data := source.recv(65536):
            sink.sendall(data)
        sink.shutdown(socket.SHUT_WR)
    except OSError:
        for end in (source, sink):
            try:
                end.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class _Pipe(socketserver.BaseRequestHandler):
    """One connection to the forwarder, piped both ways to the relay's socket."""

    def handle(self):
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as relay:
            try:
                relay.connect(self.server.socket_path)
            except OSError:
                return
            back = threading.Thread(target=_copy, args=(relay, self.request), daemon=True)
            back.start()
            _copy(self.request, relay)
            back.join()


class _Forwarder(_Running, socketserver.ThreadingTCPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True

    def handle_error(self, request, client_address):  # a broken connection just ends; nothing is printed
        pass


def forward(port, socket_path):
    """Listen on 127.0.0.1:`port` (inside the sandbox's network namespace) and pipe each connection, both ways, to the
    relay's Unix socket; returns the forwarder serving in a daemon thread (`address`, `wait()`, `close()`)."""
    if not Path(socket_path).exists():
        raise FileNotFoundError(f'no relay socket at {socket_path}')
    server = _Forwarder(('127.0.0.1', int(port)), _Pipe)
    server.socket_path = str(socket_path)
    return server.start()


# ---------------------------------------------------------------------- CLI
def _configured_pool_dir():
    from ..__main__ import DEFAULT_CONFIG
    return configured_pool_dir(json.loads(DEFAULT_CONFIG.read_text()))


def main(argv=None):
    """The CLI (module docstring): prints one JSON line once listening, then serves until SIGINT or SIGTERM."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    serve_parser = commands.add_parser('serve', help='the relay, outside the sandbox (the real key in its environment)')
    serve_parser.add_argument('--who', required=True, help='who spends from the pool (its by_who entry)')
    serve_parser.add_argument('--socket', type=Path, required=True, help='the Unix socket to make and listen on')
    serve_parser.add_argument('--ledger', type=Path, required=True, help='the JSONL ledger to append to')
    serve_parser.add_argument('--pool', type=Path, help='the pool directory (default: baselines.llm_budget.pool_dir)')
    serve_parser.add_argument('--upstream', default=DEFAULT_UPSTREAM, help='the API (a local fake in tests)')
    forward_parser = commands.add_parser('forward', help='pipe 127.0.0.1:PORT to the relay socket, inside the sandbox')
    forward_parser.add_argument('--port', type=int, required=True)
    forward_parser.add_argument('--socket', type=Path, required=True)
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # SIGTERM stops it as SIGINT does
    if args.command == 'serve':
        server = serve(args.pool or _configured_pool_dir(), args.who, socket_path=args.socket,
                       ledger_path=args.ledger, upstream=args.upstream)
        started = {'relay': str(args.socket), 'who': args.who, 'pool': str(server.pool.dir), 'ledger': str(args.ledger),
                   'upstream': args.upstream, 'models': list(server.prices),
                   'pool_left_usd': server.pool.status()['left_usd']}
    else:
        server = forward(args.port, args.socket)
        started = {'forward': f'127.0.0.1:{server.address[1]}', 'port': server.address[1], 'socket': str(args.socket)}
    try:
        print(json.dumps(started), flush=True)
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
