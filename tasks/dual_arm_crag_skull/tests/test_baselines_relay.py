import http.client
import json
import os
import select
import shutil
import signal
import socket
import ssl
import stat
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tasks.dual_arm_crag_skull.baselines import relay
from tasks.dual_arm_crag_skull.baselines.budget import BudgetPool
from tasks.dual_arm_crag_skull.baselines.dexagent import dexenv

MODEL = 'claude-opus-5-5'
REAL_KEY = f'sk-ant-relay-test-{uuid.uuid4().hex}'
DUMMY_KEY = 'sk-ant-dummy-key-of-the-sandbox'
MARKER = 'body-marker-7d1f'  # in every request body, never in a ledger
SOURCE_ROOT = Path(relay.__file__).parents[3]  # the directory holding the `tasks` package these tests import
CLIENT_HEADERS = {'Content-Type': 'application/json', 'Accept': 'application/json', 'anthropic-version': '2023-06-01',
                  'anthropic-beta': 'claude-code-20250219,interleaved-thinking-2025-05-14', 'x-api-key': DUMMY_KEY,
                  'Authorization': f'Bearer {DUMMY_KEY}', 'Proxy-Authorization': f'Basic {DUMMY_KEY}',
                  'Connection': 'keep-alive, x-hop', 'Keep-Alive': 'timeout=5', 'x-hop': '1', 'TE': 'trailers',
                  'Upgrade': 'h2c', 'x-stainless-lang': 'js', 'User-Agent': 'claude-cli/test'}


def event(kind, **fields):
    return f'event: {kind}\ndata: {json.dumps({"type": kind, **fields})}\n\n'.encode()


def chunk(data):
    return b'%x\r\n%b\r\n' % (len(data), data)


START_USAGE = {'input_tokens': 1000, 'cache_creation_input_tokens': 2000, 'cache_read_input_tokens': 10000,
               'cache_creation': {'ephemeral_5m_input_tokens': 500, 'ephemeral_1h_input_tokens': 1500},
               'output_tokens': 1}
# message_delta carries the cumulative output and, as with server tools, an updated input count
EVENTS = [event('message_start', message={'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'model': MODEL,
                                          'content': [], 'stop_reason': None, 'usage': START_USAGE}),
          event('content_block_start', index=0, content_block={'type': 'text', 'text': ''}),
          event('ping'),
          event('content_block_delta', index=0, delta={'type': 'text_delta', 'text': 'Grasp the cranium last.'}),
          event('content_block_stop', index=0),
          event('message_delta', delta={'stop_reason': 'end_turn', 'stop_sequence': None},
                usage={'input_tokens': 1200, 'output_tokens': 300}),
          event('message_stop')]
# USD per MTok: 4 in and 20 out; cache writes 1.25x (5 min) and 2x (1 h) the input price, reads 0.05x
STREAM_USD = (1200 * 4 + 500 * 4 * 1.25 + 1500 * 4 * 2 + 10000 * 4 * 0.05 + 300 * 20) / 1e6


class FakeUpstream:
    """The API on localhost: keeps what reaches it (path, headers, body) and answers with `reply(handler)`."""

    def __init__(self):
        self.seen, self.reply = [], None
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                upstream.seen.append({'path': self.path, 'body': body,
                                      'headers': [(name.lower(), value) for name, value in self.headers.items()]})
                upstream.reply(self)

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.block_on_close = False
        threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True).start()
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}'

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def stream_reply(events, *, cut_after=None, gate=None):
    """Answer 200 with these SSE events, a chunk each. With `cut_after` the connection drops after that many events
    (no last chunk); with `gate` the events after the first wait for it to open (the waits' results are kept)."""
    def reply(handler):
        handler.send_response(200)
        for name, value in (('Content-Type', 'text/event-stream; charset=utf-8'), ('request-id', 'req_stream'),
                            ('anthropic-ratelimit-tokens-remaining', '12345'), ('x-envoy-upstream-service-time', '7'),
                            ('Transfer-Encoding', 'chunked')):
            handler.send_header(name, value)
        handler.end_headers()
        for index, data in enumerate(events):
            if index == cut_after:
                handler.close_connection = True
                return
            if gate is not None and index == 1:
                reply.waits.append(gate.wait(10))
            handler.wfile.write(chunk(data))
        handler.wfile.write(b'0\r\n\r\n')
    reply.waits = []
    return reply


def json_reply(status, value, headers=()):
    def reply(handler):
        data = json.dumps(value).encode()
        handler.send_response(status)
        for name, header in (('Content-Type', 'application/json'), ('Content-Length', str(len(data))),
                             ('request-id', 'req_json'), *headers):
            handler.send_header(name, header)
        handler.end_headers()
        handler.wfile.write(data)
    return reply


def message(stream=True, **fields):
    """A Messages request body like Claude Code's (MARKER in its text)."""
    return json.dumps({'model': MODEL, 'max_tokens': 4096, 'stream': stream,
                       'system': [{'type': 'text', 'text': 'You write robot programs.'}],
                       'messages': [{'role': 'user', 'content': f'{MARKER}: put the cranium on last'}],
                       **fields}).encode()


def worst_usd(body, max_tokens=4096, images=0, rate=8):
    """Every request byte an input token, plus 1,000 the API adds and 6,000 per image, at the 1 h cache-write rate
    ($8 per MTok); max_tokens out at $20."""
    return ((len(body) + 1000 + 6000 * images) * rate + max_tokens * 20) / 1e6


def post(address, body, *, path='/v1/messages?beta=true', method='POST', headers=CLIENT_HEADERS):
    """(status, headers, body, whole) of one request on a new connection; `whole` is False for a cut answer."""
    connection = http.client.HTTPConnection(*address, timeout=30)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        try:
            data, whole = response.read(), True
        except http.client.IncompleteRead as cut:
            data, whole = cut.partial, False
        return response.status, {name.lower(): value for name, value in response.getheaders()}, data, whole
    finally:
        connection.close()


def read_exactly(response, size):
    data = b''
    while len(data) < size:
        piece = response.read1(size - len(data))
        assert piece, 'the answer ended early'
        data += piece
    return data


def ledger(tmp_path):
    path = tmp_path / 'ledger.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def pool(tmp_path):
    return BudgetPool(tmp_path / 'pool').status()


def pool_events(tmp_path):
    return [json.loads(line)['event'] for line in (tmp_path / 'pool' / 'pool.jsonl').read_text().splitlines()]


def eventually(check, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not check():
        assert time.monotonic() < deadline, 'timed out'
        time.sleep(0.02)


@pytest.fixture
def upstream():
    fake = FakeUpstream()
    yield fake
    fake.close()


@pytest.fixture
def protected(monkeypatch):
    """protect_process's calls: the real one would leave the test process non-dumpable (the CLI test runs it)."""
    calls = []
    monkeypatch.setattr(dexenv, 'protect_process', lambda: calls.append(True))
    return calls


@pytest.fixture
def start(tmp_path, monkeypatch, protected):
    """start(url, total=100.0, **serve options): a relay on 127.0.0.1 spending from a new pool in tmp_path/pool."""
    servers = []

    def make(url, total=100.0, **options):
        BudgetPool.create(tmp_path / 'pool', total, note='test')
        monkeypatch.setenv('ANTHROPIC_API_KEY', REAL_KEY)
        servers.append(relay.serve(tmp_path / 'pool', 'aspire:test', ledger_path=tmp_path / 'ledger.jsonl',
                                   upstream=url, **options))
        return servers[-1]
    yield make
    for server in servers:
        server.close()


def test_a_streamed_message_reaches_the_client_unchanged_and_is_charged_its_measured_cost(start, upstream, tmp_path):
    upstream.reply = stream_reply(EVENTS)
    server = start(upstream.url)
    body = message()
    status, headers, data, whole = post(server.address, body)
    assert status == 200 and whole and data == b''.join(EVENTS)
    assert headers['transfer-encoding'] == 'chunked' and headers['content-type'] == 'text/event-stream; charset=utf-8'
    assert headers['request-id'] == 'req_stream' and headers['anthropic-ratelimit-tokens-remaining'] == '12345'
    assert 'x-envoy-upstream-service-time' not in headers
    held = pool(tmp_path)
    assert held['spent_usd'] == pytest.approx(STREAM_USD, rel=1e-9) and held['calls'] == 1 and held['open'] == 0
    assert held['by_who'] == {'aspire:test': pytest.approx(STREAM_USD, rel=1e-9)}
    [line] = ledger(tmp_path)
    assert line['usage'] == {'input_tokens': 1200, 'output_tokens': 300, 'cache_write_5m_tokens': 500,
                             'cache_write_1h_tokens': 1500, 'cache_read_tokens': 10000, 'service_tier': None,
                             'inference_geo': None}
    assert (line['outcome'], line['status'], line['stream'], line['model'], line['path'], line['request_id']) == (
        'call', 200, True, MODEL, '/v1/messages', 'req_stream')
    assert line['usd'] == pytest.approx(STREAM_USD, rel=1e-9) and line['reserved_usd'] == pytest.approx(worst_usd(body))
    assert line['seconds'] >= 0 and line['time_utc'].endswith('+00:00')


def test_events_split_anywhere_with_crlf_lines_and_comments_are_metered_the_same(start, upstream, tmp_path):
    stream = (b': a comment line\n\n' + b''.join(EVENTS)).replace(b'\n', b'\r\n')
    upstream.reply = stream_reply([stream[i:i + 7] for i in range(0, len(stream), 7)])  # lines cut across chunks
    server = start(upstream.url)
    status, _, data, whole = post(server.address, message())
    assert status == 200 and whole and data == stream and ledger(tmp_path)[0]['outcome'] == 'call'
    assert pool(tmp_path)['spent_usd'] == pytest.approx(STREAM_USD, rel=1e-9)


def test_an_answer_reaches_the_client_as_it_arrives(start, upstream, tmp_path):
    gate = threading.Event()
    upstream.reply = stream_reply(EVENTS, gate=gate)  # the second event waits until the client holds the first
    server = start(upstream.url)
    connection = http.client.HTTPConnection(*server.address, timeout=30)
    try:
        connection.request('POST', '/v1/messages?beta=true', body=message(), headers=CLIENT_HEADERS)
        response = connection.getresponse()
        first = read_exactly(response, len(EVENTS[0]))
        gate.set()
        rest = response.read()
    finally:
        connection.close()
    assert first == EVENTS[0] and first + rest == b''.join(EVENTS)
    assert upstream.reply.waits == [True]  # the gate was opened by the client, not by its timeout


def test_the_upstream_gets_the_real_key_and_only_the_headers_the_api_needs(start, upstream, tmp_path):
    upstream.reply = stream_reply(EVENTS)
    server = start(upstream.url)
    body = message()
    post(server.address, body)
    post(server.address, body, headers={**CLIENT_HEADERS, 'Connection': 'keep-alive, accept'})  # accept: hop-by-hop
    first, second = (dict(seen['headers']) for seen in upstream.seen)
    assert upstream.seen[0]['path'] == '/v1/messages?beta=true' and upstream.seen[0]['body'] == body
    assert sorted(first) == ['accept', 'accept-encoding', 'anthropic-beta', 'anthropic-version', 'content-length',
                             'content-type', 'host', 'x-api-key']
    assert first['x-api-key'] == REAL_KEY and first['anthropic-beta'] == CLIENT_HEADERS['anthropic-beta']
    assert first['anthropic-version'] == '2023-06-01' and first['content-type'] == 'application/json'
    assert 'accept' not in second and second['x-api-key'] == REAL_KEY
    assert not any(DUMMY_KEY in value for seen in upstream.seen for _, value in seen['headers'])


def test_a_non_stream_message_is_charged_from_its_json_usage_at_the_requested_cache_ttl(start, upstream, tmp_path):
    usage = {'input_tokens': 50, 'output_tokens': 70, 'cache_creation_input_tokens': 1000, 'cache_read_input_tokens': 0}
    answer = {'id': 'msg_2', 'type': 'message', 'role': 'assistant', 'model': MODEL, 'stop_reason': 'end_turn',
              'content': [{'type': 'text', 'text': 'ok'}], 'usage': usage}  # no breakdown of the writes by TTL
    upstream.reply = json_reply(200, answer)
    server = start(upstream.url)
    system = [{'type': 'text', 'text': 'rules', 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}}]
    status, headers, data, whole = post(server.address, message(stream=False, system=system))
    assert status == 200 and whole and data == json.dumps(answer).encode()
    assert headers['content-type'] == 'application/json' and headers['transfer-encoding'] == 'chunked'
    expected = (50 * 4 + 1000 * 4 * 2 + 70 * 20) / 1e6  # the writes at the 1 h rate the request asked for
    assert pool(tmp_path)['spent_usd'] == pytest.approx(expected, rel=1e-9)
    [line] = ledger(tmp_path)
    assert line['outcome'] == 'call' and line['stream'] is False and line['usage']['cache_write_1h_tokens'] == 1000


@pytest.mark.parametrize('usage', [None, {'input_tokens': -5_000_000, 'output_tokens': 10}],
                         ids=['no_usage', 'negative_usage'])
def test_an_answer_that_cannot_be_priced_is_charged_its_worst_case(start, upstream, tmp_path, usage):
    answer = {'id': 'msg_4', 'type': 'message', 'role': 'assistant', 'model': MODEL, 'content': [],
              'stop_reason': 'end_turn', **({'usage': usage} if usage else {})}
    upstream.reply = json_reply(200, answer)
    server = start(upstream.url)
    body = message(stream=False)
    assert post(server.address, body)[0] == 200
    held = pool(tmp_path)
    assert held['spent_usd'] == pytest.approx(worst_usd(body)) and held['open'] == 0 and held['calls'] == 0
    [line] = ledger(tmp_path)
    assert line['outcome'] == 'broken_stream' and line['usd'] == pytest.approx(worst_usd(body))


def test_an_overloaded_answer_passes_through_unchanged_and_costs_nothing(start, upstream, tmp_path):
    overloaded = {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}
    upstream.reply = json_reply(529, overloaded, headers=[('retry-after', '3'), ('x-should-retry', 'true')])
    server = start(upstream.url)
    status, headers, data, whole = post(server.address, message())
    assert status == 529 and whole and data == json.dumps(overloaded).encode()
    assert headers['retry-after'] == '3' and headers['x-should-retry'] == 'true' and headers['request-id'] == 'req_json'
    held = pool(tmp_path)
    assert held['spent_usd'] == 0.0 and held['open'] == 0 and held['calls'] == 0
    assert pool_events(tmp_path) == ['created', 'reserve', 'settle']
    [line] = ledger(tmp_path)
    assert (line['outcome'], line['status'], line['usd']) == ('error', 529, 0.0)


def test_a_stream_cut_midway_is_charged_its_worst_case(start, upstream, tmp_path):
    upstream.reply = stream_reply(EVENTS, cut_after=3)
    server = start(upstream.url)
    body = message()
    status, _, data, whole = post(server.address, body)
    assert status == 200 and not whole and data == b''.join(EVENTS[:3])
    assert pool(tmp_path)['spent_usd'] == pytest.approx(worst_usd(body), rel=1e-12)
    [line] = ledger(tmp_path)
    assert (line['outcome'], line['usage']) == ('broken_stream', None) and line['usd'] == pytest.approx(worst_usd(body))


def test_an_error_event_is_passed_on_and_charged_the_worst_case(start, upstream, tmp_path):
    events = EVENTS[:4] + [event('error', error={'type': 'overloaded_error', 'message': 'Overloaded'})] + EVENTS[4:]
    upstream.reply = stream_reply(events)
    server = start(upstream.url)
    body = message()
    status, _, data, whole = post(server.address, body)
    assert status == 200 and whole and data == b''.join(events)
    assert pool(tmp_path)['spent_usd'] == pytest.approx(worst_usd(body), rel=1e-12)
    [line] = ledger(tmp_path)
    assert line['outcome'] == 'broken_stream' and 'overloaded_error' in line['detail']


def test_a_client_that_goes_away_midstream_is_charged_the_worst_case(start, upstream, tmp_path):
    gone = threading.Event()

    def reply(handler):
        handler.send_response(200)
        handler.send_header('Content-Type', 'text/event-stream')
        handler.send_header('Transfer-Encoding', 'chunked')
        handler.end_headers()
        handler.wfile.write(chunk(EVENTS[0]))
        gone.wait(10)
        for _ in range(300):  # the model goes on until the relay hangs up
            try:
                handler.wfile.write(chunk(event('ping')))
            except OSError:
                return
            time.sleep(0.01)
        handler.wfile.write(b'0\r\n\r\n')
    upstream.reply = reply
    server = start(upstream.url)
    body = message()
    connection = http.client.HTTPConnection(*server.address, timeout=30)
    connection.request('POST', '/v1/messages', body=body, headers=CLIENT_HEADERS)
    response = connection.getresponse()
    assert response.status == 200 and read_exactly(response, len(EVENTS[0])) == EVENTS[0]
    response.close()
    connection.close()
    gone.set()
    eventually(lambda: ledger(tmp_path))
    [line] = ledger(tmp_path)
    assert line['outcome'] == 'broken_stream' and 'client' in line['detail']
    assert line['usd'] == pytest.approx(worst_usd(body))
    assert pool(tmp_path)['spent_usd'] == pytest.approx(worst_usd(body)) and pool(tmp_path)['open'] == 0


def test_a_call_whose_worst_case_does_not_fit_is_refused_and_never_forwarded(start, upstream, tmp_path):
    upstream.reply = stream_reply(EVENTS)
    body = message()
    at_5m = worst_usd(body, rate=5)
    server = start(upstream.url, total=(at_5m + worst_usd(body)) / 2)  # fits at the 5 min write rate, not at 1 h
    status, headers, data, whole = post(server.address, body)
    error = json.loads(data)
    assert status == 403 and error['type'] == 'error' and error['error']['type'] == 'permission_error'
    assert 'shared model budget is exhausted' in error['error']['message'] and headers['x-should-retry'] == 'false'
    assert upstream.seen == [] and pool_events(tmp_path) == ['created', 'refused']
    [line] = ledger(tmp_path)
    assert (line['outcome'], line['status'], line['usd']) == ('refused', 403, 0.0)


@pytest.mark.parametrize('change, words', [({'model': 'claude-haiku-5-5'}, 'model'),
                                           ({'max_tokens': None}, 'max_tokens'),
                                           ({'max_tokens': '4096'}, 'max_tokens')],
                         ids=['other_model', 'no_max_tokens', 'text_max_tokens'])
def test_a_model_the_relay_does_not_serve_or_no_max_tokens_is_a_400(start, upstream, tmp_path, change, words):
    request = {key: value for key, value in {**json.loads(message()), **change}.items() if value is not None}
    server = start(upstream.url)
    status, _, data, _ = post(server.address, json.dumps(request).encode())
    error = json.loads(data)
    assert status == 400 and set(error) == {'type', 'error'} and error['type'] == 'error'
    assert error['error']['type'] == 'invalid_request_error' and words in error['error']['message']
    assert upstream.seen == [] and pool_events(tmp_path) == ['created']
    [line] = ledger(tmp_path)
    assert (line['outcome'], line['status'], line['usd']) == ('error', 400, 0.0)


PIXEL = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
UNBOUNDED = {  # what the API may bill beyond the request body
    'url_document': {'messages': [{'role': 'user', 'content': [
        {'type': 'document', 'source': {'type': 'url', 'url': 'https://example.com/a.pdf'}},
        {'type': 'text', 'text': 'x'}]}]},
    'file_image_in_tool_result': {'messages': [
        {'role': 'user', 'content': 'go'},
        {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 't1', 'name': 'Read', 'input': {}}]},
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 't1', 'content': [
            {'type': 'image', 'source': {'type': 'file', 'file_id': 'file_01'}}]}]}]},
    'server_tool': {'tools': [{'type': 'web_search_20260209', 'name': 'web_search'}]},
    'mcp_servers': {'mcp_servers': [{'type': 'url', 'url': 'https://mcp.example.com', 'name': 'remote'}]},
    'container_upload': {'messages': [{'role': 'user', 'content': [{'type': 'container_upload', 'file_id': 'f1'}]}]},
    'fallbacks': {'fallbacks': 'default'},
    'fast_mode': {'speed': 'fast'},
    'regional_inference': {'inference_geo': 'us'},
}
INLINE = {  # content the request carries, and the input tokens the relay reserves for it beyond the body's bytes
    'image_and_text_document': [
        {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': PIXEL}},
        {'type': 'document', 'source': {'type': 'text', 'media_type': 'text/plain', 'data': 'notes'}}],
    'pdf': [{'type': 'document', 'source': {'type': 'base64', 'media_type': 'application/pdf', 'data': 'JVBERi0x'}}],
}


@pytest.mark.parametrize('fields', list(UNBOUNDED.values()), ids=list(UNBOUNDED))
def test_a_call_whose_cost_its_body_cannot_bound_is_a_400_and_never_forwarded(start, upstream, tmp_path, fields):
    server = start(upstream.url)
    status, _, data, _ = post(server.address, message(**fields))
    error = json.loads(data)
    assert status == 400 and error['error']['type'] == 'invalid_request_error'
    assert 'cannot bound' in error['error']['message']
    assert upstream.seen == [] and pool_events(tmp_path) == ['created']


@pytest.mark.parametrize('name', list(INLINE))
def test_custom_tools_and_inline_images_and_documents_are_forwarded_with_their_bound(start, upstream, tmp_path, name):
    upstream.reply = stream_reply(EVENTS)
    server = start(upstream.url)
    tools = [{'name': 'Bash', 'description': 'run a command', 'input_schema': {'type': 'object'}},
             {'type': 'custom', 'name': 'Read', 'description': 'read a file', 'input_schema': {'type': 'object'}}]
    messages = [{'role': 'user', 'content': 'go'},
                {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 't1', 'name': 'Read', 'input': {}}]},
                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 't1', 'content': INLINE[name]}]}]
    body = message(tools=tools, messages=messages)
    status, _, _, whole = post(server.address, body)
    assert status == 200 and whole and len(upstream.seen) == 1
    # an image costs up to ~4,784 tokens however few its bytes; a PDF's pages are read as text and images
    reserved = worst_usd(body, images=1) if name == 'image_and_text_document' else (1e6 * 8 + 4096 * 20) / 1e6
    [line] = ledger(tmp_path)
    assert line['outcome'] == 'call' and line['reserved_usd'] == pytest.approx(reserved)


def test_a_target_that_is_not_printable_ascii_is_a_400_and_never_reserved(start, upstream, tmp_path):
    server = start(upstream.url)
    body = message()
    with socket.create_connection(server.address, timeout=30) as raw:  # http.client itself would refuse to send it
        raw.sendall(b'POST /v1/messages?beta=\xff HTTP/1.1\r\nHost: relay\r\nContent-Type: application/json\r\n'
                    b'Content-Length: %d\r\n\r\n%b' % (len(body), body))
        response = http.client.HTTPResponse(raw)
        response.begin()
        error = json.loads(response.read())
    assert response.status == 400 and error['error']['type'] == 'invalid_request_error'
    assert upstream.seen == [] and pool_events(tmp_path) == ['created']
    [line] = ledger(tmp_path)
    assert (line['outcome'], line['status'], line['usd']) == ('error', 400, 0.0)


@pytest.mark.parametrize('raw', [
    b'{"model": "claude-opus-5-5", "max_tokens": 64000, "max_tokens": 1, "messages": []}',
    b'{"model": "claude-opus-5-5", "max_tokens": 1, "messages": [{"role": "user", "content": [{"type": "image", '
    b'"source": {"type": "url", "type": "base64", "media_type": "image/png", "data": "x"}}]}]}'],
    ids=['max_tokens_twice', 'nested_source_type_twice'])
def test_a_body_that_repeats_a_key_is_a_400_and_never_reserved(start, upstream, tmp_path, raw):
    server = start(upstream.url)  # the API might read the other value than the relay's json.loads (the last one)
    status, _, data, _ = post(server.address, raw)
    error = json.loads(data)
    assert status == 400 and error['error']['type'] == 'invalid_request_error'
    assert 'repeats' in error['error']['message']
    assert upstream.seen == [] and pool_events(tmp_path) == ['created']


@pytest.mark.parametrize('method, path', [('POST', '/v1/complete'), ('POST', '/v1/messages/batches'),
                                          ('POST', '/v1/messages/'), ('GET', '/v1/messages'), ('GET', '/v1/models'),
                                          ('DELETE', '/v1/messages'), ('PATCH', '/v1/messages/count_tokens')])
def test_any_other_path_or_method_is_a_404(start, upstream, tmp_path, method, path):
    server = start(upstream.url)
    status, _, data, _ = post(server.address, message() if method in ('POST', 'PATCH') else None, path=path,
                              method=method)
    error = json.loads(data)
    assert status == 404 and error['type'] == 'error' and error['error']['type'] == 'not_found_error'
    assert upstream.seen == [] and pool_events(tmp_path) == ['created']


def test_count_tokens_is_forwarded_but_free(start, upstream, tmp_path):
    upstream.reply = json_reply(200, {'input_tokens': 4321})
    server = start(upstream.url, total=1e-9)  # no message would fit; counting needs none of it
    body = json.dumps({'model': MODEL, 'messages': [{'role': 'user', 'content': MARKER}]}).encode()
    status, _, data, whole = post(server.address, body, path='/v1/messages/count_tokens?beta=true')
    assert status == 200 and whole and json.loads(data) == {'input_tokens': 4321}
    [seen] = upstream.seen
    assert seen['path'] == '/v1/messages/count_tokens?beta=true' and seen['body'] == body
    assert dict(seen['headers'])['x-api-key'] == REAL_KEY
    assert pool_events(tmp_path) == ['created'] and pool(tmp_path)['spent_usd'] == 0.0
    [line] = ledger(tmp_path)
    assert (line['path'], line['outcome'], line['usd']) == ('/v1/messages/count_tokens', 'call', 0.0)
    assert line['model'] == MODEL


@pytest.mark.parametrize('case', ['unreachable', 'hangs_up'])
def test_no_answer_from_upstream_is_a_502_charged_only_if_the_request_went_out(start, upstream, tmp_path, case):
    if case == 'unreachable':
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            url = f'http://127.0.0.1:{probe.getsockname()[1]}'  # nothing listens there once the probe is closed
    else:
        upstream.reply = lambda handler: setattr(handler, 'close_connection', True)  # reads it, answers nothing
        url = upstream.url
    server = start(url)
    body = message()
    status, _, data, _ = post(server.address, body)
    assert status == 502 and json.loads(data)['error']['type'] == 'api_error'
    charged = 0.0 if case == 'unreachable' else worst_usd(body)  # a request that went out may have been billed
    assert pool(tmp_path)['spent_usd'] == pytest.approx(charged) and pool(tmp_path)['open'] == 0
    [line] = ledger(tmp_path)
    assert (line['outcome'], line['status']) == ('error', 502) and line['usd'] == pytest.approx(charged)


def test_a_call_dearer_than_its_reservation_is_charged_in_full_and_stops_the_relay(start, upstream, tmp_path):
    usage = {'input_tokens': 5_000_000, 'output_tokens': 10}  # $20, far more than this request could cost
    upstream.reply = json_reply(200, {'id': 'msg_3', 'type': 'message', 'role': 'assistant', 'model': MODEL,
                                      'content': [], 'stop_reason': 'end_turn', 'usage': usage})
    server = start(upstream.url)
    assert post(server.address, message(stream=False))[0] == 200
    assert pool(tmp_path)['spent_usd'] == pytest.approx((5_000_000 * 4 + 10 * 20) / 1e6)
    status, _, data, _ = post(server.address, message(stream=False))
    assert status == 403 and 'worst case' in json.loads(data)['error']['message'] and len(upstream.seen) == 1
    first, second = ledger(tmp_path)
    assert first['exceeded_reservation'] is True and second['outcome'] == 'refused'


def test_a_call_the_pool_cannot_settle_still_ends_and_stops_the_relay(start, upstream, tmp_path, monkeypatch):
    upstream.reply = stream_reply(EVENTS)
    server = start(upstream.url)

    def unsettled(*args, **kwargs):
        raise OSError('no space left on the device holding pool.json')
    monkeypatch.setattr(server.pool, 'settle', unsettled)
    status, _, data, whole = post(server.address, message())
    assert status == 200 and whole and data == b''.join(EVENTS)  # the answer still ends, so it is not retried
    status, _, data, _ = post(server.address, message())
    assert status == 403 and 'could not settle' in json.loads(data)['error']['message'] and len(upstream.seen) == 1
    first, second = ledger(tmp_path)
    assert 'settling failed' in first['detail'] and second['outcome'] == 'refused'
    assert pool(tmp_path)['open'] == 1  # held until this process ends; then the pool charges its worst case


def test_tls_secrets_are_not_logged_even_with_sslkeylogfile_set(start, upstream, tmp_path, monkeypatch):
    monkeypatch.setenv('SSLKEYLOGFILE', str(tmp_path / 'tls-secrets.log'))
    assert ssl.create_default_context().keylog_filename == str(tmp_path / 'tls-secrets.log')  # what it guards against
    assert start(upstream.url).tls.keylog_filename is None


def test_the_key_leaves_the_environment_and_never_reaches_the_ledger(start, upstream, tmp_path, protected):
    upstream.reply = stream_reply(EVENTS)
    server = start(upstream.url)
    assert 'ANTHROPIC_API_KEY' not in os.environ and protected == [True]
    child = subprocess.run([sys.executable, '-c', 'import os; print(os.environ.get("ANTHROPIC_API_KEY"))'],
                           capture_output=True, text=True, timeout=60)
    assert child.stdout.strip() == 'None'  # nothing the relay starts inherits it
    post(server.address, message())
    post(server.address, message(model='claude-haiku-5-5'))
    post(server.address, message(), path='/v1/messages/batches')
    text = (tmp_path / 'ledger.jsonl').read_text()
    assert len(text.splitlines()) == 3
    for secret in (REAL_KEY, DUMMY_KEY, MARKER, 'robot programs', 'claude-code-20250219', 'Grasp the cranium'):
        assert secret not in text


@pytest.mark.parametrize('key', [None, '', ' \n', f'{REAL_KEY}\nx-injected: 1'],
                         ids=['unset', 'empty', 'blank', 'header_injection'])
def test_the_relay_refuses_to_start_without_a_usable_key(tmp_path, monkeypatch, protected, key):
    BudgetPool.create(tmp_path / 'pool', 100.0, note='test')
    if key is None:
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    else:
        monkeypatch.setenv('ANTHROPIC_API_KEY', key)
    with pytest.raises(RuntimeError, match='ANTHROPIC_API_KEY') as refused:
        relay.serve(tmp_path / 'pool', 'aspire:test', socket_path=tmp_path / 'relay.sock',
                    ledger_path=tmp_path / 'ledger.jsonl')
    assert REAL_KEY not in str(refused.value) and 'ANTHROPIC_API_KEY' not in os.environ
    assert not (tmp_path / 'relay.sock').exists()


def test_exception_text_never_carries_the_key(start, upstream, tmp_path, monkeypatch):
    server = start(upstream.url)

    class Unreachable:
        def request(self, method, target, body, headers):
            raise ConnectionRefusedError(f'no route for a request with key {headers["x-api-key"]}')

        def close(self):
            pass
    monkeypatch.setattr(server, 'connect', Unreachable)
    status, _, data, _ = post(server.address, message())
    assert status == 502 and b'ConnectionRefusedError' in data and REAL_KEY.encode() not in data
    text = (tmp_path / 'ledger.jsonl').read_text()
    assert 'ConnectionRefusedError' in text and REAL_KEY not in text


def forwarder_lib(tmp_path):
    """relay.py and budget.py alone in an empty package skeleton: all of ours the agent's sandbox gets."""
    baselines = tmp_path / 'lib' / 'tasks' / 'dual_arm_crag_skull' / 'baselines'
    baselines.mkdir(parents=True)
    for package in (baselines.parents[1], baselines.parent, baselines):
        (package / '__init__.py').write_text('')
    for name in ('relay.py', 'budget.py'):
        shutil.copy(Path(relay.__file__).with_name(name), baselines / name)
    return tmp_path / 'lib'


def run_cli(*args, env, cwd=SOURCE_ROOT):
    """Start `python -m ...baselines.relay ARGS` in `cwd`; returns the process and its first line of output (JSON)."""
    process = subprocess.Popen([sys.executable, '-B', '-m', 'tasks.dual_arm_crag_skull.baselines.relay',
                                *map(str, args)], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    ready, _, _ = select.select([process.stdout], [], [], 120)
    line = process.stdout.readline() if ready else ''
    if not line:
        process.kill()
        raise AssertionError(f'relay {args[0]} did not start: {process.communicate(timeout=30)[1][-3000:]}')
    return process, json.loads(line)


def stop(process):
    process.send_signal(signal.SIGTERM)
    process.communicate(timeout=30)
    return process.returncode


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='prctl and /proc are Linux')
def test_the_cli_relays_through_its_unix_socket_and_the_sandbox_side_forwarder(upstream, tmp_path):
    upstream.reply = stream_reply(EVENTS)
    BudgetPool.create(tmp_path / 'pool', 100.0, note='test')
    sock = tmp_path / 'relay.sock'
    env = {**{k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}, 'PYTHONDONTWRITEBYTECODE': '1'}
    served, started = run_cli('serve', '--who', 'aspire:cli', '--socket', sock, '--ledger', tmp_path / 'ledger.jsonl',
                              '--pool', tmp_path / 'pool', '--upstream', upstream.url,
                              env={**env, 'ANTHROPIC_API_KEY': REAL_KEY})
    processes = [served]
    try:
        assert started['relay'] == str(sock) and stat.S_IMODE(sock.stat().st_mode) == 0o600
        if os.geteuid() != 0:  # non-dumpable: no other process of the user reads the key from /proc
            with pytest.raises(PermissionError):
                Path(f'/proc/{served.pid}/environ').read_bytes()
        forwarder, forwarding = run_cli('forward', '--port', 0, '--socket', sock, env=env, cwd=forwarder_lib(tmp_path))
        processes.insert(0, forwarder)
        connection = http.client.HTTPConnection('127.0.0.1', forwarding['port'], timeout=30)
        try:
            for _ in range(2):  # two calls on one kept-alive connection
                connection.request('POST', '/v1/messages?beta=true', body=message(), headers=CLIENT_HEADERS)
                response = connection.getresponse()
                assert response.status == 200 and response.read() == b''.join(EVENTS)
        finally:
            connection.close()
    finally:
        codes = [stop(process) for process in processes]
    assert codes == [0, 0] and not sock.exists()
    assert pool(tmp_path)['by_who'] == {'aspire:cli': pytest.approx(2 * STREAM_USD, rel=1e-9)}
    assert [dict(seen['headers'])['x-api-key'] for seen in upstream.seen] == [REAL_KEY, REAL_KEY]
    assert [line['outcome'] for line in ledger(tmp_path)] == ['call', 'call']
