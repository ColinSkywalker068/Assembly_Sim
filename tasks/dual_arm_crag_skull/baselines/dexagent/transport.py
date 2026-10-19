"""Claude behind DexAgent's external model transport (agent/external.py), paid from the shared budget pool.

DexAgent's StageRunner writes every model request to <transport_dir>/requests/<session>-<turn>.json (system,
messages as DexAgent keeps them, tool schemas; images as files under images/) and waits for
responses/<id>.json with text and tool_use blocks. DexAgent keeps only those blocks, so the responder keeps its
own copy of every assistant turn exactly as the API returned it (thinking blocks and signatures included) and
sends that copy back to the API, after checking that DexAgent's history is exactly what was answered.

Every call is priced (claude-opus-5-5: 4 / 20 USD per MTok in / out; prompt-cache writes 1.25x input for the
5-minute TTL and 2x for 1 hour, reads 0.05x) and written to a JSONL ledger. Money comes from one pool shared by every
live baseline run (baselines/budget.py: DexAgent and ASPIRE together, no per-baseline or per-trial amount): before
a call its worst case (every input token at the dearest input rate, max_tokens of output) is reserved there, and if
it does not fit in what is left the call is not made. After the call the reservation is settled at the measured
cost; a failed attempt at its worst case if it may have been billed, else at 0, and a retry reserves again. A call
that cost more than its reservation is charged in full and stops the run (its bound was wrong). The input bound is
the previous call's measured input plus its output plus the UTF-8 bytes of what was added since (at most one token
per byte) and w*h/750 per new image.

A stop writes operator_checkpoint.json {"kind": "administrative_checkpoint"} into the transport directory and
answers with a report_blocked call, which DexAgent's runner accepts at once and ends the stage without verifying
(agent/runner.py `_operator_checkpoint`). Stops: `budget`; `refusal`, `max_tokens`, `context_window` and any
other stop_reason but end_turn / tool_use (that turn's tool calls are never passed on); `transcript` (history or
session mismatch); `api_error` (after retries); `error`. Server-side fallbacks are not used: a fallback model
would change the agent being evaluated and has no fixed price.

Thinking is always on for claude-opus-5-5 (effort is the control). Each thinking block is bound to the system
prompt, the tools and the earlier messages it was produced with, and DexAgent's tool list grows when the agent adds
library tools, so requests set `prefix_mismatch_behavior: drop_block` (beta thinking-binding-controls-2026-08-01):
the API drops the earlier blocks such an edit invalidates, unbilled, instead of failing the request (the default
on accounts created on or after 2026-08-31). The drops are logged per call (`input_transformations`).

The API key is read only in this process (anthropic.Anthropic() from the environment); DexAgent's processes get
a scrubbed environment (dexenv.py). `client` is injectable for tests.
"""
from __future__ import annotations

import base64
import json
import math
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ...s2.claude import DEFAULT_MODEL
from ..budget import BudgetExhausted, BudgetPool

SCHEMA = 'dexagent.external.v1'
# USD per MTok in and out, and the cache-read multiplier of the input price (claude-opus-5-5 reads at $0.20).
PRICES_USD_PER_MTOK = {'claude-opus-5-5': (4.0, 20.0, 0.05)}
CACHE_WRITE = {'5m': 1.25, '1h': 2.0}
THINKING_BINDING_BETA = 'thinking-binding-controls-2026-08-01'
THINKING = {'type': 'adaptive', 'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}
STOP_KINDS = ('budget', 'refusal', 'max_tokens', 'context_window', 'stop_reason', 'transcript', 'api_error', 'error')
PASS_THROUGH_STOPS = ('end_turn', 'tool_use')
IMAGE_SUFFIXES = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}
PLACEHOLDER = '(no visible output in this turn)'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
    temp.write_text(json.dumps(value, indent=1, allow_nan=False))
    temp.replace(path)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


@dataclass(frozen=True)
class Price:
    """USD per million tokens in and out; `cache_read` is the multiplier of the input price for cache reads."""

    input: float
    output: float
    cache_read: float

    @classmethod
    def for_model(cls, model, table=None):
        table = PRICES_USD_PER_MTOK if table is None else table
        if model not in table:
            raise ValueError(f'no price for {model!r}; the hard cap needs one (PRICES_USD_PER_MTOK)')
        return cls(*map(float, table[model]))

    def cost(self, usage):
        """USD of one call from its usage record (see usage_record)."""
        tokens_in = (usage['input_tokens'] + CACHE_WRITE['5m'] * usage['cache_write_5m_tokens']
                     + CACHE_WRITE['1h'] * usage['cache_write_1h_tokens'] + self.cache_read * usage['cache_read_tokens'])
        return (tokens_in * self.input + usage['output_tokens'] * self.output) / 1e6

    def worst_case(self, input_tokens, output_tokens, cache_ttl):
        rate = self.input * max(1.0, CACHE_WRITE.get(cache_ttl, 1.0))
        return (input_tokens * rate + output_tokens * self.output) / 1e6


def _field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def usage_record(usage, cache_ttl):
    """Token counts of one response; cache writes split by TTL (the breakdown when given, else `cache_ttl`)."""
    if usage is None:
        raise ValueError('the response carries no usage; it cannot be priced')
    created = int(_field(usage, 'cache_creation_input_tokens') or 0)
    breakdown = _field(usage, 'cache_creation')
    five = int(_field(breakdown, 'ephemeral_5m_input_tokens') or 0) if breakdown is not None else None
    hour = int(_field(breakdown, 'ephemeral_1h_input_tokens') or 0) if breakdown is not None else None
    if five is None or five + hour != created:
        five, hour = (0, created) if cache_ttl == '1h' else (created, 0)
    return {'input_tokens': int(_field(usage, 'input_tokens') or 0),
            'output_tokens': int(_field(usage, 'output_tokens') or 0),
            'cache_write_5m_tokens': five, 'cache_write_1h_tokens': hour,
            'cache_read_tokens': int(_field(usage, 'cache_read_input_tokens') or 0),
            'service_tier': _field(usage, 'service_tier'), 'inference_geo': _field(usage, 'inference_geo')}


def block_dict(block):
    """A response content block as the dict the API accepts back (SDK model, dict or attribute object)."""
    if isinstance(block, dict):
        return {k: v for k, v in block.items() if v is not None}
    if hasattr(block, 'model_dump'):
        return block.model_dump(mode='json', exclude_none=True, by_alias=True)
    return {k: v for k, v in vars(block).items() if v is not None and not k.startswith('_')}


def _transformations(response):
    """What the API changed in the input before the model saw it (dropped thinking blocks), or None if not reported."""
    value = getattr(response, 'input_transformations', None)
    return None if value is None else [block_dict(item) for item in value]


def _image_tokens(path):
    try:
        from PIL import Image
        with Image.open(path) as image:
            width, height = image.size
    except Exception:  # noqa: BLE001 - an unreadable image gets the largest bound below
        return 8000
    return int(math.ceil(width * height / 750 * 1.2)) + 100


class Stop(Exception):
    def __init__(self, kind, reason):
        if kind not in STOP_KINDS:
            raise ValueError(f'stop kind must be one of {STOP_KINDS}')
        super().__init__(reason)
        self.kind, self.reason = kind, str(reason)


class Responder:
    """Answers DexAgent's external transport with Claude; see the module docstring."""

    def __init__(self, transport_dir, *, pool, who, client=None, model=DEFAULT_MODEL, effort='high', max_tokens=64000,
                 cache_ttl='1h', price=None, ledger=None, transcript=None, retries=3, backoff_s=30.0, sleep=time.sleep):
        if not isinstance(pool, BudgetPool):
            raise ValueError('pool must be the shared BudgetPool (baselines/budget.py)')
        if not isinstance(who, str) or not who:
            raise ValueError('who must name the run that spends from the pool')
        if cache_ttl not in (None, '5m', '1h'):
            raise ValueError("cache_ttl must be None, '5m' or '1h'")
        if effort not in ('low', 'medium', 'high', 'xhigh', 'max'):
            raise ValueError('effort must be an API effort level')
        self.dir = Path(transport_dir).resolve()
        self.pool, self.who, self.model, self.effort = pool, who, model, effort
        self.max_tokens, self.cache_ttl = int(max_tokens), cache_ttl
        self.price = price or Price.for_model(model)
        self.ledger = Path(ledger) if ledger else self.dir.parent / 'llm_ledger.jsonl'
        self.transcript = Path(transcript) if transcript else self.dir.parent / 'llm_transcript.jsonl'
        self.retries, self.backoff_s, self.sleep = int(retries), float(backoff_s), sleep
        self._client = client
        self.session = self.system = None
        self.turn = 0
        self.history = None          # DexAgent messages of the last request plus our answer to it
        self.assistant = []          # the API's content of every answered turn, in order
        self.spent_usd = 0.0
        self.calls = 0
        self.last = None             # {'input': measured input tokens, 'output': tokens, 'messages': n, 'tools': {...}}
        self.stopped = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            # credentials from this process's environment only; no SDK retries, so every attempt reaches the ledger
            self._client = anthropic.Anthropic(max_retries=0)
        return self._client

    # ------------------------------------------------------------------ files
    def pending(self):
        requests = self.dir / 'requests'
        if not requests.is_dir():
            return []
        out = []
        for path in requests.glob('*.json'):
            if not (self.dir / 'responses' / path.name).exists():
                out.append(path)
        return sorted(out, key=lambda p: p.stem)

    def answer_pending(self):
        paths = self.pending()
        for path in paths:
            self.answer(path)
        return len(paths)

    def serve(self, process, *, poll_s=0.2, deadline=None, clock=time.monotonic):
        """Answer until `process` exits (a request left behind by an exited process is never paid for); returns
        'exited', or 'deadline' when `deadline` (clock time) passes."""
        while True:
            if process.poll() is not None:
                return 'exited'
            self.answer_pending()
            if deadline is not None and clock() > deadline:
                return 'deadline'
            self.sleep(poll_s)

    def _log(self, path, record):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as handle:
            handle.write(json.dumps({'time_utc': _now(), **record}, default=str) + '\n')

    def _respond(self, request, content):
        atomic_json(self.dir / 'responses' / f'{request["id"]}.json',
                    {'session': request['session'], 'id': request['id'], 'content': content})

    # ------------------------------------------------------------------ one request
    def answer(self, path):
        request = json.loads(Path(path).read_text())
        try:
            if request.get('schema') != SCHEMA or not all(k in request for k in ('session', 'id', 'turn', 'messages')):
                raise Stop('transcript', f'{Path(path).name} is not a {SCHEMA} request')
            if self.stopped is not None:
                raise Stop(self.stopped['kind'], self.stopped['reason'])
            new = self._check_history(request)
            api_messages = self._api_messages(request['messages'])
            bound = self._bound(request, new)
            response = self._call(request, api_messages, bound)
            content = self._accept(request, response)
        except Stop as stop:
            self._stop(request, stop)
            return
        except Exception as exc:  # noqa: BLE001 - an unattended run ends cleanly, the reason is kept
            self._stop(request, Stop('error', f'{type(exc).__name__}: {exc}'))
            return
        self._respond(request, content)

    def _check_history(self, request):
        """DexAgent's messages must be what it sent last time, our answer, and one new user message."""
        messages = request['messages']
        if self.session is None:
            if len(messages) != 1 or messages[0].get('role') != 'user':
                raise Stop('transcript', 'the first request must hold exactly one user message')
            self.session, self.system = request['session'], request.get('system')
            return messages
        if request['session'] != self.session:
            raise Stop('transcript', f'request session {request["session"]} is not {self.session}')
        if request.get('system') != self.system:
            raise Stop('transcript', 'the system prompt changed within the session')
        if request['turn'] != self.turn + 1:
            raise Stop('transcript', f'expected turn {self.turn + 1}, got {request["turn"]}')
        prefix = self.history
        if messages[:len(prefix)] != prefix or len(messages) != len(prefix) + 1 or messages[-1].get('role') != 'user':
            raise Stop('transcript', "DexAgent's history differs from what was answered (assistant turns or earlier "
                                     'messages changed, or not exactly one new user message)')
        return messages[-1:]

    def _image(self, block):
        if 'path' in block:
            path = Path(block['path']).resolve()
            if path.parent != (self.dir / 'images') or path.suffix.lower() not in IMAGE_SUFFIXES:
                raise Stop('transcript', f'image outside the transport image folder: {block["path"]}')
            data = base64.standard_b64encode(path.read_bytes()).decode('ascii')
            media_type = IMAGE_SUFFIXES[path.suffix.lower()]
        else:
            data, media_type = block.get('data'), block.get('mime')
            if media_type not in IMAGE_SUFFIXES.values() or not isinstance(data, str):
                raise Stop('transcript', 'inline image needs a png/jpeg mime and base64 data')
        return {'type': 'image', 'source': {'type': 'base64', 'media_type': media_type, 'data': data}}

    def _user_block(self, block):
        kind = block.get('type')
        if kind == 'text':
            return {'type': 'text', 'text': block.get('text') or '(empty)'}
        if kind == 'image':
            return self._image(block)
        if kind == 'tool_result':
            inner = [self._user_block(b) for b in block.get('content') or []] or [{'type': 'text', 'text': '(empty)'}]
            return {'type': 'tool_result', 'tool_use_id': block['tool_use_id'], 'content': inner}
        raise Stop('transcript', f'unsupported user content block {kind!r}')

    def _api_messages(self, messages):
        out, answered = [], iter(self.assistant)
        for message in messages:
            if message.get('role') == 'assistant':
                out.append({'role': 'assistant', 'content': next(answered)})
            else:
                content = message.get('content')
                content = [{'type': 'text', 'text': content}] if isinstance(content, str) else content
                out.append({'role': 'user', 'content': [self._user_block(b) for b in content]})
        return out

    def _size_bound(self, blocks):
        tokens = 0
        for block in blocks:
            if block.get('type') == 'image' and 'path' in block:
                tokens += _image_tokens(block['path'])
            elif block.get('type') == 'image':
                tokens += 8000
            elif block.get('type') == 'tool_result':
                tokens += 50 + self._size_bound(block.get('content') or [])
            else:
                tokens += 50 + len(json.dumps(block, ensure_ascii=False).encode('utf-8'))
        return tokens

    def _bound(self, request, new_messages):
        """Upper bound (tokens) on this call's input, then the USD check."""
        tools = {t['name']: json.dumps(t, sort_keys=True) for t in request.get('tools') or []}
        added = sum(len(text.encode('utf-8')) for name, text in tools.items()
                    if self.last is None or self.last['tools'].get(name) != text)
        new_tokens = sum(self._size_bound(m.get('content') or []) if isinstance(m.get('content'), list)
                         else len(str(m.get('content')).encode('utf-8')) for m in new_messages)
        if self.last is None:
            estimate = len(str(request.get('system') or '').encode('utf-8')) + added + new_tokens + 2000
        else:
            estimate = self.last['input'] + self.last['output'] + added + new_tokens + 500
        worst = self.price.worst_case(estimate, self.max_tokens, self.cache_ttl)
        return {'input_tokens_bound': estimate, 'usd_bound': worst, 'tools': tools}

    def _reserve(self, request, bound, attempt):
        try:
            return self.pool.reserve(bound['usd_bound'], who=self.who,
                                     what=f'{request["id"]} attempt {attempt} ({bound["input_tokens_bound"]} input '
                                          f'tokens at most, {self.max_tokens} output)')
        except BudgetExhausted as exc:
            raise Stop('budget', str(exc)) from None

    def request_kwargs(self, request, api_messages):
        kwargs = {'model': self.model, 'max_tokens': self.max_tokens, 'messages': api_messages,
                  'system': [{'type': 'text', 'text': request.get('system') or ''}],
                  'tools': [{'name': t['name'], 'description': t.get('description', ''),
                             'input_schema': t['input_schema']} for t in request.get('tools') or []],
                  'output_config': {'effort': self.effort}, 'thinking': THINKING, 'betas': [THINKING_BINDING_BETA]}
        if self.cache_ttl:
            kwargs['cache_control'] = {'type': 'ephemeral', 'ttl': self.cache_ttl}
        return kwargs

    def _call(self, request, api_messages, bound):
        kwargs = self.request_kwargs(request, api_messages)
        for attempt in range(self.retries + 1):
            reservation = self._reserve(request, bound, attempt)
            started = time.monotonic()
            try:
                with self.client.beta.messages.stream(**kwargs) as stream:
                    response = stream.get_final_message()
            except Exception as exc:  # noqa: BLE001 - classified below
                retryable, maybe_billed = _classify(exc)
                charge = bound['usd_bound'] if maybe_billed else 0.0
                self.pool.settle(reservation, charge, outcome='api_error',
                                 details={'request_id': request['id'], 'error': type(exc).__name__})
                self.spent_usd += charge
                self._log(self.ledger, {'event': 'api_error', 'request_id': request['id'], 'turn': request['turn'],
                                        'attempt': attempt, 'error': f'{type(exc).__name__}: {exc}'[:2000],
                                        'charged_usd': charge, 'spent_usd': round(self.spent_usd, 6)})
                if not retryable or attempt == self.retries:
                    raise Stop('api_error', f'{type(exc).__name__}: {str(exc)[:500]}') from exc
                self.sleep(self.backoff_s * 2 ** attempt)
                continue
            try:
                usage = usage_record(getattr(response, 'usage', None), self.cache_ttl)
                cost = self.price.cost(usage)
            except Exception:  # an answered call that cannot be priced is charged at its worst case
                self.pool.settle(reservation, bound['usd_bound'], outcome='unpriced', details={'request_id': request['id']})
                raise
            exceeded = self.pool.settle(reservation, cost, outcome='call',
                                        details={'request_id': request['id'], 'response_id': getattr(response, 'id', None)})
            self.spent_usd += cost
            self.calls += 1
            self._log(self.ledger, {'event': 'call', 'request_id': request['id'], 'turn': request['turn'],
                                    'attempt': attempt, 'response_id': getattr(response, 'id', None),
                                    'model': getattr(response, 'model', None), 'stop_reason': response.stop_reason,
                                    'usage': usage, 'input_transformations': _transformations(response),
                                    'cost_usd': round(cost, 6), 'spent_usd': round(self.spent_usd, 6),
                                    'bound_usd': round(bound['usd_bound'], 6),
                                    'input_tokens_bound': bound['input_tokens_bound'],
                                    'seconds': round(time.monotonic() - started, 2)})
            measured = usage['input_tokens'] + usage['cache_write_5m_tokens'] + usage['cache_write_1h_tokens'] + \
                usage['cache_read_tokens']
            self.last = {'input': measured, 'output': usage['output_tokens'], 'tools': bound['tools']}
            if exceeded['exceeded']:
                raise Stop('budget', f'the call cost ${cost:.4f}, more than its reserved worst case '
                                     f'${bound["usd_bound"]:.4f}: the bound is wrong, so the run stops')
            return response
        raise Stop('api_error', 'no response')  # pragma: no cover - the loop returns or raises

    def _accept(self, request, response):
        """Check stop_reason, then keep the API content and pass DexAgent its text and tool_use blocks."""
        reason = response.stop_reason
        content = [block_dict(b) for b in response.content]
        self._log(self.transcript, {'request_id': request['id'], 'turn': request['turn'], 'stop_reason': reason,
                                    'content': content})
        if reason == 'refusal':
            details = getattr(response, 'stop_details', None)
            raise Stop('refusal', f'the model declined (category {_field(details, "category")}); its tool calls '
                                  f'were not run')
        if reason == 'max_tokens':
            raise Stop('max_tokens', f'the turn hit max_tokens={self.max_tokens}; its tool calls were not run')
        if reason == 'model_context_window_exceeded':
            raise Stop('context_window', 'the conversation exceeded the context window; tool calls were not run')
        if reason not in PASS_THROUGH_STOPS:
            raise Stop('stop_reason', f'unexpected stop_reason {reason!r}; tool calls were not run')
        content = [b for b in content if not (b.get('type') == 'text' and not b.get('text'))]
        visible = []
        for block in content:
            if block.get('type') == 'text':
                visible.append({'type': 'text', 'text': block['text']})
            elif block.get('type') == 'tool_use':
                visible.append({'type': 'tool_use', 'id': block['id'], 'name': block['name'],
                                'input': dict(block.get('input') or {})})
        if not visible:
            placeholder = {'type': 'text', 'text': PLACEHOLDER}
            content, visible = content + [placeholder], [placeholder]
        names = {t['name'] for t in request.get('tools') or []}
        if any(b['type'] == 'tool_use' and b['name'] not in names for b in visible):
            raise Stop('error', 'the model called a tool absent from this request')
        self.assistant.append(content)
        self.history = request['messages'] + [{'role': 'assistant', 'content': json.loads(json.dumps(visible))}]
        self.turn = request['turn']
        return visible

    def _stop(self, request, stop):
        first = self.stopped is None
        if first:
            self.stopped = {'kind': stop.kind, 'reason': stop.reason, 'request_id': request.get('id'),
                            'turn': request.get('turn'), 'time_utc': _now()}
            atomic_json(self.dir / 'operator_checkpoint.json', {
                'kind': 'administrative_checkpoint', 'stop': stop.kind, 'reason': stop.reason,
                'requested_by': 'Assembly_Sim DexAgent transport', 'time_utc': self.stopped['time_utc']})
            self._log(self.ledger, {'event': 'stop', **self.stopped, 'spent_usd': round(self.spent_usd, 6),
                                    'calls': self.calls})
        names = {t.get('name') for t in request.get('tools') or []}
        text = f'[operator] Administrative stop ({stop.kind}): {stop.reason}'
        content = [{'type': 'text', 'text': text}]
        if 'report_blocked' in names:
            content.append({'type': 'tool_use', 'id': f'operator_stop_{request.get("turn")}', 'name': 'report_blocked',
                            'input': {'reason': text}})
        if request.get('id') and request.get('session'):
            self._respond(request, content)

    def summary(self):
        return {'spent_usd': round(self.spent_usd, 6), 'pool': str(self.pool.dir), 'who': self.who,
                'pool_left_usd': round(self.pool.status()['left_usd'], 6), 'calls': self.calls, 'turns': self.turn,
                'model': self.model, 'effort': self.effort, 'max_tokens': self.max_tokens, 'cache_ttl': self.cache_ttl,
                'stop': self.stopped, 'ledger': str(self.ledger), 'transcript': str(self.transcript)}


def _classify(exc):
    """(retryable, maybe_billed) for an exception raised by the SDK call. A dropped connection, a timeout and an error
    event inside a stream that had already started (the SDK reports the stream's own status, below 400) may have
    been billed; rate limits and server errors (500-599: overloaded, unavailable, deadline exceeded) are rejections
    before any work; other client errors (400-499) are final."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - the default client needs the SDK anyway
        return False, True
    if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return True, True
    if isinstance(exc, anthropic.APIStatusError):
        code = int(getattr(exc, 'status_code', 0) or 0)
        if code < 400:
            return True, True
        return (True, False) if code == 429 or code >= 500 else (False, False)
    if isinstance(exc, anthropic.AnthropicError):
        return False, True  # another SDK error, e.g. a received response that failed validation
    return False, False  # a local error: no request left this process


def ledger_totals(path):
    """(USD spent, successful calls) from a ledger, as RunRecord cost_usd and llm_calls."""
    spent, calls = 0.0, 0
    path = Path(path)
    if not path.exists():
        return 0.0, 0
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get('event') == 'call':
            spent += float(record['cost_usd'])
            calls += 1
        elif record.get('event') == 'api_error':
            spent += float(record.get('charged_usd') or 0.0)
    return round(spent, 6), calls
