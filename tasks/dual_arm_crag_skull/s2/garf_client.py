"""Client for the JSON-lines GARF tool process with per-call timeouts and an input-keyed cache.

The runner (`python -m tasks.dual_arm_crag_skull.tools ...`) loads the model once and answers one
JSON line per request line. It runs in its own process group so a timeout kills wrappers such as
`uv run` or `sh -c` together with the real runner; the next call starts a fresh one. Lines that are
not JSON objects (stray output from native code) are skipped. A JSON answer that does not match the
request (observation, seeds, fragment set) means the stream is desynchronized: the runner is killed.
Results depend only on registered geometry, fragment IDs, anchor and seeds, so they are cached
across observations; only cache hits are re-stamped with the current observation ID.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path


def _child_environment():
    # The runner never needs planner credentials.
    return {key: value for key, value in os.environ.items() if not key.startswith('ANTHROPIC_')}


def answer_mismatch(envelope, result):
    """Why `result` cannot be the answer to `envelope`, or None."""
    if envelope.get('tool_name') != 'propose_assembly_candidates' or result.get('status') not in ('candidates', 'incomplete'):
        return None
    request = envelope.get('request', {})
    candidates = result.get('candidates')
    if not isinstance(candidates, list) or len(candidates) != len(request.get('seeds', ())):
        return 'candidate count differs from the requested seeds'
    fragments = set(request.get('fragment_ids', ()))
    for candidate, seed in zip(candidates, request.get('seeds', ())):
        if candidate.get('status') != 'candidate':
            continue
        if candidate.get('seed') != seed:
            return 'candidate seeds differ from the request'
        if candidate.get('observation_id') != envelope.get('current_observation_id'):
            return 'candidate observation differs from the request'
        if set(candidate.get('assembly_T_fragment', {})) != fragments:
            return 'candidate fragment set differs from the request'
    return None


class GarfProcessClient:
    def __init__(self, command, *, cwd=None, env=None, call_timeout_s=120.0, startup_timeout_s=900.0,
                 stderr_path=None):
        if not command:
            raise ValueError('runner command is required')
        if not 0 < call_timeout_s <= startup_timeout_s:
            raise ValueError('need 0 < call_timeout_s <= startup_timeout_s')
        self.command = list(command)
        self.cwd = cwd
        self.env = _child_environment() if env is None else dict(env)
        self.call_timeout_s, self.startup_timeout_s = float(call_timeout_s), float(startup_timeout_s)
        self.stderr_path = None if stderr_path is None else Path(stderr_path)
        self._process = None
        self._lines = None
        self._answered = 0
        self.starts = 0
        self.stray_lines = 0

    @property
    def restarts(self):
        return max(0, self.starts - 1)

    def _start(self):
        self.starts += 1
        stderr = subprocess.DEVNULL if self.stderr_path is None else open(self.stderr_path, 'ab')
        try:
            self._process = subprocess.Popen(self.command, cwd=self.cwd, env=self.env, stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1,
                                             start_new_session=True)
        finally:
            if stderr is not subprocess.DEVNULL:
                stderr.close()
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self._process.stdout, self._lines), daemon=True).start()
        self._answered = 0

    @staticmethod
    def _pump(stream, lines):
        try:
            for line in stream:
                lines.put(line)
        except (OSError, ValueError):
            pass
        lines.put(None)

    def _stop(self):
        process, self._process = self._process, None
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)  # the whole session: wrappers and the runner
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        for stream in (process.stdin, process.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    def close(self):
        self._stop()

    def _read_answer(self, deadline):
        """Next JSON object line before `deadline` (monotonic), skipping non-JSON lines."""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, 'timeout'
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                return None, 'timeout'
            if line is None:
                return None, 'exited'
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                self.stray_lines += 1
                continue
            if isinstance(value, dict):
                return value, None
            self.stray_lines += 1

    def call(self, envelope):
        """Send one request envelope; always returns a dict and never raises for runner faults."""
        if self._process is None or self._process.poll() is not None:
            self._stop()
            self._start()
        limit = self.startup_timeout_s if self._answered == 0 else self.call_timeout_s
        try:
            self._process.stdin.write(json.dumps(envelope, allow_nan=False) + '\n')
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._stop()
            return {'status': 'failed', 'reason': f'runner input closed: {exc}'}
        result, problem = self._read_answer(time.monotonic() + limit)
        if problem == 'timeout':
            self._stop()
            return {'status': 'failed', 'reason': f'timeout after {limit:.0f} s; runner killed'}
        if problem == 'exited':
            self._stop()
            return {'status': 'failed', 'reason': 'runner exited without answering'}
        mismatch = answer_mismatch(envelope, result)
        if mismatch:
            self._stop()
            return {'status': 'failed', 'reason': f'desynchronized runner answer ({mismatch}); runner killed'}
        self._answered += 1
        return result


def registry_fingerprint(*paths):
    """Content hash of registry files and every file they map to (geometry identity for caching)."""
    digest = hashlib.sha256()
    for path in paths:
        if path is None:
            continue
        registry = json.loads(Path(path).read_text())
        digest.update(json.dumps(registry, sort_keys=True).encode())
        for key in sorted(registry):
            digest.update(key.encode())
            digest.update(hashlib.sha256(Path(registry[key]).read_bytes()).digest())
    return digest.hexdigest()


class CachedCandidateSource:
    """propose_assembly_candidates with caching; failed or partial results are never cached."""

    def __init__(self, client, geometry_fingerprint):
        self.client = client
        self.geometry_fingerprint = geometry_fingerprint
        self._cache = {}
        self.hits = 0

    def propose(self, request, *, current_observation_id):
        key = (self.geometry_fingerprint, tuple(request.get('fragment_ids', ())), request.get('anchor_id'),
               tuple(request.get('seeds', ())))
        if request.get('observation_id') == current_observation_id and key in self._cache:
            self.hits += 1
            result = copy.deepcopy(self._cache[key])
            for candidate in result.get('candidates', []):
                candidate['observation_id'] = current_observation_id
            return {**result, 'cache_hit': True}
        result = self.client.call({'tool_name': 'propose_assembly_candidates',
                                   'current_observation_id': current_observation_id, 'request': dict(request)})
        candidates = result.get('candidates', [])
        if result.get('status') == 'candidates' and candidates and all(c.get('status') == 'candidate' for c in candidates):
            self._cache[key] = copy.deepcopy(result)
        return {**result, 'cache_hit': False}
