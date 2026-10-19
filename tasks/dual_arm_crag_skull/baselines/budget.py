"""One USD budget for every live baseline run (DexAgent and ASPIRE together): no per-baseline or per-trial amount,
nothing tops it up, and failed, retried and corrective calls all count.

The pool is a directory: pool.json (the total, what was spent, the open reservations) and pool.jsonl (every event),
changed only under an exclusive flock of pool.lock, so runs in several processes cannot spend it twice. A model call
first reserves its worst case (every input token at the dearest rate, max_tokens of output; the relay computes it)
and is made only if spent + reserved + that worst case stays within the total; afterwards the reservation is
settled at the measured cost (a failed call that may have been billed at its worst case; one that never left the
process at 0). A reservation whose process died on this host is settled at its worst case before anything else is
reserved: the call may have been billed.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import socket
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 'assembly_sim.llm_budget_pool.v1'


def configured_total_usd(config):
    """baselines.llm_budget.total_usd of configs/jing.json: the one amount every live baseline run shares."""
    total = config['baselines']['llm_budget']['total_usd']
    if isinstance(total, bool) or not isinstance(total, (int, float)) or not math.isfinite(total) or total <= 0:
        raise ValueError('baselines.llm_budget.total_usd must be a positive number of US dollars')
    return float(total)


def configured_pool_dir(config):
    """baselines.llm_budget.pool_dir of configs/jing.json, resolved against the repository root."""
    from ..workspace import repo_path
    return repo_path(config['baselines']['llm_budget']['pool_dir'])


class BudgetExhausted(Exception):
    """The worst case of the next call does not fit in what is left of the pool."""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _usd(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite, non-negative number of US dollars')
    return float(value)


def _process_start(pid):
    """The process's start time in clock ticks since boot (/proc/<pid>/stat field 22), or None if it is gone: a pid
    alone can be reused."""
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
    except OSError:
        return None
    return int(stat.rsplit(')', 1)[1].split()[19])


class BudgetPool:
    """A pool made by `create`; see the module docstring."""

    def __init__(self, directory):
        self.dir = Path(directory).resolve()
        self.state_path, self.log_path = self.dir / 'pool.json', self.dir / 'pool.jsonl'
        if not self.state_path.exists():
            raise FileNotFoundError(f'no budget pool in {self.dir} (BudgetPool.create makes one)')

    @classmethod
    def create(cls, directory, total_usd, *, note):
        total = _usd(total_usd, 'total_usd')
        if total <= 0:
            raise ValueError('total_usd must be positive')
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        state = {'schema': SCHEMA, 'total_usd': total, 'spent_usd': 0.0, 'calls': 0, 'by_who': {}, 'reservations': {},
                 'created_utc': _now(), 'note': str(note)}
        try:
            with open(directory / 'pool.json', 'x') as handle:
                handle.write(json.dumps(state, indent=1) + '\n')
        except FileExistsError:
            raise FileExistsError(f'{directory} already holds a budget pool; a pool is never replaced') from None
        pool = cls(directory)
        pool._log({'event': 'created', 'total_usd': total, 'note': str(note)})
        return pool

    # ------------------------------------------------------------------ state under the lock
    @contextmanager
    def _locked(self):
        with open(self.dir / 'pool.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                state = json.loads(self.state_path.read_text())
                if state.get('schema') != SCHEMA:
                    raise ValueError(f'{self.state_path} is not a {SCHEMA} pool')
                self._reap(state)
                try:
                    yield state  # the callers validate before they change anything, so a refusal saves the reaping only
                finally:
                    temp = self.state_path.with_name(f'pool.json.{uuid.uuid4().hex}.tmp')
                    temp.write_text(json.dumps(state, indent=1) + '\n')
                    os.replace(temp, self.state_path)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _log(self, record):
        with open(self.log_path, 'a') as handle:
            handle.write(json.dumps({'time_utc': _now(), **record}) + '\n')

    def _charge(self, state, who, usd):
        state['spent_usd'] += usd
        state['by_who'][who] = state['by_who'].get(who, 0.0) + usd

    def _reap(self, state):
        host = socket.gethostname()
        for key, held in list(state['reservations'].items()):
            if held['host'] == host and _process_start(held['pid']) != held['pid_start']:
                del state['reservations'][key]
                self._charge(state, held['who'], held['usd'])
                self._log({'event': 'settle', 'reservation': key, 'who': held['who'], 'usd': held['usd'],
                           'outcome': 'process_died', 'spent_usd': state['spent_usd']})

    # ------------------------------------------------------------------ the API
    def reserve(self, usd, *, who, what):
        """Hold `usd` (a call's worst case) for `who`; raises BudgetExhausted if it does not fit."""
        usd = _usd(usd, 'usd')
        with self._locked() as state:
            held = sum((r['usd'] for r in state['reservations'].values()), 0.0)
            left = state['total_usd'] - state['spent_usd'] - held
            if usd > left + 1e-12:
                self._log({'event': 'refused', 'who': who, 'what': what, 'usd': usd, 'left_usd': left})
                raise BudgetExhausted(f'the call could cost up to ${usd:.2f}; ${left:.2f} of the ${state["total_usd"]:.2f} '
                                      f'pool is left (${state["spent_usd"]:.2f} spent, ${held:.2f} held by calls in flight)')
            key = uuid.uuid4().hex
            state['reservations'][key] = {'who': str(who), 'what': str(what), 'usd': usd, 'pid': os.getpid(),
                                          'pid_start': _process_start(os.getpid()), 'host': socket.gethostname(),
                                          'time_utc': _now()}
            self._log({'event': 'reserve', 'reservation': key, 'who': who, 'what': what, 'usd': usd})
        return key

    def settle(self, reservation, usd, *, outcome, details=None):
        """Close a reservation at what the call cost; returns {'exceeded': whether that was more than was reserved}.
        A call dearer than its reservation is charged in full all the same (the money is spent): its caller must stop,
        since its bound was wrong."""
        usd = _usd(usd, 'usd')
        with self._locked() as state:
            held = state['reservations'].get(reservation)
            if held is None:
                raise KeyError(f'no open reservation {reservation}')
            exceeded = usd > held['usd'] + 1e-9
            del state['reservations'][reservation]
            self._charge(state, held['who'], usd)
            if outcome == 'call':
                state['calls'] += 1
            self._log({'event': 'settle', 'reservation': reservation, 'who': held['who'], 'usd': usd,
                       'reserved_usd': held['usd'], 'exceeded_reservation': exceeded, 'outcome': outcome,
                       'spent_usd': state['spent_usd'], **({'details': details} if details else {})})
        return {'exceeded': exceeded}

    def status(self):
        with self._locked() as state:
            held = sum((r['usd'] for r in state['reservations'].values()), 0.0)
            return {'total_usd': state['total_usd'], 'spent_usd': state['spent_usd'], 'reserved_usd': held,
                    'left_usd': state['total_usd'] - state['spent_usd'] - held, 'calls': state['calls'],
                    'by_who': dict(state['by_who']), 'open': len(state['reservations'])}
