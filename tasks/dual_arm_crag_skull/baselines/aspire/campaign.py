"""One ASPIRE campaign on the skull task: the coding agent (Claude Code, headless) plays upstream's iterative-debugging
fix-loop subagent for this one task, in a sandbox.

`prepare` lays out the campaign directory:

  campaign.json            the settings (trial rules, GPUs, model, the prompt's sha256 and upstream's)
  prompt.md                prompts/fix_loop_subagent.md with the campaign's values filled in
  workspace/               the agent's TASK_DIR, /workspace in both sandboxes
  debug/                   the trial service's run directories and logs, /workspace/debug (read-only for the agent)
  reference/embodiment/    adapter.py, ik.py and embodiment.md (README sections), /aspire-embodiment read-only
  bin/aspire-trial         the trial client's launcher, /opt/aspire/bin in the agent's sandbox
  lib/                     what the agent's sandbox runs of ours, copied: the trial client, and relay.py with
                           budget.py in an empty package skeleton (the forwarder imports only those), /opt/aspire/lib
  agent_home/              the agent's HOME and Claude Code configuration (transcripts land here too)
  service/                 the trial service's ledger, frozen programs, trial records and Isaac caches

`run` starts the trial service (trials.py) and, for a live campaign, the billing relay (baselines/relay.py) on Unix
sockets in sockets/, then runs the agent in its sandbox (`agent_command`): no GPU, no network (a forwarder inside
turns 127.0.0.1:FORWARD_PORT into the relay's socket, the only way out besides the trial service's), no home
directory, the workspace and its own home writable, the patched ASPIRE checkout and the embodiment notes read-only.
The API key stays in this process (the relay adds it to each request); the agent gets a placeholder. Claude Code runs
with `--print` (not `--bare`, which cuts the tools to Bash, Edit and Read), only Bash, Read, Write, Edit, Glob and Grep
(upstream's subagent tools), every model slot set to
the campaign's model (the relay refuses any other) and `rm`/`sudo` denied. Its stream-json transcript goes to
transcript.jsonl. `summarize` then counts what the campaign used (sim trials from the service's ledger, cost and
calls from the relay's) and freezes fix_code.py for Stage 2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .. import sandbox
from ...workspace import REPO_ROOT
from . import trials

HERE = Path(__file__).resolve().parent
PROMPT = HERE / 'prompts' / 'fix_loop_subagent.md'
UPSTREAM_PROMPT = Path('.claude/skills/iterative-debugging/subagent-prompt.md')
UPSTREAM_PROMPT_SHA256 = 'ca78d55c8910d116f6ca0a02bc064c27d875a10e4981cba8fde96318581ac1db'
EMBODIMENT_SECTIONS = ('## Conventions', '## Every command goes through S0\'s checks')
AGENT_TOOLS = ('Bash', 'Read', 'Write', 'Edit', 'Glob', 'Grep')
DENIED = ('Bash(rm:*)', 'Bash(sudo:*)')
MODEL = 'claude-opus-5-5'
MODEL_VARIABLES = ('ANTHROPIC_MODEL', 'ANTHROPIC_DEFAULT_OPUS_MODEL', 'ANTHROPIC_DEFAULT_SONNET_MODEL',
                   'ANTHROPIC_DEFAULT_HAIKU_MODEL', 'ANTHROPIC_SMALL_FAST_MODEL', 'CLAUDE_CODE_SUBAGENT_MODEL')
FORWARD_PORT = 8787
PLACEHOLDER_KEY = 'sandbox-placeholder-the-relay-adds-the-key'
BASH_TIMEOUT_MS = 7_200_000
HELDOUT_COUNT = 100
INSIDE = {'workspace': '/workspace', 'home': '/home/agent', 'aspire': '/aspire', 'embodiment': '/aspire-embodiment',
          'claude': '/opt/claude-code', 'bin': '/opt/aspire/bin', 'tools': '/opt/aspire/lib',
          'prompt': '/opt/aspire/prompt.md',
          'trials_socket': '/run/aspire/trials.sock', 'relay_socket': '/run/aspire/relay.sock'}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def embodiment_notes(readme):
    """The README sections a program author needs (conventions and S0's checks), verbatim."""
    text, notes = Path(readme).read_text(), []
    for heading in EMBODIMENT_SECTIONS:
        start = text.index(heading + '\n')
        end = text.find('\n## ', start + len(heading))
        notes.append(text[start:end if end >= 0 else len(text)].rstrip() + '\n')
    return '# The dual-CRX skull embodiment (fanuc_crx)\n\n' + '\n'.join(notes)


def fill_prompt(template, settings):
    values = {'MAX_STEPS': settings['max_steps'], 'DEV_SEEDS': ' '.join(map(str, settings['dev_seeds'])),
              'UNSEEN_SEEDS': ' '.join(map(str, settings['unseen_seeds'])), 'HELDOUT_COUNT': HELDOUT_COUNT,
              'MAX_PROBES': settings['max_probes']}
    for name, value in values.items():
        template = template.replace(f'<<{name}>>', str(value))
    if '<<' in template:
        raise ValueError('the prompt template has a placeholder this campaign does not fill')
    return template


def claude_package(claude):
    """The installed Claude Code package directory (bin/claude.exe and what sits beside it) of the `claude` command."""
    executable = Path(claude).resolve()
    if executable.parent.name != 'bin':
        raise ValueError(f'{executable} is not in a package bin/ directory')
    return executable.parent.parent, executable.relative_to(executable.parent.parent)


def campaign_protocol(config):
    """The scaled protocol of configs/jing.json baselines.aspire.campaign: development seeds (practice seeds), the
    unseen-check seeds (a whole split that is not held out) and the budget."""
    from ..task import canonical_split
    spec = config['baselines']['aspire']['campaign']
    first, last = spec['dev_seeds']
    dev = list(range(first, last + 1))
    low, high = config['baselines']['seed_splits'][spec['unseen_split']]
    unseen = list(range(low, high + 1))
    if not dev or {canonical_split(seed) for seed in dev} != {'practice'}:
        raise ValueError('baselines.aspire.campaign.dev_seeds must be practice seeds')
    if not unseen or {canonical_split(seed) for seed in unseen} != {'selection'}:
        raise ValueError('baselines.aspire.campaign.unseen_split must be the selection seeds')
    budget = {name: spec[name] for name in ('max_fix_versions', 'max_probes', 'max_reruns')}
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in budget.values()):
        raise ValueError('the campaign budget must be whole numbers')
    return dev, unseen, budget


def prepare(root, *, task, gpus, isaac_python, claude, config, max_steps=18000, trial_timeout_s=3600,
            max_wall_s=16 * 3600, home=Path.home()):
    """Lay out a new campaign under `root` (which must not exist) and return its campaign.json content."""
    from ..task import AssemblyTask
    root = Path(root).absolute()
    if root.exists():
        raise FileExistsError(f'{root} exists; a campaign directory is made once')
    if len(os.fsencode(root / 'sockets' / 'trials.sock')) > trials.SOCKET_PATH_MAX:
        raise ValueError(f'{root} is too long a path for the Unix sockets in its sockets/ directory')
    if AssemblyTask.from_record(json.loads(Path(task).read_text())).seed is not None:
        raise ValueError('the campaign task is the base task (seed None)')
    dev, unseen, budget = campaign_protocol(config)
    aspire = config['baselines']['aspire']
    checkout = REPO_ROOT / aspire['checkout_path']
    if _sha256(checkout / UPSTREAM_PROMPT) != UPSTREAM_PROMPT_SHA256:
        raise ValueError(f'{UPSTREAM_PROMPT} in {checkout} is not the upstream prompt this adaptation was made from')
    claude_dir, executable = claude_package(claude)
    environment, kit_cache, _ = trials.kit_paths(isaac_python)
    for directory in ('workspace/debug', 'debug', 'reference/embodiment', 'bin', 'agent_home/.claude', 'service',
                      'sockets', 'lib/tasks/dual_arm_crag_skull/baselines'):
        (root / directory).mkdir(parents=True)
    settings = {'workspace': str(root / 'workspace'), 'debug': str(root / 'debug'),
                'task': str(Path(task).absolute()), 'dev_seeds': dev, 'unseen_seeds': unseen, **budget,
                'gpus': [int(gpu) for gpu in gpus], 'isaac_python': str(isaac_python),
                'checkout': aspire['checkout_path'], 'max_steps': max_steps, 'trial_timeout_s': trial_timeout_s,
                'kit_cache': str(root / 'service' / 'kit_cache')}
    shutil.copytree(kit_cache, settings['kit_cache'], symlinks=True)
    for cache in ('.cache/ov', '.nv/ComputeCache'):  # warm Isaac caches, copied: a trial never writes the originals
        if (Path(home) / cache).is_dir():
            shutil.copytree(Path(home) / cache, root / 'service' / 'isaac_home' / cache, symlinks=True)
    readme = HERE / 'README.md'
    (root / 'reference/embodiment/embodiment.md').write_text(embodiment_notes(readme))
    for name in ('adapter.py', 'ik.py'):
        shutil.copy2(HERE / name, root / 'reference/embodiment' / name)
    launcher = root / 'bin' / 'aspire-trial'
    launcher.write_text(f'#!/bin/sh\nexec /usr/bin/python3 -I {INSIDE["tools"]}/trial_client.py "$@"\n')
    launcher.chmod(0o755)
    package = root / 'lib' / 'tasks'
    for directory in (package, package / 'dual_arm_crag_skull', package / 'dual_arm_crag_skull' / 'baselines'):
        (directory / '__init__.py').write_text('')
    copies = {'trial_client.py': (HERE / 'trial_client.py', root / 'lib' / 'trial_client.py'),
              'relay.py': (HERE.parent / 'relay.py', package / 'dual_arm_crag_skull' / 'baselines' / 'relay.py'),
              'budget.py': (HERE.parent / 'budget.py', package / 'dual_arm_crag_skull' / 'baselines' / 'budget.py')}
    for source, copy in copies.values():
        shutil.copy2(source, copy)
    prompt = fill_prompt(PROMPT.read_text(), settings)
    (root / 'prompt.md').write_text(prompt)
    campaign = {'created': _now(), 'trials': settings,
                'agent': {'model': MODEL, 'claude_package': str(claude_dir), 'claude_executable': str(executable),
                          'claude_version': subprocess.run([str(Path(claude).resolve()), '--version'],
                                                           capture_output=True, text=True).stdout.strip(),
                          'tools': list(AGENT_TOOLS), 'denied': list(DENIED), 'max_wall_s': max_wall_s},
                'prompt': {'template': str(PROMPT.relative_to(REPO_ROOT)), 'sha256': hashlib.sha256(
                    prompt.encode()).hexdigest(), 'upstream': str(UPSTREAM_PROMPT),
                    'upstream_sha256': UPSTREAM_PROMPT_SHA256, 'upstream_commit': aspire['upstream_commit']},
                'isaac_environment': str(environment),
                'sandbox_copies': {name: _sha256(copy) for name, (_, copy) in copies.items()}}
    (root / 'campaign.json').write_text(json.dumps(campaign, indent=1) + '\n')
    return campaign


def agent_environment():
    """The agent's whole environment: nothing inherited, a placeholder key, every model slot on MODEL."""
    return {'PATH': f'{INSIDE["bin"]}:/usr/bin:/bin', 'HOME': INSIDE['home'],
            'CLAUDE_CONFIG_DIR': f'{INSIDE["home"]}/.claude', 'LANG': 'C.UTF-8', 'TERM': 'dumb',
            'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{FORWARD_PORT}', 'ANTHROPIC_API_KEY': PLACEHOLDER_KEY,
            **{name: MODEL for name in MODEL_VARIABLES},
            'DISABLE_AUTOUPDATER': '1', 'DISABLE_TELEMETRY': '1', 'DISABLE_ERROR_REPORTING': '1',
            'DISABLE_BUG_COMMAND': '1', 'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
            'BASH_DEFAULT_TIMEOUT_MS': str(BASH_TIMEOUT_MS), 'BASH_MAX_TIMEOUT_MS': str(BASH_TIMEOUT_MS),
            'ASPIRE_TRIAL_SOCKET': INSIDE['trials_socket'], 'ASPIRE_WORKSPACE': INSIDE['workspace']}


def claude_arguments(campaign):
    executable = Path(INSIDE['claude']) / campaign['agent']['claude_executable']
    return [str(executable), '--print', '--output-format', 'stream-json', '--verbose', '--model', MODEL,
            '--tools', ','.join(AGENT_TOOLS), '--allowedTools', *AGENT_TOOLS, '--disallowedTools', *DENIED,
            '--permission-mode', 'dontAsk', '--strict-mcp-config']


def agent_command(root, campaign, *, agent=None):
    """argv of the agent's sandbox. `agent` replaces Claude Code with another command (a scripted fake agent for a
    smoke test); the prompt arrives on standard input either way."""
    root = Path(root).absolute()
    checkout = REPO_ROOT / campaign['trials']['checkout']
    readonly = [(checkout, Path(INSIDE['aspire'])), (root / 'reference' / 'embodiment', Path(INSIDE['embodiment'])),
                (root / 'bin', Path(INSIDE['bin'])), (root / 'lib', Path(INSIDE['tools'])),
                (Path(campaign['agent']['claude_package']), Path(INSIDE['claude'])),
                (root / 'prompt.md', Path(INSIDE['prompt']))]
    writable = [(root / 'workspace', Path(INSIDE['workspace'])), (root / 'agent_home', Path(INSIDE['home']))]
    sockets = [(root / 'sockets' / 'trials.sock', Path(INSIDE['trials_socket']))]
    if (root / 'sockets' / 'relay.sock').exists():
        sockets.append((root / 'sockets' / 'relay.sock', Path(INSIDE['relay_socket'])))
    shadows = [(root / 'debug', Path(INSIDE['workspace']) / 'debug')]  # the service's: never a path the agent made
    prefix = sandbox.bwrap_prefix(writable=writable, readonly=readonly, cwd=INSIDE['workspace'], sockets=sockets,
                                  shadows=shadows)
    inner = agent or claude_arguments(campaign)
    forwarder = (f"import sys; sys.path.insert(0, '{INSIDE['tools']}'); "
                 'from tasks.dual_arm_crag_skull.baselines.relay import main; '
                 f"main(['forward', '--port', '{FORWARD_PORT}', '--socket', '{INSIDE['relay_socket']}'])")
    forward = (f'/usr/bin/python3 -I -c "{forwarder}" > {INSIDE["home"]}/forward.log 2>&1 & '
               f'for attempt in $(seq 100); do (exec 3<>/dev/tcp/127.0.0.1/{FORWARD_PORT}) 2>/dev/null && break; '
               'sleep 0.1; done; ') if len(sockets) == 2 else ''
    script = forward + f'exec "$@" < {INSIDE["prompt"]}'
    return [*prefix, '/bin/bash', '-c', script, 'agent', *inner]


def _start_thread(target, *args, **kwargs):
    """Run the server `target` in a daemon thread; return once it is ready, or raise what stopped it."""
    ready, failed = threading.Event(), []

    def body():
        try:
            target(*args, **kwargs, ready=ready)
        except BaseException as exc:
            failed.append(exc)
            ready.set()

    thread = threading.Thread(target=body, daemon=True)
    thread.start()
    if not ready.wait(30):
        raise RuntimeError(f'{getattr(target, "__name__", target)} did not start within 30 s')
    if failed:
        raise RuntimeError(f'{getattr(target, "__name__", target)} failed to start') from failed[0]
    return thread


def run(root, *, live, agent=None, launch=None, pool_dir=None, upstream=None):
    """Run the campaign at `root` (prepared): the trial service, the relay when `live` (it takes the API key out of
    this process's ANTHROPIC_API_KEY; `pool_dir` defaults to configs/jing.json's pool, `upstream` to the API), and the
    agent in its sandbox until it exits or campaign.json's max_wall_s passes. Returns summarize(root)."""
    root = Path(root).absolute()
    campaign = json.loads((root / 'campaign.json').read_text())
    if (root / 'transcript.jsonl').exists():
        raise FileExistsError(f'{root} has run; a campaign runs once')
    if agent is None and not live:
        raise ValueError('Claude Code needs the relay: run it live, or give a scripted agent')
    relay_server = None
    if live:  # first: the relay takes the key out of the environment before anything else starts
        from .. import relay
        from ..budget import configured_pool_dir
        from ...__main__ import DEFAULT_CONFIG
        pool_dir = pool_dir or configured_pool_dir(json.loads(DEFAULT_CONFIG.read_text()))
        relay_server = relay.serve(pool_dir, 'aspire', socket_path=root / 'sockets' / 'relay.sock',
                                   ledger_path=root / 'relay' / 'ledger.jsonl', models=(MODEL,),
                                   **({'upstream': upstream} if upstream else {}))
    elif os.environ.get('ANTHROPIC_API_KEY'):
        raise ValueError('ANTHROPIC_API_KEY is set but the campaign is not live; unset it')
    service = trials.TrialService(root, campaign['trials'], launch=launch)
    _start_thread(trials.serve, root / 'sockets' / 'trials.sock', service)
    command = agent_command(root, campaign, agent=agent)
    started = time.monotonic()
    with open(root / 'transcript.jsonl', 'wb') as transcript, open(root / 'agent.stderr.log', 'wb') as errors:
        process = subprocess.Popen(command, stdout=transcript, stderr=errors, stdin=subprocess.DEVNULL,
                                   env=agent_environment(), start_new_session=True)
        try:
            code, _, killed = trials.watch(process, timeout_s=float(campaign['agent']['max_wall_s']))
        finally:
            if relay_server is not None:
                relay_server.close()
    (root / 'agent_exit.json').write_text(json.dumps({'exit_code': code, 'killed': killed,
                                                      'wall_s': time.monotonic() - started, 'ended': _now()}) + '\n')
    return summarize(root)


def summarize(root):
    """What the campaign used and produced, from the service's and the relay's ledgers; freezes fix_code.py."""
    root = Path(root)
    rows = trials.ledger_rows(root / 'service' / 'ledger.jsonl')
    ran = [row for row in rows if row['status'] == 'ran']
    passed = {}
    debug = Path(json.loads((root / 'campaign.json').read_text())['trials']['debug'])
    for row in ran:  # a rerun's results are its request's (one row per seed)
        results = trials.ledger_rows(debug / Path(row['output']).relative_to('debug') / 'results.jsonl')
        passed[row['label']] = [sum(bool(line.get('task_success')) for line in results), len(results)]
    frozen = None
    try:
        data = trials.read_regular(root / 'workspace', ('fix_code.py',))
    except trials.Refused:
        data = None
    if data is not None:
        frozen = root / 'frozen' / 'fix_code.py'
        frozen.parent.mkdir(exist_ok=True)
        frozen.write_bytes(data)
    relay_ledger = trials.ledger_rows(root / 'relay' / 'ledger.jsonl')
    made = [call for call in relay_ledger if call.get('path') == '/v1/messages' and call.get('status') == 200]
    summary = {'sim_trials': sum(len(row['trials']) for row in ran),
               'versions': [row['label'] for row in ran if row['kind'] == 'sweep'],
               'probes': len([row for row in ran if row['kind'] == 'probe']),
               'refused_requests': len([row for row in rows if row['status'] == 'refused']),
               'passed': passed, 'fix_code_sha256': None if frozen is None else _sha256(frozen),
               'selected': next((row['label'] for row in ran if frozen is not None and row['kind'] == 'sweep'
                                 and row['sha256'] == _sha256(frozen)), None),
               'llm_calls': len(made), 'refused_calls': len([call for call in relay_ledger
                                                             if call.get('outcome') == 'refused']),
               'cost_usd': round(sum(float(call.get('usd') or 0) for call in relay_ledger), 6),
               'agent': json.loads((root / 'agent_exit.json').read_text()) if (root / 'agent_exit.json').exists()
               else None}
    (root / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    return summary


def main(argv=None):
    from ...__main__ import DEFAULT_CONFIG, validate
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    make = commands.add_parser('prepare', help='lay out a new campaign directory')
    make.add_argument('--root', type=Path, required=True)
    make.add_argument('--task', type=Path, required=True, help='the base task record (seed None)')
    make.add_argument('--gpus', required=True, help='GPU indices for trials, e.g. 1,2,5')
    make.add_argument('--isaac-python', type=Path, required=True)
    make.add_argument('--claude', type=Path, default=Path(shutil.which('claude') or 'claude'))
    make.add_argument('--max-steps', type=int, default=18000)
    go = commands.add_parser('run', help='run a prepared campaign')
    go.add_argument('--root', type=Path, required=True)
    go.add_argument('--live', action='store_true', help='Claude Code through the relay (needs ANTHROPIC_API_KEY)')
    go.add_argument('--agent', nargs=argparse.REMAINDER, help='a scripted agent command instead (no model calls)')
    show = commands.add_parser('summarize', help="recount a campaign's use")
    show.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        config = json.loads(DEFAULT_CONFIG.read_text())
        validate(config)
        campaign = prepare(args.root, task=args.task, gpus=args.gpus.split(','), isaac_python=args.isaac_python,
                           claude=args.claude, config=config, max_steps=args.max_steps)
        print(json.dumps({'root': str(args.root), 'gpus': campaign['trials']['gpus'],
                          'prompt_sha256': campaign['prompt']['sha256']}))
        return 0
    if args.command == 'run':
        summary = run(args.root, live=args.live, agent=args.agent or None)
    else:
        summary = summarize(args.root)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
