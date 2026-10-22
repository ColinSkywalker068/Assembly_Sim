#!/usr/bin/python3
"""aspire-trial: run a program on the simulator through the campaign's trial service, the only way to run one here.

  aspire-trial sweep code/v00_initial.py           the program on every development seed
  aspire-trial probe code/probe_scene.py --seed 101   an observation program on one development seed
  aspire-trial unseen                              fix_code.py on the unseen-check seeds, once
  aspire-trial rerun v03                           the seeds of v03 the harness failed on (uses no budget)
  aspire-trial status                              what is used and what is left
  aspire-trial wait                                block until a running request has finished, then status

Program paths are relative to the current directory or absolute, inside the workspace. The command blocks until the
trials have finished and prints each event as a JSON line; it exits 0 when the request ran, 3 when the service refused
it (nothing ran) and 1 when the service could not be reached or failed. Standard library only: it runs inside the
agent's sandbox, where the service's socket is the only way out.
"""
import argparse
import json
import os
import socket
import sys

SOCKET = os.environ.get('ASPIRE_TRIAL_SOCKET', '/run/aspire/trials.sock')
WORKSPACE = os.environ.get('ASPIRE_WORKSPACE', '/workspace')
EXIT_CODES = {'done': 0, 'status': 0, 'refused': 3}


def workspace_relative(path):
    full = os.path.normpath(os.path.join(os.getcwd(), path))
    if not full.startswith(WORKSPACE.rstrip('/') + '/'):
        raise SystemExit(f'aspire-trial: {path} is not inside {WORKSPACE}')
    return full[len(WORKSPACE.rstrip('/')) + 1:]


def build_request(argv):
    parser = argparse.ArgumentParser(prog='aspire-trial', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='kind', required=True)
    commands.add_parser('sweep', help='a numbered version on every development seed').add_argument('program')
    probe = commands.add_parser('probe', help='an observation program on one development seed')
    probe.add_argument('program')
    probe.add_argument('--seed', type=int, required=True)
    commands.add_parser('unseen', help='fix_code.py on the unseen-check seeds, once')
    commands.add_parser('rerun', help="a request's seeds that failed in the harness").add_argument('label')
    commands.add_parser('status', help='what is used and what is left')
    commands.add_parser('wait', help='wait for a running request, then show the status')
    args = parser.parse_args(argv)
    request = {'kind': args.kind}
    if args.kind in ('sweep', 'probe'):
        request['program'] = workspace_relative(args.program)
    if args.kind == 'probe':
        request['seed'] = args.seed
    if args.kind == 'rerun':
        request['label'] = args.label
    return request


def main(argv=None):
    request = build_request(argv)
    final = None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(SOCKET)
            connection.sendall((json.dumps(request) + '\n').encode())
            for line in connection.makefile('r', encoding='utf-8'):
                print(line.rstrip('\n'), flush=True)
                final = json.loads(line)
    except OSError as exc:
        print(f'aspire-trial: the trial service at {SOCKET} failed: {exc}', file=sys.stderr)
        return 1
    return 1 if final is None else EXIT_CODES.get(final.get('event'), 1)


if __name__ == '__main__':
    sys.exit(main())
