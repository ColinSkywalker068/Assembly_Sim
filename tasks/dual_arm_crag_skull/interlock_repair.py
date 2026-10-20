"""Explicit ownership repair so the fragments can be put together by straight moves, keeping every voxel.

On a coarse grid neighbouring fragments hook into each other by a cube (experiments/assembly_paths.py): with the
anchor fixed, some cannot leave, nor so be put back, along any straight line. A fragment leaves along an axis
direction d exactly when, on every line of the grid along d, the cells it owns come after every other present
fragment's cells. `straighten` takes the fragments out one at a time: for each it finds, line by line, the fewest
cells that change owner to make that so (it takes the other fragments' cells ahead of its last one, or gives its
cells behind to the nearest other owner on the line), takes out the fragment whose changes split no fragment into
more pieces and are fewest (straight up first on a tie), and repeats. That rule is short-sighted about the first one
out (the last one in, often a cover over the rest), so every first exit is tried and finished this way, and the
cheapest plan in all is kept. Last, a stray cube or piece of a moving
fragment (not its largest) goes to a neighbour, if one can take it without blocking that order. The union of cells is unchanged; only who owns a cell at a seam changes.

The report's `plan` is the assembly that results, put together: the fragments in the order they go in, each sliding
in along the reverse of its `exit_direction` with the anchor and the fragments before it in place. By default it is
the repair's own order reversed; an `insertion_order` given instead keeps the repair's directions and is refused
unless every fragment goes in straight in it (several orders often do, and only some stand up under gravity). The
stray-cube handover then keeps the given order clear, so its cells can differ from the default's, and the report's
`order` records how the seams were cut rather than an order that still goes in straight.
"""
from __future__ import annotations

import itertools
from collections import defaultdict

from .voxel_repair import components

AXES = ((0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))  # never down: the assembly stands on the table


def line_flips(labels, name):
    """For one line's owners in order along d, {index: new owner} with the fewest changes that leave `name` owning
    exactly a suffix; the earliest such suffix on a tie (so a cell behind always has another owner to go to)."""
    m = len(labels)
    costs = [sum(label == name for label in labels[:s]) + sum(label != name for label in labels[s:]) for s in range(m + 1)]
    s = costs.index(min(costs))
    others = [i for i in range(s) if labels[i] != name]
    flips = {i: name for i in range(s, m) if labels[i] != name}
    for i in range(s):
        if labels[i] == name:
            flips[i] = labels[min(others, key=lambda j: (abs(j - i), -j))]
    return flips


def exit_flips(owner_of, name, present, direction):
    """{cell: new owner} that lets `name` leave the fragments `present` along the axis `direction`."""
    axis = next(a for a, v in enumerate(direction) if v)
    sign = direction[axis]
    lines = defaultdict(list)
    for cell, owner in owner_of.items():
        if owner in present:
            lines[cell[:axis] + cell[axis + 1:]].append(cell)
    flips = {}
    for cells in lines.values():
        ordered = sorted(cells, key=lambda c: sign * c[axis])
        labels = [owner_of[c] for c in ordered]
        if name in labels:
            flips.update({ordered[i]: owner for i, owner in line_flips(labels, name).items()})
    return flips


def pieces(owner_of, flips, names):
    """{name: how many six-connected pieces it would be in after `flips`} for the fragments `names`."""
    cells = defaultdict(set)
    for cell, owner in owner_of.items():
        owner = flips.get(cell, owner)
        if owner in names:
            cells[owner].add(cell)
    return {name: len(components(cells[name])) for name in names}


FACES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def blocked(owner_of, order, anchor):
    """The first fragment of `order` that cannot leave, in that order and direction, without a change; None when all
    leave and only the anchor stays."""
    present = set(owner_of.values())
    for step in order:
        if exit_flips(owner_of, step['fragment'], present, step['direction']):
            return step['fragment']
        present.discard(step['fragment'])
    return None if present <= {anchor} else sorted(present - {anchor})[0]


def clear(owner_of, order, anchor):
    """Whether every fragment of `order` still leaves, in that order and direction, without a change."""
    return blocked(owner_of, order, anchor) is None


def insertion_orders(owner_of, directions, anchor):
    """Every order, sorted, in which the fragments of `directions` {name: exit direction} go in straight, each along
    the reverse of its direction with the anchor and the ones before it in place."""
    return sorted(tuple(reversed(removal)) for removal in itertools.permutations(sorted(directions))
                  if clear(owner_of, [{'fragment': name, 'direction': directions[name]} for name in removal], anchor))


def plan(owner_of, order, anchor, insertion_order=None):
    """[{'fragment', 'exit_direction'}] in the order the fragments go in: `insertion_order` (names, without the anchor)
    with the directions of the repair's `order`, or by default that order reversed."""
    directions = {step['fragment']: list(step['direction']) for step in order}
    names = [step['fragment'] for step in reversed(order)] if insertion_order is None else list(insertion_order)
    if sorted(names) != sorted(directions):
        raise ValueError(f'the assembly order must name every fragment but the anchor once: {sorted(directions)}')
    stuck = blocked(owner_of, [{'fragment': name, 'direction': directions[name]} for name in reversed(names)], anchor)
    if stuck is not None:
        raise ValueError(f'{stuck!r} cannot go in straight along {[-v for v in directions[stuck]]} in the order '
                         f'{names}')
    return [{'fragment': name, 'exit_direction': directions[name]} for name in names]


def merge_islands(owner_of, order, anchor):
    """Hand each piece of a moving fragment other than its largest (stray cubes the voxelization left) to the
    neighbour it touches most that keeps `order` clear; a piece no neighbour can take stays. The anchor keeps its
    pieces: the fixture holds them all. Returns what moved."""
    moved = []
    for name in sorted(set(owner_of.values()) - {anchor}):
        groups = sorted(components({c for c, o in owner_of.items() if o == name}), key=lambda g: (-len(g), min(g)))
        for group in groups[1:]:
            touching = defaultdict(int)
            for cell in group:
                for face in FACES:
                    other = owner_of.get(tuple(a + b for a, b in zip(cell, face)))
                    if other is not None and other != name:
                        touching[other] += 1
            for other in sorted(touching, key=lambda o: (-touching[o], o)):
                trial = {**owner_of, **{cell: other for cell in group}}
                if clear(trial, order, anchor):
                    owner_of.update({cell: other for cell in group})
                    moved.append({'fragment': name, 'cells': len(group), 'to': other})
                    break
    return moved


def exits(owner_of, present, anchor):
    """Every straight exit of a fragment of `present` that keeps one of its cells, cheapest first:
    (fragments it cuts into more pieces, cells that change owner, not straight up, name, direction, {cell: owner})."""
    found = []
    for name in sorted(present - {anchor}):
        size = sum(owner == name for owner in owner_of.values())
        for direction in AXES:
            flips = exit_flips(owner_of, name, present, direction)
            if sum(owner_of[c] == name and new != name for c, new in flips.items()) < size:  # keep a cell
                touched = {owner_of[c] for c in flips} | set(flips.values())
                before, after = pieces(owner_of, {}, touched), pieces(owner_of, flips, touched)
                split = sum(max(after[n] - before[n], 0) for n in touched)  # one fragment's cut is not undone
                found.append((split, len(flips), direction != AXES[0], name, direction, flips))
    return sorted(found, key=lambda e: e[:4])


def take_out(owner_of, present, anchor, first):
    """(owners after, exits taken) taking out `first` and then, one at a time, the cheapest exit left."""
    owner_of, present, steps = dict(owner_of), set(present), []
    step = first
    while True:
        owner_of.update(step[5])
        steps.append(step)
        present.discard(step[3])
        if len(present) <= 1:
            return owner_of, steps
        options = exits(owner_of, present, anchor)
        if not options:
            raise ValueError(f'no straight exit keeps a cell of any of {sorted(present - {anchor})}')
        step = options[0]


def straighten(owned, anchor, insertion_order=None):
    """(cells of each fragment after the repair, report) for `owned` {name: cells} with `anchor` fixed in place; the
    report's plan follows `insertion_order` when given (module docstring)."""
    if anchor not in owned:
        raise ValueError(f'the anchor {anchor} is not one of the fragments')
    owner_of = {tuple(int(v) for v in cell): name for name, cells in owned.items() for cell in cells}
    if len(owner_of) != sum(len(cells) for cells in owned.values()):
        raise ValueError('a cell is owned by two fragments')
    plans = [] if len(owned) > 1 else [(owner_of, [])]
    for first in exits(owner_of, set(owned), anchor):
        try:
            plans.append(take_out(owner_of, set(owned), anchor, first))
        except ValueError:
            continue
    if not plans:
        raise ValueError(f'no straight exit keeps a cell of any of {sorted(set(owned) - {anchor})}')
    owner_of, steps = min(plans, key=lambda plan: tuple(sum(step[k] for step in plan[1]) for k in range(3)))
    order = [{'fragment': name, 'direction': list(direction), 'flipped': count}
             for _, count, _, name, direction, _ in steps]
    steps_in = plan(owner_of, order, anchor, insertion_order)
    islands = merge_islands(owner_of, [{'fragment': step['fragment'], 'direction': step['exit_direction']}
                                       for step in reversed(steps_in)], anchor)
    result = {name: [] for name in owned}
    for cell, owner in owner_of.items():
        result[owner].append(cell)
    result = {name: tuple(sorted(cells)) for name, cells in result.items()}
    changed = {}
    for name, cells in owned.items():
        before, after = {tuple(int(v) for v in c) for c in cells}, set(result[name])
        if before != after:
            changed[name] = {'lost': len(before - after), 'gained': len(after - before)}
    report = {'anchor': anchor, 'flipped': sum(step['flipped'] for step in order), 'order': order, 'islands': islands,
              'changed': changed, 'plan': steps_in, 'plan_source': 'repair' if insertion_order is None else 'given'}
    return result, report
