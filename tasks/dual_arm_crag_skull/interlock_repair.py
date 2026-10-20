"""Explicit ownership repair so the fragments can be put together by straight moves, keeping every voxel.

On a coarse grid neighbouring fragments hook into each other by a cube (experiments/assembly_paths.py): with the
anchor fixed, some cannot leave, nor so be put back, along any straight line. A fragment leaves along an axis
direction d exactly when, on every line of the grid along d, the cells it owns come after every other present
fragment's cells. `straighten` takes the fragments out one at a time: for each it finds, line by line, the fewest
cells that change owner to make that so (it takes the other fragments' cells ahead of its last one, or gives its
cells behind to the nearest other owner on the line), takes out the fragment whose changes split no fragment into
more pieces and are fewest (straight up first on a tie), and repeats. The union of cells is unchanged; only who owns a cell at a seam changes.
"""
from __future__ import annotations

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


def straighten(owned, anchor):
    """(cells of each fragment after the repair, report) for `owned` {name: cells} with `anchor` fixed in place."""
    if anchor not in owned:
        raise ValueError(f'the anchor {anchor} is not one of the fragments')
    owner_of = {tuple(int(v) for v in cell): name for name, cells in owned.items() for cell in cells}
    if len(owner_of) != sum(len(cells) for cells in owned.values()):
        raise ValueError('a cell is owned by two fragments')
    present, order = set(owned), []
    while len(present) > 1:
        candidates = []
        for name in sorted(present - {anchor}):
            size = sum(owner == name for owner in owner_of.values())
            for direction in AXES:
                flips = exit_flips(owner_of, name, present, direction)
                if sum(owner_of[c] == name and new != name for c, new in flips.items()) < size:  # keep a cell
                    touched = {owner_of[c] for c in flips} | set(flips.values())
                    before, after = pieces(owner_of, {}, touched), pieces(owner_of, flips, touched)
                    split = sum(max(after[n] - before[n], 0) for n in touched)  # one fragment's cut is not undone
                    candidates.append((split, len(flips), direction != AXES[0], name, direction, flips))
        if not candidates:
            raise ValueError(f'no straight exit keeps a cell of any of {sorted(present - {anchor})}')
        _, count, _, name, direction, flips = min(candidates, key=lambda c: c[:4])
        owner_of.update(flips)
        order.append({'fragment': name, 'direction': list(direction), 'flipped': count})
        present.remove(name)
    result = {name: [] for name in owned}
    for cell, owner in owner_of.items():
        result[owner].append(cell)
    result = {name: tuple(sorted(cells)) for name, cells in result.items()}
    changed = {}
    for name, cells in owned.items():
        before, after = {tuple(int(v) for v in c) for c in cells}, set(result[name])
        if before != after:
            changed[name] = {'lost': len(before - after), 'gained': len(after - before)}
    report = {'anchor': anchor, 'flipped': sum(step['flipped'] for step in order), 'order': order, 'changed': changed}
    return result, report
