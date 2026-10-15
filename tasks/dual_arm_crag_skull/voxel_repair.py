"""Explicit one-cell bridge repair, preserving every original voxel and ownership."""
OFFSETS = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))


def components(cells):
    remaining = set(cells)
    groups = []
    while remaining:
        start = min(remaining); remaining.remove(start)
        group = {start}; todo = [start]
        while todo:
            p = todo.pop()
            for d in OFFSETS:
                q = tuple(a+b for a,b in zip(p,d))
                if q in remaining:
                    remaining.remove(q); group.add(q); todo.append(q)
        groups.append(group)
    return groups


def bridge_one_cell(cells, forbidden):
    occupied = set(cells); added = []
    while True:
        groups = components(occupied)
        if len(groups) <= 1:
            return tuple(sorted(occupied)), added
        frontier = {}
        for index, group in enumerate(groups):
            for p in group:
                for d in OFFSETS:
                    q = tuple(a+b for a,b in zip(p,d))
                    if q not in occupied and q not in forbidden:
                        frontier.setdefault(q,set()).add(index)
        choices = [q for q,g in frontier.items() if len(g)>1]
        if not choices:
            raise ValueError('components cannot be joined by unoccupied one-cell bridges')
        q = min(choices, key=lambda q:(-len(frontier[q]),q))
        occupied.add(q); added.append(q)
