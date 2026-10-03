# Architecture

The dependency direction is deliberately one-way:

```text
external dataset -> task preprocessing -> task metadata
                                      + core preset
                                             |
                                             v
                              core workcell + task additions
                                             |
                                             v
                                   external experiment output
```

`core` is a stable workcell platform. It owns the environment, robot models,
canonical robot placements, physics defaults, camera mounts, robot controls,
and generic scene lifecycle. It never imports a task.

Each package under `tasks/` has a distinct purpose and is its own composition
root. A task may import `core`, but tasks do not import one another. Dataset
adapters, object representations, staging policies, choreography, and other
task behavior stay local by default. A utility is promoted to `core` only after
it represents a stable task-independent contract.

Generated assembly metadata references processed mesh files relative to its
own `layout.json`, and records a named core preset rather than copying the
workcell configuration. Consequently, a processed assembly can be relocated
and loaded without consulting the raw dataset.

Repository source lives only in `core/`, `tasks/`, and `docs/`. Workspace data
uses `/local_data/yz11445/datasets`, experiment results use
`/local_data/yz11445/experiments`, and disposable intermediates use
`/local_data/yz11445/scratch`.
