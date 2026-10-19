# Baselines: DexAgent and ASPIRE against S2/S1/S0

Status (2026-10-19): integration only, **no baseline results**.
- One hardware set for every method (user, 2026-10-19): two CRX-10iA/L with the YAM linear_4310, on Isaac Sim.
  - ASPIRE is on it: YAM rigs at S0's 60 Hz, `--backend isaac` (`sim/backend.py`), contact declarations through
    `fanuc_crx.declare_contact`. A hand-written reference program ran end to end on Isaac (not a baseline result: it
    grasped and carried a fragment, then jammed lowering it straight onto the fixed anchor).
  - DexAgent is being ported (user, 2026-10-19; `dexagent/README.md`, "Port to the shared hardware"): it plans and
    verifies in MuJoCo on a CRX + YAM bench built from the same robot models (`--bench crx_yam`, a code patch), S0
    checks each delivery in its gate suite, and the delivery runs on `AssemblyBackend` through S0 (`execute`). A
    scripted delivery (not an agent run) ran on Isaac through S0. Left: DexAgent's tool library on the bench, the
    MuJoCo <-> Isaac calibration, a mock end-to-end trial. Its native YAM + Sharpa bench is unchanged.
- Neither baseline has made a model API call; ASPIRE's coding-agent loop has not run.

Live runs execute model-written code. Decided 2026-10-19: they run in a bwrap sandbox (`sandbox.py`, below) and pay
from one model budget for both baselines together (`budget.py`: $100, no per-baseline or per-trial amount, failed and
retried calls count, nothing tops it up); first one practice seed per baseline, held-out seeds stay closed.

| `method` | What it is | Here | Status |
|---|---|---|---|
| `s2_s1_s0` | ours | executor over `AssemblyBackend` not written yet | TODO P6 |
| `aspire` | [NVlabs/ASPIRE](https://github.com/NVlabs/ASPIRE) `fa00f0e`, Apache-2.0: a coding agent writes and repairs one program per task against a typed robot API, then the frozen program is evaluated | [`aspire/`](aspire/README.md): patched checkout, CRX `EnvironmentAdapter` over `AssemblyBackend`, S0 checks on every command | adapter and frozen-program runner tested offline; loop not run |
| `dexagent` | [dexagent1/DexAgent](https://github.com/dexagent1/DexAgent) `e29bdad`, MIT: Human2Sim2Robot; an LLM agent with a self-extending tool library plans and verifies trajectories in MuJoCo | [`dexagent/`](dexagent/README.md): Stage 3 only, inputs constructed from the task, our model transport; unmodified on its native bench, or ported (`--bench crx_yam`: CRX + YAM bench, delivery executed through S0 on `AssemblyBackend`) | export and Stage 3 pass DexAgent's gates with a mock LLM; port in progress; no live trial |
| `scripted` | hand-written programs | tests only | never reported |

## Common layer

An independent review of this layer (2026-10-18) found 4 HIGH and 6 MEDIUM issues; all are fixed with tests. Of the
review's surviving mutations and the new behaviour, 54 mutations are now caught.

- **Task** (`task.py`)
  - `AssemblyTask` holds:
    - fragment meshes and their sha256;
    - start and goal `world_T_fragment`;
    - the anchor;
    - `goal_source`;
    - the table, pad and exclusion zones;
    - the success criterion (tolerances and `settle_s`);
    - the seed and split.
  - `task_from_layout` builds it from a schema-2 layout and checks the geometry:
    - meshes inside the layout directory;
    - each goal mesh over the pad and not below the table;
    - pad and exclusions on the table;
    - staged footprints on the table, off the pad and apart.
  - The default task id carries the layout's sha256 prefix, so two layouts with the same object id get different ids.
  - `initial_condition(task, seed)` shifts (≤ 3 cm) and yaws (≤ 30°) each staged fragment, keeping footprints:
    - `gap_m` apart, the anchor included;
    - on the table with that margin;
    - off the pad (touching allowed);
    - `gap_m` from the robot exclusions.

    It raises instead of breaking a constraint.
- **One protocol.**
  - The protocol lives in `configs/jing.json`:
    - seed bands;
    - placement parameters;
    - the success criterion (`s2.completion_defaults.place`, a `released_stable` predicate: 3 mm / 3° after `settle_s` = 1 s);
    - the held-out claims directory.
  - `require_canonical(config)` refuses any other configuration.
  - `verify_task(task)` rebuilds the task from its layout and refuses one that differs in any way: tolerance, placement, split label, layout or meshes.
  - Both runners verify every task before a trial. Seed bands are provisional; fix them before any method sees a held-out seed.

  | Band | Seeds | Use |
  |---|---|---|
  | practice | 101–125 | a method's own development or learning |
  | selection | 126–130 | choosing a variant |
  | held-out | 1–100 | evaluating each frozen variant once |

- **Score** (`scoring.py`)
  - `score_assembly(task, final_poses, settled=...)` takes the simulator's privileged final poses. Each moving fragment is judged relative to the anchor's final pose, and the anchor itself must also stay within tolerance of its goal.
  - These count as failures:
    - a missing pose;
    - a non-finite or non-rigid pose (recorded in `invalid_ids`);
    - poses not read after release and settling.
  - A task whose goals are GARF candidates is refused, because scoring needs the true assembly; carrying both sets of goals is a TODO.
  - Each score records what it was judged against: layout sha256, start-pose digest, goal source and tolerances.
- **Records**
  - `RunRecord` holds:
    - method, variant, task, seed, and the split, which must be the seed's canonical band;
    - `backend` and `test_only` (physics-free double, mock model);
    - status;
    - the score;
    - `cost_usd`, `llm_calls`, `wall_s` and `sim_trials`.
  - `summarize` gives each (method, variant, task, backend, test_only, split) group its success rate with a Wilson 95 % interval. It refuses:
    - test-only runs, unless asked;
    - a held-out seed recorded twice;
    - a stored success that its score contradicts;
    - a non-canonical split;
    - a group judged against different layouts or tolerances.
- **Held-out seeds are used once.**
  - `claim_heldout` creates a claim file exclusively in `baselines.heldout_claims_dir`, before the trial touches anything. A second claim fails, whatever records file or process asks.
  - A trial that crashes or is interrupted keeps its claim, and the runner records it as `error`.
  - `summarize` lists claims that have no record among the files it was given.
  - A new variant (for example an edited ASPIRE program) may claim the same seed again; every such claim stays visible.
- **Backend** (`backend.py`)
  - `AssemblyBackend` is the simulator boundary:
    - `reset(task)` and `robot_state()`, which is validated: finite clock, joints and widths, rigid TCPs;
    - `observed_fragment_poses()`, each pose with its source and `time_s`;
    - `camera_frames()`, each frame with its sample time and rectified intrinsics;
    - `step(joints, widths)` for one control period;
    - `privileged_fragment_poses()`, used for scoring only.
  - **End of every trial:** open both grippers fully and hold the arms for `settle_s`, through the usual command checks, then score. A fragment still held or still sliding is judged where it comes to rest.
  - `KinematicBackend` is a physics-free test double. Its results are never reported.
- **S0 check for direct joint commands.** `s0.adapter.check_joint_path` reads joint targets as reached at their times with constant acceleration from the measured state:
  - the first segment needs `2 (v1 − qd_now) / dt`;
  - later knots use the second difference over the mean spacing;
  - the arm must be able to brake inside the joint limits after the last target;
  - arm-arm clearance, link and tool-box clearance above the table, and TCP height are checked as for chunks.

  It rejects a non-finite or malformed measured state or clock. It never clips or re-times.
- **CLI**
  - `python -m tasks.dual_arm_crag_skull.baselines task --layout L --anchor ID [--seed N [--allow-heldout]] --output T.json`
  - `... summarize RUNS.jsonl... [--include-test-only]`

  The real 8-fragment skull tasks, regenerated under the protocol, are in `runs/baseline_tasks/`: the base task and seeds 101 and 1. The task id is `view_gt-78b9204e3da1-anchor-fragment_45266904a743c03b`; all 130 seeds place in 0.6 s.

## Fairness protocol

1. **Same task and information.** Every method gets:
   - the same goals;
   - the same observed fragment poses, with their source and time;
   - the camera frames.

   No method gets privileged poses, its seed or its split. Compare methods only with the same goal source and pose source; report them with every result. Today these are `oracle_layout` goals and `oracle_sim` poses.
2. **Same execution checks.**
   - ASPIRE: every joint command passes `check_joint_path`, plus geometry checks between control periods.
   - Ours: S1 chunks pass S0.
   - DexAgent on its native bench executes on its own robot and physics, so it cannot be held to our S0 limits (a confound below). The port is: its gate suite runs S0 on each delivery, and the delivery runs through S0 in chunks of at most 1 s, with contact windows declared from its phase labels.
3. **Same scorer, tolerances and ending.**
   - ASPIRE: `check_success` releases, settles for 1 s through S0, and scores the backend's privileged poses.
   - DexAgent on its native bench: its accepted delivery has passed its own `release_settle` gate (0.3 s settle, 4 mm / 10° by default). The last recorded frame is mapped back to world and scored at our 3 mm / 3°.
   - The DexAgent port: after its delivery, completed or rejected by S0, the same ending as ASPIRE's (open, settle 1 s through S0, score the backend's privileged poses).
4. **Seeds.**
   - Held-out trials are claimed first, must be written to a records file, and run once per variant and task.
   - Practice and selection seeds are free.
5. **Costs are part of the result.** Report per run:
   - dollars and model calls (from the transport's ledger);
   - simulator episodes the method ran for itself;
   - wall time.

## Confounds that remain (reported with any comparison)

- **DexAgent runs on its own robot.**
  - It uses two YAM arms with five-finger Sharpa hands, versus our CRX arms with parallel grippers.
  - Its physics is MuJoCo 3.10, not Isaac/PhysX.
  - It publishes only after its whole gate suite passes; our score is pose error only.
  - It starts from a stand-in video without hands.
  - The port (variant `crx_yam_stage3`) is listed separately. It removes the robot difference; MuJoCo planning and
    open-loop execution on Isaac remain (first paired probe: the carried fragment ends 0.49 mm / 0.08° from where
    MuJoCo put it; more to do).
- **A fixture holds the anchor at its goal for every method** (`baselines.anchor = "fixture"` in `configs/jing.json`, part of the protocol `require_canonical` checks; decided 2026-10-18). Alone it is not statically stable (below).
  - `KinematicBackend` never lets a gripper take the anchor.
  - A DexAgent trial refuses `anchor_mode='free'`; `export` can still write a free anchor for diagnostics.
  - ASPIRE's task context says the anchor is held by a fixture (`anchor_fixture`).
  - The Isaac backend still has to implement it (a kinematic anchor body).
- **Settling differs.** DexAgent's own release check settles for 0.3 s at 4 mm / 10°. Ours scores after 1 s at 3 mm / 3°.
- **The LLM differs.**
  - ASPIRE's loop is meant to run under Claude Code.
  - DexAgent runs through our transport: Claude Opus 5.5, effort `high`, thinking kept, a hard USD cap.
  - Our S2 uses its own client.

## Blockers for live comparisons (TODO P6)

- **S0 values and the gripper are provisional.** `configs/jing.json` has no S0 values yet (acceleration limit,
  clearances, contact phase), so every record is `test_only`; the YAM adapter plate and its finger drive (20 N,
  0.1 m/s) are provisional.
- **No protective stop on Isaac.** The CRX drives push without a force limit: a held fragment lowered onto the fixed
  anchor jammed, the joints oscillated at 1–5 rad/s, and on release joint 5 reached 16.7 rad/s. The real CRX would
  stop on the collision. One stop rule (tracking error or joint effort) must hold for every method before physics
  runs.
- **DexAgent's port is not finished** (above): its tool library on the bench, the MuJoCo <-> Isaac calibration, a
  mock end-to-end trial.
- **One step budget** for every method (ASPIRE's default 18,000 control steps is provisional).
- **S0 reads a path's first segment differently from its later knots.** `check_joint_path` takes the measured
  `qd_now` as instantaneous and needs 2 (v1 - qd_now) / dt for the first segment, the velocity change over the
  period at later knots. Every simulated backend reports the velocity of the period just commanded: the kinematic
  double and the replay by construction, Isaac as measured (a constant-acceleration ramp on joint 1: the reported
  PhysX velocity is the commanded segment's to 0.3 mrad/s RMS, not that plus a dt/2 as a smooth motion's end
  velocity would be; a step from rest reads at once, through the velocity feed-forward). So every path or chunk
  boundary reads twice the acceleration the same motion reads inside one path, for every method (S1 chunks, ASPIRE
  commands, DexAgent chunks). Done 2026-10-19 (user decision): every backend declares what its velocity averages
  over (`RobotSnapshot.velocity_window_s`) and S0 reads a first step over half the step plus that window, limits
  unchanged; the new reading is never stricter, and no recorded verdict changes (every recorded command passed the
  old one; the only refusal, ASPIRE's final hold at 16.7 rad/s, is a velocity check). ASPIRE's speed bound doubles
  with it (`aspire/README.md`). The real FANUC backend has no code yet: it declares its own window when written.
- **The fragments interlock: the task cannot be assembled as it stands (found 2026-10-19, every method).** The
  fragments are 8 mm voxels (their collision boxes' smallest edge is 8 mm), so neighbours meet in 8 mm steps and
  hook into each other by about one voxel. Moving straight out of its goal (rotation fixed, the DexAgent export's
  collision boxes) with only the fixed anchor present, a fragment's best exit along the 26 lattice directions runs
  into the anchor by: blue, yellow, magenta 0 mm; cyan 7.0, red 7.07, orange 8.0, green 10.0 mm (6.8-8.9 mm over 800
  arbitrary directions; none goes in from above). With every fragment placed only yellow comes out within 3 mm;
  even at 8 mm green stays locked. Another anchor does not help: for each of the 8 choices at most one fragment comes
  out within 6 mm (the fixed anchor's own exits taken by symmetry from its neighbours'); ASPIRE's reference program
  jammed on the anchor for this reason. Fixing it changes the task (for the user):
  reassigning interface voxels so each fragment has a straight, interference-free insertion in some order, or finer
  voxels / assembly clearance (the interlock is about one voxel edge).
- **The anchor tips on its own.**
  - Found by DexAgent's export: the anchor fragment, standing alone at its assembled pose, tilts 4.8° within 1 s in MuJoCo. Isaac hides this because its bodies start asleep.
  - A staged fragment (`fragment_36d5ca6a07ba2b2f`) tips 18° while settling.
  - Decided: a fixture holds the anchor for all methods (above). Starts must still be settled before they are recorded, which needs the Isaac backend.

## Tests and environments

| Environment | Runs | Command (from the repository root) |
|---|---|---|
| `envs/asm` | all task tests; the DexAgent unit and integration tests (the latter need `envs/dexagent` and `third_party/DexAgent`, else they skip); ASPIRE tests skip | `PYTHONDONTWRITEBYTECODE=1 envs/asm/bin/python -B -m pytest -p no:cacheprovider -q tasks core` |
| `envs/aspire` | `tests/test_baselines_aspire.py`, `tests/test_s2_aspire.py` (need the patched `cap_harness`) | `PYTHONDONTWRITEBYTECODE=1 envs/aspire/bin/python -B -m pytest -p no:cacheprovider -q tasks/dual_arm_crag_skull/tests/test_baselines_aspire.py tasks/dual_arm_crag_skull/tests/test_s2_aspire.py` |
| `envs/dexagent` | DexAgent itself, as a subprocess of our runner (`DEXAGENT_PYTHON`) | not run directly |

## Security

- Model-written code runs in a bwrap sandbox (`sandbox.py`): an empty root with its own namespaces, so inside there
  is no network (only the sandbox's own loopback), no home directory (a venv's interpreter directory alone, read-only),
  no other run, environment or upstream clone; the system directories and the listed code read-only, and only the
  trial directory writable. The model is called from outside: the relays hold the API key and talk to the sandboxed
  process through files in the trial directory.
- `python -m tasks.dual_arm_crag_skull.baselines sandbox-check` probes that sandbox with DexAgent's venv and its
  scrubbed environment and exits 1 unless all of it holds; `tests/test_baselines_sandbox.py` checks the same and
  that killing a sandbox leaves no process behind.
- DexAgent: every process of a live trial after the export runs in it (Stage 3, the delivery re-check). ASPIRE's
  programs run in-process with Isaac Sim on the GPU and are not sandboxed yet: no live ASPIRE run until they are.
- Load the API key only in the process that calls the API, and never put it on a command line.
- DexAgent's processes get a scrubbed environment, and the runner makes itself non-dumpable before a live trial. Neither stops code running as the same user from reading the user's files.
- Details are in each baseline's README.
