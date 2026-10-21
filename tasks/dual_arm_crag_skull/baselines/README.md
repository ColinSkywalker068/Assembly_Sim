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
- The task (user, 2026-10-20): the red anchor's repaired geometry (`runs/skull_red_plan/`), and its assembly plan is
  ground truth that every method is given (`baselines.assembly_plan = "given"`): the order the fragments go in and the
  direction of each one's last straight move (below, "The fragments interlock").

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
    - the assembly plan (`PlanStep`s: each fragment but the anchor once, in the order they go in, with the world
      direction of its last straight move into its goal);
    - the table, pad and exclusion zones;
    - the success criterion (tolerances and `settle_s`);
    - the seed and split.
  - `task_from_layout` builds it from a schema-2 layout, takes the plan from the layout's `ownership_repair.plan`
    (`prepare --straight-assembly-anchor`, checked there to go in straight on the voxel grid; refused for another
    anchor) and checks the geometry:
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
    - the anchor fixture and the assembly plan (`baselines.anchor = "fixture"`, `baselines.assembly_plan = "given"`);
    - the seam clearance (`baselines.collider_inset_m` = 0.25 mm: every fragment's collision boxes are that much smaller on each face than its voxel cells, so each seam is open by 0.5 mm);
    - the held-out claims directory.
  - `require_canonical(config)` refuses any other configuration.
  - `verify_task(task)` rebuilds the task from its layout and refuses one that differs in any way: tolerance, placement, split label, layout, meshes or plan; and, since the protocol gives every method the plan, a task without one.
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

  The real 8-fragment skull tasks are in `runs/baseline_tasks/` (2026-10-20): the base task and seed 101 (no held-out
  seed is written). The task id is `view_gt-00f0cc2853e9-anchor-fragment_038755611176a8c9`; all 130 seeds place in
  0.5 s. The earlier task (the
  unrepaired diagnostic geometry, white anchor, no plan) is kept in `runs/baseline_tasks_diag_white/`; the protocol
  refuses it.

## Fairness protocol

1. **Same task and information.** Every method gets:
   - the same goals;
   - the same assembly plan (ASPIRE: `metadata.assembly_plan`; DexAgent: the subgoal order, each subgoal's last
     straight move, the stand-in video and `stage3.notes`; ours: S2's task context, when its executor is written);
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
- **Every seam is open by 0.5 mm for every method** (`baselines.collider_inset_m = 0.00025`, part of the protocol; decided 2026-10-20 at 1 mm, changed 2026-10-21). The layout's fragments meet face to face; with the seams closed, our scripted reference (`sim/assemble.py`) brought each fragment to the start of its insertion within 0.1 mm in Isaac and the insertion still jammed (orange turned 35°, blue stopped one voxel high). With 1 mm seams the same insertions ended 0.1 mm off, but a fragment resting on fragments sank 1 mm a layer, and cyan (on blue and on red) tipped 3.8° by itself once in place, in the way of the next insertion (Isaac: every prefix of the plan put at its goals and settled). With 0.5 mm seams the worst of every prefix is 0.96 mm / 1.9°; insetting only the sideways faces kept them at 0.03° but jammed a sideways insertion. The Isaac backend insets each fragment's box colliders, and DexAgent's export its MJCF boxes, by the protocol's value; the meshes shown, the cells, the goals and the plan are unchanged.
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
- **Protective stop (done 2026-10-19, user decision).** The CRX drives push without a force limit, so a held fragment
  lowered onto the fixed anchor jammed (joints oscillating at 1-5 rad/s, joint 5 at 16.7 rad/s on release) where the
  real CRX would stop. The Isaac backend now applies one rule for every method each control period
  (`configs/jing.json` `s0.protective_stop`, frozen; `backend.StopMonitor`): a joint more than 10 mrad from the target
  it was commanded to for 3 periods in a row, or faster than 1.1 x its maximum speed, raises `ProtectiveStop`; the
  trial ends there, unscored, recorded as `protective_stop` (a failure) with the periods before it. Normal
  pick-and-place on Isaac lags 0.32 mrad at most (the DexAgent orange delivery with its set-down; ASPIRE's reference
  0.18; a free 1.2 rad/s ramp 0.07); the jam reached 279 mrad, and the rule would have stopped it 0.3 s after contact.
  Added 2026-10-21 (`finger_reach`): fingers reaching further out (`RobotSnapshot.spans`) than S0 boxed them for the
  motion stop the robot too. Every method's executor (ASPIRE's adapter, DexAgent's executor and its S0 replay, the
  scripted reference) calls `AssemblyBackend.bound_reach` with S0's opening plus its margin once S0 accepts a motion
  and before stepping it. S0 reads the opening where a motion begins; a fragment jammed between the fingers pried them
  6-12 mm open before the seams were opened, and normal runs since widened them 1.15 mm at most.
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
- **The fragments interlock (found 2026-10-19, every method; resolved 2026-10-20: the red anchor's repair and its plan
  are the task).** The
  fragments are 8 mm voxels, so neighbours meet in 8 mm steps and hook into each other by about one voxel. On the
  voxel grid (`experiments/assembly_paths.py`, exact for cubes moved straight, rotation fixed), with only the fixed
  anchor present red, green, cyan and orange have no free lattice direction, and for none of the 8 anchors does any
  order take all of them apart. In MuJoCo (`experiments/assembly_depths.py`, the DexAgent export's collision boxes,
  the contacts its own pipeline makes) their best exits run into the anchor by red 3.55, cyan 3.55, orange 3.76 and
  green 10.0 mm. (The 7.0-10.0 mm reported before came from `mj_geomDistance`, which reports millimetres of
  penetration for boxes that only share an edge.) ASPIRE's reference program jammed on the anchor for this reason.
  **Candidate fixes; the user chose the red anchor and made its assembly plan ground truth, given to every method
  (10-20).** The task now uses it (`runs/skull_red_plan/`: the same meshes byte for byte as `runs/skull_straight_red/`,
  plus the plan; `prepare --assembly-order` checks a given order on the voxel grid with the repair's directions):
  `prepare --straight-assembly-anchor ID` (`interlock_repair.py`) hands seam cubes to the neighbour across the seam,
  none added or removed, so that with the anchor fixed every other fragment goes in along an axis in some order, then
  gives a moving fragment's stray cubes to a neighbour that keeps that order clear. Two anchors were tried, the only
  two fragments on the table:
  - **red (the base, 40 cubes on the table):** 62 of 2116 cubes change owner; blue, green, yellow, magenta and white
    (the cranium, last) go in from above, orange and cyan slide in. The repair's own order is orange, blue, cyan,
    green, yellow, magenta, white; keeping each fragment's direction, 36 orders are collision-free. Released one by one
    under gravity (every placed set started from the goals and settled): in Isaac, which scores every method, all 36
    orders stay within 0.0004 mm / 0.0007 deg at every step (`experiments/assembly_settle_isaac.py`,
    `runs/assembly_paths/settle_red_plan_isaac.json`); in MuJoCo, with the robot at its home keyframe, all stay within
    3 mm / 3 deg and only blue slides, 1.9 mm, when it goes in before orange (`experiments/assembly_settle.py`). So
    only collisions constrain the order. The plan's order (orange, blue, green, yellow, cyan, magenta, white) moves
    nothing in either engine. A recorded Isaac replay of the plan (`experiments/assembly_replay_isaac.py`: each fragment
    carried kinematically with collisions on, slid in along its insertion direction, let go, settled 1 s) leaves every
    placed fragment within 0.0003 mm of its goal after all 7 steps (`runs/assembly_paths/replay_red_plan_isaac/`).
    Correction (10-20): an earlier MuJoCo check reported that only 6 orders move nothing and
    the others tip cyan 3.9 mm / 6.3 deg; it started the robot at qpos0 (hands inside the skull, pressed into the
    table, touching no fragment), and the tipping is gone with the robot at its keyframe
    (`runs/assembly_paths/settle_red_orders.json` is that earlier result). A robot must carry white (1414 cubes, about 0.47 kg). Geometry
    `runs/skull_straight_red/`, tasks `runs/baseline_tasks_straight_red/`. (An earlier version, 92 cubes with white
    second, came from taking the cheapest fragment out at every step; now every first exit is tried and the cheapest
    plan in all is kept. It is in the `_v1` paths.)
  - **white (the other candidate):** 106 cubes change owner (108 handovers; two cubes change twice); all 7 slide in
    horizontally (one from above already costs 167 handovers, all from above cuts fragments into pieces), order green,
    orange, red, cyan, blue, magenta, yellow; nothing moves under gravity. Geometry `runs/skull_straight/`, tasks
    `runs/baseline_tasks_straight/`.

  For both: every fragment one connected piece, the grid check takes all 7 apart, MuJoCo contacts along the order
  are 0 mm, DexAgent's export checks pass (seed 101). Either way the order matters and some insertions are
  horizontal, so a method that lowers every fragment from above still collides; the DexAgent export now follows the
  plan (subgoal order, and each fragment's last straight move in its subgoal and the stand-in video), where it used to
  lower every fragment from above, bottom layer first.
  The original (unvoxelized) meshes do not avoid this: they are closed solids of a solid skull with zero-gap seams,
  and with every fragment shrunk 0.5 mm, orange, yellow, cyan and green each cannot leave white along any sampled
  direction (about 1,100 on the upper hemisphere and their reverses); shrunk 1 mm, green is still held inside white
  (its best direction sweeps through 1.4 cm3 of white; green is 19 cm3), and it comes free only with every fragment
  shrunk 4 mm. `experiments/mesh_paths.py`; `runs/assembly_paths/mesh/`, `runs/assembly_paths/mesh_wide/`.
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
