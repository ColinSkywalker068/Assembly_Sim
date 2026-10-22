# DexAgent baseline (native YAM + Sharpa bench, Stage 3 only)

[DexAgent](https://github.com/dexagent1/DexAgent) (MIT) is an agentic Human2Sim2Robot framework: from a human video
it builds scene semantics (Stage 1), a verified MuJoCo reconstruction (Stage 2) and a physically executed robot
trajectory (Stage 3), with an LLM agent that calls and extends a tool library. This folder runs it as a baseline
for the skull assembly, pinned at `e29bdad8ed3e95b2c9ee2eea9044a6b4115ec633`, **as published**: unmodified code,
its own bench (two I2RT YAM 6-DoF arms with Sharpa Wave hands, `setup/`), Stage 3 only.

Nothing here has made a model API call yet; every result so far is offline (mock LLM, fake API client).

## Port to the shared hardware (in progress, user decision 2026-10-19)

Every method runs on one hardware set: two CRX-10iA/L with YAM linear_4310 grippers, scored on Isaac. DexAgent is
being ported to it: it plans and verifies in MuJoCo on a bench built from the same robot models (`bench/`), with a
patch to its runtime and tools, and its delivery is executed on Isaac through S0 like every other method. The native
bench below stays as it is until the port replaces it.

| Step | What | Status |
|---|---|---|
| 1 | Bench (`bench/`): CRX MJCF on S0's chain (`crx.py`); YAM as a DexAgent hand (`yam.py`, `yam_pieces.py`); scene and setup.yaml (`scene.py`), to replace `setup/` in each trial clone | done, tested (MuJoCo and DexAgent's own HandRig read it) |
| 2 | Patch DexAgent (`patches/0001-crx-yam-port.patch`, 21 files, developed in `third_party/dexagent-e29bdad-crx_yam`): joint-position arm servos with gravity compensation (Isaac's arm is effectively rigid), 20 N / 0.1 m/s fingers, slide-joint units (cinch, press depth, opening, rate caps), symmetric jaws, no wraps, per-joint CRX/S0 rate limits, arm joint ranges clipped (DexAgent treats any range >= 2 pi as free), gate thresholds in slide units, fixed stations, hidden hand-only tools, prompt text; the planner slows fingers in S0's contact band to 0.8 x its contact speed; `s0_delivery_check` in the gate suite; for the jaws, `generate_rigid_grasps` pinches across chords of the object's own material at every 15 deg (the bounding-box pinches it had found nothing on fragments wider than the jaws), nearest the centre of mass first | done; the full gate suite on a scripted episode passes every robot, arm and hand check (its failures were two export bugs, fixed, and the scripted plan's own quality) |
| 3 | Export on the bench (`--bench crx_yam`): rig = world shifted down to the tabletop; Isaac-settled starts | done: seed 101 passes DexAgent's checks; Isaac and MuJoCo settle it alike (every fragment within 0.09 mm / 0.17 deg; one tips 18 deg in both), so the export keeps MuJoCo's start and the execution refuses a start more than 1 mm / 1 deg from it |
| 4 | Library triage on 1-3 fragment probes until the gate suite passes a scripted episode | in progress: every moving fragment of seed 101 gets 20-24 jaw pinches (12-23 the arm reaches) and the top two of each lift in `grasp_lift_test`; the orange fragment is grasped, lifted and set down at a free spot within S0 (the planner's band slowdown costs 1.5 s); setting a fragment down on its goal meets the fixed anchor |
| 5 | Executor: delivery -> 60 Hz -> S0 (contact windows from phase labels) -> backend -> release, settle, score (`executor.py`, `execute_trial.py`, CLI `execute`) | done, tested on the kinematic double; the scripted orange episode ran on Isaac through S0 (738 steps, both windows declared, settled) |
| 6 | MuJoCo <-> Isaac calibration with paired scripted probes | first pair: after the orange episode the carried fragment ends 0.49 mm / 0.08 deg from where DexAgent's MuJoCo recording has it (every other fragment within 0.09 mm, its settled start); more probes (other fragments, a place on the goal, heavier contact) to do |
| 7 | Mock end-to-end run; one live practice seed after the user approves budget and host | todo |

## Why this shape

- **Native embodiment.** The YAM + Sharpa robot is hard-wired into DexAgent's runtime (`tool_library/common/
  rigid_robot.py`, `hand.py`, the gate suite's robot and station integrity checks). Porting it to our CRX arms and
  parallel grippers would be a different method, so the agent works on its own bench and we compare outcomes.
- **Stage 3 only.** There is no human video of this task. Stages 1 and 2 exist to turn such a video into what
  Stage 3 needs; we construct those deliverables directly from the `AssemblyTask` (`export.py`) instead.

## What we fabricate, and the gate each input has to pass

| Input | What it is | Gate it satisfies |
|---|---|---|
| `video.mp4` | **A stand-in, not a demonstration**: the Stage 2 scene rendered (MuJoCo, OSMesa) through `video_cam` straight above the table, robot hidden, no hands; each moving fragment glides to its goal during its subgoal's window and goes in by the last straight move of the task's assembly plan (`export.waypoints`: it comes down outside the skull, clear of what is already in place, then moves straight in). H.264 via DexAgent's `write_video`. | Stage 3 pre-agent identity gate: `agent/input_identity.py:32-78` called at `run_stage3.py:504` |
| `stage1/scene_semantics.json` | One `place` subgoal per moving fragment, in the order of the task's assembly plan (bottom layer first for a task without one), its action and hand role naming the last straight move ("with a last straight move downward", "... along rig -x (toward the left station)" on DexAgent's bench, "(toward the left arm)" on the shared one), a note that the subgoals follow the plan; acting hand = the arm on the fragment's side when DexAgent's `object_reach_check` finds a pose for it there, else the other arm with a pose, else the closest (`export.choose_hand`; there is no demonstrator to copy), `time_window_s` = the video windows, goal "The X is released." (a supported atom, `final_goal_language.py:374`) plus a trusted `relative_pose` predicate to the anchor at the task tolerances (3 mm / 3 deg from `configs/jing.json`); `task_success` "the X is released" for each moving fragment and "the A has not moved" for the anchor (its invariant, `final_goal_language.py:314`: nothing lets go of a fixture-held anchor, and "released" would compile to `release_settle`, which a fixed body touching nothing fails as falling), with `task_success_predicates` for every fragment and the anchor (`reference_frame: 0`); `input_identity` from DexAgent's own `video_identity`, as Stage 1's finalize writes it (`run_stage1.py:114-119`). | `scene_semantic_schema_check`, `scene_semantic_consistency_check`, `goal_language_coverage_check` (Stage 1's verifiers); predicates parsed by `task_gates.py:2610` and used at `:3061`, `:3148`. Windows and `video.duration_s` are needed because `timing_realism_audit` is blocking under `strict_quality` and a skipped input fails it (`stage3_gate_suite.py:75,271`; `timing_realism_audit.py:123,175`). |
| `stage2/scene.xml` | The clone's `setup/scene.xml` with one body per fragment staged by DexAgent's own `stage_on_bench`, `video_cam` added from `camera.json`, then settled with the robot clamped and the rest poses written to keyframe 0. | `initial_keyframe_state_check` (`stage3_gate_suite.py:197`) and `start_state_audit` (its `AUDITS`), both blocking, need a keyframe at rest. |
| `stage2/assets/<id>/` | OBJ visual mesh (massless, never collides: the exterior of the same voxel solid, `geometry.voxel_surface`, closed and with every edge shared by two faces: the layout's voxel surface is not watertight where two cells meet along an edge only, 6 of the 8 skull fragments, and DexAgent's blocking `object_visual_penetration_audit` cannot run on an open mesh, nor measure depth right with faces inside the solid) and an MJCF body (a free joint, except for the fixed anchor) whose collision is the exact box decomposition of the voxel solid, each box inset as in Isaac by the protocol's `baselines.collider_inset_m` (0.25 mm on the faces sideways where the fragment ends, 0.1 mm on its top and bottom: `baselines.task.box_axis_insets`; the exports on disk predate any inset and need re-exporting), explicit contact parameters (DexAgent's own 6 ms defaults, stated so no rigid pairs are added, `contact_defaults.py:24,64`). Density 650 kg/m3 and friction 0.8 from `core/config/workcell.json`. | Compiles in DexAgent's rigid runtime (`episode.build_execution_model` + `rigid_robot.compile_rigid`); `hand_scene_from_recon` builds hand scenes from it. |
| `stage2/frame0.png`, `masks/<id>.png`, `reference_evidence_recipes.json` | DexAgent's CPU `segment_source_frame` (GrabCut in a box around each fragment's projection) on frame 0, run through DexAgent's own `reference_evidence` registry, which writes the recipes. | Stage 3 pre-agent provenance gate: `revalidate_source_recipes` re-runs every recipe on the video and needs byte-identical frame and masks (`reference_evidence.py:591-697`, called at `run_stage3.py:510`). No `masks/hands.png` (it would need SAM hand prompts, `:293-307`). |
| `stage2/scene_recon.json`, `camera.json`, `frame0_compare.png` | Objects with `asset_xml`, `root_body` = `<id>_<id>`, planar `pose` (plus exact `quat_wxyz`), mask, IoU and verifier reports from DexAgent's `silhouette_iou_check` and `settle_stability_check`; camera from DexAgent's `rigcam`. | Read by Stage 3 tools. `scene_recon_manifest_check` (a Stage 2 gate Stage 3 does not re-run) flags only `skill` / `skill_origin`: our builder is not a library skill, and we name it honestly (`skill_origin: external`). |
| `stage3.notes` | One factual paragraph: no human demonstration, the stand-in video, the fixed anchor, box collision, and the assembly plan (the order and each fragment's last straight move in rig terms). | DexAgent's documented hint field (`dexagent.yaml` `stage3.notes`). |

`export_checks.json` records every check above, run with DexAgent's code in a fresh process, plus
`object_reach_check` (informational) and the goal-pose interpenetration of boxes versus hulls. A trial refuses to
start when a required check fails (`export.REQUIRED_CHECKS`).

## Frames and ids

- Rig (DexAgent, `setup/setup.yaml:10-16`): origin between the arm mounts on the tabletop, x toward the right
  station, y away from the operator, z up, tabletop z = 0.
- `rig_T_world` = yaw (default 0) + translation putting the pad centre at rig (0.00, 0.40) on the tabletop.
  World x maps to rig x, so our left lane meets the left YAM. The pad depth came from DexAgent's
  `object_reach_check`, which is inconclusive by design (finite probes): 2-5 of the 7 staged skull fragments get a
  pose witness at pad depths 0.30-0.60 m, the most (5) at 0.40. DexAgent may move stations itself
  (`place_arm_stations`), which its gates allow when recorded.
- Body frame = fragment frame (`body_T_fragment` = identity); scoring uses
  `world_T_fragment = inverse(rig_T_world) @ rig_T_body`.
- Ids are `<colour>_fragment` in fragment-id order (`red`, `blue`, ...); names `"<colour> skull fragment"`. The map,
  `rig_T_world`, the task record, the settled and goal poses, the file sha256s and the DexAgent commit are in
  `inputs/export_manifest.json`.

## Collision geometry and the anchor (findings)

- DexAgent's mesh builder (`construct_mesh_asset`) offers only `hull`, `vessel` or `box` collision, recentres the
  origin and levels the object. For the skull's concave fragments, hulls cannot reach the assembled state: at the
  goal poses of the real layout (`runs/skull_diagnostic`), **convex hulls interpenetrate by up to 80 mm (22 of 23
  touching pairs deeper than 1 mm)**, exact boxes by 0 mm (16 pairs touching).
- Our own `asset_xml` (exact voxel boxes) **is accepted by Stage 3**: both pre-agent gates pass, the rigid runtime
  compiles, `hand_scene_from_recon` works, `initial_keyframe_state_check` passes, and a mock Stage 3 run reaches the
  agent loop. The boxes are identical (up to translation) to the Isaac colliders of
  `tasks/dual_arm_breakingbad/scene/geometry.merge_voxel_cells` for all 8 real fragments. Cost: 502 boxes give 19,443
  hand-object contact pairs; DexAgent's runtime build (`build_execution_model` + `compile_rigid`) takes about 110 s
  per call on this machine (stepping is fast: 500 steps in 0.4 s).
- **The assembled anchor is not statically stable alone on the table**: its centre of mass is outside the footprint
  of its lowest cells; in MuJoCo it tilts 4.8 deg within 1 s (Isaac hides this because bodies start asleep). One
  staged fragment (`blue`, `fragment_36d5ca6a07ba2b2f`) tips 18 deg from its staging pose. DexAgent's gates need a
  resting keyframe, so starts are settled (recorded in `scene_recon.json` and the manifest), and the anchor is fixed
  at its assembled pose by default (`--anchor-mode fixed`; `free` lets it settle, after which our anchor tolerance
  cannot be met). This affects the task definition itself, not only DexAgent.
- DexAgent judges our goals the way `score_assembly` does: on kinematic replays of the 3-cube fixture and of the
  8-fragment skull, its `stage1_goal_gate` passes `task_success` when every fragment is at its goal and fails it with
  one fragment 5 mm off ("5.0 mm from its target pose relative to white_fragment ... tolerance 3.0 mm"). Its subgoal
  `release_settle` checks need a real hand release, which a hand-free replay cannot show.

## LLM transport and budget

DexAgent's own `AnthropicLLM` (`agent/llm.py:23-47`) sends non-streaming calls with `max_tokens=8000`, drops thinking
blocks and never checks `stop_reason`. We use `agent.provider: external` (`agent/external.py`): DexAgent writes each
request to `transport/requests/`, `transport.py` answers it with Claude (`claude-opus-5-5`, effort `high`,
streaming, `max_tokens` 64000 because thinking counts toward it, prompt caching with 1 h TTL) and:

- keeps every assistant turn as the API returned it (thinking and signatures) and sends that back, after checking
  that DexAgent's history equals what was answered (else stop);
- sets `thinking.block_binding.prefix_mismatch_behavior: drop_block` (beta `thinking-binding-controls-2026-08-01`).
  Thinking on `claude-opus-5-5` is always on, and each thinking block is bound to the system prompt, the tools and
  the earlier messages it was produced with. DexAgent's tool list grows when the agent adds library tools, which on
  accounts created on or after 2026-08-31 would otherwise fail the next request with a 400; with `drop_block` the API
  drops the invalidated earlier blocks (unbilled) and the ledger records each drop (`input_transformations`);
- never passes on tool calls of a `refusal`, `max_tokens`, `model_context_window_exceeded` or other non
  `end_turn`/`tool_use` turn;
- pays from the budget pool every live baseline run shares (`../budget.py`, `baselines.llm_budget` in
  `configs/jing.json`: $100 for DexAgent and ASPIRE together, no per-trial amount): before each attempt its worst
  case is reserved there, and the call is not made if it does not fit in what is left; worst case = measured input
  of the previous call + its output + one token per UTF-8 byte added since (w*h/750 per new image), all at the 1 h
  cache-write rate ($8/MTok), plus `max_tokens` at $20/MTok. Prices: $4 / $20 per MTok in / out, cache writes 1.25x
  (5 min) or 2x (1 h), reads 0.05x ($0.20/MTok). The reservation is settled at the measured cost; a call that cost
  more than its reservation is charged in full and stops the run.
  Rate limits, 5xx and 529 overload are retried (3 times, 30/60/120 s); a dropped connection or an error event inside
  a started stream is retried too but charged its worst case, since it may have been billed;
- stops by writing `transport/operator_checkpoint.json` `{"kind": "administrative_checkpoint"}` and answering
  `report_blocked`, which DexAgent accepts at once in Stage 3 (`agent/runner.py:257-275,391-398`);
- logs every call and stop to `llm_ledger.jsonl` (tokens, USD) and every turn to `llm_transcript.jsonl`;
- reads the API key only in this process (see Security) and turns the SDK's own retries off, so every attempt is in
  the ledger. No server-side fallbacks: a fallback model would change the agent and has no fixed price. DexAgent's
  own `agent.budget_usd` does not apply to the external provider (`agent/budget.py:22,78`).

## Security

Model-written code (`run_python`, `library_create` tests and every created tool) runs in the bwrap sandbox of
`../sandbox.py`: Stage 3 and the delivery re-check of a trial run inside it (`runner.stage3_launch`,
`runner.sandbox_paths`), with the trial directory as the only writable place; DexAgent's venv and its interpreter
directory, and this repository's `core/` and `tasks/` (the S0 delivery verifier imports them), read-only; no network,
no home directory, no other run (`python -m tasks.dual_arm_crag_skull.baselines sandbox-check`). A live trial cannot
turn it off. Inside it the key file, the runner's `/proc/<pid>/environ` and the user's other files do not exist,
which is what the separate user below was asked for. Each trial also gets a fresh clone of the pinned commit
(verified clean, ignored files included), its own HOME/TMPDIR/XDG directories and an environment built from an allow
list (no `ANTHROPIC_*`, `OPENAI_*`, cloud or SSH variables; `dexenv.scrubbed_env`). What the earlier, unsandboxed
design relied on still holds:

- a scrubbed environment does not hide the parent's: a child of the same user reads `/proc/<pid>/environ` (shown by
  `test_protect_process_hides_the_runner_environment_from_children`). A live run therefore first makes the runner
  non-dumpable (`prctl(PR_SET_DUMPABLE, 0)`, `dexenv.protect_process`), which blocks that read. Every other
  process of the user stays readable, including the shell the runner was started from: load the key in a subshell
  that then becomes the runner, `( set -a; . ~/.config/anthropic.env; set +a; exec python -B -m ... )`, so no other
  process holds it. Never put the key on a command line (`ps` and `/proc/<pid>/cmdline` show arguments to every
  user, and the shell history keeps them). The key file itself stays readable by DexAgent's code under the same
  user, which is why live trials need a separate user (next paragraph);
- files readable by the user stay readable and writable. Scoring therefore re-checks the exported inputs against the
  sha256s held in memory since export, the DexAgent venv against its package RECORDs, and DexAgent's acceptance from a
  second fresh clone, which also re-runs `execution_scene_integrity_check` against our Stage 2 and
  `physics_recording_check`. This detects tampering with what scoring reads; it is not authentication against code
  running as the same user (`agent/artifacts.py:141-145` says the same of DexAgent's own certificate).

The export (before any model call) runs DexAgent's own code outside the sandbox. `examples.reuse` is false (no
example results are copied in).

## Setup

```bash
uv venv --python 3.11 envs/dexagent   # from the repository root
uv pip install --python envs/dexagent/bin/python --no-deps \
    -r tasks/dual_arm_crag_skull/baselines/dexagent/requirements-dexagent.txt
# Defaults, under the repository root: envs/dexagent/bin/python and third_party/DexAgent (the clone every trial
# is cloned from). Override with absolute paths:
export DEXAGENT_PYTHON=/path/to/python DEXAGENT_SOURCE=/path/to/DexAgent
```

Pinned (`requirements-dexagent.txt`): Python 3.11.13, numpy 1.26.4, mujoco 3.10.0, opencv-python 4.11.0.86, scipy
1.17.1, trimesh 5.1.1, imageio 2.38.1, imageio-ffmpeg 0.6.0 (ffmpeg 7.0.2), scikit-image 0.26.0, matplotlib 3.11.2,
h5py 3.16.0, sympy 1.14.0, pillow 13.0.0, pyyaml 6.0.3, jsonschema 4.26.0, anthropic 1.15.0, openai 3.35.1, rich
15.0.0 (+ pytest 9.1.1). No Stage 4, SAM, HaMeR or GPU packages. Rendering uses OSMesa (EGL is not available here).

## Run

```bash
python -m tasks.dual_arm_crag_skull.baselines task --layout LAYOUT.json --anchor ID --seed 101 --output task.json
python -m tasks.dual_arm_crag_skull.baselines.dexagent export --task task.json --out runs/dexagent/export1
python -m tasks.dual_arm_crag_skull.baselines.dexagent run --task task.json --trial runs/dexagent/t1 --llm mock:turns.json
python -m tasks.dual_arm_crag_skull.baselines budget create --note 'practice round'   # once, for both baselines
( set -a; . ~/.config/anthropic.env; set +a; exec python -B -m tasks.dual_arm_crag_skull.baselines.dexagent run \
    --task task.json --trial runs/dexagent/t2 --live --records runs/dexagent/runs.jsonl )
python -m tasks.dual_arm_crag_skull.baselines.dexagent score --trial runs/dexagent/t2 --records runs/dexagent/runs.jsonl
```

`run` clones, exports (about 2.5 min for the 8-fragment skull, mostly the runtime compile check), writes
`dexagent.yaml` (provider external, `examples.reuse: false`, `stage3.methods: []` unless `--methods published`,
`stage4_draws: 3`, `strict_quality: true`), starts `run_stage3.py ... --semantic --recon --video --name skull` in its
own process group (it has no `--yes`; only `run_dexagent.py` asks between stages) and kills the group at
`max_wall_minutes + 120`. The task must be its layout under the canonical protocol (`baselines.task.verify_task`).
Held-out seeds need `--allow-heldout` and `--records`, are claimed atomically before anything is cloned
(`baselines.scoring.claim_heldout`), and run once per variant whatever records file is given; a trial that ends in an
exception or Ctrl-C is recorded as `error` before the exception propagates, and `score --records` refuses to add a
second record of a held-out seed. The `RunRecord` is `method='dexagent'`, `backend='dexagent_native_yam_sharpa_mujoco'`,
`test_only` for mock-LLM trials,
`variant='native_yam_sharpa_stage3'`; `cost_usd`/`llm_calls` come from the ledger, `sim_trials` counts the
`execute_subgoal_plan` calls that ran (not dry runs, suppressed duplicates or failed calls) plus the variants
`search_subgoal_variants` evaluated. A score needs a delivery DexAgent accepts (`agent/artifacts.py:138-168`), re-checked
as described under Security; such a delivery has passed DexAgent's own `release_settle` gate (0.3 s settle, 4 mm /
10 deg by default, `task_gates.py:981`), so it is scored as settled, at our 3 mm / 3 deg on the last recorded frame; status: transport stop `budget` -> `budget_exhausted`, `refusal` -> `rejected`, other
stops -> `error`; our deadline -> `timeout`; a failed input gate or model-backend failure -> `error`; DexAgent's round or
money budget -> `budget_exhausted`, wall budget -> `timeout`, `report_blocked`/stall/no-progress -> `blocked`; an
accepted, re-checked delivery -> `completed`; changed inputs or venv -> `error`. Relative `--trial` paths are resolved.

### On the shared hardware (the port)

```bash
python -m tasks.dual_arm_crag_skull.baselines.dexagent run --task task.json --trial runs/dexagent/t3 \
    --llm mock:turns.json --bench crx_yam
"$ISAAC_PYTHON" -m tasks.dual_arm_crag_skull.baselines.dexagent execute --trial runs/dexagent/t3 --backend isaac \
    --records runs/dexagent/runs.jsonl
```

- `--bench crx_yam` (`export`, `run`): the trial's DexAgent is a fresh clone with the port patch applied and `setup/`
  replaced by the bench generated for the trial's S0 limits (`bench/checkout.py` checks both, also for the clone that
  re-checks the delivery). The limits are ASPIRE's TEST values with the YAM's width range, written to
  `inputs/s0_spec.json`; until `configs/jing.json` has real S0 values every shared-hardware record is `test_only`.
- In Stage 3 the gate suite's `s0_delivery_check` (in the patch) runs `s0_check.check_trajectory` on a delivery: the
  executor's own commands and contact windows on a replay that follows them exactly, with the fragments where
  DexAgent's recording has them. DexAgent accepts nothing that S0 would refuse on a perfect tracker. Since S0 takes
  the grippers' boxes at their widest opening (2026-10-21, `finger_opening_margin_m`), the replay's grippers are as wide
  as commanded or, where the recording has the jaws wider (a fragment between them), as recorded; the executor passes
  each chunk's widest opening from the measured and commanded widths.
- A delivered trial ends awaiting execution (`result.json`, no record; DexAgent's own MuJoCo outcome is only a note).
  `execute` runs it once (`execution.started` is created before anything moves; an execution that fails is recorded
  as `error`; a held-out seed needs `--records`). The backend resets to the task (Isaac settles the fragments first;
  its start is saved as `settled_start.json` and must be within 1 mm / 1 deg of the start DexAgent planned from), the
  delivery's commanded reference is ramped at S0's 60 Hz from its recorded start state and checked in chunks, and
  then, completed or rejected, every method's ending: both grippers open, the settle hold through S0, the score. A
  chunk starts at each command where an arm enters S0's contact band, or one command before it where the commanded
  velocity changes less, and the arm's window is declared then from the phase label of its first command in the band,
  on what is observed (a grasp on the fragment where it is; a place on its goal when the fragment is within S0's
  radius of it, else on the fragment; never on the fixture-held anchor); other chunks are at most 1 s and end where
  the commanded velocity changes least. A chunk S0 rejects changes no window. The RunRecord keeps the
  Stage 3 trial's costs and names the backend as ASPIRE's records do.
- S0 reads the first segment of a chunk as a constant acceleration from the measured velocity, 2 (v1 - qd_now) / dt,
  and its later knots as the velocity change over the period; both backends report the last period's velocity, so a
  chunk boundary reads twice the acceleration a knot inside a chunk reads (for every method, TODO P6). On DexAgent's
  30 fps ramp a chunk ending between frames reads nothing there, but every frame's velocity change falls in one 60 Hz
  period, so S0 reads twice the planned acceleration: the bench plans at most 0.8 x 30/60 of S0's acceleration.
- `score` re-checks a shared-hardware trial's delivery from a fresh port checkout under the trial's own S0 limits.

## Confounds when comparing with S2/S1/S0

Different robot (YAM arms and five-finger Sharpa hands vs CRX arms and parallel grippers), physics engine (MuJoCo 3.10
vs Isaac/PhysX: contact model, friction, no sleeping), success gates (DexAgent publishes only after its whole gate suite,
including timing, visibility and Stage 4 randomization, passes; our score is pose error only), the fixed anchor,
settled starts, a stand-in video without hands, bench placement (`rig_T_world`), and the LLM (Claude through our
transport, not DexAgent's own client). Contact parameters differ slightly from DexAgent's own convention: its defaults
for unspecified rigid assets go on explicit object-object pairs only (`contact_defaults.py:5-8`), while ours are on each
box, so fragment-table contacts mix 6 ms with the table's 5 ms instead of the 20 ms engine default (hand-fragment
contacts use DexAgent's explicit hand pairs either way).

## Tests

`tasks/dual_arm_crag_skull/tests/test_baselines_dexagent.py`: unit tests in venv-asm (geometry, plan and frames,
Stage 1 content, export orchestration with a fake worker, transport with a fake client, trajectory import, runner
status) and integration tests that skip without the DexAgent venv/clone: export through DexAgent's own checks, Stage 3
with a mock LLM passing both input gates (and failing them on a changed mask or video identity), and the responder
driving DexAgent's real `ExternalLLM` runner, including a budget stop.

The port (`test_baselines_dexagent_bench*.py`, `_executor.py`, `_s0_check.py`, `_execute_trial.py`): the CRX MJCF
against S0's chain (to 1e-7) and MuJoCo's own reading of it, the masses at the stated density; the YAM pieces'
silhouettes exact against brute force, the window kept open, the hand read by DexAgent's own HandRig; the scene and
setup.yaml (rig, cameras against the measured intrinsics, servo gains, planning share of S0, S0's contact band);
the checkout refusing a tampered tree or other S0 limits; the executor (labels to windows, the ramp, jaw targets to
widths, S0's rejections, windows cleared out of the band, the tool box's band, every ending released and scored, the
recorded start), the replay of the S0 verifier (rig frames, frame by frame), and a trial's execution, rescoring and
CLI. A mutation check of 35 changes to the port's code is caught 34 times (the one survivor is equivalent).

## What remains

- No live run yet: the shared pool and the sandbox are in place (2026-10-19); the first live run is one practice
  seed, whose cost, model calls, wall time and failure reasons decide how the rest of the pool is used. No accepted
  delivery has been produced, so `trajectory.py` is tested on synthetic and shipped example recordings only.
- The anchor stability finding is decided for the task itself: a fixture holds the anchor for every method (`baselines.anchor` in `configs/jing.json`), so a trial refuses `anchor_mode='free'`.
- `object_reach_check` finds no pose witness for some staged fragments; DexAgent may need to move stations.
- Runtime build cost (about 110 s per call) slows DexAgent's executions and gate suite; fewer boxes (set cover with
  overlapping boxes cut 502 to 399 in a probe) would need explicit inertia.
- The port (table above): DexAgent's tool library on the bench (step 4), the MuJoCo <-> Isaac calibration (step 6),
  a mock end-to-end trial (step 7). Its S0 limits are TEST values, so every shared-hardware record is `test_only`.
