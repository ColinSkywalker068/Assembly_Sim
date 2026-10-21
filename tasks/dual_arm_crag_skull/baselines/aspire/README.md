# ASPIRE baseline (integration only, no results yet)

[ASPIRE](https://github.com/NVlabs/ASPIRE) (NVlabs, Apache-2.0) is a Code-as-Policy harness: a coding agent
(Claude Code or Codex following `.claude/skills/iterative-debugging`) writes and repeatedly repairs one Python
program per task against a typed robot API (`cap_harness`), using simulator trials on practice seeds. The frozen
program (`fix_code.py`) is then evaluated on held-out seeds, and a trial succeeds when the adapter's
`check_success()` says so. We pin commit `fa00f0ef2bc1d91d594b9aa502c22133c9f538f7` (`configs/jing.json`
`baselines.aspire`).

## What this does and does not do

Does:

- puts the dual CRX-10iA/L workcell behind ASPIRE's `EnvironmentAdapter` as a 6-DoF embodiment `fanuc_crx`
  (`adapter.py`, over our `baselines.backend.AssemblyBackend`), with S0's DLS IK as an ASPIRE IK provider (`ik.py`);
- runs a frozen program for one seeded trial through ASPIRE's own `run_program_with_api` and `RunRecorder`, so ASPIRE
  writes its usual per-trial run directory, and scores the trial with `baselines.scoring` (`runner.py`);
- evaluates a frozen program on held-out seeds and appends `RunRecord`s to a JSONL file.

Does not: run ASPIRE's coding-agent loop, call any language model, or start SAM3/Contact-GraspNet/cuRobo/PyRoki.

Every rig is the one hardware set all compared methods use: two CRX arms with the YAM linear_4310 gripper
(`s0.kinematics.yam_model`, provisional adapter plate), S0 at its 60 Hz servo rate. Backends: `kinematic`, the
physics-free `KinematicBackend` (a test double), and `isaac`, the Isaac Sim `AssemblyBackend` (`sim/backend.py`).
The S0 limits are TEST values on both because `configs/jing.json` has none, so every record is `test_only`.
`reference_program.py` (kinematic capture) and `reference_program_physics.py` (top-down grasp with contact
declarations) are hand-written programs that prove the path end to end; they are not ASPIRE results.

## The patch

`patches/0001-fanuc-crx-embodiment.patch` (four hunks in three files):

1. `contracts.py` `JOINT_DIMENSIONS` gains `"fanuc_crx": 6`. Every contract with joints validates its embodiment
   against this mapping, and `CapApi` silently replaces an unknown embodiment with the 7-DoF `robosuite`
   (`api.py:108-111`).
2. `contracts.py` `MotionStrategy` accepts the IK solver name `"crx_dls"`. It only accepts pyroki/curobo/mink, so
   otherwise `plan_motion()`/`move_to_pose()` could reach our IK only under another solver's name, mislabelling the
   artifacts.
3. `api.py` `CapApi.register_tools` lists `declare_contact` and `clear_contact` among the adapter extensions.
4. `registry.py` approves the `fanuc_crx.` extension prefix and names its only public extensions
   (`FANUC_CRX_PUBLIC_TOOL_NAMES`); `s2/aspire.py` builds the direct-mode registry with that allowlist.

No CLI or benchmark code changes.

`checkout.py` clones the reference clone (read only) into an empty directory, detaches at the commit, runs
`git apply --check`, applies, and verifies that the working tree differs from the commit by exactly the patch, with
no untracked files and no ignored ones besides `src/cap_harness.egg-info/` (stale bytecode could otherwise run
instead of the verified sources; always use `-B`).
`runner.py` re-verifies the checkout `cap_harness` is imported from before every trial.

## Conventions

- Frames: ASPIRE's base frame `robot_base` is our world frame (metres, z up). All poses a program reads or sends are
  in it, quaternions `wxyz`. Camera poses are `robot_base_T_camera` with OpenCV axes; intrinsics are passed
  unchanged (ASPIRE back-projects integer pixel indices like `perception.depth_to_world`).
- Arms: native `left`/`right` (`contracts.actions.ARMS`). `resolve_arm` maps ASPIRE's default `primary` to `left`
  and `secondary` to `right` for the calls that take an arm name; `RobotAction`, trajectories and `set_grippers`
  must use `left`/`right`.
- Grippers: 0 is the minimum and 1 the maximum of S0's configured width range, linear in width; intermediate widths
  are held. `set_gripper` steps until the width is reached or stops changing (`completion` in diagnostics). A
  measured opening up to 5 % of the stroke outside the range reads as 0 or 1; further out the adapter raises.
- Task identity: `TaskContext(suite='fragment_assembly', task_id=0, task_name=<AssemblyTask.task_id>)`; ASPIRE
  needs an integer task id and there is one task per task file.

## Every command goes through S0's checks

`step`, `execute_trajectory` and `go_home` build the joint and width targets of every control period (held arms keep
their last commanded targets) and pass the whole sequence to `s0.adapter.check_joint_path` from the measured state:
joint position, velocity and acceleration limits, arm-arm clearance, link and tool-box clearance above the table
(since 2026-10-21 the gripper's boxes at its widest opening over the command, from the measured opening, the last
commanded width and the command's widths: S0's `finger_opening_margin_m`, 4 mm in the TEST limits),
TCP height, plus the width range; the geometry is also checked between periods at S0's `servo_hz` (joints
interpolated linearly). `set_gripper` checks each of its hold periods the same way, `set_grippers` its one period.
Any violation rejects the command before the backend steps: nothing moves, the program gets a typed failure with
S0's reason, and `rejected_commands` counts it. Nothing is clipped or re-timed. `step`, trajectories and `go_home`
return `error.code == 'safety_interlock'`; a refused `set_gripper`/`set_grippers` raises inside the adapter, which
CapApi reports as `execution_failed`/`controller_failed` with the `safety_interlock` message, because CapApi keeps a
returned-but-failed gripper target as the opening of later motions. Consequences a program author must know (also in
the task context):

- a joint target is reached in one control period, so `step()`/`move_to_joints()` only work for small moves; large
  moves go through `plan_motion`/`move_to_pose`, whose joint interpolation is sized by the adapter's
  `action_bounds`, 90 % of min(v_max, a_max (w + dt) / 2): ASPIRE's interpolation starts and stops within one period
  dt, and S0 reads that step over half the period plus the window w the measured velocity averages over (the
  backends report each period's velocity, w = dt: a_max dt; until 2026-10-19 S0 read every first step as from an
  instantaneous velocity, a_max dt / 2, so ASPIRE's joint speeds have doubled, 0.15 -> 0.30 rad/s at 60 Hz and the
  TEST 20 rad/s^2);
  the adapter's `go_home` hook moves to the reset joints the same way (CapApi's fallback would jump in one period);
- every trajectory gets one extra hold period at its last waypoint, checked with it, so the commanded motion ends at
  rest;
- targets are read as reached at the backend's control period (60 Hz, S0's servo rate, on both backends) with constant
  acceleration from the measured state: the first period needs 2 (v - qd_now) / dt, later ones the second difference,
  and the arm must be able to brake inside the joint limits after the last target (S0 checks S1 chunks as splines at
  `servo_hz`). A real backend must step at the rate its joint targets are executed at for the limits to mean the same
  thing, and a measured state or clock that is not finite is rejected;
- close to the table only inside a declared contact window (S0's contact phase, the rule our S1 chunks follow): a
  program calls `fanuc_crx.declare_contact(kind, fragment_id, arm=...)` before grasping (`'grasp'`, the window
  centred on the fragment's observed pose) or placing (`'place'`, centred on its goal), and
  `fanuc_crx.clear_contact(arm=...)` after leaving the band. The program names a fragment, never coordinates; neither
  call is allowed while the arm is inside the band (lift first), so a window stays where it was declared. While a
  window is declared `action_bounds` shrink so that no finger corner exceeds 80 % of S0's contact speed (joint speed
  times the summed lever arms to the farthest finger corner, at the measured pose). S0 checks nothing between targets
  next to a contact, so it refuses targets there more than one servo period apart: contact motions need a backend
  stepping at S0's servo rate (both rigs do). The rule and the calls are in the task context
  (`metadata['contact_rule']` and the language);
- `move_synchronized` needs a `curobo-integrated` planner and returns ASPIRE's typed `unsupported` failure.

The limits are passed explicitly (`S0Limits`) because the S0 values in `configs/jing.json` are still null; the CLI
uses the labelled test values of `tests/test_s0_adapter.py` plus the contact phase of the S0 tests, and marks every
record `test_only`.

## Information parity with our method

Our S2 gets goal poses (GARF candidates or the oracle layout) and the observed fragment poses with their source; so
does the program, in `get_task_context().metadata`: `fragment_ids`, `anchor_id`, `moving_ids`, `anchor_fixture`,
`goal_source`, `goal_poses`, `assembly_plan` (the task's plan, ground truth since 2026-10-20: each moving fragment's
id and the world direction of its last straight move, in the order they go in; the task language says how to read
it), `fragment_local_bounds` (fragment frame), `placement_tolerance`, `table_top_z_m`,
`contact_rule`, `controller` (limits and check semantics), and, refreshed on every call and every observation,
`observed_fragment_poses`, `observed_pose_sources` and `observed_pose_times_s` from
`backend.observed_fragment_poses()`. ASPIRE has no object-pose slot in `Observation`, and the patch adds only the
two contact tools as extensions, so the task context carries them. Privileged poses are read only by
`check_success()` after the program has finished, and are not in ASPIRE's artifacts. `check_success()` first ends
the trial as every method must (`baselines/backend.py`): both grippers open fully and the arms hold their last
targets for the task's `settle_s` (1 s), checked by S0 like any command but outside the program's step budget; if S0
refuses that hold (the program ended with the arm moving too fast to stop), the score is not settled and not a
success (`settle_error` in the record's notes). Both backends report `oracle_sim` poses and the layout gives
`oracle_layout` goals, so a fair comparison must give both methods the same pose and goal sources. The program does
not get the seed or its split.

## Setup

```bash
cd /local_data/jz6676/RoboAssembly   # the repository root; the paths below are relative to it
python3.12 -B -m tasks.dual_arm_crag_skull.baselines.aspire checkout --source third_party/ASPIRE
uv venv --python /usr/bin/python3.12 envs/aspire
uv pip install --python envs/aspire/bin/python numpy==1.26.4 scipy==1.12.0 \
  trimesh==4.12.2 networkx==3.4.2 pytest==8.4.2 Pillow==11.0.0 rtree==1.4.1 "imageio[ffmpeg]==2.37.0" \
  imageio-ffmpeg==0.6.0 -e third_party/aspire-fa00f0e-fanuc_crx
```

`checkout` defaults to `third_party/aspire-fa00f0e-fanuc_crx` (git-ignored) and refuses a
non-empty destination. Install the patched checkout, never the reference clone. Check afterwards that numpy is
still 1.26.4.

## Running

`--task` is the JSON record (`AssemblyTask.to_record()`) of the base task (seed `None`), as written by
`python -m tasks.dual_arm_crag_skull.baselines task --layout LAYOUT.json --anchor ID --output task.json` without
`--seed`. The adapter checks it against its layout under the canonical protocol (`baselines.task.verify_task`)
and draws the seed's initial condition. `--mode practice` runs practice and selection seeds, `--mode evaluation`
only held-out seeds; any other seed raises before the backend is touched, and the configuration must carry the
protocol of `configs/jing.json` (another one cannot relabel a held-out seed or change the criterion). A held-out
trial is claimed atomically before the backend is touched (`baselines.scoring.claim_heldout`, in
`baselines.heldout_claims_dir`), always appended to `--records`, and a program variant never runs twice on the same
held-out seed of a task, whatever records file is given; a trial that ends in an exception or Ctrl-C is recorded as
`error` before the exception propagates.

```bash
P=envs/aspire/bin/python
PYTHONDONTWRITEBYTECODE=1 $P -B -m tasks.dual_arm_crag_skull.baselines.aspire run --program fix_code.py \
  --task task.json --seed 101 --mode practice --output-root runs --backend kinematic
PYTHONDONTWRITEBYTECODE=1 $P -B -m tasks.dual_arm_crag_skull.baselines.aspire evaluate --program fix_code.py \
  --task task.json --seeds 1-100 --records aspire.jsonl --output-root runs --backend kinematic
# On Isaac Sim: the interpreter of the Isaac Sim environment (on maxwell the conda env with isaacsim 5.1.0, which
# already has cap_harness's only dependencies numpy, scipy, PyYAML and requests), the patched checkout's src on
# PYTHONPATH, nothing installed into that environment.
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=.:third_party/aspire-fa00f0e-fanuc_crx/src PYTHONDONTWRITEBYTECODE=1 \
  "$ISAAC_PYTHON" -B -m tasks.dual_arm_crag_skull.baselines.aspire run --program fix_code.py \
  --task task.json --seed 101 --mode practice --output-root runs --backend isaac --no-videos
```

`--max-steps` defaults to 18,000 control steps per trial, five minutes of simulated time at 60 Hz; provisional, and
every compared method must get the same budget. On `isaac` one backend serves every seed of the process (each trial
resets it), the trial starts where the fragments settled (`settled_start.json` in the run directory), and the
process ends through the backend's `shutdown`, Kit's way out.

`evaluate` (or `runner.evaluate_frozen_program`) guards every seed and checks the records first, runs each seed on
the backend the factory returns (a fresh kinematic double, or the one Isaac backend reset), appends one `RunRecord`
per seed as it finishes, and stops if the program bytes change (the copy ASPIRE saved and executed must match them
too). The variant is the program's sha256, so any edit is a new variant; evaluate one frozen program per ASPIRE run.
Each trial writes ASPIRE's run directory under
`<output-root>/dual_arm_crag_skull/fragment_assembly/00-<task>/<seed>/<run-id>/` (`run.json`, `outcome.json`,
`manifest.json`, `source/`, `episode/`, `trace/`, `media/`) and returns `RunRecord(method='aspire', variant=<program
sha256[:12]>, sim_trials=1, cost_usd=None, backend=<label>, test_only=<TEST S0 values or kinematic>)`; `summarize`
refuses test-only records unless asked. Status: `program_completed`/`task_succeeded` -> `completed`, `step_limit` ->
`budget_exhausted`, `environment_truncated` -> `timeout`, `program_error` and `environment_terminated` -> `error`;
success is the score's. Harness failures and interrupts are recorded as `error` (when a records file is given) and
re-raised.

Tests: `PYTHONDONTWRITEBYTECODE=1 $P -B -m pytest tasks/dual_arm_crag_skull/tests/test_baselines_aspire.py -q`
(the file skips where `cap_harness` is not installed).

## Security

ASPIRE executes generated Python in this process. Its runtime validates the program's AST and restricts builtins and
imports, but that is not a sandbox, and the coding-agent loop runs arbitrary code by design. Run programs and the
loop on an isolated host without credentials, tokens or sensitive mounts, with restricted network access, and never
with access to robot hardware.

## What remains

- Real S0 limits in `configs/jing.json` (all records are `test_only` until then), the YAM adapter plate and finger
  force (provisional), and one step budget for every compared method.
- (Done 2026-10-19) the protective stop, the same for every method (`configs/jing.json` `s0.protective_stop`): the
  adapter turns a `ProtectiveStop` into a terminated episode (ASPIRE records `environment_terminated`), refuses every
  later command, does not release, settle or score, and the runner records `protective_stop` with the stop's log in
  the run directory (`protective_stop.json`). `reference_program_physics.py` jammed on the fixed anchor before
  (`runs/aspire_isaac_ref_20261019_005140`), because the fragments interlock (`../README.md`).
- The planner slows both arms while either has a contact declared (ASPIRE reads one joint bound for all arms).
- The adapter takes an observation, cameras included, after every control step; on Isaac the backend re-renders at
  most at 10 Hz, but three 1920x1440 RTX cameras still cost wall time.
- A wall-clock limit per trial: refused commands use no steps and no simulated time, so a program that retries a
  refused command forever never ends (ASPIRE has no wall-clock limit either).
- A benchmark profile and task context (instructions, tool notes) for ASPIRE's coding-agent loop, with a counting
  shim around `run_program` that refuses held-out seeds and trials beyond the practice budget and records how many
  trials the loop spent (`RunRecord.sim_trials`).
- Cost metering of the loop (model calls and dollars into `RunRecord.cost_usd`/`llm_calls`).

## Shared S2 program runtime

Our S2 and this baseline now share `s2/aspire.py` and ASPIRE's `ToolRegistry` / `ProgramExecutor`.
The baseline uses `direct` mode with CapApi and the checked CRX adapter. Our default S2 uses `skills`
mode: a frozen program requests GARF candidates and submits one validated skill to the existing
S1/S0 orchestrator. Only the selected mode's tools are registered. Direct trials launched from the
S2 CLI still record method `aspire`, with `execution_mode=direct` in artifacts; they are the same method.
This is runtime/interface integration, not a completed coding-agent learning campaign or physical rollout.
