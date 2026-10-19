# S1: conditional Diffusion Policy

Status: the data contract, dataset, policy, training CLI and checks are implemented and tested on
synthetic episodes only. No skull, simulator or robot data has been recorded or trained on yet.
Revised 2026-10-18 after an S1 code review (3 HIGH, 8 MEDIUM, 4 LOW findings, all fixed with tests).

| Module | Role | Needs |
|---|---|---|
| `episode.py` | Episode record (`meta.json` + npz per episode, schema 2; `privileged.npz` never loaded by the policy loaders), pose and camera streams (exposure + receive times), skill segments with provenance, causal state sampling, gap measurement, runtime buffers | numpy, scipy |
| `dataset.py` | Windows with end-of-segment hold padding, conditioning encoder shared with `live_sample`, split by initial condition, train-only normalizer, trained (skill, arms) combinations | numpy, scipy |
| `chunking.py` | One prediction (16, 20) -> validated `contracts.actions.ActionChunk` (held arms: exact pose hold and last commanded width; width clips counted) | numpy |
| `policy.py` | Chi et al. conditional 1D U-Net, per-camera ResNet-18 with GroupNorm, DDPM/DDIM (`sampler_eta` sets the DDIM noise, default 0), EMA, MSE regression baseline, layout-checked checkpoints | torch, torchvision |
| `train.py` | Training CLI: config JSON, seeds, checkpoint (+ trained combinations, image shape, layout), metrics | torch |
| `runtime.py` | `S1Runtime(checkpoint).chunk(request, tcp, cameras, *, commanded_widths, observation_time=None)`; refuses untrained (skill, arms) combinations, which S2 also gates when given `trained_combinations` | torch |
| `synthetic.py` | Deterministic pick/place/retreat demos for both arms, two identical-looking fragments, two equally likely detour modes | numpy |
| `smoke.py` | Pre-declared PASS/FAIL criteria of the GPU smoke run (v3) | stdlib |

Sample layout (one window at grid step i), conditioning layout version 2:

- images `(2, 3, H, W, 3)` uint8 + mask `(2, 3)`;
- low-dim `(2, 29)`: each hand's pose relative to its own pose at i (9 each), both widths, `left_T_right` (9);
- skill index into `contracts.skills.SKILL_IDS`; active-arm mask `(2,)` in ARMS order (arms are canonicalised);
- goal pose and observed fragment pose, each relative to the first active hand's anchor (9 + mask);
- action `(16, 20)`: future poses at i+1..i+16 relative to each hand's TCP at i, encoded with
  `contracts.actions.hand_vector`; steps past the segment end hold the segment's last pose and width.

Anchors (absolute TCP poses) are returned for decoding only and never reach the network. Checkpoints
store `SKILL_IDS`, ARMS, cameras, horizons and the layout version and refuse to load under another.

Timing. Observations use the latest TCP sample at or before each step (strictly causal; history may
reach into the previous skill segment, as the robot's real history does, and repeats the first state
only at the recording start). Camera frames are usable once received and their age counts from the
exposure. Commanded supervision is stamped at the time each command is meant to be reached.
Waypoint k of a chunk is stamped `observation_time + k / 10`; latency is absorbed by S0 skipping
waypoints already in the past and refusing chunks after `valid_until`. `S1Runtime` defaults
`observation_time` to the newest time both TCP buffers have reached.

Data hygiene. Only episodes with `success is True` train unless `include_failures`; ground-truth
goals (`oracle_ground_truth`) or simulator fragment poses (`oracle_sim`) raise unless
`allow_oracle`; windows whose supervision interpolates across a pose-stream gap longer than 0.1 s,
or whose state is older than 0.1 s, are dropped and counted. Splits group episodes by
`initial_condition_id` (else geometry hash + seed), so retries of one layout never straddle splits.

A degenerate predicted rotation raises `contracts.actions.ActionRejected`; callers treat that as a
failed step, never as motion. S1 clips predicted widths only at 0 (no YAM width limit is known yet);
S0 must enforce the gripper's real range and keep holds at the last commanded width.

```bash
python -m pytest tasks/dual_arm_crag_skull/tests/test_s1_*.py tasks/dual_arm_crag_skull/tests/test_contracts.py  # 114 tests on 2026-10-18
python -m tasks.dual_arm_crag_skull.s1.train --synthetic 300 --output RUN --config CONFIG.json
python -m tasks.dual_arm_crag_skull.s1.train --episodes EPISODES_DIR --output RUN
```

On a shared CPU machine set `OMP_NUM_THREADS` (e.g. 8): torch otherwise starts one thread per core.

GPU smoke results: `../reports/s1_smoke/README.md`. On 2026-10-18 the maxwell run of this revision with the
default deterministic DDIM-16 sampler failed 2 of the 10 checks fixed before the run (criteria v3): C3 (21.7%
of the samples taken just before the carry clip the obstacle; limit 15%) and C10 (the raw prediction for the
arm that should stay still moves 2.0 cm; limit 1 cm; deployment overwrites that arm with an exact hold).
A pre-registered rerun on fresh episodes with `sampler_eta = 1` and 16000 training steps passes both (no hits,
2.7 mm) but fails C5 and C6 on latency (DDIM-16 315 ms, every chunk past `valid_until`) on a GPU that other
users kept at 99-100% utilisation; that FAIL stands. On an uncontended torch L40S the same bundle passes all
ten checks (job 20232734, pre-registered separately): DDIM-16 52 ms, deployment path 54 ms, and training-loss
logs equal to the maxwell run value for value, so the maxwell latency came from contention.
