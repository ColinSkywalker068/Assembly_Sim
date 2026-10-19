# S1 GPU smoke, 2026-10-18 (synthetic data)

Paths: on 2026-10-18 at 17:06 the workspace moved from `/local_data/jz6676/repos_review/` to `/local_data/jz6676/RoboAssembly/` (`runs/` to `RoboAssembly/runs/`, `venv-s1gpu` to `envs/s1gpu`, `venv-asm` to `envs/asm`). The records below keep the paths the runs used; the old paths are now links to the new places.

Synthetic two-arm pick/place/retreat episodes (`synthetic_pick_detour_v2`), not the skull task. Criteria v3 are in `../../s1/smoke.py`, written before these runs; the full record is `../s1_smoke_2026-10-18.json`.

## Latest: v5 bundle, FAIL on shared maxwell, PASS on torch (both pre-registered)

**Verdicts.** maxwell-v5-20261018_042157: **FAIL** (C5, C6). Torch 20232734, the identical bundle on an uncontended L40S with its own pre-registration (`/local_data/jz6676/repos_review/runs/s1_smoke_torch_v5_preregistration.md`, sha256 2283baa1050c, written 2026-10-18T05:35:37-04:00, before submission): **PASS**, all ten checks. The maxwell FAIL stands as pre-registered; the torch PASS shows that the code_v5 configuration meets criteria v3 on a GPU it does not share.

The v5 pre-registration (`/local_data/jz6676/repos_review/runs/s1_smoke_maxwell_v5_20261018_042157/PREREGISTRATION.md`, sha256 894e862e32ff, written 2026-10-18T04:21:57-04:00, before training) keeps criteria v3 unchanged and changes the configuration only: `policy.sampler_eta` 0 → 1 (DDIM-16 with DDPM posterior noise, still 16 network calls), diffusion training 8000 → 16000 steps, `synthetic_seed` 0 → 1 so the test windows are new, expected tests 101 → 114. Both changes were chosen from the post hoc diagnosis of the v4 run below.

| Check | Threshold | v4 maxwell | v4 torch 20226851 | v5 maxwell | v5 torch 20232734 |
|---|---|---|---|---|---|
| C1 | pytest exit 0, expected test count (v4 101, v5 114), failures/errors/skipped 0 | exit 0, 101 tests, 0/0/0 pass | exit 0, 101 tests, 0/0/0 pass | exit 0, 114 tests, 0/0/0 pass | exit 0, 114 tests, 0/0/0 pass |
| C2 | reload max abs diff <= 1e-5 (diffusion / regression) | 0.0e+00 / 0.0e+00 pass | 0.0e+00 / 0.0e+00 pass | 0.0e+00 / 0.0e+00 pass | 0.0e+00 / 0.0e+00 pass |
| C3 | place_first hits <= 0.15 / left fraction in [0.25, 0.75] / both modes >= 0.5 / final error <= 0.02 m | 0.217 / 0.60 / 0.93 / 0.0174 **FAIL** | 0.217 / 0.60 / 0.93 / 0.0174 **FAIL** | 0.000 / 0.53 / 1.00 / 0.0057 pass | 0.000 / 0.53 / 1.00 / 0.0057 pass |
| C4 | regression place_first hits >= diffusion hits + 0.15 | 0.800 vs 0.217 pass | 0.800 vs 0.217 pass | 0.867 vs 0.000 pass | 0.867 vs 0.000 pass |
| C5 | DDIM-16 batch-1 median <= 100 ms | 80.7 ms pass | 70.3 ms pass | 315.3 ms **FAIL** | 52.2 ms pass |
| C6 | every deployment-path chunk valid and executable | 64/64 valid, 0 not executable pass | 64/64 valid, 0 not executable pass | 64/64 valid, 64 not executable **FAIL** | 64/64 valid, 0 not executable pass |
| C7 | pick_first chosen fragment rate >= 0.9 | 0.983 pass | 0.983 pass | 1.000 pass | 1.000 pass |
| C8 | place_mid mode agreement >= 0.9 / hits <= 0.15 | 0.987 / 0.004 pass | 0.987 / 0.004 pass | 1.000 / 0.000 pass | 1.000 / 0.000 pass |
| C9 | place_late final error <= 0.02 m / final width >= 0.07 m | 0.0044 / 0.0797 pass | 0.0044 / 0.0797 pass | 0.0011 / 0.0799 pass | 0.0011 / 0.0799 pass |
| C10 | raw inactive-arm excursion median <= 0.01 m / active-arm median >= 0.05 m | 0.0201 / 0.1972 **FAIL** | 0.0201 / 0.1972 **FAIL** | 0.0027 / 0.1642 pass | 0.0027 / 0.1642 pass |

**Same code, same policy, different latency.** The torch run used the same 20 files (per-file sha256 checked) and `code_sha256` 389ee4243f86. Its training-loss means (160 diffusion, 40 regression) and validation losses equal the maxwell run value for value, and of the 228 evaluation numbers 17 differ, all by a relative 6e-14 or less, so they agree to 13 significant digits. The checkpoint files are not byte-identical; torch.save of identical contents is byte-reproducible, so some saved tensor differs (not inspected: the torch checkpoints were not copied). The v4 bundle shows the same pattern on torch 20226851: C3 and C10 fail with the same values as on maxwell, so the v4 failures belong to that configuration, not to the maxwell GPU.

| Run | GPU | DDIM-16 median / p90 | DDPM-100 median | deployment path median | training |
|---|---|---|---|---|---|
| v4 maxwell | maxwell GPU 2, RTX 6000 Ada (79 % used by others at start) | 80.7 / 101.3 ms | 487.1 ms | 64.9 ms | 8000 steps at 17.56 steps/s |
| v4 torch 20226851 | torch gl065, L40S | 70.3 / 70.6 ms | 416.7 ms | 72.1 ms | 8000 steps at 22.44 steps/s |
| v5 maxwell | maxwell GPU 5, RTX 6000 Ada (99-100 % used by others) | 315.3 / 329.8 ms | 2803.4 ms | 395.5 ms | 16000 steps at 4.68 steps/s |
| v5 torch 20232734 | torch gl067, L40S | 52.2 / 52.3 ms | 306.7 ms | 53.5 ms | 16000 steps at 25.45 steps/s |

Why C5 and C6 failed on maxwell. Both are latency checks: DDIM-16 took 315.3 ms per sample (limit 100 ms) and the deployment path 395.5 ms, so all 64 contract-valid chunks were already past `valid_until` (250 ms). GPU 5 stayed at 99–100 % utilisation from other users in all 143 samples taken during the run, and samplers that do not depend on eta slowed as much as DDIM did. On the uncontended L40S the same bundle takes 52.2 ms for DDIM-16 and 53.5 ms on the deployment path, with all 64 chunks executable, which confirms contention as the cause of the maxwell latency.

v5 training: diffusion 16000 steps in 3419.3 s on maxwell and 628.7 s on torch, final val loss 0.0028 (v4 0.0073); regression 4000 steps, final val loss 0.0016. Train windows 10560. v5 place_first: diffusion min clearance median 0.0580 m, regression hits 0.867 with clearance -0.0215 m; inactive-arm p90 0.0039 m.

Torch outputs: `/scratch/jz6676/jing_assembly/runs/s1_smoke_<job>/` (checkpoints stay there); local copies without checkpoints: `/local_data/jz6676/repos_review/runs/s1_smoke_torch_<job>/`.

## v4 run (after the S1 code review, criteria v3)

**Verdict: FAIL** for maxwell-20261018_033008 (C3, C10 fail). Torch job 20226851 (same code, L40S) ran after this section was written (06:02–06:14) and fails the same two checks with the same values; it is a comparison, not a verdict (table above).

| Run | Role | Node / GPU | State | pytest | Verdict |
|---|---|---|---|---|---|
| maxwell-20261018_033008 | maxwell local GPU reference run (shared RTX 6000 Ada, GPU 2), same code_v4 bundle and criteria v3 as torch job 20226851; not a torch run | maxwell (GPU 2, shared) / NVIDIA RTX 6000 Ada Generation | local run (no Slurm), script exit 1 = verdict FAIL, 00:12:32 | 101 run, 0 failed, 0 skipped | FAIL |
| 20226851 | jing-s1-smoke, submitted 2026-10-18T03:24:55 (code_v4, expected_tests 101); same-hardware comparison, not a verdict | gl065 / NVIDIA L40S | FAILED (script exit 1 = verdict FAIL), 00:11:52 | 101 run, 0 failed, 0 skipped | FAIL (C3, C10) |
| 20219996 | initial S1 code (faf31f75), 74 tests (superseded: synthetic_detour_v1, earlier criteria) | gl038 / NVIDIA L40S | COMPLETED, 00:13:28 | 74 run, 0 failed, 0 skipped | PASS |
| 20221507 | final S1 code (9498adb9; contracts as shipped 01:42), 80 tests (superseded: synthetic_detour_v1, earlier criteria) | gl063 / NVIDIA L40S | COMPLETED, 00:12:42 | 80 run, 0 failed, 0 skipped | PASS |

### Criteria v3 against the measurements

| Check | Threshold | maxwell-20261018_033008 |
|---|---|---|
| C1 | pytest exit 0, 101 tests, failures/errors/skipped 0 | exit 0, 101 tests, 0/0/0 pass |
| C2 | reload max abs diff <= 1e-5 (diffusion / regression) | 0.0e+00 / 0.0e+00 pass |
| C3 | place_first hits <= 0.15 / left fraction in [0.25, 0.75] / both modes >= 0.5 / final error <= 0.02 m | 0.217 / 0.60 / 0.93 / 0.0174 **FAIL** |
| C4 | regression place_first hits >= diffusion hits + 0.15 | 0.800 vs 0.217 pass |
| C5 | DDIM-16 batch-1 median <= 100 ms | 80.7 ms pass |
| C6 | every deployment-path chunk valid and executable | 64/64 valid, 0 not executable pass |
| C7 | pick_first chosen fragment rate >= 0.9 | 0.983 pass |
| C8 | place_mid mode agreement >= 0.9 / hits <= 0.15 | 0.987 / 0.004 pass |
| C9 | place_late final error <= 0.02 m / final width >= 0.07 m | 0.0044 / 0.0797 pass |
| C10 | raw inactive-arm excursion median <= 0.01 m / active-arm median >= 0.05 m | 0.0201 / 0.1972 **FAIL** |

Held-out test metrics, run maxwell-20261018_033008 (8 samples per window; demonstrations: the recorded action, 1 per window):

| Phase | Metric | Diffusion | Regression | Demonstrations |
|---|---|---|---|---|
| pick_first | chosen_fragment_rate | 0.983 | 0.967 | 1.000 |
| pick_first | final_error_m_median | 0.0139 | 0.0572 | 0.0002 |
| place_first | obstacle_hit_rate | 0.217 | 0.800 | 0.000 |
| place_first | left_fraction_of_clear | 0.60 | 0.67 | 0.53 |
| place_first | episodes_with_both_modes | 0.93 | 0.00 | 0.00 |
| place_first | final_error_m_median | 0.0174 | 0.0264 | 0.0002 |
| place_first | min_clearance_m_median | 0.0267 | -0.0246 | 0.0589 |
| place_mid | mode_agreement_of_clear | 0.987 | 1.000 | 1.000 |
| place_mid | obstacle_hit_rate | 0.004 | 0.000 | 0.000 |
| place_late | final_error_m_median | 0.0044 | 0.0034 | 0.0002 |
| place_late | final_width_m_median | 0.0797 | 0.0810 | 0.0800 |
| arms | active_motion_m_median | 0.1972 | 0.1464 | 0.1648 |
| arms | inactive_motion_m_median | 0.0201 | 0.0069 | 0.0006 |
| arms | inactive_motion_m_p90 | 0.0514 | 0.0171 | 0.0008 |

Batch-1 sampling latency on the GPU (median ms):

| Run | DDIM 1 | DDIM 2 | DDIM 4 | DDIM 8 | DDIM 16 | DDPM 100 | regression |
|---|---|---|---|---|---|---|---|
| maxwell-20261018_033008 | 10.8 | 14.7 | 23.8 | 43.1 | 80.7 | 487.1 | 5.3 |
| 20219996 (superseded) | 7.4 | 10.7 | 17.0 | 29.7 | 55.1 | 332.9 | 4.9 |
| 20221507 (superseded) | 10.4 | 14.4 | 23.1 | 40.7 | 75.7 | 441.3 | 5.7 |

Training (EMA weights evaluated):

| Run | Policy | Steps | Seconds | Params (M) | Train windows | Final val loss | Final test loss | Reload max abs diff | Deployment-path chunks |
|---|---|---|---|---|---|---|---|---|---|
| maxwell-20261018_033008 | diffusion | 8000 | 455.5 | 53.45 | 10560 | 0.0073 | 0.0070 | 0.0e+00 | 64/64 valid, 0 not executable, median 64.9 ms |
| maxwell-20261018_033008 | regression | 4000 | 159.0 | 35.5 | 10560 | 0.0019 | 0.0017 | 0.0e+00 | 64/64 valid, 0 not executable, median 9.4 ms |

Training windows per (skill, arms), all above the 16-window minimum: pick:left 1495, pick:right 1625, place:left 2185, place:right 2375, retreat:left 1380, retreat:right 1500.

### Exploratory sampler diagnosis (post hoc)

Post hoc, after the v3 verdict of the maxwell run; changes no verdict, criterion or code. reports/s1_smoke/diagnose_sampler.py re-evaluates the saved EMA checkpoint of the maxwell run on the same test windows with the same evaluation code and deterministic kernels; only the sampler changes. The ddim_16 variant reproduces metrics.json exactly (max abs diff 0.0). A first pass without deterministic kernels moved rates by up to 1/30 episodes and distances by up to 0.14 mm.

Run s1_smoke_maxwell_20261018_033008, checkpoint 4636aa291081, same test windows:

| Sampler | place_first hits | min clearance (m) | final error (m) | pick_first chosen | inactive median (m) | inactive p90 (m) | active median (m) |
|---|---|---|---|---|---|---|---|
| ddim_16 | 0.217 | 0.0267 | 0.0174 | 0.983 | 0.0201 | 0.0514 | 0.1972 |
| ddim_100 | 0.171 | 0.0327 | 0.0158 | 0.988 | 0.0131 | 0.0390 | 0.1906 |
| ddpm_100 | 0.096 | 0.0497 | 0.0096 | 1.000 | 0.0034 | 0.0054 | 0.1642 |
| ddim_16_eta_0.5 | 0.204 | 0.0339 | 0.0131 | 1.000 | 0.0081 | 0.0169 | 0.1741 |
| ddim_16_eta_1 | 0.167 | 0.0414 | 0.0117 | 1.000 | 0.0048 | 0.0074 | 0.1646 |
| ddim_16_unclipped_eps_direction | 0.217 | 0.0276 | 0.0146 | 1.000 | 0.0126 | 0.0258 | 0.1765 |
| ddim_16_no_x0_clip | 0.096 | 0.0951 | 32.2864 | 0.675 | 50.4816 | 72.7207 | 56.2228 |

The same network sampled with DDPM-100 meets both failed thresholds (place_first hits 0.096, inactive-arm median 3.4 mm) but takes 487 ms per sample. At the deployed cost of 16 network calls, DDIM with eta = 1 gives 0.167 hits (still above 0.15) and 4.8 mm. Deterministic DDIM-100 gives 0.171 and 13.1 mm, so the gap comes mostly from the deterministic sampler, not from too few steps. Without x0 clipping DDIM-16 diverges (tens of metres). Not tested: a verdict run with eta = 1 or with longer training (val loss still fell 5% over the last 1000 of 8000 steps); both need criteria declared before the run.

Verdict checks:

- maxwell-20261018_033008: C1_pytest pass, C2_trained_and_reloadable pass, C3_diffusion_modes FAIL, C4_regression_averages_modes pass, C5_latency pass, C6_inference_chunks pass, C7_pick_conditioned_on_fragment pass, C8_mid_window_follows_mode pass, C9_late_window_releases_at_goal pass, C10_only_requested_arm_moves FAIL
- 20219996 (superseded, earlier criteria and data): C1_pytest pass, C2_trained_and_reloadable pass, C3_diffusion_modes pass, C4_regression_averages_modes pass, C5_latency pass
- 20221507 (superseded, earlier criteria and data): C1_pytest pass, C2_trained_and_reloadable pass, C3_diffusion_modes pass, C4_regression_averages_modes pass, C5_latency pass, C6_inference_chunks pass

Notes:

- Verdict of the maxwell reference run: FAIL. C3 fails on the obstacle hit rate alone (0.217 > 0.15; left fraction 0.60, both modes in 28/30 episodes, final error 1.7 cm all pass). C10 fails on the inactive arm (median raw excursion 2.0 cm > 1 cm, p90 5.1 cm; active arm 19.7 cm). C1, C2, C4-C9 pass.
- C10 measures the network output only. On the deployment path S1 overwrites the inactive arm with an exact hold at its last commanded width; all 64 deployment-path chunks were contract-valid and executable (C6).
- The arm and fragment conditioning works: pick_first goes to the requested fragment in 98.3% of samples (regression 96.7%), and the requested arm moves about 10 times as far as the other one.
- Regression baseline: place_first hits 0.80 with no episode showing both modes and a median clearance of -2.5 cm (through the obstacle), so C4 passes by a wide margin (0.80 >= 0.217 + 0.15).
- Compared with the superseded synthetic_detour_v1 runs (diffusion hits 0.017, clearance 7.4 cm), v2 adds the arm choice, two fragments and three skills with the same 8000 training steps, and its demonstrations keep a smaller clearance (median 5.9 cm versus 7.9 cm).
- DDIM-16 latency on the shared maxwell GPU: median 80.7 ms, p90 101.3 ms (budget 100 ms for the median). Deployment-path live_sample -> ActionChunk: median 64.9 ms, max 108.7 ms, under the 250 ms valid_until.
- Torch job 20226851 (same bundle, L40S) ran after this section was written: FAIL on C3 and C10 with the same values; it is a same-hardware comparison, not part of the v4 verdict.

Limitations:

- Synthetic data only: no Isaac, UMI or robot episodes exist yet, so nothing here measures skull assembly ability.
- Images are 32x32 top-down rasters; the real three-view resolution, crops and a pretrained encoder are untested.
- Training preloads the whole split onto the GPU; real datasets need a streaming loader.
- The maxwell GPU was shared with other users during the run, so its latency numbers include contention.
- All synthetic episodes succeed and carry no oracle inputs or stream gaps; the success filter, oracle refusal and gap dropping are exercised only by the unit tests.
- Mode-coverage metrics depend on privileged ground truth and the synthetic generator; they are a sanity check of the method, not a benchmark.
