"""Inspect the proposed configuration without importing Isaac Sim."""

import argparse
import json
from pathlib import Path

from .grippers import add_gripper_arguments, selected_gripper, require_scene_gripper

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "jing.json"


def validate(config):
    selected_gripper(config)
    if config["schema_version"] != 1:
        raise ValueError("unsupported schema version")
    if config["dataset"]["minimum_fragments"] < 6:
        raise ValueError("the skull task requires more than five fragments")
    if len(config["cameras"]) != 3 or set(config["cameras"]) != {"left_wrist", "right_wrist", "agent"}:
        raise ValueError("exactly three named camera views are required")
    policy = config["s1"]
    if policy["action_dim"] != 2 * (3 + 6 + 1):
        raise ValueError("action must contain position, rotation-6D and width for each arm")
    if not 0 < policy["execution_steps"] <= policy["prediction_steps"]:
        raise ValueError("invalid receding-horizon settings")
    control = config["s0"]
    rates = [control[k] for k in ("physics_hz", "servo_hz", "camera_hz")]
    if any(not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0 for rate in rates):
        raise ValueError("rates must be positive integers")
    if rates[0] % rates[1] or rates[1] % rates[2]:
        raise ValueError("simulation rates must align on integer step boundaries")
    if control["hardware_execution_enabled"]:
        raise ValueError("this scaffold cannot authorize or execute hardware motion")
    from .baselines.backend import StopRule
    try:
        StopRule.from_config(config)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"s0.protective_stop is incomplete or malformed: {exc!r}") from exc
    validate_s2(config["s2"])
    validate_baselines(config)


def validate_baselines(config):
    try:
        _validate_baselines(config)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"baselines configuration is incomplete or malformed: {exc!r}") from exc


def _validate_baselines(config):
    from .baselines.budget import configured_pool_dir, configured_total_usd
    from .baselines.task import PLACEMENT_KEYS, check_splits, success_criterion
    baselines = config["baselines"]
    configured_total_usd(config)
    configured_pool_dir(config)
    check_splits(baselines["seed_splits"])
    if set(baselines["initial_condition"]) != set(PLACEMENT_KEYS):
        raise ValueError(f"baselines.initial_condition must have exactly {PLACEMENT_KEYS}")
    for key, upper in (("max_shift_m", 1.0), ("max_yaw_deg", 180.0), ("gap_m", 1.0)):
        value = baselines["initial_condition"][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= upper:
            raise ValueError(f"baselines.initial_condition.{key} must be a number in [0, {upper}]")
    attempts = baselines["initial_condition"]["attempts"]
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("baselines.initial_condition.attempts must be a positive integer")
    success_criterion(config)
    claims = baselines["heldout_claims_dir"]
    if not isinstance(claims, str) or not claims or Path(claims).is_absolute() or ".." in Path(claims).parts:
        raise ValueError("baselines.heldout_claims_dir must be a path relative to the repository root, without ..")
    for name in ("dexagent", "aspire"):
        commit = baselines[name]["upstream_commit"]
        if not isinstance(commit, str) or len(commit) != 40 or set(commit) - set("0123456789abcdef"):
            raise ValueError(f"baselines.{name}.upstream_commit must be a full 40-character commit hash")
        if baselines[name]["live_run_verified"] not in (True, False):
            raise ValueError(f"baselines.{name}.live_run_verified must be a boolean")


def validate_s2(planner):
    try:
        _validate_s2(planner)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"s2 configuration is incomplete or malformed: {exc!r}") from exc


def _validate_s2(planner):
    from .contracts.skills import MAX_TIMEOUT_S, SKILL_IDS, SkillRejected, check_completion
    if planner['planner'] not in ('aspire_program', 'claude_tools'):
        raise ValueError('s2.planner must be aspire_program or claude_tools')
    if planner['execution_mode'] not in ('skills', 'direct'):
        raise ValueError('s2.execution_mode must be skills or direct')
    if planner['execution_mode'] == 'direct' and planner['planner'] != 'aspire_program':
        raise ValueError('direct execution requires aspire_program')
    if planner['program_path'] is not None and (not isinstance(planner['program_path'], str) or not planner['program_path'].strip()):
        raise ValueError('s2.program_path must be a non-empty path or null')
    if planner["effort"] not in ("low", "medium", "high", "xhigh", "max"):
        raise ValueError("s2.effort must be an API effort level")
    if planner["server_fallbacks"] not in (None, "default"):
        raise ValueError("s2.server_fallbacks must be null or 'default'")
    turns = planner["max_turns"]
    if isinstance(turns, bool) or not isinstance(turns, int) or not 1 <= turns <= 20:
        raise ValueError("s2.max_turns must be an integer in 1..20")
    limit = planner["max_planning_s"]
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not 0 < limit <= 600:
        raise ValueError("s2.max_planning_s must be in (0, 600]")
    for name in ("completion_defaults", "timeouts_s"):
        if set(planner[name]) != set(SKILL_IDS):
            raise ValueError(f"s2.{name} must cover exactly {SKILL_IDS}")
    for skill, predicate in planner["completion_defaults"].items():
        try:
            check_completion(skill, predicate)
        except SkillRejected as exc:
            raise ValueError(f"s2.completion_defaults.{skill}: {exc}") from exc
    for skill, timeout in planner["timeouts_s"].items():
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= MAX_TIMEOUT_S:
            raise ValueError(f"s2.timeouts_s.{skill} must be in (0, {MAX_TIMEOUT_S}]")


def readiness(config):
    missing = []
    for section, keys in {
        "workcell": ("gripper_model", "gripper_asset", "calibration_file"),
        "dataset": ("source_path", "voxel_pitch_m", "target_length_m", "manufacturing_clearance_m"),
        "s2": (("program_path",) if config["s2"]["planner"] == "aspire_program" else ("api_model_id",)),
        "s1": ("checkpoint",),
        "s0": ("max_joint_acceleration_rad_s2", "gripper_width_range_m", "flange_T_tcp", "arm_arm_clearance_m",
               "table_clearance_m", "tcp_min_height_m", "max_observation_age_s", "anchor_position_tolerance_m",
               "anchor_rotation_tolerance_rad", "contact_table_clearance_m", "contact_region_radius_m",
               "contact_max_speed_m_s"),
    }.items():
        missing.extend(f"{section}.{key}" for key in keys if config[section].get(key) is None)
    pending = ["real CRAG sample and manufacturing validation", "YAM model and calibration",
               "YAM scene and GPU physics validation", "S1 training on simulated expert episodes",
               "S1/S0 executor inside Isaac Sim", "GARF re-registration of verified clusters",
               "UMI import and synchronized collection"]
    if config["s2"]["planner"] == "aspire_program":
        pending.append("ASPIRE S2 program generation and refinement loop")
    elif not config["s2"].get("api_live_call_verified"):
        pending.append("S2 live API call")
    pending.append("Isaac Sim AssemblyBackend for the baselines")
    for name, label in (("dexagent", "DexAgent"), ("aspire", "ASPIRE")):
        if not config["baselines"][name].get("live_run_verified"):
            pending.append(f"{label} baseline live run")
    return {
        "configuration_valid": True,
        "simulator_ready": False,
        "hardware_ready": False,
        "missing_configuration": missing,
        "pending_implementation": pending,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "validate", "prepare", "launch"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pitch", type=float)
    parser.add_argument("--target-length", type=float)
    parser.add_argument("--allow-disconnected-rigid-proxy", action="store_true")
    parser.add_argument("--bridge-single-cell-gaps", action="store_true")
    parser.add_argument("--straight-assembly-anchor", help="hand seam cells over so every fragment goes in straight with this one fixed")
    parser.add_argument("--simulation-only", action="store_true")
    parser.add_argument("--source-is-assembled-gt", action="store_true")
    add_gripper_arguments(parser)
    parser.add_argument("--assembly", type=Path)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--smoke-frames", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if not args.source_is_assembled_gt or any(v is None for v in (args.input, args.output, args.pitch, args.target_length)):
            parser.error("prepare requires input, output, pitch, target-length and explicit assembled-GT declaration")
        from .preparation import prepare
        print(prepare(args.input, args.output, pitch=args.pitch, target_length=args.target_length, simulation_only=args.simulation_only, bridge_single_cell_gaps=args.bridge_single_cell_gaps, allow_disconnected_rigid_proxy=args.allow_disconnected_rigid_proxy, straight_assembly_anchor=args.straight_assembly_anchor))
        return
    config = json.loads(args.config.read_text())
    validate(config)
    try:
        gripper = selected_gripper(config, args.gripper, args.allow_robotiq_proxy)
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "launch":
        if args.assembly is None:
            parser.error("launch requires --assembly")
        try:
            require_scene_gripper(gripper)
        except ValueError as exc:
            parser.error(str(exc))
        from tasks.dual_arm_breakingbad.domain.layout import load_layout
        layout = load_layout(args.assembly)
        if len(layout.fragment_names) < 6:
            parser.error("at least six fragments required")
        from tasks.dual_arm_breakingbad.scene.runtime import run_interactive
        return run_interactive(args.assembly, stream=args.stream, smoke_frames=args.smoke_frames,
                               output_directory=args.output)

    result = config if args.command == "inspect" else readiness(config)
    result["selected_gripper"] = gripper
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
