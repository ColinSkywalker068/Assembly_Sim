"""Inspect the proposed configuration without importing Isaac Sim."""

import argparse
import json
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "jing.json"


def validate(config):
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


def readiness(config):
    missing = []
    for section, keys in {
        "workcell": ("gripper_model", "gripper_asset", "calibration_file"),
        "dataset": ("source_path", "voxel_pitch_m", "target_length_m", "manufacturing_clearance_m"),
        "s2": ("api_model_id",),
        "s1": ("checkpoint",),
    }.items():
        missing.extend(f"{section}.{key}" for key in keys if config[section].get(key) is None)
    return {
        "configuration_valid": True,
        "simulator_ready": False,
        "hardware_ready": False,
        "missing_configuration": missing,
        "pending_implementation": ["real CRAG sample and manufacturing validation", "YAM model and calibration", "YAM scene and GPU physics validation", "S0 constrained controller", "S1 training and inference", "S2 provider and GARF service adapters", "UMI import and synchronized collection"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "validate", "prepare", "launch"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pitch", type=float)
    parser.add_argument("--target-length", type=float)
    parser.add_argument("--source-is-assembled-gt", action="store_true")
    parser.add_argument("--allow-robotiq-proxy", action="store_true")
    parser.add_argument("--assembly", type=Path)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--smoke-frames", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if not args.source_is_assembled_gt or any(v is None for v in (args.input, args.output, args.pitch, args.target_length)):
            parser.error("prepare requires input, output, pitch, target-length and explicit assembled-GT declaration")
        from .preparation import prepare
        print(prepare(args.input, args.output, pitch=args.pitch, target_length=args.target_length))
        return
    if args.command == "launch":
        if not args.allow_robotiq_proxy or args.assembly is None:
            parser.error("launch currently requires --assembly and --allow-robotiq-proxy; YAM is not implemented")
        from tasks.dual_arm_breakingbad.domain.layout import load_layout
        layout = load_layout(args.assembly)
        if len(layout.fragment_names) < 6:
            parser.error("at least six fragments required")
        from tasks.dual_arm_breakingbad.scene.runtime import run_interactive
        return run_interactive(args.assembly, stream=args.stream, smoke_frames=args.smoke_frames,
                               output_directory=args.output)

    config = json.loads(args.config.read_text())
    validate(config)
    print(json.dumps(config if args.command == "inspect" else readiness(config), indent=2))


if __name__ == "__main__":
    main()
