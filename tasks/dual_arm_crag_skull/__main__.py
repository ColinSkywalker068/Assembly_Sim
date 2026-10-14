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
        "pending_implementation": ["CRAG loader and printable mesh export", "YAM model and calibration", "dual-arm scene runner", "S0 constrained controller", "S1 training and inference", "S2 provider adapter", "UMI import and synchronized collection"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "validate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate(config)
    print(json.dumps(config if args.command == "inspect" else readiness(config), indent=2))


if __name__ == "__main__":
    main()
