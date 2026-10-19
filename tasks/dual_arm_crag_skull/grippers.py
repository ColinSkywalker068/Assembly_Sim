"""Explicit gripper selection for the skull task's scene entry points."""
GRIPPERS = ('yam', 'robotiq')


def add_gripper_arguments(parser):
    parser.add_argument('--gripper', choices=GRIPPERS,
                        help='gripper selection (default: configured family, YAM)')
    parser.add_argument('--allow-robotiq-proxy', action='store_true',
                        help='compatibility alias for --gripper robotiq')


def selected_gripper(config, gripper=None, allow_robotiq_proxy=False):
    if allow_robotiq_proxy:
        if gripper not in (None, 'robotiq'):
            raise ValueError('--allow-robotiq-proxy conflicts with --gripper yam')
        gripper = 'robotiq'
    selected = gripper if gripper is not None else config['workcell']['gripper_family'].lower()
    if selected not in GRIPPERS:
        raise ValueError(f'unsupported gripper: {selected!r}; choose yam or robotiq')
    return selected


def require_scene_gripper(gripper):
    if gripper not in GRIPPERS:
        raise ValueError(f'unsupported gripper: {gripper!r}')
    if gripper == 'yam':
        raise ValueError('YAM is the default gripper, but its scene adapter, model/asset and TCP calibration '
                         'are not available yet. Configure the YAM integration before launching, '
                         'or explicitly select --gripper robotiq to use the Robotiq 2F-85 proxy.')
