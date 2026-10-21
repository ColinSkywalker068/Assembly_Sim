"""JSON-lines S2 tool process: model loads once; stdout contains tool results only."""
import argparse
import contextlib
import json
import sys
import numpy as np
from .garf import GarfTool, propose_candidates
from .garf_backend import GarfBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--geometry', required=True, help='JSON mapping IDs to NPZ paths: points/normals in metres')
    parser.add_argument('--mesh-registry', help='Optional IDs to closed local meshes in the same metre frames as the NPZ registry')
    parser.add_argument('--model-unit-m', type=float, help='Explicit metres per model coordinate unit; required for official preprocessing')
    parser.add_argument('--legacy-preprocessing', action='store_true', help='Explicit historical baseline only')
    parser.add_argument('--minimum-points-per-part', type=int, default=20, help='Explicit experimental sampling override, 20..400; total remains 5000')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.legacy_preprocessing and args.minimum_points_per_part != 20:
        parser.error("minimum point override requires official preprocessing")
    geometry = {}
    for key, path in json.load(open(args.geometry)).items():
        with np.load(path, allow_pickle=False) as data:
            geometry[key] = (data['points'], data['normals'])
    meshes=None
    if args.mesh_registry:
        import trimesh
        meshes={k:trimesh.load_mesh(p) for k,p in json.load(open(args.mesh_registry)).items()}
    with contextlib.redirect_stdout(sys.stderr):
        if args.legacy_preprocessing:
            backend = GarfBackend(args.repository, args.checkpoint, args.device)
        else:
            if meshes is None or args.model_unit_m is None:
                parser.error('official mode requires --mesh-registry and --model-unit-m; point density is not surface area')
            from .official_backend import OfficialGarfBackend
            backend = OfficialGarfBackend(args.repository, args.checkpoint, meshes, args.model_unit_m, args.device, args.minimum_points_per_part)
        tool = GarfTool(backend, geometry, backend.checkpoint_sha256)
    for line in sys.stdin:
        try:
            envelope = json.loads(line)
            # Observation ID is supplied by orchestrator, independently of S2 args.
            with contextlib.redirect_stdout(sys.stderr):
                if envelope.get('tool_name') == 'propose_assembly_candidates':
                    result = propose_candidates(tool, envelope['request'], current_observation_id=envelope['current_observation_id'])
                elif envelope.get('tool_name', 'propose_assembly') == 'propose_assembly':
                    result = tool(envelope['request'], current_observation_id=envelope['current_observation_id'])
                else:
                    result = {'status':'rejected','reason':'unknown tool name'}
                if meshes is not None and result.get('status') in ('candidate','candidates','incomplete'):
                    from .candidate_validation import screen_result
                    selected={k:meshes[k] for k in envelope['request']['fragment_ids']}
                    result=screen_result(selected,result)
        except Exception as exc:
            result = {'status':'failed', 'reason':str(exc)}
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
