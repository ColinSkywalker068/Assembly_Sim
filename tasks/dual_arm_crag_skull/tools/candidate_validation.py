"""Geometry-only candidate screening. No GT; sampled checks are not collision proof."""
import numpy as np
import trimesh
from ..perception import rigid_transform


def score_candidate(meshes, poses, *, samples=256, tolerance_m=.001, contact_m=.003):
    if set(meshes)!=set(poses) or samples<32 or not 0 < tolerance_m < contact_m:
        raise ValueError('invalid geometry screening inputs')
    world, clouds, transforms, probes = {}, {}, {}, {}
    for index,(key,mesh) in enumerate(meshes.items()):
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError('solid penetration requires closed consistently oriented meshes')
        t=rigid_transform(poses[key]); transforms[key]=t
        m=mesh.copy(); m.apply_transform(t); world[key]=m
        points,faces=trimesh.sample.sample_surface(m,samples,seed=index)
        clouds[key]=points
        probes[key]=points-2*tolerance_m*m.face_normals[faces]
    pairs=[]; ids=list(meshes); adjacency={key:set() for key in ids}
    for i,left in enumerate(ids):
        for right in ids[i+1:]:
            penetrations=[]; contacts=[]
            for a,b in ((left,right),(right,left)):
                points=clouds[a]
                # Broad phase avoids expensive ray tests for well separated meshes.
                near=(points >= world[b].bounds[0]-contact_m).all(1)&(points <= world[b].bounds[1]+contact_m).all(1)
                inside=np.zeros(len(points),bool); touching=np.zeros(len(points),bool)
                if near.any():
                    # Query the unchanged local mesh so its spatial index is reused
                    # across noise candidates; rigid transforms preserve distance.
                    t=transforms[b]
                    local=(points[near]-t[:3,3]) @ t[:3,:3]
                    _,distance,_=trimesh.proximity.closest_point(meshes[b],local)
                    contained=meshes[b].contains(local)
                    inside[near]=contained & (distance>tolerance_m)
                    touching[near]=distance<contact_m
                # Surface-only tests miss coincident solids. Inward probes catch
                # this; require containment in both solids to handle thin shells.
                probe=probes[a]
                near_probe=(probe>=world[b].bounds[0]).all(1)&(probe<=world[b].bounds[1]).all(1)
                overlap=np.zeros(samples,bool)
                if near_probe.any():
                    t=transforms[b]; local_b=(probe[near_probe]-t[:3,3])@t[:3,:3]
                    _,distance,_=trimesh.proximity.closest_point(meshes[b],local_b)
                    deep=meshes[b].contains(local_b)&(distance>tolerance_m)
                    if deep.any():
                        t=transforms[a]; local_a=(probe[near_probe][deep]-t[:3,3])@t[:3,:3]
                        indices=np.flatnonzero(near_probe)[deep]
                        overlap[indices]=meshes[a].contains(local_a)
                penetrations.append(float((inside|overlap).mean())); contacts.append(float(touching.mean()))
            contact=min(contacts)
            if contact>.01:
                adjacency[left].add(right); adjacency[right].add(left)
            pairs.append({'pair':[left,right],'penetration_fraction':max(penetrations),'contact_fraction':contact})
    seen=set(); components=0
    for node in ids:
        if node in seen: continue
        components+=1; todo=[node]; seen.add(node)
        while todo:
            for neighbor in adjacency[todo.pop()]-seen:
                seen.add(neighbor); todo.append(neighbor)
    penetration=max((p['penetration_fraction'] for p in pairs),default=0)
    return {'max_penetration_fraction':penetration,'contact_components':components,'pairs':pairs,
            'rejected':penetration>.02 or components>1,
            'sample_count_per_part':samples,'tolerance_m':tolerance_m,'contact_m':contact_m,
            'motion_feasibility_checked':False,'sampling_can_miss_collisions':True}


def rank_candidates(meshes,candidates,**kwargs):
    results=[]
    for candidate in candidates:
        if candidate.get('status')!='candidate': continue
        score=score_candidate(meshes,candidate['assembly_T_fragment'],**kwargs)
        results.append({'seed':candidate['seed'],**score})
    # Rank a complete set lexicographically, do not hide rejected candidates.
    return sorted(results,key=lambda r:(r['rejected'],r['max_penetration_fraction'],r['contact_components'],r['seed']))


def screen_result(meshes, result, **kwargs):
    """Attach geometry screening; never authorize robot motion from this score."""
    candidates=result.get('candidates', [result])
    valid=[c for c in candidates if c.get('status')=='candidate']
    if not valid:
        return result
    rankings=rank_candidates(meshes, valid, **kwargs)
    eligible=[r['seed'] for r in rankings if not r['rejected']]
    return {**result, 'geometry_screening':rankings,
            'geometry_status':'needs_motion_validation' if eligible else 'all_candidates_rejected',
            'geometrically_eligible_seeds':eligible,
            'selected_candidate':None, 'motion_authorized':False}


if __name__=='__main__':
    import argparse
    import json
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh-registry',required=True)
    parser.add_argument('--candidates',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--samples',type=int,default=256)
    args=parser.parse_args()
    meshes={k:trimesh.load_mesh(p) for k,p in json.loads(Path(args.mesh_registry).read_text()).items()}
    result=screen_result(meshes,json.loads(Path(args.candidates).read_text()),samples=args.samples)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
