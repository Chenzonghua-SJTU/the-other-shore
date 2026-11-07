"""Independent E16 evaluator. Never imported by the search/refit stage."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

_REFERENCE_CACHE = {}


def reference_states(world, public, *, count=256, seed=160016):
    # Reference access is confined to this post-selection evaluator process.
    from development_worlds.worlds import DevelopmentSimulator
    from development_worlds import catalog
    config = catalog.config(str(world))
    reference = DevelopmentSimulator(str(world))
    rng = np.random.default_rng(seed)
    rows = []
    region = config['launch_region']
    if region['kind'] != 'annulus':raise ValueError('Unregistered held-out sampler region')
    initial = public['background_initial_state']
    for _ in range(count):
        radius = np.sqrt(rng.uniform(region['radius_min']**2, region['radius_max']**2))
        theta = rng.uniform(-np.pi, np.pi)
        position = np.asarray(region['center'])+radius*np.array([np.cos(theta), np.sin(theta)])
        angle = rng.uniform(-np.pi, np.pi);speed = rng.uniform(0, config['initial_speed_max'])
        velocity = speed*np.array([np.cos(angle), np.sin(angle)])
        t = round(rng.uniform(0, config['maximum_physical_time'])/.005)*.005
        pp = {k: rng.uniform(v['min'], v['max']) for k, v in config['probe_properties'].items()}
        sp = {k: rng.uniform(v['min'], v['max']) for k, v in config['source_properties'].items()}
        expected = reference.acceleration(t, position[None, :], source_properties=sp, probe_properties=[pp])[0]
        rows.append({'t': t, 'pos': position, 'vel': velocity, 'pp': pp, 'sp': sp, 'expected': expected})
    background = {}
    if initial['ids']:
        times = sorted({row['t'] for row in rows})
        # The public simulator accepts <=50 strictly increasing readouts.
        for start in range(0, len(times), 50):
            history = reference.background(times[start:start+50])
            if list(history['ids']) != initial['ids']:raise ValueError('Reference background ID mismatch')
            for j, ti in enumerate(history['times']):
                background[round(float(ti)/.005)] = (history['positions'][j], history['velocities'][j])
    for row in rows:
        bodies = []
        positions, velocities = background.get(round(row['t']/.005), ([], []))
        for i, identifier in enumerate(initial['ids']):
            props = copy.deepcopy(initial.get('properties', [{} for _ in initial['ids']])[i])
            if identifier == public['source_property_target_id']:props.update(row['sp'])
            bodies.append({'id': identifier, 'kind': 'background', 'pos': np.asarray(positions[i]),
                           'vel': np.asarray(velocities[i]), 'properties': props})
        bodies.append({'id': 'heldout_probe', 'kind': 'probe', 'pos': row['pos'], 'vel': row['vel'], 'properties': row['pp']})
        row['bodies'] = tuple(bodies)
    return rows


def recovery(world, law_path, public, *, count=256, seed=160016):
    from rift import load_law
    key = (str(world), count, seed, json.dumps(public, sort_keys=True))
    if key not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[key] = reference_states(world, public, count=count, seed=seed)
    states = _REFERENCE_CACHE[key];law = load_law(law_path)
    errors, reference_values = [], []
    for row in states:
        predicted = np.asarray(law.acceleration(row['t'], row['bodies'], law.params), dtype=float)[-1]
        if predicted.shape != (2,) or not np.isfinite(predicted).all():
            return {'correct': False, 'reason': 'nonfinite_or_invalid_acceleration', 'states': len(errors)}
        errors.append(predicted-row['expected']);reference_values.append(row['expected'])
    denominator = max(float(np.mean(np.sum(np.square(reference_values), axis=1))), 1e-24)
    with np.errstate(over='ignore', invalid='ignore'):
        nrmse = float(np.sqrt(np.mean(np.sum(np.square(errors), axis=1))/denominator))
    if not np.isfinite(nrmse):return {'correct': False, 'reason': 'overflow_in_error_metric', 'states': count}
    return {'correct': nrmse <= .01, 'normalized_acceleration_rmse': nrmse,
            'criterion': 'NRMSE<=0.01 on 256 predetermined independent legal-domain states',
            'states': count, 'reference_seed': seed,
            'reference_velocity_policy': 'Public simulator acceleration has no velocity argument; independently varying candidate velocity tests spurious dependence.',
            'background_state_policy': 'Clean autonomous background at the same sampled absolute time; evaluator-only reference state.',
            'interpretation': 'Numerical law recovery; not exact symbolic-equivalence proof'}


def evaluate(directory, world):
    from development_worlds.evaluation import evaluate_law
    directory = Path(directory)
    selection_path = directory/'selection.json'
    frozen_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    selection = json.loads(selection_path.read_text())
    public = json.loads((directory/'public-mission.json').read_text())
    constants = selection['scoring_constants'];rows = []
    for candidate in selection['candidates']:
        rank = candidate['sr_rank'];started = time.monotonic()
        row = {'sr_rank': rank, 'score': {'total': 0., 'prediction': 0.},
               'recovery': {'correct': False, 'reason': 'no_successful_raw_refit'}}
        row['sr_returned_score'] = {'total': 0., 'prediction': 0.}
        row['sr_returned_recovery'] = {'correct': False, 'reason': 'no_compiled_SR_formula'}
        if candidate.get('sr_law_file'):
            sr_law = directory/candidate['sr_law_file']
            if hashlib.sha256(sr_law.read_bytes()).hexdigest() != candidate['sr_law_sha256']:
                raise ValueError('Original SR candidate changed before scoring')
            try:row['sr_returned_score'] = evaluate_law(str(world), sr_law, constants['claims'], constants['energy_used'])
            except Exception as exc:row['sr_returned_evaluation_error'] = type(exc).__name__
            try:row['sr_returned_recovery'] = recovery(world, sr_law, public)
            except Exception as exc:row['sr_returned_recovery'] = {'correct': False, 'reason': type(exc).__name__}
        if candidate.get('law_file'):
            law = directory/candidate['law_file']
            if hashlib.sha256(law.read_bytes()).hexdigest() != candidate['law_sha256']:
                raise ValueError('Selected candidate source changed before scoring')
            try:row['score'] = evaluate_law(str(world), law, constants['claims'], constants['energy_used'])
            except Exception as exc:row['evaluation_error'] = type(exc).__name__
            try:row['recovery'] = recovery(world, law, public)
            except Exception as exc:row['recovery'] = {'correct': False, 'reason': type(exc).__name__}
        row['evaluation_seconds'] = time.monotonic()-started
        rows.append(row)
    by_rank = {row['sr_rank']: row for row in rows}
    empty = {'score': {'total': 0., 'prediction': 0.}, 'recovery': {'correct': False, 'reason': 'no_complete_candidate'}}
    original_one = by_rank.get(1)
    report = {'world': world, 'selection_sha256': frozen_hash,
              'selected': by_rank.get(selection['selected_sr_rank'], empty),
              'sr_rank_one': {'score': original_one['sr_returned_score'], 'recovery': original_one['sr_returned_recovery']} if original_one else empty,
              'any_top_three_recovered': any(row['sr_returned_recovery']['correct'] for row in rows),
              'any_top_three_after_refit_recovered': any(row['recovery']['correct'] for row in rows),
              'candidate_evaluations': rows, 'scoring_constants': constants,
              'scope': 'Official clean fixed-case score with frozen original claims/energy; standalone SR, not an Agent trial'}
    if hashlib.sha256(selection_path.read_bytes()).hexdigest() != frozen_hash:
        raise ValueError('Selection changed during evaluation')
    (directory/'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--directory', required=True)
    parser.add_argument('--world', required=True, type=int)
    args = parser.parse_args();print(json.dumps(evaluate(args.directory, args.world), ensure_ascii=False))
