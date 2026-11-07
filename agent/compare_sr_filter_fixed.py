"""E16 fixed-input SR ablation. No Agent/model calls and no parameter tuning.

The scientific stage imports no reference simulator/evaluator. Scoring happens
in a separate process, after the raw-training selection has been persisted.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def save(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_candidate(rows):
    usable = [r for r in rows if isinstance(r.get('raw_position_rmse'), (int, float))
              and math.isfinite(r['raw_position_rmse']) and r.get('law_file')]
    return min(usable, key=lambda r: (r['raw_position_rmse'], r['sr_rank']))['sr_rank'] if usable else None


def read_inputs(baseline, world):
    directory = Path(baseline)/'R07'/f'world-{world}'/'agent-output'
    task = json.loads((directory/'mission.json').read_text())['task']
    original = json.loads((directory/'result.json').read_text())
    files = sorted((directory/'records/probe').glob('*.json'))
    probes = [json.loads(path.read_text()) for path in files]
    if not probes:raise ValueError('Missing fixed observations')
    for probe in probes:
        probe['observation_noise'] = {'position_sigma': task['position_noise_sigma'],
                                      'velocity_sigma': task['velocity_noise_sigma']}
    return probes, {'claims': original['submission']['claims'], 'energy_used': original['budget']['energy_used'],
                    'source_files': {str(p.relative_to(directory)): sha(p) for p in files},
                    'mission_sha256': sha(directory/'mission.json'), 'baseline_result_sha256': sha(directory/'result.json')}


def replay_environment(public, probes):
    """Rebuild only the public observation record schema used by official rollout."""
    import numpy as np
    from .pipeline_bridge.compiler import property_mapping
    mapping = property_mapping(public)
    acquired = {}
    for probe in probes:
        initial = probe['initial_conditions']
        pp = {k: v['default'] for k, v in public['controls']['probe_properties'].items()}
        sp = {k: v['default'] for k, v in public['controls']['source_properties'].items()}
        for name, value in initial['properties'].items():
            scope, key = mapping[name]
            (pp if scope == 'probe' else sp)[key] = value
        record = {'t_launch': initial['t0'], 'times': probe['times'],
                  'readout': [t-initial['t0'] for t in probe['times']],
                  'probes': [{'pos': initial['r0'], 'vel': initial['v0'], 'properties': pp}],
                  'source_properties': sp,
                  'positions': np.asarray(probe['r_obs'])[:, None, :].tolist(),
                  'velocities': np.asarray(probe['v_obs'])[:, None, :].tolist()}
        acquired[probe['probe_id']] = {'_official_record': record}
    return SimpleNamespace(public_mission=public, acquired=acquired)


def trial(args):
    from .pipeline_bridge.bootstrap import repository
    repository()
    from .pipeline_bridge.trajectory_filter import filter_probe, digest, PAYLOAD_KEY, VERSION
    from .pipeline_bridge.filter_capability import verify
    from .pipeline_bridge.backend import install
    from .repository.shore_agent.symbolic_service import search_remote
    from .repository.shore_agent import numerics, expressions
    # Public mission only, for property alias encoding. No reference force fields.
    from development_worlds.console import DevelopmentConsole
    from development_worlds import catalog
    output = Path(args.output);output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    probes, constants = read_inputs(args.baseline, args.world)
    snapshot = digest(probes)
    save(output/'input-provenance.json', dict(constants, augmented_probe_sha256=snapshot))
    result = {'world': args.world, 'seed': args.seed, 'arm': args.arm, 'status': 'running',
              'model_calls': 0, 'average_model_output_tokens': None, 'original_probe_sha256': snapshot}
    filter_rows = []
    try:
        verify(args.endpoint, wait_timeout=120.)
        search_probes = copy.deepcopy(probes)
        begin = time.monotonic()
        if args.arm == 'joint':
            deadline = begin+2.
            for probe in search_probes:
                try:row = filter_probe(probe, deadline=deadline)
                except (TimeoutError, ArithmeticError, ValueError, RuntimeError) as exc:
                    from .pipeline_bridge.trajectory_filter import validate, raw_digest
                    validate(probe)
                    row = {'probe_id': probe['probe_id'], 'version': VERSION, 'status': 'bypass',
                           'raw_sha256': raw_digest(probe), 'reason': type(exc).__name__}
                filter_rows.append(row);probe[PAYLOAD_KEY] = row
        result['filter_seconds'] = time.monotonic()-begin
        result['filtered_probes'] = sum(r['status'] == 'filtered' for r in filter_rows)
        save(output/'filter-artifacts.json', filter_rows)
        begin = time.monotonic()
        search = search_remote(search_probes, 47., endpoint=args.endpoint, random_state=args.seed)
        result['search_seconds'] = time.monotonic()-begin
        save(output/'search.json', search)
        expected = {r['probe_id']: r for r in filter_rows if r['status'] == 'filtered'}
        consumed = {r['probe_id']: r for r in search.get('data_quality', {}).get('local_derivative_quality', []) if r.get('method') == VERSION}
        verified = bool(expected) and all(k in consumed and consumed[k].get('output_sha256') == v['output_sha256'] for k, v in expected.items())
        result['worker_consumption'] = 'verified' if verified else 'not_required' if not expected else 'failed'
        if expected and not verified:raise ValueError('Filter targets were not consumed')
        candidates = search.get('candidates', [])[:3]
        result['returned_candidates'] = len(candidates)
        # The mission supplies only public metadata; its generated first-light
        # samples are discarded. Search/refit use the archived observations only.
        public = DevelopmentConsole(str(args.world), catalog.noise_sigma(str(args.world)), 42, catalog.config(str(args.world))).mission()
        environment = replay_environment(public, probes)
        backend = install(environment, output/'raw-refit')
        save(output/'public-mission.json', {k: v for k, v in public.items() if k != 'first_light'})
        rows = []
        begin = time.monotonic()
        for rank, candidate in enumerate(candidates, 1):
            row = {'sr_rank': rank, 'candidate': candidate}
            fit_started = time.monotonic()
            try:
                expression = expressions.Expression(candidate['law_spec'], candidate['parameter_spec'])
                sr_parameters = {p['name']: float(p['initial']) for p in candidate['parameter_spec']}
                original_law = output/f'candidate-{rank}-sr.py'
                original_law.write_text(backend.source(expression, sr_parameters, public))
                row.update(sr_law_file=original_law.name, sr_law_sha256=sha(original_law))
                fitted = numerics.fit(candidate['law_spec'], candidate['parameter_spec'], probes, [],
                    max_nfev=80, timeout_seconds=8., position_noise_sigma=probes[0]['observation_noise']['position_sigma'])
                save(output/f'candidate-{rank}-fit.json', fitted)
                row['fit_status'] = fitted.get('fit_status')
                if fitted.get('errors') and fitted.get('fit_status') != 'failed':
                    row['raw_position_rmse'] = fitted['errors']['overall']['position_rmse']
                    row['raw_velocity_rmse'] = fitted['errors']['overall']['velocity_rmse']
                    expression = expressions.Expression(candidate['law_spec'], candidate['parameter_spec'])
                    source = backend.source(expression, fitted['fitted_parameters'], public)
                    law = output/f'candidate-{rank}.py';law.write_text(source)
                    row.update(law_file=law.name, law_sha256=sha(law), fitted_parameters=fitted['fitted_parameters'])
            except Exception as exc:
                row['fit_error'] = type(exc).__name__
            row['refit_seconds'] = time.monotonic()-fit_started
            rows.append(row)
        result['refit_seconds'] = time.monotonic()-begin
        result['raw_official_rollout_calls'] = backend.rollout_calls
        result['observations_unchanged'] = digest(probes) == snapshot
        if not result['observations_unchanged']:raise ValueError('Original observations mutated')
        selection = {'selected_sr_rank': select_candidate(rows), 'original_sr_rank_one': 1 if rows else None,
                     'selection_rule': 'Minimum raw-training position RMSE; original SR rank breaks ties',
                     'scoring_constants': {'claims': constants['claims'], 'energy_used': constants['energy_used']},
                     'candidates': rows}
        save(output/'selection.json', selection)
        result['selection_sha256_before_scoring'] = sha(output/'selection.json')
        result['status'] = 'scientific_stage_completed'
        save(output/'trial.json', result)
        # Separate process: reference simulator and held-out states cannot leak
        # back into SR, refit, ranking, or filtering.
        evaluation_started = time.monotonic()
        command = [sys.executable, '-B', '-m', 'agent.evaluate_sr_filter_fixed', '--directory', str(output), '--world', str(args.world)]
        with (output/'evaluation.log').open('w') as stream:
            scored = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        result['evaluation_seconds'] = time.monotonic()-evaluation_started
        result['evaluation_exit_code'] = scored.returncode
        if scored.returncode:raise RuntimeError('Independent evaluator failed')
        if sha(output/'selection.json') != result['selection_sha256_before_scoring']:raise ValueError('Selection changed after truth access')
        result['evaluation'] = json.loads((output/'evaluation.json').read_text())
        result['status'] = 'completed'
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__, error_message=str(exc)[:500])
        import traceback
        (output/'error.txt').write_text(traceback.format_exc())
    result['total_seconds'] = time.monotonic()-started
    save(output/'trial.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True);parser.add_argument('--output', required=True)
    parser.add_argument('--endpoint', required=True);parser.add_argument('--world', type=int, choices=range(1, 7), required=True)
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    parser.add_argument('--arm', choices=['legacy', 'joint'], required=True)
    result = trial(parser.parse_args())
    print(json.dumps({k: v for k, v in result.items() if k != 'evaluation'}, ensure_ascii=False), flush=True)
    return 0 if result['status'] == 'completed' else 2


if __name__ == '__main__':raise SystemExit(main())
