"""Reproducible public-world numerical and interface validation (not Agent results).

python -m development_worlds.validate_suite --output validation/report.json
Oracle laws are evaluator fixtures. They must never be fed to an Agent as evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np

from . import catalog
from .evaluation import calibrate_world, claims_truth, evaluate_law
from .oracles import oracle_source
from .worlds import DevelopmentSimulator

ROOT = Path(__file__).resolve().parents[1]


def _upstream_module(name):
    path = ROOT.parent / 'shore-contest-starter-v15' / 'development_worlds' / (name + '.py')
    if not path.is_file():
        path = ROOT / 'validation' / 'upstream_v15' / (name + '.py')
    manifest = json.loads((ROOT / 'upstream-manifest.json').read_text())
    expected = manifest['sha256']['development_worlds/' + name + '.py']
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Original parity fixture hash mismatch: ' + str(path))
    spec = importlib.util.spec_from_file_location('_original_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_parity(world):
    """Compare all old validation trajectories and backgrounds against original code."""
    if str(world) not in ('1', '2', '3', '4', '5', '6'):
        return None
    original = _upstream_module('worlds').DevelopmentSimulator(world)
    old_catalog = _upstream_module('catalog')
    current = DevelopmentSimulator(world)
    old_cases, cases = old_catalog.validation_cases(world), catalog.validation_cases(world)
    checks = [old_cases == cases, old_catalog.first_light(world) == catalog.first_light(world)]
    for case in cases + [catalog.first_light(world)]:
        request = {k: case[k] for k in ('probes', 't_launch', 'readout', 'source_properties')}
        a, b = original.launch(**request), current.launch(**request)
        checks.extend(np.array_equal(a[k], b[k]) for k in ('positions', 'velocities', 'times'))
    times = [0., 1., catalog.LIMITS[str(world)][2]]
    a, b = original.background(times), current.background(times)
    checks.extend(np.array_equal(a[k], b[k]) for k in ('positions', 'velocities', 'times'))
    checks.append(a['ids'] == b['ids'])
    return {'passed': bool(all(checks)), 'comparisons': len(checks),
            'coverage': 'all validation cases, first light and background at 0, 1, maximum time',
            'comparison': 'exact array equality against unmodified upstream'}


def random_domain_check(world, seed=20261108):
    """Eight reproducible legal initial states at full duration; not a domain proof."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(world)]))
    lo, hi, tmax, duration, vmax = catalog.LIMITS[world]
    angles = rng.uniform(-np.pi, np.pi, 8)
    radii = np.sqrt(rng.uniform(lo * lo, hi * hi, 8))
    radii[:2] = [lo, hi]
    direction = rng.uniform(-np.pi, np.pi, 8)
    speed = rng.uniform(0., vmax, 8)
    speed[:2] = [0., vmax]
    positions = radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    velocities = speed[:, None] * np.column_stack((np.cos(direction), np.sin(direction)))
    cfg = catalog.config(world)
    properties = []
    for i in range(8):
        properties.append({k: (v['min'] if i % 2 == 0 else v['max'])
                           for k, v in cfg['probe_properties'].items()})
    source = {k: v['max'] for k, v in cfg['source_properties'].items()}
    result = DevelopmentSimulator(world).integrate(positions, velocities,
        t_launch=tmax-duration, duration=duration,
        source_properties=source, probe_properties=properties)
    finite = all(np.isfinite(result[k]).all() for k in ('positions', 'velocities'))
    return {'passed': bool(finite), 'seed': seed, 'probe_count': 8,
            'duration': duration, 't_launch': tmax-duration,
            'max_radius': float(np.linalg.norm(result['positions'], axis=-1).max()),
            'max_speed': float(np.linalg.norm(result['velocities'], axis=-1).max()),
            'limitation': 'finite-state stress sample, not all-domain stability or convergence proof'}


def validate_world(world):
    started = time.monotonic()
    reference = calibrate_world(world)
    with tempfile.TemporaryDirectory(prefix='shore-oracle-') as tmp:
        law_path = Path(tmp) / 'law.py'
        law_path.write_text(oracle_source(world), encoding='utf-8')
        scored = evaluate_law(world, law_path, claims_truth(world), 0., calibration=reference)
    parity = original_parity(world)
    random_check = random_domain_check(world)
    passed = bool(reference['all_cases_converged'] and not scored['invalid_cases']
                  and scored['raw_MSE'] is not None and scored['raw_MSE'] < 1e-16
                  and scored['total'] > 99.999999 and random_check['passed']
                  and (parity is None or parity['passed']))
    return {'world': world, 'passed': passed,
            'validation_cases': len(reference['cases']),
            'probes': reference['N_probes'], 'samples_per_probe': reference['T_samples'],
            'convergence': reference['convergence'],
            'all_cases_converged': reference['all_cases_converged'],
            'oracle': {k: scored[k] for k in ('prediction', 'law', 'saving', 'total',
                                            'raw_MSE', 'invalid_cases', 'law_sha256')},
            'ballistic_baseline': reference['ballistic_baseline'],
            'random_domain_check': random_check, 'original_parity': parity,
            'reference_sha256': reference['reference_sha256'],
            'elapsed_seconds': round(time.monotonic() - started, 3)}


def upstream_integrity():
    manifest = json.loads((ROOT / 'upstream-manifest.json').read_text())
    source = ROOT.parent / manifest['source']
    if not source.exists():
        checked, mismatched = [], []
        for name in ('worlds.py', 'catalog.py'):
            path = ROOT / 'validation' / 'upstream_v15' / name
            checked.append(name)
            if (not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()
                    != manifest['sha256']['development_worlds/' + name]):
                mismatched.append(name)
        return {'available': True, 'passed': not mismatched, 'files_checked': len(checked),
                'mismatched': mismatched, 'scope': 'Bundled original simulator/catalog fixtures; full starter not adjacent.'}
    mismatched = []
    for name, digest in manifest['sha256'].items():
        path = source / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            mismatched.append(name)
    return {'available': True, 'passed': not mismatched,
            'files_checked': len(manifest['sha256']), 'mismatched': mismatched}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worlds', nargs='+', choices=catalog.WORLDS, default=list(catalog.WORLDS))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace an existing report.')
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite:
        parser.error('Output already exists; choose a new path or pass --overwrite.')
    if args.workers < 1:
        parser.error('--workers must be positive')
    started = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(validate_world, world): world for world in dict.fromkeys(args.worlds)}
        for future in as_completed(futures):
            world = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {'world': world, 'passed': False, 'error': type(exc).__name__ + ': ' + str(exc)}
            rows.append(row)
            print(f"World {world}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
    rows.sort(key=lambda r: int(r['world']))
    integrity = upstream_integrity()
    sources = sorted(set(ROOT.glob('development_worlds/*.py')) | set(ROOT.glob('rift/*.py'))
                     | set(ROOT.glob('rift_numerics/*.py')))
    report = {'suite': 'shore-worlds-20', 'kind': 'numerical_and_interface_validation_not_agent_score',
              'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
              'world_count': len(rows), 'all_passed': all(r['passed'] for r in rows)
                     and integrity['passed'] is not False,
              'elapsed_seconds': round(time.monotonic() - started, 3),
              'upstream_integrity': integrity,
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sources},
              'worlds': rows,
              'limitations': ['No actual LLM, GPU training or policy learning run is implied.',
                  'Step-halving covers catalog validation cases only; random tests check finiteness only.',
                  'Worlds 1 and 4 retain the upstream radial-direction discontinuity at the origin.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w' if args.overwrite else 'x', encoding='utf-8') as out:
        out.write(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    print(str(args.output.resolve()), flush=True)
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
