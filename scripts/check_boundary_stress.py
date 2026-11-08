#!/usr/bin/env python3
"""Reproduce finite boundary-condition convergence checks for synthetic worlds.

Runs eight legal probes simultaneously, including maximum launch radius/speed
and maximum source-to-inertia ratio. This is not a proof over the full domain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from development_worlds import catalog
from development_worlds.worlds import DevelopmentSimulator, SINGLE_CENTER, VERSION
from rift_numerics import DT, NUMERICS_VERSION


def check_world(world):
    simulator = DevelopmentSimulator(world)
    angle = np.arange(8)*2*np.pi/8
    radial = np.column_stack((np.cos(angle), np.sin(angle)))
    position = np.array([.5, 1., 1.5, 2., 3., 4., 4.8, 5.])[:, None]*radial
    velocity = 3*np.column_stack((-np.sin(angle+.31), np.cos(angle+.31)))
    source = {'p1': 2.} if world in SINGLE_CENTER else {}
    properties = {'p2': .5} if world in SINGLE_CENTER else {'mass': 1.}
    kwargs = {'t_launch': 12., 'duration': 20., 'source_properties': source,
              'probe_properties': properties}
    coarse = simulator.integrate(position, velocity, **kwargs)
    fine = simulator.integrate(position, velocity, dt=DT/2, **kwargs)
    dx = float(np.max(np.linalg.norm(coarse['positions']-fine['positions'][::2], axis=-1)))
    dv = float(np.max(np.linalg.norm(coarse['velocities']-fine['velocities'][::2], axis=-1)))
    return {'world': world, 'inputs': {'positions': position.tolist(),
             'velocities': velocity.tolist(), **kwargs},
            'simulator_cache_token': simulator.cache_token,
            'max_position_difference': dx, 'max_velocity_difference': dv,
            'max_radius': float(np.linalg.norm(coarse['positions'], axis=-1).max()),
            'all_states_finite': all(np.isfinite(history[key]).all().item()
                                    for history in (coarse, fine)
                                    for key in ('positions', 'velocities')),
            'passed': dx < .001 and dv < .01}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worlds', nargs='+', choices=tuple(str(i) for i in range(7, 21)),
                        default=[str(i) for i in range(7, 21)])
    parser.add_argument('--output', type=Path, default=ROOT/'validation'/'boundary-stress.json')
    args = parser.parse_args(argv)
    names = ['development_worlds/worlds.py', 'development_worlds/world_specs.py',
             'development_worlds/catalog.py', 'rift_numerics/__init__.py',
             'rift_numerics/rollout.py', 'scripts/check_boundary_stress.py']
    rows = []
    for world in args.worlds:
        row = check_world(world)
        rows.append(row)
        print(f"world {world}: dx={row['max_position_difference']:.3g}, "
              f"dv={row['max_velocity_difference']:.3g}, passed={row['passed']}", flush=True)
    result = {'schema_version': '1.0.0', 'kind': 'finite_boundary_stress',
              'scope': 'Eight deterministic legal initial states per new world; not a full-domain proof.',
              'seed': None, 'sampling': 'deterministic; no random sampling',
              'python_version': platform.python_version(), 'numpy_version': np.__version__,
              'worlds_version': VERSION, 'catalog_version': catalog.VERSION,
              'numerics_version': NUMERICS_VERSION, 'coarse_dt': DT, 'fine_dt': DT/2,
              'tolerances': {'position_strictly_below': .001, 'velocity_strictly_below': .01},
              'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                for name in names},
              'worlds': rows, 'passed': all(r['passed'] and r['all_states_finite'] for r in rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    print(f'Report: {args.output}')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
