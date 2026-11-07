"""Body state estimates computed only from public initial states and observations."""
import math

import numpy as np


def estimate_body_trajectories(observations, initial_states=None, position_noise_sigma=None):
    """Group by absolute time, never average a moving body's different epochs.

    Known t=0 positions override noisy samples at that epoch. Between observed
    times the numerical tools interpolate linearly; no motion law is assumed.
    """
    sigma = position_noise_sigma
    if sigma is not None and (isinstance(sigma, bool) or not math.isfinite(sigma) or sigma < 0):
        raise ValueError('Invalid public position noise scale')
    initial_states = initial_states or []
    initial = {}
    for body in initial_states:
        body_id, position = body['body_id'], np.asarray(body['position'], dtype=float)
        if not isinstance(body_id, str) or body_id in initial or position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError('Invalid known body initial states')
        initial[body_id] = position.tolist()
    records, samples = {}, {}
    body_ids = list(initial)
    for observation in observations:
        rid = observation['observation_id']
        if rid in records:
            if records[rid] != observation:
                raise ValueError('Conflicting body observation records')
            continue
        records[rid] = observation
        ids, times = observation['body_ids'], observation['times']
        if (not ids or len(set(ids)) != len(ids) or not times
                or any(not isinstance(v, str) for v in ids)):
            raise ValueError('Invalid body observation axes')
        if not body_ids:
            body_ids = list(ids)
        if set(ids) != set(body_ids):
            raise ValueError('Body observation identifiers do not match the task')
        times = np.asarray(times, dtype=float)
        positions = np.asarray(observation['positions_obs'], dtype=float)
        if (times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0)
                or np.any(np.diff(times) <= 0) or positions.shape != (len(times), len(ids), 2)
                or not np.isfinite(positions).all()):
            raise ValueError('Invalid observed body positions or absolute times')
        for t, row in zip(times, positions):
            epoch = samples.setdefault(round(float(t), 10), {})
            for body_id, position in zip(ids, row):
                epoch.setdefault(body_id, []).append(position)
    if initial:
        samples.setdefault(0.0, {})
    if not samples or not body_ids:
        raise ValueError('Body positions require public initial states or acquired observations')
    times = sorted(samples)
    positions, counts, errors = [], [], []
    for t in times:
        row, row_counts, row_errors = [], [], []
        for body_id in body_ids:
            if t == 0 and body_id in initial:
                row.append(initial[body_id]); row_counts.append(0); row_errors.append([0., 0.])
                continue
            values = samples[t].get(body_id, [])
            if not values:
                raise ValueError('Incomplete body observations at an acquired time')
            row.append(np.mean(values, axis=0).tolist())
            row_counts.append(len(values))
            row_errors.append(None if sigma is None else [float(sigma / math.sqrt(len(values)))] * 2)
        positions.append(row); counts.append(row_counts); errors.append(row_errors)
    return {'kind': 'observed_body_trajectories', 'version': 'body_trajectory_' + str(len(records)),
            'method': 'piecewise_linear_observed_positions', 'body_ids': body_ids,
            'times': times, 'positions': positions, 'time_range': [times[0], times[-1]],
            'observation_ids': list(records), 'sample_counts': counts, 'position_standard_errors': errors,
            'known_initial_positions': initial,
            'bodies': [{'body_id': bid, 'position': position, 'position_time': times[-1]}
                       for bid, position in zip(body_ids, positions[-1])],
            'assumption': 'Linear interpolation between acquired epochs; no extrapolation or stationary-body assumption.',
            'uncertainty_note': 'Sample standard errors describe observation noise only, not interpolation bias; t=0 initial positions are supplied as known states.'}


def estimate_static_bodies(observations, position_noise_sigma=None):
    """Average independent position samples; never read simulator configuration/truth.

    A repeated record ID is counted once. The standard error uses the public
    observation-noise scale, not the unknown distance to a true body position.
    """
    records = {}
    for observation in observations:
        rid = observation['observation_id']
        if rid in records:
            if records[rid] != observation:
                raise ValueError('Conflicting body observation records')
            continue
        records[rid] = observation
    if not records:
        raise ValueError('Body positions require acquired observations')
    sigma = position_noise_sigma
    if sigma is not None and (isinstance(sigma, bool) or not math.isfinite(sigma) or sigma < 0):
        raise ValueError('Invalid public position noise scale')
    samples = {}
    for obs in records.values():
        ids, times = obs['body_ids'], obs['times']
        if not ids or len(set(ids)) != len(ids) or not times:
            raise ValueError('Invalid body observation axes')
        positions = np.asarray(obs['positions_obs'], dtype=float)
        if positions.shape != (len(times), len(ids), 2) or not np.isfinite(positions).all():
            raise ValueError('Invalid observed body positions')
        for index, body_id in enumerate(ids):
            samples.setdefault(body_id, []).extend(positions[:, index, :].tolist())
    bodies = []
    for body_id, values in samples.items():
        count = len(values)
        error = None if sigma is None else [float(sigma / math.sqrt(count))] * 2
        bodies.append({'body_id': body_id, 'position': np.mean(values, axis=0).tolist(),
                       'sample_count': count, 'position_standard_error': error})
    return {'version': 'body_estimate_' + str(len(records)),
            'method': 'mean_of_acquired_static_observations',
            'observation_ids': list(records),
            'sample_count_per_body': {b['body_id']: b['sample_count'] for b in bodies},
            'bodies': bodies,
            'assumption': 'Visible bodies are static; all acquired independent position samples have equal weight.',
            'uncertainty_note': 'Per-coordinate standard error uses the public independent noise sigma / sqrt(sample_count); it is not the realized error relative to truth.'}
