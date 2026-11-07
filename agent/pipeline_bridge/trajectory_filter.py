"""Observation-only joint spline smoothing. No simulator, model, or world inputs."""
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy import sparse
from scipy.sparse.linalg import spsolve, MatrixRankWarning

VERSION = 'joint_spline_v1'
LAMBDAS = (1e-8, 1e-6, 1e-4, 1e-2, 1., 100.)
PAYLOAD_KEY = '_joint_spline_search'
CONSUMER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def capability():
    return {'version': VERSION, 'consumer_sha256': CONSUMER_SHA256}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def raw_digest(probe):
    return digest({k: v for k, v in probe.items() if k != PAYLOAD_KEY})


def validate(probe):
    t, r, v = (np.asarray(probe[k], dtype=float) for k in ('times', 'r_obs', 'v_obs'))
    if (t.ndim != 1 or not 3 <= len(t) <= 1000 or r.shape != (len(t), 2)
            or v.shape != r.shape or not all(np.isfinite(a).all() for a in (t, r, v))
            or not np.all(np.diff(t) > 0)):
        raise ValueError('Invalid trajectory arrays or timestamps')
    noise = probe['observation_noise']
    sigmas = np.array([noise['position_sigma'], noise['velocity_sigma']], dtype=float)
    if not np.isfinite(sigmas).all() or (sigmas < 0).any():
        raise ValueError('Public noise scales must be finite and nonnegative')
    initial = probe['initial_conditions']
    t0 = float(initial.get('t0', t[0]))
    r0, v0 = (np.asarray(initial[k], dtype=float) for k in ('r0', 'v0'))
    if not np.isfinite(t0) or t0 > t[0] or r0.shape != (2,) or v0.shape != (2,) or not np.isfinite([r0, v0]).all():
        raise ValueError('Invalid controlled initial state')
    return t, r, v, sigmas, t0, r0, v0


def _basis(knots, points, order):
    degree = 5
    transform = sparse.eye(len(knots)-degree-1, format='csr')
    for _ in range(order):
        n = len(knots)-degree-1
        factor = degree/(knots[degree+1:degree+n]-knots[1:n])
        transform = sparse.diags((-factor, factor), (0, 1), shape=(n-1, n)) @ transform
        knots = knots[1:-1]
        degree -= 1
    return sparse.csr_matrix(BSpline.design_matrix(points, knots, degree)) @ transform


def filter_probe(probe, *, deadline=float('inf')):
    """Return full-timestamp r/v/a and diagnostic metadata; never mutate input.

    A bypass result has no derived arrays: the caller must use the original
    legacy path. CV is discovery data reuse, not independent law validation.
    """
    started = time.monotonic()
    t, r, v, sigmas, t0, r0, v0 = validate(probe)
    base = {'probe_id': probe['probe_id'], 'version': VERSION,
            'raw_sha256': raw_digest(probe), 'sigma_r': float(sigmas[0]),
            'sigma_v': float(sigmas[1]), 'samples': len(t)}
    if not sigmas.any() or len(t) < 12:
        return dict(base, status='bypass', reason='zero_noise' if not sigmas.any() else 'short_trajectory')
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('joint_spline_budget')
    check()
    T = t[-1]-t0
    u = (t-t0)/T
    scale = max(float(np.max(np.std(r, axis=0))), float(np.max(np.abs(v)))*T,
                float(sigmas[0]), float(sigmas[1])*T, 1e-12)
    targets = [(r-r0)/scale, v*T/scale]
    sigma = sigmas * np.array([1., T])/scale
    positive = sigma > 0
    precision = np.zeros(2)
    precision[positive] = (sigma[positive].min()/sigma[positive])**2
    # First observation can coincide with t0; no repeated interior endpoint.
    knots = np.r_[np.zeros(6), u[(u > 0) & (u < 1)], np.ones(6)]
    B = [_basis(knots, u, order) for order in range(3)]
    initial_B = [_basis(knots, np.array([0.]), order) for order in (0, 1)]
    intervals = np.unique(knots)
    nodes, weights = np.polynomial.legendre.leggauss(3)
    mid, half = (intervals[1:]+intervals[:-1])/2, np.diff(intervals)/2
    q = (mid[:, None]+half[:, None]*nodes).ravel()
    qw = (half[:, None]*weights).ravel()
    D = _basis(knots, q, 3)
    penalty = D.T @ sparse.diags(qw) @ D

    def solve(lam, keep):
        check()
        H = lam*penalty
        rhs = np.zeros((H.shape[0], 2))
        constraints, values = [], []
        for channel in (0, 1):
            design = B[channel][keep]
            target = targets[channel][keep]
            if sigma[channel] > 0:
                H = H + precision[channel]*(design.T @ design)
                rhs += precision[channel]*(design.T @ target)
            else:
                constraints.append(design)
                values.append(target)
            # Avoid duplicate exact constraints at a noise-free t0 observation.
            if sigma[channel] > 0 or not (keep[0] and u[0] == 0):
                constraints.append(initial_B[channel])
                values.append(np.zeros((1, 2)) if channel == 0 else (v0*T/scale)[None, :])
        C = sparse.vstack(constraints, format='csr')
        value = np.vstack(values)
        hscale = max(float(np.max(np.abs(H.diagonal()))), 1e-30)
        system = sparse.bmat([[H/hscale, C.T], [C, None]], format='csc')
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            coef = spsolve(system, np.vstack((rhs/hscale, value)))[:H.shape[0]]
        check()
        if not np.isfinite(coef).all() or np.max(np.abs(C @ coef-value)) > 1e-7:
            raise ValueError('joint_spline_constraint_or_condition_failure')
        return coef

    # Fixed interior two-point blocks, same held-out times for both channels.
    blocks = [np.array([i, i+1]) for i in np.linspace(2, len(t)-4, 3).astype(int)]
    scores = []
    for lam in LAMBDAS:
        errors = []
        for block in blocks:
            keep = np.ones(len(t), dtype=bool)
            keep[block] = False
            coef = solve(lam, keep)
            for channel in (0, 1):
                if sigma[channel] > 0:
                    errors.extend(((B[channel][block] @ coef-targets[channel][block])/sigma[channel]).ravel()**2)
        scores.append(float(np.mean(errors)))
    best = min(scores)
    selected = next(i for i, score in enumerate(scores) if score <= best*1.01+1e-12)
    coef = solve(LAMBDAS[selected], np.ones(len(t), dtype=bool))
    rf, vf, af = B[0] @ coef*scale+r0, B[1] @ coef*scale/T, B[2] @ coef*scale/T**2
    if not np.isfinite([rf, vf, af]).all():
        raise ValueError('nonfinite_filter_output')
    result = dict(base, status='filtered', **{'lambda': LAMBDAS[selected]},
                  arrays={'r': rf.tolist(), 'v': vf.tolist(), 'a': af.tolist()},
                  cv_scores=dict(zip(map(str, LAMBDAS), scores)),
                  boundary_warning=selected in (0, len(LAMBDAS)-1),
                  position_correction_rms=float(np.sqrt(np.mean((rf-r)**2))),
                  velocity_correction_rms=float(np.sqrt(np.mean((vf-v)**2))),
                  elapsed_seconds=time.monotonic()-started,
                  uncertainty_scope='No calibrated acceleration uncertainty; CV reuses discovery observations.')
    result['output_sha256'] = digest(result['arrays'])
    return result


def consume(probe, t, r, v, legacy):
    """SR-only dispatch; derived targets never replace dataset['probes']."""
    payload = probe.get(PAYLOAD_KEY)
    if payload is None:
        return legacy(probe, t, r, v)
    if payload['version'] != VERSION or payload['raw_sha256'] != raw_digest(probe):
        raise ValueError('Stale or unsupported filter artifact')
    validate(probe)
    if payload['status'] != 'filtered':
        return legacy(probe, t, r, v)
    arrays = payload['arrays']
    if payload['output_sha256'] != digest(arrays):
        raise ValueError('Filter output hash mismatch')
    rf, vf, af = (np.asarray(arrays[k], dtype=float) for k in ('r', 'v', 'a'))
    if any(a.shape != r.shape or not np.isfinite(a).all() for a in (rf, vf, af)):
        raise ValueError('Invalid filter output shape')
    detail = {k: val for k, val in payload.items() if k not in ('arrays', 'cv_scores')}
    # Preserve the original raw-observation stencil sensitivity for the existing
    # candidate simplicity tolerance. This is NOT a second pass over joint data
    # and NOT a calibrated joint-spline variance estimate.
    raw_diagnostics = legacy(probe, t, r, v)
    raw_quality, raw_resolution = raw_diagnostics[3:5]
    detail.update(method=VERSION, resolution_source='legacy_raw_observation_stencil_sensitivity',
                  scope='Search-only joint spline; original raw stencil diagnostic retained, not joint statistical uncertainty.')
    return rf[1:-1], vf[1:-1], af[1:-1], raw_quality, raw_resolution, detail


def annotate_report(report, probes):
    filtered = [p['probe_id'] for p in probes if p.get(PAYLOAD_KEY, {}).get('status') == 'filtered']
    if not filtered:
        return report
    mixed = len(filtered) != len(probes)
    report['target'] = 'mixed_joint_spline_and_legacy_acceleration' if mixed else 'joint_spline_acceleration'
    report.setdefault('data_quality', {}).update(
        acceleration_source=report['target'], joint_spline_probe_ids=filtered,
        resolution_source='Original raw-observation stencil sensitivity, not joint spline confidence.')
    old = 'Search loss uses finite-difference acceleration, not independent prediction.'
    report['warnings'] = [w for w in report.get('warnings', []) if w != old]
    note = 'Search uses observation-only joint spline derivatives'+(' with legacy fallback targets' if mixed else '')+'; this is not independent prediction. Resolution tolerance retains the original raw-observation stencil sensitivity.'
    if note not in report['warnings']:
        report['warnings'].append(note)
    return report


def install_consumer():
    """Install in an opted-in worker before search, leaving source hashes intact."""
    from ..repository.shore_agent import symbolic_quality, symbolic_regression, symbolic_service
    if getattr(symbolic_quality.derivatives, '_joint_filter_consumer', False):
        return
    original = symbolic_quality.derivatives
    def derivatives(probe, t, r, v):
        return consume(probe, t, r, v, original)
    derivatives._joint_filter_consumer = True
    symbolic_quality.derivatives = derivatives
    original_search = symbolic_regression.run_search
    def run_search(probes, *args, **kwargs):
        callback = kwargs.get('progress_callback')
        if callback is not None:
            kwargs['progress_callback'] = lambda report: callback(annotate_report(report, probes))
        return annotate_report(original_search(probes, *args, **kwargs), probes)
    symbolic_regression.run_search = run_search
    warmup = symbolic_service._warmup_models
    def warmup_models(directory):
        result = warmup(directory)
        result['trajectory_filter'] = capability()
        return result
    symbolic_service._warmup_models = warmup_models
