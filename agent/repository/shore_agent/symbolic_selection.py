"""Observation-only error bands: prefer simpler laws at comparable accuracy.

Bands are engineering tolerances, not confidence intervals or submission gates.
All elementary operators have equal cost; no world/force templates are used.
"""
from __future__ import annotations

import ast
import math


RELATIVE_MSE_TOLERANCE = 0.05


def finite_error(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def structural_complexity(candidate):
    """Count the exported two-axis expression, consistently across backends."""
    parameters = {p['name'] for p in candidate['parameter_spec']}
    used = set()

    def size(node):
        if isinstance(node, ast.Name):
            if node.id in parameters:
                used.add(node.id)
            return 1, 1
        if isinstance(node, ast.Constant):
            return 1, 1
        if isinstance(node, ast.Call):
            # A public scalar property is one input, just like x or t.
            if isinstance(node.func, ast.Name) and node.func.id == 'q':
                return 1, 1
            children = node.args
        elif isinstance(node, ast.BinOp):
            children = [node.left, node.right]
        elif isinstance(node, ast.UnaryOp):
            children = [node.operand]
        else:
            raise ValueError('Unsupported symbolic complexity node')
        values = [size(child) for child in children]
        return 1 + sum(n for n, _ in values), 1 + max(d for _, d in values)

    axes = [size(ast.parse(candidate['law_spec'][axis], mode='eval').body)
            for axis in ('ax', 'ay')]
    return {'nodes': sum(n for n, _ in axes), 'parameters': len(used),
            'depth': max(d for _, d in axes)}


def position_mse_tolerance(probes):
    """Numerical floor and public-noise energy sampling scale; no hidden labels.

    Each trajectory has equal weight. The noise term is the standard deviation
    of averaged 2D squared independent Gaussian noise, not a general confidence
    interval for a fitted dynamical model (correlated/model errors remain).
    """
    noise_variances, scales = [], [1.0]
    for probe in probes:
        sigma = probe.get('observation_noise', {}).get('position_sigma') or 0.0
        if not finite_error(sigma):
            raise ValueError('invalid_position_noise')
        points = probe.get('r_obs', [])
        noise_variances.append(4 * sigma**4 / max(1, len(points)))
        scales.extend(abs(float(x)) for point in points for x in point)
    noise = math.sqrt(sum(noise_variances)) / max(1, len(probes))
    numerical = 2 * (5 * (1e-10 + 2e-8 * max(scales)))**2
    return max(noise, numerical)


def candidate_error(candidate, stage='final'):
    residuals = candidate.get('residuals', {})
    if stage == 'final':
        overall = residuals.get('trajectory_training', {}).get('overall', {})
        error = overall.get('position_mse')
        if not finite_error(error) and finite_error(overall.get('position_rmse')):
            error = overall['position_rmse']**2
        if finite_error(error):
            return 0, 'trajectory_position_mse', error
    else:
        error = residuals.get('observed_velocity_integral', {}).get('relative_increment_rmse')
        if finite_error(error):
            return 0, 'relative_velocity_increment_mse', error**2
    error = candidate.get('normalized_acceleration_search_loss')
    return 1, 'normalized_acceleration_loss', error if finite_error(error) else float('inf')


def rank_candidates(candidates, probes, *, stage='final', acceleration_tolerance=0.0):
    """Order by observed risk, comparable metric, then simplicity in error bands.

    Never compare acceleration loss numerically with position MSE. Keep a
    materially more accurate complicated law ahead of a simple underfit law.
    Unknown/unfinished checks are reported separately, not certified stable.
    """
    groups = {}
    for candidate in candidates:
        integral = candidate.get('residuals', {}).get('observed_velocity_integral', {})
        risk = bool(integral.get('domain_error_probe_ids') or
                    integral.get('denominator_sign_change_probe_ids'))
        # A partial path check must not beat a completed check merely because
        # its reported error covers fewer/easier observed trajectories.
        risk_level = 2 * int(risk) + int(integral.get('complete') is False)
        family, metric, error = candidate_error(candidate, stage)
        candidate['structural_complexity'] = structural_complexity(candidate)
        groups.setdefault((risk_level, family, metric), []).append((error, candidate))
    ranked = []
    position_tolerance = position_mse_tolerance(probes)
    for (risk_level, _, metric), entries in sorted(groups.items()):
        pending = sorted(entries, key=lambda item: item[0])
        while pending:
            best = pending[0][0]
            absolute = (position_tolerance if metric == 'trajectory_position_mse' else
                        max(0.0, acceleration_tolerance) if metric == 'normalized_acceleration_loss' else 1e-20)
            tolerance = max(absolute, RELATIVE_MSE_TOLERANCE * best) if math.isfinite(best) else 0.0
            band = [item for item in pending if item[0] <= best + tolerance]
            pending = [item for item in pending if item[0] > best + tolerance]
            band.sort(key=lambda item: (item[1]['structural_complexity']['nodes'],
                                       item[1]['structural_complexity']['parameters'],
                                       item[1]['structural_complexity']['depth'], item[0]))
            for error, candidate in band:
                candidate['selection_diagnostics'] = {
                    'metric': metric, 'error': error if math.isfinite(error) else None,
                    'band_best_error': best if math.isfinite(best) else None,
                    'absolute_tolerance': absolute, 'relative_mse_tolerance': RELATIVE_MSE_TOLERANCE,
                    'observed_domain_or_pole_risk': risk_level >= 2,
                    'rule': 'simplest exported expression within comparable-error band',
                    'independent_validation': False}
                ranked.append(candidate)
    return ranked
