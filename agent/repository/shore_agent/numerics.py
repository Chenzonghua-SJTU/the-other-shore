"""Deterministic candidate fitting and integration using observed records only.

This module deliberately has no simulator imports. dt denotes sampling spacing;
solve_ivp chooses integration steps independently and receives absolute time.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .expressions import Expression, ExpressionError, body_positions


class NumericalError(ValueError):
    def __init__(self, message, code="numerical_failure"):
        super().__init__(message)
        self.code = code


MAX_STATE = 1e7
MAX_RHS_CALLS = 12000
MAX_ROLLOUT_SECONDS = 4.0
MAX_FIT_SECONDS = 25.0
MAX_PHYSICAL_SPAN = 100.0


def _prepare_probe(probe, observations=False):
    try:
        times = np.asarray(probe["times"], dtype=float)
        initial = probe["initial_conditions"]
        r0 = np.asarray(initial["r0"], dtype=float)
        v0 = np.asarray(initial["v0"], dtype=float)
        t0 = float(initial.get("t0", times[0]))
        properties = initial.get("properties", {})
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise NumericalError("invalid probe record", "invalid_record") from exc
    # A grid value such as 527*0.005 may serialize as 2.6350000000000002
    # while the first sample is rounded to 2.635. Normalize round-off only;
    # genuinely earlier observations remain invalid below.
    if times.ndim == 1 and len(times) and math.isclose(float(times[0]), t0, rel_tol=0, abs_tol=1e-10):
        t0 = float(times[0])
    if (times.ndim != 1 or not 1 <= len(times) <= 1000 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or not math.isfinite(t0)
            or times[0] < t0 or times[-1] - t0 > MAX_PHYSICAL_SPAN
            or np.max(np.abs(times)) > 1e6):
        raise NumericalError("times must be finite, strictly increasing and span at most 100 time units", "invalid_record")
    if (r0.shape != (2,) or v0.shape != (2,) or not np.isfinite(r0).all()
            or not np.isfinite(v0).all() or np.max(np.abs(np.r_[r0, v0])) > MAX_STATE):
        raise NumericalError("invalid controlled initial state", "invalid_record")
    if not isinstance(properties, dict):
        raise NumericalError("properties must be an object", "invalid_record")
    if observations:
        for key in ("r_obs", "v_obs"):
            obs = np.asarray(probe.get(key), dtype=float)
            if obs.shape != (len(times), 2) or not np.isfinite(obs).all():
                raise NumericalError(f"{key} must be finite with shape [K,2]", "invalid_record")
    return times, t0, np.r_[r0, v0], properties


def _positions_at(bodies, t):
    # Python providers are an internal evaluator adapter, never a model DSL
    # capability. Tools pass only JSON observations or historical static lists.
    supplied = bodies(t) if callable(bodies) else bodies
    return body_positions(supplied, t)


def _check_body_coverage(expression, bodies, probe):
    if expression.uses_bodies:
        times, t0, *_ = _prepare_probe(probe)
        _positions_at(bodies, t0)
        _positions_at(bodies, float(times[-1]))


def _integrate(expression, parameters, probe, bodies, deadline=None):
    times, t0, y0, properties = _prepare_probe(probe)
    _check_body_coverage(expression, bodies, probe)
    local_deadline = time.monotonic() + MAX_ROLLOUT_SECONDS
    if deadline is not None:
        local_deadline = min(local_deadline, deadline)
    calls = 0

    def acceleration(t, state):
        nonlocal calls
        calls += 1
        if calls > MAX_RHS_CALLS:
            raise NumericalError("integration exceeded derivative-evaluation limit", "integration_limit")
        if time.monotonic() > local_deadline:
            raise NumericalError("numerical wall-time budget exhausted", "numerical_timeout")
        if not np.isfinite(state).all() or np.max(np.abs(state)) > MAX_STATE:
            raise NumericalError("candidate trajectory diverged", "integration_diverged")
        try:
            positions = _positions_at(bodies, t) if expression.uses_bodies else []
            return expression.acceleration(t, state[:2], state[2:], positions, properties, parameters)
        except ExpressionError as exc:
            code = exc.code if exc.code == 'body_time_coverage' else 'expression_domain'
            raise NumericalError(str(exc), code) from exc

    def rhs(t, state):
        return np.r_[state[2:], acceleration(t, state)]

    if times[-1] == t0:
        states = y0[None, :]
    else:
        try:
            sol = solve_ivp(rhs, (t0, float(times[-1])), y0, t_eval=times,
                            method="DOP853", rtol=2e-8, atol=1e-10)
        except NumericalError:
            raise
        except (ValueError, OverflowError, FloatingPointError) as exc:
            raise NumericalError(f"integration failed: {type(exc).__name__}") from exc
        if not sol.success or sol.y.shape != (4, len(times)):
            raise NumericalError("integration failed: " + str(sol.message))
        states = sol.y.T
    accelerations = np.asarray([acceleration(float(t), state) for t, state in zip(times, states)])
    if not np.isfinite(states).all() or not np.isfinite(accelerations).all():
        raise NumericalError("candidate produced non-finite predictions")
    return {"probe_id": probe.get("probe_id"), "times": times.tolist(),
            "r_pred": states[:, :2].tolist(), "v_pred": states[:, 2:].tolist(),
            "a_pred": accelerations.tolist()}


def rollout(law_spec, parameters, probe, bodies, timeout_seconds=MAX_ROLLOUT_SECONDS):
    """Replay controlled initial state with frozen parameters and a caller deadline.

    The effective budget is min(timeout_seconds, MAX_ROLLOUT_SECONDS), allowing
    callers to pass the remaining session time without extending per-rollout work.
    bodies is a static list or an acquired observed_body_trajectories context.
    A Python callable bodies(t) is also accepted for the private post-session
    evaluator's instantaneous-state adapter; no DSL or JSON tool can create one.
    """
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds < 0):
        raise NumericalError("rollout timeout must be a finite nonnegative number")
    if timeout_seconds == 0:
        raise NumericalError("numerical wall-time budget exhausted", "numerical_timeout")
    deadline = time.monotonic() + min(timeout_seconds, MAX_ROLLOUT_SECONDS)
    if not isinstance(parameters, dict):
        raise ExpressionError("parameters must be a mapping")
    spec = [{"name": name, "initial": value} for name, value in parameters.items()]
    expression = Expression(law_spec, spec)
    return _integrate(expression, parameters, probe, bodies, deadline=deadline)


def errors_for_rollouts(probes, rollouts):
    """Per-trajectory mean squared Euclidean norm, then equal trajectory weights."""
    if not probes or len(probes) != len(rollouts):
        raise NumericalError("errors need matching nonempty probe and rollout lists", "invalid_record")
    per_probe = []
    for probe, prediction in zip(probes, rollouts):
        times, *_ = _prepare_probe(probe, observations=True)
        if probe.get("probe_id") != prediction.get("probe_id"):
            raise NumericalError("prediction and probe IDs do not match", "invalid_record")
        if not np.array_equal(times, np.asarray(prediction["times"], dtype=float)):
            raise NumericalError("prediction and observation times do not match", "invalid_record")
        metrics = {"probe_id": probe.get("probe_id"), "n_samples": len(times)}
        for label, pred_key, obs_key in (("position", "r_pred", "r_obs"), ("velocity", "v_pred", "v_obs")):
            pred, obs = np.asarray(prediction[pred_key]), np.asarray(probe[obs_key])
            if pred.shape != obs.shape or not np.isfinite(pred).all():
                raise NumericalError("invalid prediction shape or values")
            mse = float(np.mean(np.sum((pred - obs) ** 2, axis=1)))
            metrics[label + "_mse"] = mse
            metrics[label + "_rmse"] = math.sqrt(mse)
        per_probe.append(metrics)
    overall = {label + "_mse": float(np.mean([p[label + "_mse"] for p in per_probe]))
               for label in ("position", "velocity")}
    overall.update({label + "_rmse": math.sqrt(overall[label + "_mse"])
                    for label in ("position", "velocity")})
    return {"overall": overall, "per_probe": per_probe,
            "definition": "sqrt(mean_trajectories(mean_samples(sum_coordinates((pred-obs)^2))))"}


def fit(law_spec, parameter_spec, probes, bodies, max_nfev=80, timeout_seconds=MAX_FIT_SECONDS,
        position_noise_sigma=None):
    """Local bounded least squares, minimizing position MSE with equal trajectory weights.

    Velocity is diagnostic only. Failed integrations have bounded penalties; if
    no usable fit remains, the returned status is failed. No global optimum claim.
    """
    if not isinstance(probes, list) or not probes or len(probes) > 102:
        raise NumericalError("fit needs between 1 and 102 observed trajectories", "invalid_record")
    ids = [p.get("probe_id") for p in probes]
    if len(set(ids)) != len(ids):
        raise NumericalError("duplicate training trajectory IDs", "invalid_record")
    for probe in probes:
        _prepare_probe(probe, observations=True)
    if type(max_nfev) is not int or not 1 <= max_nfev <= 300:
        raise NumericalError("max_nfev must be an integer between 1 and 300")
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120:
        raise NumericalError("fit timeout must be in (0,120] seconds")
    if position_noise_sigma is not None and (
            isinstance(position_noise_sigma, bool) or not isinstance(position_noise_sigma, (int, float))
            or not math.isfinite(position_noise_sigma) or position_noise_sigma < 0):
        raise NumericalError("position_noise_sigma must be nonnegative and finite, or unknown")
    expression = Expression(law_spec, parameter_spec)
    # Missing external-input coverage cannot be repaired by adjusting force-law
    # coefficients. Report it before optimization, rather than penalizing a law.
    for probe in probes:
        _check_body_coverage(expression, bodies, probe)
    normalized = expression.parameter_spec
    names = [item["name"] for item in normalized]
    initial = np.asarray([item["initial"] for item in normalized])
    deadline = time.monotonic() + timeout_seconds
    last_failure = None
    objective_calls = 0
    max_objective_calls = max_nfev * (len(names) + 2)
    size = sum(2 * len(probe["times"]) for probe in probes)
    invalid_evaluations = 0
    best = None

    def predictions(values):
        params = dict(zip(names, map(float, values)))
        return [_integrate(expression, params, probe, bodies, deadline) for probe in probes]

    def residual(values):
        nonlocal last_failure, objective_calls, invalid_evaluations, best
        objective_calls += 1
        if time.monotonic() > deadline or objective_calls > max_objective_calls:
            raise NumericalError("fit exceeded time or evaluation budget", "fit_limit")
        try:
            predicted = predictions(values)
            vector = np.concatenate([(np.asarray(p["r_pred"]) - np.asarray(obs["r_obs"])).ravel()
                                     / math.sqrt(len(obs["times"]) * len(probes))
                                     for obs, p in zip(probes, predicted)])
            cost = float(np.dot(vector, vector))
            if best is None or cost < best[0]:
                best = (cost, values.copy(), predicted)
            return vector
        except NumericalError as exc:
            if exc.code in ("numerical_timeout", "fit_limit"):
                raise
            last_failure = str(exc)
            invalid_evaluations += 1
            return np.full(size, 1e8 / math.sqrt(size))

    fitted = initial
    converged = False
    status = "failed"
    message = "fit did not execute"
    try:
        if names:
            result = least_squares(residual, initial,
                                   bounds=([p["lower"] for p in normalized], [p["upper"] for p in normalized]),
                                   max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8,
                                   x_scale="jac")
            fitted = result.x
            converged = bool(result.success)
            status = "converged" if converged else "not_converged"
            message = str(result.message)
            predicted = predictions(fitted)
        else:
            predicted = predictions(initial)
            converged, status, message = True, "no_parameters", "No parameters to optimize; evaluated the supplied structure."
        metrics = errors_for_rollouts(probes, predicted)
    except (NumericalError, ValueError, OverflowError, FloatingPointError) as exc:
        converged, status, message = False, "failed", str(exc)
        predicted, metrics = [], None
        if best is not None:
            _, fitted, predicted = best
            metrics = errors_for_rollouts(probes, predicted)
            status = "not_converged"
            message += " Returning the best finite iterate; optimization did not converge."
    if last_failure:
        message += f" Invalid candidate evaluations: {invalid_evaluations}; last failure: {last_failure}"
    refinement = {'attempted': False, 'status': 'skipped', 'tolerance': 1e-12,
                  'position_noise_sigma': position_noise_sigma, 'reason': 'coarse_fit_not_converged'}
    if converged:
        # Only the task's public noise level enables polishing. A tiny residual
        # is not evidence that noisy observations are noise-free.
        if position_noise_sigma != 0:
            refinement['reason'] = 'noise_present_or_unknown'
        elif not names:
            refinement['reason'] = 'no_parameters'
        else:
            scale = max(1.0, math.sqrt(float(np.mean([
                np.mean(np.sum(np.asarray(p['r_obs']) ** 2, axis=1)) for p in probes]))))
            rmse = metrics['overall']['position_rmse']
            remaining_nfev = max_nfev - result.nfev
            refinement.update(initial_position_rmse=rmse, final_position_rmse=rmse)
            if rmse > 1e-5 * scale:
                refinement['reason'] = 'residual_above_refinement_range'
            elif result.optimality <= 1e-12:
                refinement['reason'] = 'already_at_tight_gradient_tolerance'
            elif remaining_nfev <= 0 or time.monotonic() >= deadline:
                refinement['reason'] = 'shared_fit_budget_exhausted'
            else:
                refinement.update(attempted=True, status='retained_coarse', reason=None)
                before_calls = objective_calls
                try:
                    refined = least_squares(residual, fitted,
                        bounds=([p['lower'] for p in normalized], [p['upper'] for p in normalized]),
                        max_nfev=remaining_nfev, ftol=1e-12, xtol=1e-12, gtol=1e-12, x_scale='jac')
                    candidate = predictions(refined.x)
                    candidate_metrics = errors_for_rollouts(probes, candidate)
                    if refined.success and candidate_metrics['overall']['position_mse'] <= metrics['overall']['position_mse']:
                        fitted, predicted, metrics = refined.x, candidate, candidate_metrics
                        message = str(refined.message) + ' Noiseless parameter refinement completed.'
                        refinement.update(status='refined', final_position_rmse=metrics['overall']['position_rmse'])
                    else:
                        refinement['reason'] = 'refinement_not_converged_or_not_better'
                except (NumericalError, ValueError, OverflowError, FloatingPointError) as exc:
                    # Keep the already valid coarse solution; no new deadline,
                    # extra evaluation budget, validation data or truth is used.
                    refinement['reason'] = getattr(exc, 'code', type(exc).__name__)
                refinement['objective_calls'] = objective_calls - before_calls
    return {"fitted_parameters": dict(zip(names, map(float, fitted))), "fit_status": status,
            "converged": converged, "message": message, "errors": metrics,
            "rollouts": predicted, "objective_calls": objective_calls,
            "invalid_evaluations": invalid_evaluations, "refinement": refinement,
            "objective": "equal_trajectory_position_mse", "optimizer": "scipy.least_squares (local)"}
