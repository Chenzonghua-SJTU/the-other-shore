"""Bounded, data-fitted basis proposals from acquired observations only.

These are regression candidates, not PySR discoveries or accepted physical laws.
No simulator, world identity, law catalogue, or hidden evaluation is consulted.
"""
from __future__ import annotations

import ast
import itertools
import math
import time

import numpy as np


MAX_BASIS_SECONDS = 3.0
MAX_FIT_ROWS = 768
STATE_NAMES = ("x", "y", "vx", "vy")


class _Deadline(Exception):
    pass


def _check(stop):
    if time.monotonic() >= stop:
        raise _Deadline


def _fit(matrix, targets, weights, stop):
    """Scale columns for numerical conditioning, without changing their meaning."""
    _check(stop)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite regression basis")
    weighted = matrix * np.sqrt(weights)[:, None]
    scales = np.maximum(np.linalg.norm(weighted, axis=0), 1e-30)
    coefficients, _, rank, _ = np.linalg.lstsq(weighted/scales, targets*np.sqrt(weights)[:, None], rcond=1e-10)
    _check(stop)
    coefficients = coefficients/scales[:, None]
    with np.errstate(all="ignore"):
        loss = np.sum(weights[:, None]*(matrix@coefficients-targets)**2, axis=0)
    if not np.isfinite(coefficients).all() or not np.isfinite(loss).all():
        raise ValueError("Nonfinite regression fit")
    return coefficients, loss, int(rank)


def _source(terms, coefficients):
    pieces = []
    for term, coefficient in zip(terms, coefficients):
        coefficient = float(coefficient)
        if abs(coefficient) < 1e-14:
            continue
        part = format(abs(coefficient), ".15g") + "*(" + term + ")"
        pieces.append(("-("+part+")") if coefficient < 0 else part)
    return " + ".join(pieces) or "0"


def _row(terms, coefficients, loss, family, metadata):
    source = _source(terms, coefficients)
    # Count the same elementary syntax operators/operands used in the source.
    complexity = sum(isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant))
                     for n in ast.walk(ast.parse(source, mode="eval")))
    return {"source": source, "complexity": complexity, "origin": "regression_basis",
            "basis_family": family, "basis_fit_mse": float(loss),
            "fitting_metadata": dict(metadata, independent_validation=False)}


def _sparse_paths(matrix, terms, targets, weights, stop, max_terms=6):
    """Generic forward sparse regression, plus thresholded full least squares."""
    outputs = [[], []]
    full, _, rank = _fit(matrix, targets, weights, stop)
    norms = np.sqrt(np.sum(weights[:, None]*matrix**2, axis=0))
    for axis in range(2):
        target = targets[:, axis:axis+1]
        active, residual = [], target[:, 0].copy()
        for _ in range(min(max_terms, len(terms))):
            _check(stop)
            with np.errstate(all="ignore"):
                correlation = np.abs(matrix.T@(weights*residual))/np.maximum(norms, 1e-30)
            correlation[active] = -1
            index = int(np.argmax(correlation))
            if norms[index] <= 1e-20 or correlation[index] <= 1e-14:
                break
            active.append(index)
            coefficients, loss, sparse_rank = _fit(matrix[:, active], target, weights, stop)
            if sparse_rank != len(active):
                break
            outputs[axis].append(([terms[i] for i in active], coefficients[:, 0], float(loss[0])))
            with np.errstate(all="ignore"):
                residual = target[:, 0]-matrix[:, active]@coefficients[:, 0]
        # Threshold contributions, then refit. Never retain a coefficient merely
        # because it happens to resemble another parameter or a physical constant.
        for threshold in (.001, .01, .03, .1):
            _check(stop)
            contributions = abs(full[:, axis])*norms
            selected = np.flatnonzero(contributions >= max(float(np.max(contributions))*threshold, 1e-14))
            if not 1 <= len(selected) <= max_terms:
                continue
            coefficients, loss, sparse_rank = _fit(matrix[:, selected], target, weights, stop)
            if sparse_rank == len(selected):
                outputs[axis].append(([terms[i] for i in selected], coefficients[:, 0], float(loss[0])))
    return outputs, rank


def _polynomial(table, targets, weights, stop):
    rows = [[], []]
    names = [name for name in STATE_NAMES if name in table]
    terms, columns = ["1"], [np.ones(len(targets))]
    ranks = []
    for degree in (1, 2, 3):
        for indexes in itertools.combinations_with_replacement(range(len(names)), degree):
            if time.monotonic() >= stop:
                return rows, ranks, False
            factors = [names[i] for i in sorted(set(indexes))]
            term = "*".join(name if indexes.count(names.index(name)) == 1 else
                            name+"**"+str(indexes.count(names.index(name))) for name in factors)
            columns.append(np.prod([table[names[i]] for i in indexes], axis=0))
            terms.append(term)
        try:
            paths, rank = _sparse_paths(np.column_stack(columns), terms, targets, weights, stop)
        except _Deadline:
            return rows, ranks, False
        ranks.append({"degree": degree, "columns": len(terms), "rank": rank})
        for axis in range(2):
            for selected, coefficients, loss in paths[axis]:
                rows[axis].append(_row(selected, coefficients, loss, "sparse_polynomial",
                                       {"maximum_degree": degree, "selected_terms": len(selected)}))
    return rows, ranks, True


def _periodic(table, targets, weights, stop):
    diagnostic = {"status": "unresolved", "independent_validation": False}
    if "t" not in table:
        return [[], []], dict(diagnostic, reason="no_varying_time")
    t = table["t"]
    unique = np.unique(t)
    if len(unique) < 12:
        return [[], []], dict(diagnostic, reason="insufficient_distinct_times")
    span = float(np.ptp(unique))
    gaps = np.diff(unique)
    significant_gaps = gaps[gaps > max(span*1e-9, np.finfo(float).eps)]
    step = float(np.median(significant_gaps)) if len(significant_gaps) else 0.
    if span <= 0 or step <= 0:
        return [[], []], dict(diagnostic, reason="insufficient_time_span")
    # Include quarter-cycle hypotheses, but only emit them if the measured
    # frequency profile resolves nearby alternatives. A short monotone window
    # alone is not evidence for arbitrarily slow, guessed oscillations.
    low, high = math.pi/(2*span), math.pi/step
    diagnostic.update(observed_time_span=span, median_time_step=step,
                      frequency_range_radians_per_time=[low, high], minimum_observed_cycles=.25,
                      maximum_observation_gap=float(np.max(np.diff(unique))),
                      temporal_alias_risk=bool(np.max(np.diff(unique)) > .25*span))
    names = [name for name in STATE_NAMES if name in table]
    base = np.column_stack([np.ones(len(t)), *[table[name] for name in names]])
    base_terms = ["1", *names]
    _, baseline, _ = _fit(base, targets, weights, stop)
    signal = np.maximum(np.sum(weights[:, None]*targets**2, axis=0), 1e-20)
    cache = {}
    def evaluate(frequency):
        _check(stop)
        frequency = float(frequency)
        if frequency not in cache:
            matrix = np.column_stack([base, base*np.sin(frequency*t)[:, None], base*np.cos(frequency*t)[:, None]])
            coefficients, loss, rank = _fit(matrix, targets, weights, stop)
            cache[frequency] = (float(np.mean(loss/signal)), coefficients, loss, rank)
        return cache[frequency][0]
    grid = np.geomspace(low, high, 48)
    for frequency in grid:
        evaluate(frequency)
    index = int(np.argmin([cache[float(f)][0] for f in grid]))
    if index in (0, len(grid)-1):
        return [[], []], dict(diagnostic, reason="frequency_minimum_at_resolution_boundary", frequencies_evaluated=len(cache))
    from scipy.optimize import minimize_scalar
    optimum = minimize_scalar(evaluate, bounds=(grid[index-1], grid[index+1]), method="bounded",
                              options={"maxiter": 24, "xatol": max(low*1e-7, 1e-10)})
    frequency = float(optimum.x)
    error, coefficients, loss, rank = cache.get(frequency, cache[min(cache, key=lambda f: abs(f-frequency))])
    baseline_error = float(np.mean(baseline/signal))
    diagnostic.update(frequency=frequency, frequencies_evaluated=len(cache), normalized_fit_mse=error,
                      normalized_unmodulated_mse=baseline_error, design_rank=rank, design_columns=3*len(base_terms))
    if rank < 3*len(base_terms):
        return [[], []], dict(diagnostic, reason="time_modulation_design_rank_deficient")
    if error >= .8*baseline_error or baseline_error < 1e-12:
        return [[], []], dict(diagnostic, reason="no_resolved_modulation_gain")
    nearby = [evaluate(frequency*factor) for factor in (.9, 1.1)]
    # A flat frequency profile is evidence of insufficient temporal information,
    # even if a particular sinusoid happens to fit that short observed window.
    if min(nearby)-error <= max(1e-10, error*.05):
        return [[], []], dict(diagnostic, reason="frequency_profile_unresolved", neighboring_profile_losses=nearby)
    n = len(base_terms)
    sine, cosine = coefficients[n:2*n], coefficients[2*n:]
    amplitudes = np.hypot(sine, cosine)
    # Any rotation is equivalent before sparsification. Orient the common phase
    # along the strongest measured Fourier coefficient to encourage sparse form.
    pivot = np.unravel_index(int(np.argmax(amplitudes)), amplitudes.shape)
    phase = math.atan2(float(cosine[pivot]), float(sine[pivot]))
    # Modulo pi keeps sign in the outer coefficient and shares one phase across
    # axes without asserting equal amplitudes or forcing any physical symmetry.
    if phase < 0:
        phase += math.pi
    angle = f"({frequency:.15g}*t + {phase:.15g})"
    terms = [*base_terms, *[f"({term})*sin({angle})" for term in base_terms],
             *[f"({term})*cos({angle})" for term in base_terms]]
    matrix = np.column_stack([base, base*np.sin(frequency*t+phase)[:, None], base*np.cos(frequency*t+phase)[:, None]])
    paths, _ = _sparse_paths(matrix, terms, targets, weights, stop, max_terms=5)
    rows = [[], []]
    for axis in range(2):
        for selected, fitted, sparse_loss in paths[axis]:
            rows[axis].append(_row(selected, fitted, sparse_loss, "shared_frequency_linear_modulation",
                                   {"frequency": frequency, "phase": phase, "selected_terms": len(selected),
                                    "frequency_identifiability": "resolved_on_supplied_data"}))
    diagnostic.update(status="resolved_on_supplied_data", phase=phase,
                      limitation="Local identifiability heuristic only; fresh-probe validation is required.")
    return rows, diagnostic


def _bounded_rows(rows, count):
    unique = {}
    for row in rows:
        if len(row["source"]) <= 1800 and row["complexity"] <= 120 and np.isfinite(row["basis_fit_mse"]):
            unique[row["source"]] = row
    ordered = sorted(unique.values(), key=lambda r:(r["basis_fit_mse"],r["complexity"]))
    frontier = [row for row in ordered if not any(other["complexity"] <= row["complexity"]
                and other["basis_fit_mse"] < row["basis_fit_mse"] for other in ordered)]
    if len(frontier) <= count:
        return frontier
    by_size = sorted(frontier, key=lambda r:r["complexity"])
    selected = [frontier[0]]
    if by_size[0] not in selected:
        selected.append(by_size[0])
    for index in np.linspace(0,len(by_size)-1,count,dtype=int):
        if by_size[index] not in selected:
            selected.append(by_size[index])
    return selected[:count]


def generate_basis_candidates(dataset, *, deadline, max_candidates_per_axis=8):
    """Return ``axis_rows`` and diagnostics; never fit/submit an executable law.

    ``deadline`` is absolute ``time.monotonic()`` and is additionally capped at
    three seconds. Small bounded linear solves cannot be interrupted mid-LAPACK;
    the enclosing worker's process deadline remains the final hard safeguard.
    """
    started = time.monotonic()
    if not math.isfinite(float(deadline)):
        raise ValueError("deadline must be finite")
    stop = min(float(deadline), started+MAX_BASIS_SECONDS)
    if type(max_candidates_per_axis) is not int or not 1 <= max_candidates_per_axis <= 16:
        raise ValueError("max_candidates_per_axis must be between 1 and 16")
    rows = [[], []]
    diagnostics = {"origin": "regression_basis", "independent_validation": False,
                   "data_source": "selected current-session velocity-derived acceleration",
                   "time_modulation": {"status":"not_attempted"}, "polynomial": {"status":"not_attempted"}}
    try:
        _check(stop)
        X, targets, weights = (np.asarray(dataset[key], dtype=float) for key in ("X","y","weights"))
        names = list(dataset["feature_names"])
        if X.ndim != 2 or targets.shape != (len(X),2) or weights.shape != (len(X),) or len(names) != X.shape[1]:
            raise ValueError("Invalid numerical dataset shapes")
        if not all(np.isfinite(a).all() for a in (X,targets,weights)) or np.any(weights <= 0):
            raise ValueError("Dataset must contain finite observations and positive weights")
        if len(X)<3:
            diagnostics.update(reason="insufficient_derivative_samples", observed_rows=len(X),
                               elapsed_seconds=time.monotonic()-started, budget_exhausted=False)
            return {"axis_rows":rows,"diagnostics":diagnostics}
        if len(set(names)) != len(names):
            raise ValueError("Feature names must be unique")
        indexes = np.linspace(0,len(X)-1,min(len(X),MAX_FIT_ROWS),dtype=int)
        samples = dataset.get("samples_per_probe",{})
        ids = dataset.get("used_probe_ids",[])
        if len(X)>MAX_FIT_ROWS and ids and sum(samples.get(key,0) for key in ids)==len(X):
            # Preserve each probe's original total weight even when the density
            # of its observations is very different from the others.
            indexes, sampled_weights, offset = [], [], 0
            per_probe = max(1,MAX_FIT_ROWS//len(ids))
            for key in ids:
                count = samples[key]
                selected = offset+np.linspace(0,count-1,min(count,per_probe),dtype=int)
                indexes.extend(selected)
                sampled_weights.extend(np.full(len(selected),sum(weights[offset:offset+count])/len(selected)))
                offset += count
            indexes = np.asarray(indexes,dtype=int)
        else:
            sampled_weights = weights[indexes]
        table = {name:X[indexes,i] for i,name in enumerate(names)}
        targets, weights = targets[indexes], np.asarray(sampled_weights,dtype=float)
        weights = weights/sum(weights)
        diagnostics.update(observed_rows=len(X), numerical_fit_rows=len(indexes),
                           used_probe_ids=list(dataset.get("used_probe_ids",[])),
                           omitted_properties=[name for name in ("mass","charge") if name in names],
                           unused_features=[name for name in names if name not in ("t",*STATE_NAMES)])
        # Reserve part of the same deadline for temporal hypotheses.
        polynomial_stop = min(stop, started+1.5)
        polynomial, ranks, polynomial_complete = _polynomial(table,targets,weights,polynomial_stop)
        for axis in range(2): rows[axis].extend(polynomial[axis])
        diagnostics["polynomial"] = {"status":"completed" if polynomial_complete else "budget_exhausted", "rank_by_degree":ranks}
        periodic, diagnostics["time_modulation"] = _periodic(table,targets,weights,stop)
        for axis in range(2): rows[axis].extend(periodic[axis])
    except _Deadline:
        diagnostics["budget_exhausted"] = True
    diagnostics["elapsed_seconds"] = time.monotonic()-started
    diagnostics.setdefault("budget_exhausted",False)
    return {"axis_rows":[_bounded_rows(axis,max_candidates_per_axis) for axis in rows],"diagnostics":diagnostics}
