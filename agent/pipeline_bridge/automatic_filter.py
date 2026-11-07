"""Automatic SR preprocessing, sharing the caller's search time budget."""
import json
import math
from pathlib import Path
import time

from .trajectory_filter import validate

VERSION = 'automatic-filter-v1'


def wrap(runtime, legacy_search, reference_search):
    def symbolic_regression(train_probe_ids, time_budget_seconds, preprocessing_id=None):
        # Internal explicit references take precedence; never create another
        # artifact or rescue an invalid explicit reference by automatic filtering.
        if preprocessing_id is not None:
            return reference_search(train_probe_ids, time_budget_seconds, preprocessing_id)
        if isinstance(time_budget_seconds, bool) or not isinstance(time_budget_seconds, (int, float)) or not math.isfinite(time_budget_seconds) or time_budget_seconds <= 0:
            raise ValueError('SR time budget must be finite and positive')
        started = time.monotonic()
        audit = {'policy': VERSION, 'parent_tool': 'symbolic_regression',
                 'trigger': 'missing_preprocessing_id', 'probe_ids': list(train_probe_ids),
                 'requested_time_budget_seconds': time_budget_seconds,
                 'filter_calls': 0, 'filtered_count': 0, 'bypass_count': 0,
                 'preprocessing_id': None, 'worker_consumption': 'not_required'}
        search_started = None
        try:
            total = min(float(time_budget_seconds), 60., runtime.budget()['wall_seconds_remaining']-35.)
            audit['total_budget_cap_seconds'] = max(0., total)
            if total < 1.:
                raise ValueError('Insufficient time for search and subsequent fit/validation/submission')
            from .filter_tool import with_noise
            raw = with_noise(runtime, runtime._probes(train_probe_ids))
            reasons = []
            eligible = False
            for probe in raw:
                t, _, _, sigmas, *_ = validate(probe)
                reason = 'zero_noise' if not sigmas.any() else 'short_trajectory' if len(t) < 12 else None
                reasons.append({'probe_id': probe['probe_id'], 'reason': reason})
                eligible |= reason is None
            audit['per_probe'] = reasons
            key = None
            available = total-(time.monotonic()-started)
            if eligible and available >= 2.:
                audit['filter_calls'] = 1
                preprocessing = runtime.denoise_trajectories(
                    train_probe_ids, time_budget_seconds=min(2., available-1.))
                rows = preprocessing['per_probe']
                audit['per_probe'] = [{k: v for k, v in row.items() if k in ('probe_id', 'status', 'reason', 'version')} for row in rows]
                audit['filtered_count'] = sum(row['status'] == 'filtered' for row in rows)
                audit['bypass_count'] = len(rows)-audit['filtered_count']
                audit['created_preprocessing_id'] = preprocessing['preprocessing_id']
                if audit['filtered_count']:
                    key = preprocessing['preprocessing_id']
                    audit['filter_status'] = 'applied'
                else:
                    audit['filter_status'] = 'all_bypass'
            else:
                audit['filter_status'] = 'not_eligible' if not eligible else 'insufficient_budget'
                audit['bypass_count'] = len(raw)
            audit['preprocessing_id'] = key
            # Include input validation, selection, artifact I/O and hashing in
            # preprocessing time. Original runtime then enforces remaining time.
            audit['preprocessing_seconds'] = time.monotonic()-started
            search_budget = total-audit['preprocessing_seconds']
            audit['search_requested_seconds'] = search_budget
            if search_budget < 1.:
                raise ValueError('Automatic preprocessing exhausted the shared SR time budget')
            if key is not None:
                audit['worker_consumption'] = 'pending'
            search_started = time.monotonic()
            result = (reference_search(train_probe_ids, search_budget, key) if key is not None
                      else legacy_search(train_probe_ids, search_budget))
            if key is not None:
                verified = result.get('preprocessing', {}).get('worker_consumption_verified') is True
                audit['worker_consumption'] = 'verified' if verified else 'unverified'
                if not verified:
                    raise ValueError('Missing automatic filter worker consumption receipt')
            audit['status'] = 'completed'
            return dict(result, automatic_preprocessing=audit)
        except Exception as exc:
            if audit['worker_consumption'] == 'pending':
                audit['worker_consumption'] = 'unverified'
            audit['status'] = 'failed'
            audit['error_type'] = type(exc).__name__
            raise
        finally:
            ended = time.monotonic()
            audit['search_seconds'] = ended-search_started if search_started is not None else 0.
            audit['total_seconds'] = ended-started
            audit['shared_budget_exceeded'] = audit['total_seconds'] > audit.get('total_budget_cap_seconds', 0.)
            audit['timing_scope'] = 'SR adapter validation, preprocessing/artifact I/O, dispatch, search and receipt; excludes final audit log write.'
            with (Path(runtime.run_dir)/'filter-automatic.jsonl').open('a') as stream:
                stream.write(json.dumps(audit, allow_nan=False)+'\n')
    return symbolic_regression
