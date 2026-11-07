"""Opt-in filter tool and search payload integration; vendored source is intact."""
import copy
import json
from pathlib import Path
import time
import uuid

from .trajectory_filter import VERSION, PAYLOAD_KEY, digest, raw_digest, filter_probe


def with_noise(runtime, probes):
    result = copy.deepcopy(probes)
    for probe in result:
        probe['observation_noise'] = {'position_sigma': runtime.task['position_noise_sigma'],
                                      'velocity_sigma': runtime.task['velocity_noise_sigma']}
    return result


def install(runtime, *, enabled=False, policy='explicit'):
    if policy not in ('explicit', 'automatic'):
        raise ValueError('Trajectory filter policy must be explicit or automatic')
    if not enabled:
        return runtime
    if not runtime.symbolic_enabled:
        raise ValueError('Trajectory filtering requires symbolic regression')
    if getattr(runtime, '_filter_tool_installed', False):
        return runtime
    runtime._filter_tool_installed = True
    artifacts = {}
    session = uuid.uuid4().hex
    directory = Path(runtime.run_dir)/'trajectory-filters'
    original_entries = runtime.files.entries
    def entries():
        return dict(original_entries(), **{f'analyses/{k}.json': ('json', v) for k, v in artifacts.items()})
    runtime.files.entries = entries

    def denoise_trajectories(train_probe_ids, method=VERSION, time_budget_seconds=2.):
        if method != VERSION:
            raise ValueError('Unsupported trajectory filter')
        if isinstance(time_budget_seconds, bool) or not 0 < float(time_budget_seconds) <= 5:
            raise ValueError('Filter budget must be in (0,5] seconds')
        started = time.monotonic()
        remaining = runtime.budget()['wall_seconds_remaining']
        allowed = min(float(time_budget_seconds), max(0., remaining-35.))
        deadline = started+allowed
        probes = with_noise(runtime, runtime._probes(train_probe_ids))
        rows = []
        for probe in probes:
            try:
                row = filter_probe(probe, deadline=deadline)
            except (TimeoutError, ArithmeticError, ValueError, RuntimeError) as exc:
                # Bad raw input is never hidden by fallback.
                from .trajectory_filter import validate
                validate(probe)
                row = {'probe_id': probe['probe_id'], 'version': VERSION,
                       'raw_sha256': raw_digest(probe), 'status': 'bypass',
                       'reason': 'budget_exhausted' if isinstance(exc, TimeoutError) else 'numerical_failure',
                       'failure_type': type(exc).__name__}
            rows.append(row)
        key = 'filter_'+uuid.uuid4().hex[:12]
        record = {'preprocessing_id': key, 'session': session, 'version': VERSION,
                  'probe_ids': list(train_probe_ids), 'per_probe': rows,
                  'elapsed_seconds': time.monotonic()-started,
                  'requested_time_budget_seconds': time_budget_seconds,
                  'effective_time_budget_seconds': allowed,
                  'scope': 'Search preprocessing only; all observations used for selection are discovery data.'}
        directory.mkdir(exist_ok=True)
        (directory/(key+'.json')).write_text(json.dumps(record, allow_nan=False))
        artifacts[key] = copy.deepcopy(record)
        runtime.discovery_probe_ids.update(train_probe_ids)
        return {'status': 'completed', 'preprocessing_id': key,
                'elapsed_seconds': time.monotonic()-started,
                'per_probe': [{k: v for k, v in row.items() if k not in ('arrays', 'cv_scores')}
                              for row in rows], 'scope': record['scope']}
    runtime.denoise_trajectories = denoise_trajectories

    original_search = runtime.symbolic_regression
    def symbolic_regression(train_probe_ids, time_budget_seconds, preprocessing_id=None):
        if preprocessing_id is None:
            return original_search(train_probe_ids, time_budget_seconds)
        if preprocessing_id not in artifacts:
            raise ValueError('Unknown or cross-session preprocessing_id')
        record = artifacts[preprocessing_id]
        if record['session'] != session or record['probe_ids'] != train_probe_ids:
            raise ValueError('Filter probe selection does not match this search')
        original_probes = runtime._probes
        raw = with_noise(runtime, original_probes(train_probe_ids))
        if any(raw_digest(probe) != row['raw_sha256'] for probe, row in zip(raw, record['per_probe'])):
            raise ValueError('Stale filter artifact: observations or noise scales changed')
        def search_probes(ids):
            if ids != train_probe_ids:
                raise ValueError('Unexpected filter probe selection')
            result = copy.deepcopy(raw)
            for probe, row in zip(result, record['per_probe']):
                probe[PAYLOAD_KEY] = copy.deepcopy(row)
            return result
        # Original runtime owns search provenance, retries, feedback and time caps.
        # The replacement exists only for this synchronous, single-session call.
        runtime._probes = search_probes
        before = set(runtime.store.records['symbolic_search'])
        try:
            result = original_search(train_probe_ids, time_budget_seconds)
        finally:
            runtime._probes = original_probes
        new = set(runtime.store.records['symbolic_search'])-before
        if len(new) != 1:
            raise ValueError('Missing SR consumption audit')
        search_id = new.pop()
        search = runtime.store.records['symbolic_search'][search_id]
        quality = search.get('data_quality', {}).get('local_derivative_quality', [])
        expected = {r['probe_id']: r for r in record['per_probe'] if r['status'] == 'filtered'}
        actual = {r['probe_id']: r for r in quality if r.get('method') == VERSION}
        consumed = all(k in actual and actual[k].get('output_sha256') == v['output_sha256']
                       and actual[k].get('raw_sha256') == v['raw_sha256'] for k, v in expected.items())
        audit = {'search_id': search_id, 'preprocessing_id': preprocessing_id,
                 'preprocessing_sha256': digest(record), 'worker_consumption_verified': consumed}
        with (Path(runtime.run_dir)/'filter-consumption.jsonl').open('a') as stream:
            stream.write(json.dumps(audit)+'\n')
        if not consumed:
            raise ValueError('Worker did not acknowledge filtered targets; result is not a filtered search')
        return dict(result, preprocessing=audit)
    runtime.symbolic_regression = symbolic_regression
    if policy == 'automatic':
        from .automatic_filter import wrap
        runtime.symbolic_regression = wrap(runtime, original_search, symbolic_regression)
        return runtime
    runtime.schemas = copy.deepcopy(runtime.schemas)
    for schema in runtime.schemas:
        function = schema['function']
        if function['name'] == 'symbolic_regression':
            function['parameters']['properties']['preprocessing_id'] = {'type': 'string'}
            function['description'] += ' Optional preprocessing_id from denoise_trajectories selects search-only filtered states/accelerations. Omit for original preprocessing.'
    runtime.schemas.append({'type': 'function', 'function': {
        'name': 'denoise_trajectories',
        'description': 'Optional observation-only joint position/velocity spline. Returns an artifact ID to explicitly pass to symbolic_regression. Raw fitting and validation remain unchanged. Short/zero-noise trajectories use original preprocessing. Smoothing is not law validation.',
        'parameters': {'type': 'object', 'properties': {
            'train_probe_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 128, 'uniqueItems': True},
            'method': {'type': 'string', 'enum': [VERSION]},
            'time_budget_seconds': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': 5}},
            'required': ['train_probe_ids'], 'additionalProperties': False}}})
    return runtime
