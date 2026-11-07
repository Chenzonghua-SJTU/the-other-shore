"""Scientific tools and current-session files. Reads only acquired records, never simulator internals."""
import ast
import copy
import json
import math
import time
import uuid
from datetime import datetime, timezone
import shutil
from pathlib import Path

import numpy as np
from jsonschema import validate
from .schemas import SCHEMAS, tool_schemas
from .store import Store, preview, append_event
from .observations import estimate_static_bodies, estimate_body_trajectories
from .expressions import Expression
from . import numerics


def correlation(a, b):
    if len(a) < 3:
        return {'value': None, 'reason': 'fewer than three samples'}
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return {'value': None, 'reason': 'constant or numerically constant variable'}
    return {'value': float(np.corrcoef(a, b)[0, 1]), 'reason': None}


def statistics(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {'mean': None, 'rmse': None, 'max_abs': None, 'reason': 'no internal samples'}
    return {'mean': a.mean(axis=0).tolist(), 'rmse': np.sqrt((a*a).mean(axis=0)).tolist(),
            'max_abs': np.abs(a).max(axis=0).tolist()}


class EnvironmentToolError(ValueError):
    """Preserve public environment error codes in the Agent feedback channel."""
    def __init__(self, error):
        error = error if isinstance(error, dict) else {'message': str(error)}
        code = error.get('code')
        self.code = code if isinstance(code, str) and code.isidentifier() else 'environment_error'
        super().__init__(str(error.get('message', error)))


class SymbolicSearchToolError(ValueError):
    """Expose stable public failures without forwarding worker exception text."""
    SAFE_CODES = frozenset({
        'symbolic_service_busy', 'symbolic_service_unavailable', 'symbolic_worker_failed',
        'symbolic_search_failed', 'symbolic_search_timeout', 'symbolic_search_incomplete',
        'invalid_worker_result', 'invalid_service_response', 'invalid_symbolic_endpoint',
        'service_closed', 'invalid_search_budget', 'invalid_search_seed',
        'invalid_search_probes', 'invalid_search_payload',
    })

    @classmethod
    def safe_code(cls, code):
        return code if isinstance(code, str) and code in cls.SAFE_CODES else 'symbolic_service_unavailable'

    def __init__(self, code, search_id):
        self.code = self.safe_code(code)
        super().__init__(f'{self.code}. Search record: analyses/{search_id}.json. '
                         'No usable search result; continue with existing observations and tools.')


class SubmissionRequiredNowError(ValueError):
    code = 'submission_required_now'


class SubmissionFitGateError(ValueError):
    code = 'submission_fit_gate_not_met'

    def __init__(self, details):
        self.details = details
        requirement = ("训练与未参与该规律拟合的数据验证的位置RMSE都严格<0.1（全部已采集轨迹均参与拟合时免验证），或余时严格<60秒"
                       if details.get('requires_validation_for_error_branch') else
                       f"训练位置RMSE严格<{details['fit_rmse_threshold']}，或余时严格<{details['time_remaining_threshold_seconds']}秒")
        super().__init__(f"本次submit未执行、未扣能量、未结束会话。要求{requirement}；"
                         f"当前训练RMSE={details['fit_position_rmse']}，验证RMSE={details.get('validation_position_rmse')}，"
                         f"余时={details['wall_seconds_remaining']}秒。请按证据改进并保存草稿，不要原样重复submit。")



class ToolRuntime:
    schemas = SCHEMAS

    def __init__(self, environment, run_dir, config=None, code_config=None):
        self.environment = environment
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.run_dir / 'records')
        from .workspace import SessionFiles
        self.files = SessionFiles(self.store, self.run_dir)
        self.config = config or {}
        self.code_config = code_config or {}
        self.code_enabled = self.code_config.get('enabled', False)
        if type(self.code_enabled) is not bool:
            raise ValueError('code_execution.enabled must be boolean')
        self.ruleset = getattr(environment, 'ruleset', 'e3')
        self.symbolic_enabled = self.config.get('symbolic_regression_enabled', False)
        if type(self.symbolic_enabled) is not bool:
            raise ValueError('symbolic_regression_enabled must be boolean')
        self.configure_submission_gate(self.config)
        self.schemas = tool_schemas(self.code_enabled, ruleset=self.ruleset, symbolic_enabled=self.symbolic_enabled)
        self.discovery_probe_ids = set()
        self.last_valid_draft = None
        self.exploration_started_monotonic = None
        self.exploration_deadline_monotonic = None
        self.task = None
        self.known_bodies = []
        self.body_position_estimate = None
        self._initial_mission = None
        self.session_ended = False
        self.last_draft_finalization_gate = None

    def configure_submission_gate(self, config):
        """Agent configuration may opt in/out; SR defaults to the requested gate."""
        threshold = config.get('submission_fit_rmse_threshold',
                               getattr(self, 'submission_fit_rmse_threshold', 1e-5 if self.symbolic_enabled else None))
        remaining = config.get('submission_time_remaining_seconds',
                               getattr(self, 'submission_time_remaining_seconds', 40.))
        for name, value in (('submission_fit_rmse_threshold', threshold),
                            ('submission_time_remaining_seconds', remaining)):
            if value is None and name == 'submission_fit_rmse_threshold':
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a positive finite number')
        if self.ruleset=='e4':
            threshold,remaining=.1,60.
        self.submission_fit_rmse_threshold = threshold
        self.submission_time_remaining_seconds = remaining

    def submission_gate_policy(self):
        return {'enabled': self.submission_fit_rmse_threshold is not None,
                'fit_error_field': 'selected_law.errors.overall.position_rmse',
                'fit_rmse_threshold': self.submission_fit_rmse_threshold,
                'time_remaining_threshold_seconds': self.submission_time_remaining_seconds,
                'condition': ('(training_position_rmse < 0.1 AND (all_acquired_probes_used_for_fit OR heldout_validation_position_rmse < 0.1)) OR remaining_seconds < 60' if self.ruleset=='e4' else 'finite_nonnegative_fit_rmse < threshold OR known_remaining_seconds < time_threshold'),
                'requires_validation_for_error_branch':self.ruleset=='e4',
                'validation_waiver': 'all_acquired_probes_used_for_selected_law_fit' if self.ruleset=='e4' else None,
                'validation_rmse_threshold':.1 if self.ruleset=='e4' else None,
                'force_submit_remaining_seconds':40. if self.ruleset=='e4' else None,
                'comparison': 'strict', 'independent_validation_is_hard_requirement': False,
                'save_draft_allowed_before_eligibility': True}

    def submission_eligibility(self, law, budget=None):
        if budget is None:
            try:
                budget = self.budget()
            except Exception:
                budget = {'budget_available': False}
        errors = law.get('errors') or {}
        overall = errors.get('overall') if isinstance(errors, dict) else None
        error = overall.get('position_rmse') if isinstance(overall, dict) else None
        finite_error = type(error) in (int, float) and math.isfinite(error)
        valid_error = finite_error and error >= 0
        remaining = budget.get('wall_seconds_remaining')
        known_remaining = (budget.get('budget_available') is not False
                           and type(remaining) in (int, float) and math.isfinite(remaining))
        time_source = 'environment_budget'
        if not known_remaining and self.exploration_deadline_monotonic is not None:
            remaining = max(0., self.exploration_deadline_monotonic - time.monotonic())
            known_remaining, time_source = True, 'runtime_monotonic_deadline'
        enabled = self.submission_fit_rmse_threshold is not None
        fit_allowed = bool(enabled and valid_error and error < self.submission_fit_rmse_threshold)
        validation_ids=[]; validation_errors=[]
        acquired_ids = set(self.store.records['probe'])
        fit_ids = set(law.get('train_probe_ids') or [])
        available_ids = sorted(acquired_ids - fit_ids)
        all_used_for_fit = bool(acquired_ids) and acquired_ids <= fit_ids
        if self.ruleset=='e4':
            for rid,record in self.store.records['validation'].items():
                if record.get('law_id')!=law.get('law_id'): continue
                value=((record.get('errors') or {}).get('overall') or {}).get('position_rmse')
                if type(value) in (int,float) and math.isfinite(value) and value>=0:
                    validation_ids.append(rid);validation_errors.append(value)
            fit_allowed=fit_allowed and (all_used_for_fit or (bool(validation_errors) and max(validation_errors)<.1))
        validation_error=max(validation_errors) if validation_errors else None
        time_allowed = bool(known_remaining and remaining < self.submission_time_remaining_seconds)
        return dict(self.submission_gate_policy(), law_id=law.get('law_id'),
                    fit_position_rmse=error if finite_error else None, fit_error_valid=valid_error,
                    validation_position_rmse=validation_error,validation_ids=validation_ids,
                    validation_eligible_probe_ids=available_ids,
                    all_acquired_probes_used_for_fit=all_used_for_fit,
                    validation_waived=self.ruleset=='e4' and all_used_for_fit,
                    wall_seconds_remaining=remaining if known_remaining else None,
                    remaining_time_source=time_source if known_remaining else 'unavailable',
                    fit_condition_met=fit_allowed, time_condition_met=time_allowed,
                    allowed=not enabled or fit_allowed or time_allowed)

    def budget(self):
        response = self.environment.request('budget')
        if not response['ok']:
            return {'energy_remaining': None, 'observations_remaining': None,
                    'wall_seconds_remaining': 0, 'session_ended': True, 'budget_available': False}
        return response['budget']

    def mission(self):
        response = self.environment.request('mission')
        if not response['ok']:
            return response
        data = copy.deepcopy(response['data'])
        if self.task is None:
            if data['task'].get('body_observation_protocol') != 'time_series':
                # Preserve v1-v3 observation semantics for explicit legacy runs.
                data['first_light'].pop('body_observation', None)
                data.pop('body_position_estimate', None)
                data['task'].pop('visible_bodies', None)
                data['task']['body_position_source'] = 'Call observe_body(times) to obtain body positions.'
                data['task']['first_light_note'] = (
                    'Two free standard probe trajectories, each with eight absolute times including t0. '
                    'Repeated mission reads return the same data without resetting the session.')
            self.task = copy.deepcopy(data['task'])
            light = data['first_light']
            if self._time_series_bodies():
                light['body_observation'] = self._acquire_body_observation(light['body_observation'])
            for p in light['probes']:
                self.store.add('probe', p, p['probe_id'])
            if self.code_enabled:
                from .code_response import input_hints
                data['code_inputs'] = input_hints(self.store, ['probe', 'body_observation'])
            data['workspace'] = {'scope': 'current_session_only', 'files': sorted(self.files.entries()),
                                 'note': 'Paths are shared by list_files, read_file and run_code; no historical sessions or hidden evaluation.'}
            self._initial_mission = copy.deepcopy(data)
            (self.run_dir / 'mission.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        data = copy.deepcopy(self._initial_mission)
        data['expression_syntax'] = self.expression_syntax()
        if self.submission_gate_policy()['enabled']:
            data['submission_gate'] = self.submission_gate_policy()
        response['data'] = data
        return response

    @staticmethod
    def expression_syntax():
        return ('law_spec={ax:expression,ay:expression}. Both may use t,x,y,vx,vy. '
                'Operators + - * / **; functions sin,cos,exp,log,sqrt,abs. '
                'sum_b(expression) uses bx,by at the current absolute integration time t, from acquired body states. '
                'For time-series missions, initial states and first-light body observations are free; '
                'observe_body(times) adds measured epochs. Linear interpolation requires observations covering the full rollout interval; no extrapolation. '
                'Legacy missions without acquired body positions require observe_body first. '
                "q('name') reads a public allowed scalar property: mass or charge when provided. "
                'No attributes, indexing, imports or code. All coefficients and offsets must be named fitted parameters '
                '(at most 5, parameter_spec list of {name,initial,lower?,upper?}). '
                'Only structural literals 0,1 and integer exponents -3..3; no constant arithmetic or nonzero literal offsets. '
                'Both components together allow at most 256 AST nodes and depth 16. '
                'A syntax/resource rejection does not test the physical hypothesis; simplify an equivalent expression before changing the hypothesis. '
                'For example a velocity-only candidate ax=-d*vx,ay=-d*vy is syntactically legal; '
                'this is a grammar example, not evidence about this universe. Initial conditions and times use mission units.')

    def execute(self, name, args):
        started = time.monotonic()
        try:
            schema = next((s['function']['parameters'] for s in self.schemas if s['function']['name'] == name), None)
            if schema is None:
                raise ValueError('Unknown tool')
            # JSON encoder also rejects nonfinite numbers supplied by a permissive parser.
            json.dumps(args, allow_nan=False)
            validate(args, schema)
            budget = self.budget()
            if self.session_ended or budget['session_ended'] or budget['wall_seconds_remaining'] <= 0:
                raise ValueError('Session ended or wall time exhausted')
            if self.ruleset=='e4' and budget['wall_seconds_remaining']<40 and name!='submit':
                raise SubmissionRequiredNowError('含噪版剩余时间少于40秒：本操作未执行，下一次动作必须submit，使用已拟合law及基于证据的claims；不得继续搜索、拟合或存草稿。')
            if self.task is None:
                raise ValueError('Framework must initialize mission first')
            before_files = set(self.files.entries())
            result = getattr(self, name)(**args)
            added = sorted(set(self.files.entries()) - before_files)
            if added:
                result['workspace_files'] = added
            if self.code_enabled and name in ('observe_body', 'insert_probe'):
                from .code_response import input_hints
                kind = 'body_observation' if name == 'observe_body' else 'probe'
                ids = ([result['observation_id']] if name == 'observe_body'
                       else [p['probe_id'] for p in result['probes']])
                result['code_inputs'] = input_hints(self.store, [kind], ids)
            if self.budget()['wall_seconds_remaining'] <= 0 and not self.session_ended:
                raise ValueError('Session wall time exhausted during tool execution')
            response = {'ok': True, 'data': result, 'error': None, 'budget': self.budget()}
        except Exception as exc:
            response = {'ok': False, 'data': None, 'error': {'code': getattr(exc, 'code', type(exc).__name__),
                        'message': str(exc)[:1500]}, 'budget': self.budget()}
            if isinstance(exc, SubmissionFitGateError):
                response['error'].update(submission_gate=exc.details, temporary_rejection=True,
                                         requires_changed_fit_or_remaining_time=True)
        try:
            json.dumps(args, allow_nan=False)
            logged_args = args
        except (ValueError, TypeError):
            logged_args = {'invalid_arguments': 'not finite JSON; omitted from audit'}
        append_event(self.run_dir / 'tools.jsonl', {'tool': name, 'arguments': logged_args, 'result': response,
                                                  'elapsed_seconds': time.monotonic()-started})
        return response

    def _environment_call(self, name, **kwargs):
        response = self.environment.request(name, **kwargs)
        if not response['ok']:
            raise EnvironmentToolError(response['error'])
        return response['data']

    def observe_body(self, times):
        obs = self._environment_call('observe_body', times=times)
        return preview(self._acquire_body_observation(obs))

    def _time_series_bodies(self):
        return bool(self.task and self.task.get('body_observation_protocol') == 'time_series')

    def _body_inputs(self, law=None):
        # Source states are experiment inputs, not fitted force parameters. Each
        # new validation records its own input snapshot; old law records stay fixed.
        if law is not None and law.get('body_input_protocol') != 'instantaneous_visible_state':
            return law['known_bodies']
        if self._time_series_bodies():
            return copy.deepcopy(self.body_position_estimate)
        return copy.deepcopy(self.known_bodies)

    def _acquire_body_observation(self, observation):
        records = list(self.store.records['body_observation'].values()) + [observation]
        if self._time_series_bodies():
            estimate = estimate_body_trajectories(records, self.task.get('body_initial_states'),
                                                  self.task.get('position_noise_sigma'))
        else:
            if self.task.get('static_visible_bodies') is not True:
                raise ValueError('Static position estimation requires explicitly static bodies')
            estimate = estimate_static_bodies(records, self.task.get('position_noise_sigma'))
        acquired = copy.deepcopy(observation)
        acquired['body_position_estimate'] = copy.deepcopy(estimate)
        self.store.add('body_observation', acquired, observation['observation_id'])
        self.body_position_estimate = estimate
        self.known_bodies = [{'body_id': b['body_id'], 'position': list(b['position'])}
                             for b in estimate['bodies']]
        return acquired

    def insert_probe(self, probes, t0, dt, n_samples):
        data = self._environment_call('insert_probe', probes=probes, t0=t0, dt=dt, n_samples=n_samples)
        trajectories = data['probes'] if isinstance(data, dict) else data
        for probe in trajectories:
            self.store.add('probe', probe, probe['probe_id'])
        return {'probes': [preview(p) for p in trajectories]}

    def _probes(self, ids):
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError('Specify distinct existing probe ids')
        return [self.store.get('probe', rid) for rid in ids]

    def fit_law_and_rollout(self, law_spec, parameter_spec, train_probe_ids):
        probes = self._probes(train_probe_ids)
        if not self.known_bodies:
            expression = Expression(law_spec, parameter_spec)
            if any(isinstance(node, ast.Name) and node.id == 'sum_b'
                   for tree in expression.nodes for node in ast.walk(tree)):
                raise ValueError('No body positions acquired; call observe_body(times) before fitting with sum_b.')
        body_inputs = self._body_inputs()
        result = numerics.fit(law_spec, parameter_spec, probes, body_inputs,
                              max_nfev=self.config.get('max_nfev', 80),
                              timeout_seconds=min(25.0, self.budget()['wall_seconds_remaining']),
                              position_noise_sigma=self.task.get('position_noise_sigma'))
        # Versions are numbered within a session; never reuse an archived ID.
        law_ids = set(self.store.records['law'])
        law_ids.update(p.stem for p in (self.store.directory / 'law').glob('*.json'))
        serials = [int(item[4:]) for item in law_ids
                   if item.startswith('law_') and item[4:].isdigit()]
        rid = f'law_{max([len(law_ids), *serials]) + 1}'
        record = dict(result, law_id=rid, law_spec=law_spec, parameter_spec=parameter_spec,
                      train_probe_ids=train_probe_ids, known_bodies=copy.deepcopy(self.known_bodies),
                      body_position_estimate=copy.deepcopy(self.body_position_estimate))
        if self.symbolic_enabled:
            # Conservative provenance: a later law may be influenced by every
            # structure search already observed, including unsuccessful searches.
            record['discovery_probe_ids'] = sorted(self.discovery_probe_ids)
        if self._time_series_bodies():
            record.update(body_context=body_inputs, body_input_protocol='instantaneous_visible_state')
        if self.symbolic_enabled and record.get('errors'):
            record['acquired_observation_consistency']=self._acquired_consistency(record)
            record['discovery_probe_ids']=sorted(self.discovery_probe_ids)
        record['validation_eligible_probe_ids'] = sorted(set(self.store.records['probe']) - set(train_probe_ids))
        self.store.add('law', record, rid)
        return preview(record)

    def _acquired_consistency(self,law):
        from .symbolic_quality import consistency
        if Expression(law['law_spec'],law['parameter_spec']).uses_bodies:
            return {'status':'not_checked','reason':'Observed-path check currently requires body-free expressions.'}
        probes=list(self.store.records['probe'].values())
        result=consistency(law['law_spec'],law['fitted_parameters'],probes,
                           deadline=time.monotonic()+min(.5,max(0.,self.budget()['wall_seconds_remaining'])))
        # Automatically inspected probes are development data from this point;
        # never relabel the feedback as an independent validation result.
        self.discovery_probe_ids.update(result['checked_probe_ids'])
        result['outside_training_probe_ids']=[rid for rid in result['checked_probe_ids'] if rid not in law['train_probe_ids']]
        return result

    def _law(self, law_id):
        law = self.store.get('law', law_id)
        if not law.get('fitted_parameters') and law.get('parameter_spec'):
            raise ValueError('Law has no fitted parameters')
        if law.get('fit_status') in ('failed', 'error'):
            raise ValueError('Cannot use failed fit')
        return law

    def validate_law(self, law_id, test_probe_ids):
        law = self._law(law_id)
        used = set(law['train_probe_ids'])
        if set(test_probe_ids) & used:
            raise ValueError('Training and validation trajectories overlap')
        probes = self._probes(test_probe_ids)
        discovery_overlap = sorted(set(test_probe_ids) &
                                   (set(law.get('discovery_probe_ids', [])) | self.discovery_probe_ids))
        body_inputs = self._body_inputs(law)
        rollouts = [numerics.rollout(law['law_spec'], law['fitted_parameters'], p, body_inputs,
                                    timeout_seconds=self.budget()['wall_seconds_remaining']) for p in probes]
        rid = 'validation_' + uuid.uuid4().hex[:12]
        data = {'validation_id': rid, 'law_id': law_id, 'test_probe_ids': test_probe_ids,
                'validation_scope': 'parameter_holdout' if discovery_overlap else 'fresh_observations',
                'independent_of_discovery': not bool(discovery_overlap),
                'discovery_overlap_probe_ids': discovery_overlap,
                'errors': numerics.errors_for_rollouts(probes, rollouts), 'rollouts': rollouts,
                'warnings': ['Errors are relative to observations, not hidden truth. No pass threshold is defined.',
                             'Changing a model after seeing these data makes them development data; use fresh experiments for final checking.']}
        if discovery_overlap:
            data['warnings'].append('These trajectories were not used to fit this law, but were used in discovery or diagnostics. This is parameter-holdout validation, not independent evidence of structure selection.')
        if law.get('body_input_protocol') == 'instantaneous_visible_state':
            data.update(body_context=body_inputs, body_input_protocol='instantaneous_visible_state')
            data['warnings'].append('Force formula and parameters are frozen; visible source states use the acquired observation snapshot saved in this validation. Interpolation error is part of the input uncertainty.')
        self.store.add('validation', data, rid)
        return preview(data)

    def symbolic_regression(self, train_probe_ids, time_budget_seconds):
        if not self.symbolic_enabled:
            raise ValueError('symbolic_regression is disabled')
        probes = copy.deepcopy(self._probes(train_probe_ids))
        for probe in probes:
            probe['observation_noise']={'position_sigma':self.task.get('position_noise_sigma'),
                                        'velocity_sigma':self.task.get('velocity_noise_sigma')}
        remaining = self.budget()['wall_seconds_remaining']
        allowed = min(float(time_budget_seconds), 60.0, remaining - 35.0)
        if allowed < 1.0:
            raise ValueError('Insufficient time for search and subsequent fit/validation/submission')
        started = time.monotonic()
        self.discovery_probe_ids.update(train_probe_ids)
        rid = 'sr_' + uuid.uuid4().hex[:12]
        failure_code = None
        try:
            from .symbolic_service import search_remote
            result = search_remote(probes, allowed,
                endpoint=self.config.get('symbolic_regression_endpoint', 'http://127.0.0.1:8890'),
                random_state=self.config.get('symbolic_regression_seed', 42))
        except Exception as exc:
            failure_code = SymbolicSearchToolError.safe_code(getattr(exc, 'code', None))
            result = {'status': 'unavailable', 'candidates': [],
                      'warnings': ['Symbolic search failed or is unavailable; continue with existing tools.']}
        if failure_code is None:
            try:
                if not isinstance(result, dict) or not isinstance(result.get('candidates'), list):
                    raise ValueError('Invalid result shape')
                if result.get('status') in ('error', 'failed', 'unavailable'):
                    # Only a documented code is public; discard backend messages and paths.
                    error = result.get('error')
                    code = error.get('code') if isinstance(error, dict) else None
                    failure_code = SymbolicSearchToolError.safe_code(code)
                    result = {'status': 'unavailable', 'candidates': []}
                elif result.get('status') not in ('completed', 'timeout', 'partial'):
                    raise ValueError('Invalid result status')
                else:
                    for candidate in result['candidates']:
                        Expression(candidate['law_spec'], candidate['parameter_spec'])
                        candidate['discovery_probe_ids'] = list(train_probe_ids)
                    from .symbolic_regression import validate_axis_candidate
                    axis_candidates = result.get('axis_candidates', {})
                    if not isinstance(axis_candidates, dict) or set(axis_candidates) - {'ax', 'ay'}:
                        raise ValueError('Invalid single-axis result')
                    for component, candidates in axis_candidates.items():
                        if not isinstance(candidates, list) or len(candidates) > 3:
                            raise ValueError('Invalid single-axis candidates')
                        for candidate in candidates:
                            validate_axis_candidate(candidate, component)
                            candidate['discovery_probe_ids'] = list(train_probe_ids)
                    if not result['candidates'] and not any(axis_candidates.values()) and result['status'] != 'completed':
                        failure_code = ('symbolic_search_timeout' if result['status'] == 'timeout'
                                        else 'symbolic_search_incomplete')
            except Exception:
                failure_code = 'invalid_service_response'
                result = {'status': 'unavailable', 'candidates': []}
        # The runtime owns provenance, not the worker or the language model.
        result.update(search_id=rid, used_probe_ids=list(train_probe_ids),
                      discovery_probe_ids=sorted(self.discovery_probe_ids),
                      requested_time_budget_seconds=time_budget_seconds,
                      effective_time_budget_seconds=allowed,
                      wall_seconds=time.monotonic() - started,
                      body_positions_included=False)
        result['input_coverage'] = self._symbolic_input_coverage(probes)
        has_candidates = bool(result.get('candidates') or any(result.get('axis_candidates', {}).values()))
        result['outcome'] = ('budget_ended_with_candidates' if result.get('status') == 'timeout' and has_candidates
                             else 'candidates_available' if has_candidates else 'no_candidates')
        result['status_meaning'] = ('A search budget ending with candidates is a usable bounded result, not an over-budget failure. '
                                   'Executable or integrable candidates are not certified accurate physical laws.')
        if failure_code is not None:
            result['error'] = {'code': failure_code,
                               'message': 'Symbolic search produced no usable result; continue with existing tools.'}
        self.store.add('symbolic_search', result, rid)
        if failure_code is not None:
            # Archive provenance before the generic Agent failure/retry path sees it.
            raise SymbolicSearchToolError(failure_code, rid)
        # Full diagnostics are archived above. Preserve executable fields in a
        # compact decision response instead of replaying per-probe audit tables.
        from .symbolic_feedback import search_feedback
        return search_feedback(result)

    def _symbolic_input_coverage(self, probes):
        """Describe acquired controls only; never silently add held-out data to search."""
        selected = {p['probe_id'] for p in probes}
        properties = [p.get('initial_conditions', {}).get('properties', {}) for p in probes]
        property_values, unselected_contrasts = {}, {}
        for key in ('mass', 'charge'):
            values = sorted({float(p[key]) for p in properties if key in p})
            property_values[key] = values
            if len(values) == 1 and all(key in p for p in properties):
                unselected_contrasts[key] = [rid for rid, p in self.store.records['probe'].items()
                    if rid not in selected and key in p.get('initial_conditions', {}).get('properties', {})
                    and float(p['initial_conditions']['properties'][key]) != values[0]]
        times = np.concatenate([np.asarray(p['times'], dtype=float) for p in probes])
        states = np.concatenate([np.asarray(p['r_obs'], dtype=float) for p in probes])
        steps = np.concatenate([np.diff(np.asarray(p['times'], dtype=float)) for p in probes])
        return {
            'selected_probe_count': len(probes),
            'interior_samples': sum(max(0, len(p['times'])-2) for p in probes),
            'absolute_time_range': [float(times.min()), float(times.max())],
            'launch_times': sorted({float(p.get('initial_conditions', {}).get('t0', p['times'][0])) for p in probes}),
            'position_min': states.min(axis=0).tolist(), 'position_max': states.max(axis=0).tolist(),
            'sample_interval_range': [float(steps.min()), float(steps.max())] if steps.size else None,
            'insufficient_sample_probe_ids': [p['probe_id'] for p in probes if len(p['times']) < 3],
            'selected_property_values': property_values,
            'unselected_acquired_property_contrasts': {k: v for k, v in unselected_contrasts.items() if v},
            'meaning': 'Coverage diagnostics, not an identifiability certificate. Constant properties cannot establish independence. '
                       'Unselected contrast IDs have not been added to search; choose training data explicitly and preserve fresh validation probes. '
                       'A long body observation does not extend these probe times. Coarse velocity differences can be biased even without observation noise.'}

    def list_files(self, path='', offset=0, limit=100):
        return self.files.list_files(path, offset, limit)

    def read_file(self, path, offset=0, limit=12000):
        return self.files.read_file(path, offset, limit)

    def analyze_residual(self, law_id, probe_ids):
        law = self._law(law_id)
        body_inputs = self._body_inputs(law)
        data = []
        for p in self._probes(probe_ids):
            pred = numerics.rollout(law['law_spec'], law['fitted_parameters'], p, body_inputs,
                                    timeout_seconds=self.budget()['wall_seconds_remaining'])
            res = {'r': np.asarray(pred['r_pred'])-p['r_obs'], 'v': np.asarray(pred['v_pred'])-p['v_obs'],
                   'a': np.asarray(pred['a_pred'])[1:-1]-np.asarray(p['a_est']).reshape(-1, 2)}
            features = np.column_stack([p['times'], p['r_obs'], p['v_obs']])
            correlations = {}
            for name, values in res.items():
                f = features[1:-1] if name == 'a' else features
                correlations[name] = {axis: {feature: correlation(values[:, j], f[:, k])
                   for k, feature in enumerate(('t', 'x', 'y', 'vx', 'vy'))} for j, axis in enumerate(('x', 'y'))}
            data.append({'probe_id': p['probe_id'], 'used_for_fit': p['probe_id'] in law['train_probe_ids'],
                         'times': p['times'], 'a_est_times': p['a_est_times'],
                         'residuals': {k: v.tolist() for k,v in res.items()},
                         'statistics': {k: statistics(v) for k,v in res.items()}, 'correlations': correlations})
        rid = 'residual_' + uuid.uuid4().hex[:12]
        record = {'residual_id': rid, 'law_id': law_id, 'probe_ids': probe_ids, 'trajectories': data,
                  'description': 'Prediction minus observation; acceleration prediction minus central-difference estimate.',
                  'warnings': ['Correlation is not causation; acceleration estimates have differencing bias.',
                               'Missing acceleration samples and constant-variable correlations are undefined.']}
        if law.get('body_input_protocol') == 'instantaneous_visible_state':
            record.update(body_context=body_inputs, body_input_protocol='instantaneous_visible_state')
        self.store.add('residual', record, rid)
        return preview(record)

    def run_code(self, code, files=None):
        import hashlib
        import base64
        from .code_execution import execute_code
        from .code_response import execution_preview
        if not self.code_enabled:
            raise ValueError('Code execution is disabled')
        loaded, provenance, total = {}, [], 0
        for path in files or []:
            raw = self.files.read_bytes(path)
            total += len(raw)
            if total > 16 * 1024 * 1024:
                raise ValueError('Selected files exceed 16 MiB; select fewer files')
            loaded[path] = base64.b64encode(raw).decode('ascii')
            provenance.append({'path':path, 'sha256':hashlib.sha256(raw).hexdigest(), 'bytes':len(raw)})
        rid = 'code_' + uuid.uuid4().hex[:12]
        parent = self.run_dir / 'code_executions' / rid
        parent.mkdir(parents=True, exist_ok=False)
        (parent / 'analysis.py').write_text(code, encoding='utf-8')
        (parent / 'input_files.json').write_text(json.dumps(loaded, ensure_ascii=False), encoding='utf-8')
        try:
            result = execute_code(code, {}, {}, parent / 'artifacts', self.code_config,
                                  max(0, self.budget()['wall_seconds_remaining'] - 10), files=loaded)
        except (ValueError, OSError) as exc:
            result = {'status':'unavailable', 'result':None, 'error':{'type':type(exc).__name__,
                      'message':str(exc)[:1500]}, 'stdout':'', 'stderr':'', 'artifacts':[]}
        record = dict(result, execution_id=rid, code=code, files=provenance,
                      meaning='Computed from acquired current-session files; not an independent experiment or correctness verdict.')
        self.store.add('code_execution', record, rid)
        return execution_preview(record)

    def _pair_timing(self):
        now = time.monotonic()
        if self.exploration_deadline_monotonic is not None and now >= self.exploration_deadline_monotonic:
            raise ValueError('Candidate checks did not finish before the exploration deadline')
        return {'pair_validated_at': datetime.now(timezone.utc).isoformat(),
                'pair_validated_elapsed_seconds': (now - self.exploration_started_monotonic
                    if self.exploration_started_monotonic is not None else None),
                'submission_deadline_met': True}

    def submit(self, law_id, claims, physical_explanation):
        from .submission import export_submission
        law = self._law(law_id)
        eligibility = self.submission_eligibility(law)
        if not eligibility['allowed']:
            raise SubmissionFitGateError(eligibility)
        rid = 'submission_' + uuid.uuid4().hex[:12]
        timing = {}
        try:
            checks = export_submission(self.run_dir / 'submissions' / rid, law, claims,
                                       self._probes(law['train_probe_ids']), physical_explanation)
            if self.budget()['wall_seconds_remaining'] <= 0:
                raise ValueError('Wall time exhausted during submission checks')
            if self.ruleset in ('e4', 'e5'):
                timing = self._pair_timing()
            ended = self._environment_call('end_session')
            self.session_ended = True
            status, reasons = 'accepted', []
        except Exception as exc:
            checks, status, reasons = {}, 'rejected', [str(exc)[:1000]]
        data = {'submission_id': rid, 'law_id': law_id, 'claims': claims,
                'physical_explanation': physical_explanation, 'status': status,
                'checks': checks, 'reasons': reasons, 'session_ended': self.session_ended,
                'meaning': 'Local acceptance only; no truth or hidden-test check was performed.'}
        if status == 'accepted':
            data.update(timing)
            if eligibility['enabled']:
                data['submission_gate'] = eligibility
        self.store.add('submission', data, rid)
        return data

    def save_draft(self, law_id, claims, physical_explanation):
        """Only a model's explicit, timely, validated pair can become a fallback."""
        if self.ruleset not in ('e4', 'e5'):
            raise ValueError('Draft preservation is available only in E4/E5')
        from .submission import export_submission
        from .agent import _save
        law = self._law(law_id)
        rid = 'draft_' + uuid.uuid4().hex[:12]
        checks = export_submission(self.run_dir / 'drafts' / rid, law, claims,
                                   self._probes(law['train_probe_ids']), physical_explanation)
        if self.symbolic_enabled:
            checks['acquired_observation_consistency']=self._acquired_consistency(law)
        budget = self.budget()
        if budget['wall_seconds_remaining'] <= 0 or budget.get('session_ended'):
            raise ValueError('Draft checks did not finish before the exploration deadline')
        record = {'draft_id': rid, 'law_id': law_id, 'claims': copy.deepcopy(claims),
                  'physical_explanation': physical_explanation, 'checks': checks,
                  'saved_at': datetime.now(timezone.utc).isoformat(), 'status': 'saved',
                  'session_ended': False,
                  'meaning': 'Timely local format/execution checks only; no physical correctness verdict.'}
        record.update(self._pair_timing())
        _save(self.run_dir / 'latest_draft.json', record)
        self.last_valid_draft = copy.deepcopy(record)
        return record

    def finalize_draft(self, reason):
        """Seal previously checked bytes; do not infer claims or run new science after expiry."""
        if self.ruleset not in ('e4', 'e5') or self.last_valid_draft is None:
            return None
        draft = copy.deepcopy(self.last_valid_draft)
        eligibility = self.submission_eligibility(self._law(draft['law_id']))
        self.last_draft_finalization_gate = eligibility
        if not eligibility['allowed']:
            append_event(self.run_dir / 'events.jsonl', {'event': 'draft_finalization_deferred',
                         'reason': reason, 'draft_id': draft['draft_id'], 'submission_gate': eligibility})
            return None
        rid = 'submission_' + uuid.uuid4().hex[:12]
        target = self.run_dir / 'submissions' / rid
        target.parent.mkdir(exist_ok=True)
        shutil.copytree(self.run_dir / 'drafts' / draft['draft_id'], target)
        try:
            budget = self.budget()
            if budget.get('wall_seconds_remaining', 0) > 0 and not budget.get('session_ended'):
                self._environment_call('end_session')
        except Exception:
            pass  # A timely saved pair survives an expired/unavailable console.
        self.session_ended = True
        data = {'submission_id': rid, 'law_id': draft['law_id'], 'claims': draft['claims'],
                'physical_explanation': draft['physical_explanation'], 'checks': draft['checks'],
                'status': 'accepted', 'reasons': [], 'session_ended': True,
                'acceptance_source': 'timely_saved_draft', 'draft_id': draft['draft_id'],
                'draft_saved_at': draft['saved_at'], 'finalized_at': datetime.now(timezone.utc).isoformat(),
                'finalization_reason': reason,
                'meaning': 'Sealed last timely model-selected pair, without new inference or truth access.'}
        data.update({key: draft[key] for key in ('pair_validated_at',
                     'pair_validated_elapsed_seconds', 'submission_deadline_met')})
        if eligibility['enabled']:
            data['submission_gate'] = eligibility
        self.store.add('submission', data, rid)
        return data
