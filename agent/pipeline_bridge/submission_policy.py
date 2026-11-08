"""Noisy-session submission gates, evaluated from the current server budget."""
import math


def noisy(runtime):
    task = getattr(runtime, 'task', None) or {}
    return any(type(task.get(k)) in (int, float) and task[k] > 0
               for k in ('position_noise_sigma', 'velocity_noise_sigma'))


def usable_laws(runtime):
    return [r for r in getattr(getattr(runtime, 'store', None), 'records', {}).get('law', {}).values()
            if r.get('fit_status') not in ('failed', 'error')
            and (r.get('fitted_parameters') or not r.get('parameter_spec'))]


def forced(runtime):
    return noisy(runtime) and runtime.budget().get('wall_seconds_remaining', 0) <= 60


def install(runtime):
    def configure(config):
        runtime.submission_fit_rmse_threshold = .01
        runtime.submission_time_remaining_seconds = 90.
    def policy():
        return {'enabled': noisy(runtime), 'fit_rmse_threshold': .01,
                'validation_rmse_threshold': .01, 'time_remaining_threshold_seconds': 90.,
                'force_submit_remaining_seconds': 60., 'requires_validation_for_error_branch': True,
                'validation_waiver': None, 'comparison': 'rmse_strict_time_inclusive',
                'condition': 'remaining_seconds <= 90 OR (training_position_rmse < 0.01 AND heldout_validation_position_rmse < 0.01)',
                'save_draft_allowed_before_eligibility': True}
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    def eligibility(law, budget=None):
        b = runtime.budget() if budget is None else budget
        remaining = b.get('wall_seconds_remaining')
        train = ((law.get('errors') or {}).get('overall') or {}).get('position_rmse')
        values, ids = [], []
        for rid, record in runtime.store.records['validation'].items():
            if record.get('law_id') != law.get('law_id'): continue
            probes = set(record.get('test_probe_ids') or [])
            if not probes or probes & set(law.get('train_probe_ids') or []): continue
            value = ((record.get('errors') or {}).get('overall') or {}).get('position_rmse')
            ids.append(rid)
            values.append(value)
        valid = bool(values) and all(finite(v) and v < .01 for v in values)
        fit_ok = finite(train) and train < .01 and valid
        time_ok = finite(remaining) and remaining <= 90 and b.get('budget_available') is not False
        return dict(policy(), law_id=law.get('law_id'), fit_position_rmse=train,
                    validation_position_rmse=max(values) if values and all(finite(v) for v in values) else None,
                    validation_ids=ids, validation_waived=False,
                    wall_seconds_remaining=remaining, fit_condition_met=fit_ok,
                    time_condition_met=time_ok, allowed=not noisy(runtime) or fit_ok or time_ok)
    runtime.configure_submission_gate = configure
    runtime.submission_gate_policy = policy
    runtime.submission_eligibility = eligibility
    if hasattr(runtime, 'submit'):
        original_submit = runtime.submit
        def submit(law_id, claims, physical_explanation):
            detail=eligibility(runtime._law(law_id))
            if not detail['allowed']:
                class GateError(ValueError):
                    code='submission_fit_gate_not_met'
                raise GateError('Noisy submission requires remaining seconds <=90 OR both selected-law training and held-out validation position RMSE strictly <0.01; missing validation is not waived.')
            return original_submit(law_id,claims,physical_explanation)
        runtime.submit=submit
    configure({})
