"""Pure, auditable SR experiment review using current-session public records only.

No tool execution, formula selection, simulation, archive access or truth inputs.
The caller owns the one-shot flag and decides whether to deliver this advice.
"""
from __future__ import annotations

import math


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _control_key(probe, *, include_properties=True):
    initial = probe.get('initial_conditions', {})
    r, v = initial.get('r0'), initial.get('v0')
    if not all(isinstance(a, (list, tuple)) and len(a) == 2 and all(_finite(x) for x in a)
               for a in (r, v)):
        return None
    key = tuple(round(float(x), 12) for x in (*r, *v))
    if include_properties:
        properties = initial.get('properties', {})
        key += tuple((name, round(float(value), 12)) for name, value in sorted(properties.items())
                     if _finite(value))
    return key


def _position_scale(probe):
    points = probe.get('r_obs', [])
    if not points or not all(isinstance(p, (list, tuple)) and len(p) == 2
                             and all(_finite(x) for x in p) for p in points):
        return None
    first = probe.get('initial_conditions', {}).get('r0', points[0])
    if len(first) != 2 or not all(_finite(x) for x in first):
        return None
    return math.sqrt(sum(sum((p[i]-first[i])**2 for i in range(2)) for p in points)/len(points))


def _noninertial_position_scale(probe):
    """Observed departure from its known constant-velocity initial motion."""
    points, times = probe.get('r_obs', []), probe.get('times', [])
    initial = probe.get('initial_conditions', {})
    r0, v0, t0 = initial.get('r0'), initial.get('v0'), initial.get('t0')
    if (_control_key(probe) is None or not _finite(t0) or len(points) != len(times)
            or not points or not all(_finite(t) for t in times)
            or not all(isinstance(p, (list, tuple)) and len(p) == 2
                       and all(_finite(x) for x in p) for p in points)):
        return None
    return math.sqrt(sum(sum((p[i]-r0[i]-v0[i]*(t-t0))**2 for i in range(2))
                         for p,t in zip(points,times))/len(points))


def _validation_warnings(records, probes, law_id, task):
    result = []
    sigma = task.get('position_noise_sigma', 0.)
    sigma = float(sigma) if _finite(sigma) and sigma >= 0 else 0.
    noise = math.sqrt(2)*sigma
    reference = task.get('reference_length', 1.)
    reference = abs(float(reference)) if _finite(reference) and reference else 1.
    training_error = records.get('law', {}).get(law_id, {}).get('errors', {}).get('overall', {}).get('position_rmse')
    if not _finite(training_error) or training_error < 0:
        training_error = None
    for record_id, validation in records.get('validation', {}).items():
        if validation.get('law_id') != law_id:
            continue
        for row in validation.get('errors', {}).get('per_probe', []):
            probe_id, error = row.get('probe_id'), row.get('position_rmse')
            scale = _position_scale(probes.get(probe_id, {}))
            if scale is None or not _finite(error) or error < 0:
                continue
            # The production DOP853 trajectory evaluator uses rtol=2e-8 and
            # atol=1e-10. Use a conservative, scale-aware numerical floor for
            # every diagnostic ratio; roundoff-sized deflection is not evidence
            # of a scientifically large residual in almost-free motion.
            observed_scale = max(reference, *[abs(float(value))
                for point in probes[probe_id].get('r_obs', []) for value in point])
            floor = 5*math.sqrt(2)*(1e-10+2e-8*observed_scale)
            normalized = error/max(scale, noise, floor)
            departure = _noninertial_position_scale(probes.get(probe_id, {}))
            force_normalized = error/max(departure,noise,floor) if departure is not None else None
            training_ratio = error/max(training_error,noise,floor) if training_error is not None else None
            cues = []
            if normalized >= .2: cues.append('large_relative_to_observed_change')
            if force_normalized is not None and force_normalized >= .2:
                cues.append('large_relative_to_departure_from_uniform_motion')
            if training_ratio is not None and training_ratio >= 5:
                cues.append('validation_at_least_five_times_training_residual')
            # A continuation cue, never a scientific pass/fail criterion.
            if cues and error > max(5*noise, floor):
                result.append({'validation_id': record_id, 'probe_id': probe_id,
                               'position_rmse': error, 'observed_position_rms_change': scale,
                               'relative_to_observed_change': normalized,
                               'observed_departure_from_uniform_motion_rms': departure,
                               'relative_to_departure_from_uniform_motion': force_normalized,
                               'training_position_rmse': training_error,
                               'relative_to_training_residual': training_ratio,
                               'numerical_position_floor': floor,
                               'diagnostic_cues': cues,
                               'interpretation': 'large_relative_residual; diagnostic only, not an official threshold'})
    return result


def build_probe_plan_review(records, budget, task, args):
    """Review a proposed paid batch; caller alone controls SR enablement/one-shot.

    This function never changes arguments, spends energy, or chooses observations
    as training data. Invalid requests are left to the ordinary schema validator.
    """
    if not all(isinstance(value,dict) for value in (records,budget,task,args)):
        return None
    remaining, energy = budget.get('wall_seconds_remaining'), budget.get('energy_remaining')
    proposed, t0, dt, count = (args.get(key) for key in ('probes','t0','dt','n_samples'))
    cost_each = task.get('limits', {}).get('probe_energy',1)
    if (budget.get('session_ended') or not _finite(remaining) or remaining <= 35
            or not _finite(energy) or not _finite(cost_each) or cost_each <= 0
            or not isinstance(proposed,list) or not 1 <= len(proposed) <= 8
            or not _finite(t0) or not _finite(dt) or dt <= 0
            or type(count) is not int or not 1 <= count <= 50):
        return None
    cost = len(proposed)*cost_each
    after = energy-cost
    if after > 2 or after < 0:
        return None
    window = (count-1)*dt
    converted = []
    for item in proposed:
        if (not isinstance(item,dict) or set(item)-{'r0','v0','properties'}
                or not isinstance(item.get('properties'),dict)):
            return None
        probe = {'initial_conditions':dict(item,t0=t0)}
        if _control_key(probe) is None:
            return None
        converted.append(probe)
    acquired = [(key,p) for key,p in records.get('probe',{}).items()
                if isinstance(p,dict) and _control_key(p) is not None
                and _finite(p.get('initial_conditions',{}).get('t0'))]
    all_controls = [p for _,p in acquired]+converted
    time_contrast = any(_control_key(a)==_control_key(b)
                        and abs(a['initial_conditions']['t0']-b['initial_conditions']['t0'])>1e-10
                        for i,a in enumerate(all_controls) for b in all_controls[i+1:])
    duplicate_indexes, duplicate_ids = [], {}
    for index,probe in enumerate(converted):
        same_batch = any(_control_key(p)==_control_key(probe) for p in converted[:index])
        prior = []
        for probe_id,known in acquired:
            if (_control_key(known)!=_control_key(probe)
                    or abs(known['initial_conditions']['t0']-t0)>1e-10):
                continue
            times = known.get('times',[])
            # A longer/newly dense observation is useful, even at the same t0.
            if times and all(any(_finite(t) and abs(t-(t0+j*dt))<1e-9 for t in times)
                             for j in range(count)):
                prior.append(probe_id)
        if same_batch or prior:
            duplicate_indexes.append(index)
            duplicate_ids[str(index)] = prior
    reasons = []
    if not time_contrast: reasons.append('spending_last_training_energy_without_delayed_same_state_control')
    if duplicate_indexes: reasons.append('repeated_complete_controls_with_already_covered_sampling')
    if not reasons:
        return None
    durations = [float(p['times'][-1]-p['times'][0]) for _,p in acquired
                 if isinstance(p.get('times'),list) and len(p['times'])>=2
                 and all(_finite(t) for t in (p['times'][0],p['times'][-1]))
                 and p['times'][-1]>p['times'][0]]
    planning_window = max([window,*durations],default=0.)
    grid = task.get('time_grid')
    horizon = task.get('physical_time_range')
    duration_limit = task.get('maximum_single_trajectory_duration')
    suggestions = []
    constraints = {'time_grid':grid,'physical_time_range':horizon,
                   'maximum_single_trajectory_duration':duration_limit}
    if (_finite(grid) and grid>0 and isinstance(horizon,(list,tuple)) and len(horizon)==2
            and all(_finite(v) for v in horizon) and horizon[1]>=horizon[0] and planning_window>0):
        safe_duration = min(planning_window, float(duration_limit)) if _finite(duration_limit) and duration_limit>0 else planning_window
        safe_duration = math.floor((safe_duration+1e-10)/grid)*grid
        lower = math.ceil((horizon[0]-1e-10)/grid)*grid
        upper = math.floor((horizon[1]-safe_duration+1e-10)/grid)*grid
        anchor = min([t0,*[p['initial_conditions']['t0'] for _,p in acquired]])
        for factor in (3,9):
            value = math.ceil((anchor+factor*planning_window-1e-10)/grid)*grid
            value = min(value,upper)
            if lower<=value<=upper and value>anchor+1e-10 and value not in suggestions:
                suggestions.append(float(round(value,12)))
        constraints['suggested_window_duration']=safe_duration
    reuse = None
    if acquired:
        probe_id, existing = max(acquired,key=lambda item:len(item[1].get('times',[])))
        initial = existing['initial_conditions']
        reuse = {'existing_probe_id':probe_id,
                 'probes':[{'r0':list(initial['r0']),'v0':list(initial['v0']),
                            'properties':dict(initial.get('properties',{}))}],
                 'original_t0':initial['t0'],
                 'suggested_t0_values':[value for value in suggestions if abs(value-initial['t0'])>1e-10],
                 't0_location':'batch_top_level',
                 'change_only':'t0; keep r0, v0 and every property unchanged'}
    positions = [p['initial_conditions']['r0'] for _,p in acquired]
    return {'kind':'symbolic_probe_plan_review','one_time_only':True,'advisory_only':True,
            'tool_executed':False,'reasons':reasons,'budget':dict(budget),
            'proposed_probe_cost':cost,'energy_after_proposed_batch':after,
            'training_energy_capacity_preserving_two':max(0.,energy-2),
            'duplicate_proposed_indexes':duplicate_indexes,'duplicate_existing_probe_ids':duplicate_ids,
            'existing_launch_times':sorted({p['initial_conditions']['t0'] for _,p in acquired}),
            'proposed_t0':t0,'planning_window':planning_window,'suggested_delayed_launch_times':suggestions,
            'recommended_same_state_time_control':reuse,
            'acquired_spatial_coverage':{
                'probe_count':len(positions),
                'off_axis_probe_count':sum(p[0]!=0 and p[1]!=0 for p in positions),
                'coordinate_signs':{name:{'negative':any(p[i]<0 for p in positions),
                                           'positive':any(p[i]>0 for p in positions)}
                                    for i,name in enumerate(('x','y'))}},
            'legal_time_constraints':constraints,'has_same_complete_state_time_contrast_after_batch':time_contrast,
            'instructions':[
                '本批尚未执行，参数未改变。先调整探索设计：至少保留2点能量给两枚新的独立验证探针。',
                '避免把最后探索能量全用于同一t0或已完整覆盖的重复初态；有明确噪声重复测量目的时请说明。',
                '延迟对照保持r0、v0及完整properties相同，只改批次顶层t0；不同t0需分开insert_probe调用。',
                '复用列出的existing_probe完整初态；同时改位置、速度或属性的延迟探针不能单独证明时间依赖。',
                '若列出建议t0，它由当前探针窗口的3倍/9倍动态产生并限制在网格/时间范围；不是预设周期，使用前仍核对dt、样本数和总时长。',
                '密采样局部探针可接近50个样本；把较长轨迹与延迟发射对照分开安排，勿只增加同一短窗密度。',
                '天体长时观测不延长探针时间覆盖；粗采样免费轨迹只作对照，不要为凑数量全部强塞符号回归。',
                '这是一次建议，不自动选择训练/验证记录、不强制预设任何力学形式。']}


def build_review(records, budget, task=None, *, train_probe_ids=None, law_id=None,
                 already_reviewed=False, symbolic_enabled=True):
    """Return one advisory payload, or None when disabled/already sent/too late.

    ``records`` is the existing Store.records mapping. Pass explicit training
    IDs when reviewing a search request; otherwise an existing law's training
    IDs are preferred, or acquired probes are summarized without selecting them.
    Never silently promote unselected or reserved validation probes into search.
    """
    remaining = budget.get('wall_seconds_remaining', 0)
    if (not symbolic_enabled or already_reviewed or budget.get('session_ended')
            or not _finite(remaining) or remaining <= 35):
        return None
    task = task or {}
    probes = records.get('probe', {})
    usable = {key: probe for key, probe in probes.items()
              if isinstance(probe, dict) and len(probe.get('times', [])) >= 3
              and len(probe.get('v_obs', [])) == len(probe.get('times', []))}
    if not usable:
        return None
    law = records.get('law', {}).get(law_id, {})
    requested_ids = list(train_probe_ids if train_probe_ids is not None
                         else law.get('train_probe_ids', usable))
    selected = {key: usable[key] for key in requested_ids if key in usable}
    allowed = [key for key in task.get('allowed_properties', ('mass', 'charge'))
               if key in ('mass', 'charge')]
    property_values, available_contrasts = {}, {}
    for name in allowed:
        values = sorted({float(p['initial_conditions']['properties'][name]) for p in selected.values()
                         if _finite(p.get('initial_conditions', {}).get('properties', {}).get(name))})
        property_values[name] = values
        if len(values) == 1:
            matches = [key for key, p in usable.items() if key not in selected
                       and _finite(p.get('initial_conditions', {}).get('properties', {}).get(name))
                       and p['initial_conditions']['properties'][name] != values[0]]
            if matches:
                available_contrasts[name] = matches
    time_pairs = []
    items = list(selected.items())
    for index, (first_id, first) in enumerate(items):
        key = _control_key(first)
        if key is None:
            continue
        for second_id, second in items[index+1:]:
            if _control_key(second) != key:
                continue
            t1 = first.get('initial_conditions', {}).get('t0')
            t2 = second.get('initial_conditions', {}).get('t0')
            if _finite(t1) and _finite(t2) and abs(t2-t1) > 1e-10:
                time_pairs.append({'probe_ids': [first_id, second_id], 'launch_time_difference': abs(t2-t1)})
    positions = [p.get('initial_conditions', {}).get('r0') for p in selected.values()]
    positions = [p for p in positions if isinstance(p, (list, tuple)) and len(p) == 2
                 and all(_finite(x) for x in p)]
    velocities = {tuple(p['initial_conditions']['v0']) for p in selected.values()
                  if _control_key(p) is not None}
    coordinate_coverage = {axis: {'negative': any(p[index] < 0 for p in positions),
                                  'positive': any(p[index] > 0 for p in positions)}
                           for index, axis in enumerate(('x', 'y'))}
    reasons = []
    if len(usable) >= 3 and not records.get('symbolic_search'):
        reasons.append('symbolic_search_not_attempted_with_available_training_data')
    if available_contrasts:
        reasons.append('unselected_acquired_property_contrasts')
    if any(len(values) < 2 for values in property_values.values()):
        reasons.append('open_properties_not_varied_in_training')
    if not time_pairs:
        reasons.append('no_same_complete_initial_state_delayed_launch_control')
    if positions and (not all(v['negative'] and v['positive'] for v in coordinate_coverage.values())
                      or not any(p[0] != 0 and p[1] != 0 for p in positions)):
        reasons.append('limited_sign_or_off_axis_initial_positions')
    if len(velocities) < 2:
        reasons.append('limited_initial_velocity_controls')
    if law.get('fit_status') in ('not_converged', 'failed', 'error'):
        reasons.append('selected_fit_not_converged')
    validation = _validation_warnings(records, usable, law_id, task) if law_id else []
    if validation:
        reasons.append('selected_law_large_relative_validation_residual')
    searches = list(records.get('symbolic_search', {}).values())
    latest_search = searches[-1] if searches else {}
    if latest_search and not (latest_search.get('candidates')
                             or any(latest_search.get('axis_candidates', {}).values())):
        reasons.append('latest_search_has_no_usable_candidates')
    if not reasons:
        return None
    energy = budget.get('energy_remaining')
    capacity = max(0, int(energy)-2) if _finite(energy) else None
    can_search = remaining >= 100 and len(selected) >= 3
    return {'kind': 'symbolic_experiment_review', 'one_time_only': True, 'advisory_only': True,
            'tool_executed': False, 'law_id': law_id, 'reasons': reasons,
            'budget': dict(budget), 'new_training_probe_capacity_preserving_two_energy': capacity,
            'recommended_search_seconds': 47 if can_search else None,
            'acquired_probe_ids_retained': list(usable), 'reviewed_probe_ids': list(selected),
            'unselected_acquired_property_contrasts': available_contrasts,
            'coverage': {'selected_property_values': property_values,
                         'same_complete_initial_state_time_controls': time_pairs[:4],
                         'same_complete_initial_state_time_control_count': len(time_pairs),
                         'initial_coordinate_signs': coordinate_coverage,
                         'distinct_initial_velocities': len(velocities)},
            'validation_evidence': validation[:3], 'validation_evidence_count': len(validation),
            'instructions': [
                '既有观测、候选和草稿不会因一次搜索失败失效；保留它们，勿原样反复调用。',
                '缺口只是可辨识性提醒，不预设力的形式；仅补最有区分力的对照，不为补齐清单耗尽预算。',
                ('可考虑47秒搜索已有训练数据；新增探针前先核对机动能量，保留两枚新验证能量。'
                 if can_search else '余时或数据不足以建议完整47秒搜索；优先已有候选的必要检验与及时保存草稿。'),
                ('若补实验，仅在上述机动额度内选择最缺的属性、速度或延迟t0对照。'
                 if capacity and remaining >= 70 else '当前不建议新增训练探针；可继续使用已有训练数据和候选。'),
                '未选对照编号仅供明确规划，不自动加入训练；加入搜索后不能再当独立验证。',
                '未收敛或明显验证反证应促使更换结构或补关键对照，不能自行宣布正确；残差阈值不是官方通过标准。',
                '这是一次回顾；如无可行改进，如实说明局限后仍可提交，不无限阻止交卷。']}
