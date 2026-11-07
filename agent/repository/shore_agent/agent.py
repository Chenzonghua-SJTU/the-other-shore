"""Transparent main research loop. Files are restricted to acquired current-session data."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

from .model import ModelError
from .store import append_event


def _save(path, value):
    temp = Path(path).with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


class Agent:
    def __init__(self, client, tool_runtime, run_dir: Path, config: dict):
        self.client = client
        self.runtime = tool_runtime
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        configure_gate = getattr(tool_runtime, 'configure_submission_gate', None)
        if callable(configure_gate):
            configure_gate(config)
        self.messages = []
        self.last_budget = {}
        self.submission = None
        self.session_ended = False
        self.ruleset = getattr(tool_runtime, 'ruleset', 'e3')
        self.guidance_profile = config.get('guidance_profile', 'baseline')
        if self.guidance_profile not in ('baseline', 'e3', 'concise'):
            raise ValueError('guidance_profile must be baseline, e3 or concise')
        self.e4 = self.ruleset in ('e4', 'e5')
        self.finish_reserve_seconds = 5.0 if self.ruleset=='e4' else 30.0 if self.e4 else 0.0
        self.max_no_action = config.get('max_no_action', 3)
        if isinstance(self.max_no_action, bool) or not isinstance(self.max_no_action, int) or self.max_no_action < 1:
            raise ValueError('max_no_action must be a positive integer')
        self.loop_recovery = config.get('loop_recovery', False)
        if type(self.loop_recovery) is not bool:
            raise ValueError('loop_recovery must be boolean')
        self._repeat_states = {}
        self._last_observation_version = None
        self._fallback_observation_version = 0
        self._length_recovered = False
        self._symbolic_submission_reviewed = False
        self._symbolic_plan_reviewed = False
        self._symbolic_probe_plan_reviewed = False
        self._symbolic_fit_attempts = []

    @staticmethod
    def _symbolic_law_signature(value):
        """Exact executable structure and named input parameters, not fit quality."""
        try:
            law, parameters = value['law_spec'], value['parameter_spec']
            if set(law) != {'ax', 'ay'} or not isinstance(parameters, list) or len(parameters) > 5:
                return None
            trees = tuple(ast.dump(ast.parse(law[axis].strip(), mode='eval'), include_attributes=False)
                          for axis in ('ax', 'ay'))
            return trees, sorted(parameters, key=lambda parameter: parameter['name'])
        except (AttributeError, KeyError, TypeError, ValueError, SyntaxError):
            return None

    def _untested_symbolic_joint_candidates(self, records):
        candidates = []
        for search_id, search in records.get('symbolic_search', {}).items():
            for index, candidate in enumerate(search.get('candidates', [])):
                if (not isinstance(candidate, dict) or candidate.get('joint_fit_ready') is False
                        or candidate.get('dsl_valid') is False
                        or (candidate.get('initial_domain_check') or {}).get('status') == 'failed'):
                    continue
                signature = self._symbolic_law_signature(candidate)
                if signature is not None:
                    candidates.append((search_id, index, signature))
        tested = [self._symbolic_law_signature(law) for law in records.get('law', {}).values()]
        tested.extend(self._symbolic_fit_attempts)
        if not candidates or any(signature in tested for _, _, signature in candidates):
            return None
        return {'candidate_count': len(candidates),
                'candidate_references': [{'search_id': sid, 'candidate_index': index}
                                         for sid, index, _ in candidates[:3]],
                'meaning': 'No complete returned candidate has a direct fit record or executed fit attempt. '
                           'Untested does not mean correct or incorrect; compare a promising original before rewriting, '
                           'or explain concrete reasons to reject it. No candidate is selected automatically.'}

    def _symbolic_probe_plan_review(self, name, args):
        if (name != 'insert_probe' or self._symbolic_probe_plan_reviewed
                or not getattr(self.runtime, 'symbolic_enabled', False)):
            return None
        from .symbolic_guidance import build_probe_plan_review
        records = getattr(getattr(self.runtime, 'store', None), 'records', {})
        budget = self.runtime.budget()
        review = build_probe_plan_review(records, budget, getattr(self.runtime, 'task', None) or {}, args)
        if review is None:
            return None
        self._symbolic_probe_plan_reviewed = True
        self.last_budget = budget
        self._event('symbolic_probe_plan_review', review=review, tool_executed=False, one_time_only=True)
        result = self._failure('scientific_review_needed')
        result['error']['message'] = (
            '本次insert_probe暂未执行、未扣能量。拟议批次将消耗最后机动能量；请先检查下方具体覆盖缺口、'
            '延迟t0建议与两枚独立验证余量。可修改本批参数、先使用既有数据，或说明为何原计划更合适。'
            '这是一次提醒；下一次insert_probe按你给出的原参数执行，不自动补实验或修改观测。')
        result['error']['research_review'] = review
        return result

    def _symbolic_submission_review(self, name, args):
        """One continuation opportunity, never an automatic hypothesis or search."""
        if (name != 'submit' or self._symbolic_submission_reviewed
                or not getattr(self.runtime, 'symbolic_enabled', False)):
            return None
        records = getattr(getattr(self.runtime, 'store', None), 'records', {})
        budget = self.runtime.budget()
        if budget.get('session_ended') or budget.get('wall_seconds_remaining', 0) < 100:
            return None
        gate_reader = getattr(self.runtime, 'submission_gate_policy', None)
        eligibility = getattr(self.runtime, 'submission_eligibility', None)
        if callable(gate_reader) and callable(eligibility) and gate_reader().get('enabled'):
            law = records.get('law', {}).get(args.get('law_id'), {})
            if eligibility(law, budget).get('allowed'):
                # The requested OR is sufficient: scientific diagnostics must
                # not become a third hard requirement once it is satisfied.
                return None
        from .symbolic_guidance import build_review
        review = build_review(records, budget, getattr(self.runtime, 'task', None), law_id=args.get('law_id'))
        untested = self._untested_symbolic_joint_candidates(records)
        if untested is not None:
            review = dict(review or {'kind': 'symbolic_experiment_review', 'law_id': args.get('law_id'),
                                     'budget': budget, 'one_time_only': True, 'advisory_only': True})
            review['reasons'] = list(review.get('reasons', [])) + ['returned_joint_candidates_not_directly_fitted']
            review['untested_joint_candidates'] = untested
        critical = {'symbolic_search_not_attempted_with_available_training_data',
                    'selected_fit_not_converged', 'selected_law_large_relative_validation_residual',
                    'returned_joint_candidates_not_directly_fitted'}
        reasons = [r for r in (review or {}).get('reasons', []) if r in critical]
        if not reasons:
            return None
        self._symbolic_submission_reviewed = True
        self.last_budget = budget
        self._event('symbolic_submission_review', reasons=reasons, law_id=args.get('law_id'), budget=budget,
                    tool_executed=False, one_time_only=True, evidence=review)
        result = self._failure('scientific_review_needed')
        result['error']['message'] = (
            '尚有至少100秒，本次submit暂未执行，给一次继续研究机会：' + ', '.join(reasons) + '。'
            '请检查拟合与独立验证反证；有多条轨迹时可选择覆盖不同控制量的训练集调用符号回归，'
            '保留独立验证数据。不要无证据假设中心力、各向同性或不同轴共享系数。'
            '若尚未直接测试任何完整SR候选，可先按原law_spec与parameter_spec拟合一个有希望的候选再比较；'
            '未测试不代表候选正确或错误，有具体反证时不必强行采用。'
            '可以先save_draft保留当前最佳候选。此提醒只出现一次；如已有充分理由不再探索，'
            '可如实说明局限后再次submit，系统不会无限阻止交卷。')
        result['error']['research_review'] = review
        return result

    def _symbolic_plan_review(self, name, result):
        if (name != 'insert_probe' or not result.get('ok') or self._symbolic_plan_reviewed
                or not getattr(self.runtime, 'symbolic_enabled', False)):
            return None
        records = getattr(getattr(self.runtime, 'store', None), 'records', {})
        if records.get('symbolic_search') or len(records.get('probe', {})) < 4:
            return None
        from .symbolic_guidance import build_review
        review = build_review(records, self.last_budget, getattr(self.runtime, 'task', None))
        if review is None or self.last_budget.get('wall_seconds_remaining', 0) < 100:
            return None
        self._symbolic_plan_reviewed = True
        self._event('symbolic_experiment_review', review=review)
        return json.dumps(review, ensure_ascii=False, allow_nan=False)

    _DETERMINISTIC_TOOLS = frozenset({'read_file', 'list_files', 'fit_law_and_rollout',
                                    'validate_law', 'analyze_residual', 'observe_body'})

    def _observation_version(self):
        records = getattr(getattr(self.runtime, 'store', None), 'records', None)
        if isinstance(records, dict):
            return (len(records.get('probe', {})), len(records.get('body_observation', {})))
        return self._fallback_observation_version

    def _repeat_key(self, name, args, raw_args):
        version = self._observation_version()
        if version != self._last_observation_version:
            self._repeat_states.clear()
            self._last_observation_version = version
        def normalized(value):
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, list):
                return [normalized(v) for v in value]
            if isinstance(value, dict):
                return {k: normalized(v) for k, v in value.items()}
            return value
        canonical = (json.dumps(normalized(args), sort_keys=True, separators=(',', ':'), allow_nan=False)
                     if args is not None else 'invalid_json:' + raw_args.strip())
        # File contents/listings can change after a new law, validation or code
        # artifact without acquiring another physical observation. Bind ONLY
        # this file call to its public view; do not reset fit/validate counters.
        file_view = None
        if name in ('list_files', 'read_file') and args is not None:
            reader = getattr(getattr(self.runtime, 'files', None), name, None)
            if callable(reader):
                try:
                    view = reader(**args)
                except Exception as exc:
                    view = {'error_type': type(exc).__name__, 'error': str(exc)[:500]}
                file_view = hashlib.sha256(json.dumps(view, sort_keys=True, ensure_ascii=False,
                                                      allow_nan=False).encode('utf-8')).hexdigest()
        if name == 'submit' and args is not None:
            gate_reader = getattr(self.runtime, 'submission_gate_policy', None)
            eligibility = getattr(self.runtime, 'submission_eligibility', None)
            if callable(gate_reader) and callable(eligibility) and gate_reader().get('enabled'):
                records = getattr(getattr(self.runtime, 'store', None), 'records', {})
                law = records.get('law', {}).get(args.get('law_id'), {})
                gate = eligibility(law)
                # Time passing within one side of the threshold is not new
                # evidence. Crossing it must allow a previously rejected submit.
                file_view = ('submission_gate', gate.get('allowed'), gate.get('time_condition_met'),
                             gate.get('fit_position_rmse'), gate.get('fit_error_valid'))
        return name, canonical, file_view

    def _repeat_block_reason(self, key):
        state = self._repeat_states.get(key, {})
        if state.get('same_failures', 0) >= 2:
            return 'repeated_tool_failure_limit'
        if key[0] in self._DETERMINISTIC_TOOLS and state.get('attempts', 0) >= 2:
            return 'repeated_deterministic_call_limit'
        return None

    def _record_repeat(self, key, result):
        name = key[0]
        if result.get('ok') and name in ('insert_probe', 'observe_body'):
            records = getattr(getattr(self.runtime, 'store', None), 'records', None)
            if not isinstance(records, dict):
                self._fallback_observation_version += 1
            # Actual acquired-observation counts reset on the next call. A new
            # law/validation ID never advances this version.
            if name == 'insert_probe' or self._observation_version() != self._last_observation_version:
                self._repeat_states.pop(key, None)
                return None
            # Re-reading a cached body observation is not a fresh measurement.
        state = self._repeat_states.setdefault(key, {'attempts': 0, 'same_failures': 0})
        state['attempts'] += 1
        error = result.get('error') or {}
        # Stable error codes identify the same failure even when a backend adds
        # a request ID or changing time value to its human-readable message.
        code = error.get('code')
        fingerprint = ('code:' + code if isinstance(code, str) and code else
                       json.dumps(error, sort_keys=True, ensure_ascii=False, allow_nan=False))
        if result.get('ok'):
            state['same_failures'] = 0
            state.pop('error_fingerprint', None)
        else:
            state['same_failures'] = state['same_failures'] + 1 if state.get('error_fingerprint') == fingerprint else 1
            state['error_fingerprint'] = fingerprint
        if result.get('ok') and (name not in self._DETERMINISTIC_TOOLS or state['attempts'] < 2):
            return None
        notice = {'tool': name, 'identical_attempts_without_new_observations': state['attempts'],
                  'same_error_failures': state['same_failures'], 'budget': copy.deepcopy(self.last_budget)}
        if name in ('list_files', 'read_file'):
            notice['relevant_file_view_unchanged'] = True
        if not result.get('ok'):
            notice['last_error'] = {'code': error.get('code'), 'message': str(error.get('message', ''))[:500]}
        notice['instruction'] = (
            '本次操作没有取得新的探针或天体观测。请修改参数、改用其他必要操作，或保存/提交已有候选；'
            '不要把新的law_id或validation_id当作新证据。首次失败最多允许一次原样重试；'
            '同错误失败两次或确定性调用执行两次后，再原样调用会被阻止，并给一次改变操作的机会；'
            '下一轮仍原样重复才结束本轮探索，保留已有有效草稿。')
        if code == 'submission_fit_gate_not_met':
            notice['temporary_submission_gate_rejection'] = True
            notice['instruction'] = (
                '当前提交条件尚未达到；不要把临时拒绝当作物理反证，也不要原样反复提交。'
                '可用已有数据继续拟合或比较不同候选，并save_draft保留选择。'
                '仅所选训练位置RMSE严格低于阈值、或剩余时间严格低于门槛时才允许结束。'
                '同样的拒绝仍受有界重复保护；早停且未达门槛的草稿保留为incomplete，不会自动绕过提交。')
        self._event('loop_recovery_feedback', **notice)
        return json.dumps(notice, ensure_ascii=False, allow_nan=False)

    def _persist(self):
        _save(self.run_dir / 'messages.json', self.messages)

    def _event(self, kind, **data):
        append_event(self.run_dir / 'events.jsonl', {'event': kind, **data})

    def _failure(self, code):
        return {'ok': False, 'data': None, 'error': {'code': code, 'message': code},
                'budget': self.last_budget}

    def _finish(self, status, reason):
        if self.e4 and self.submission is None and status != 'submitted':
            draft = self.runtime.finalize_draft(reason)
            if draft is not None:
                self.submission = draft
                self.session_ended = True
                status = 'submitted'
                reason = ('deadline_saved_draft' if 'wall' in reason or 'deadline' in reason
                          else 'saved_draft_after_' + reason)
        budget_reader = getattr(self.runtime, 'budget', None)
        if callable(budget_reader):
            try:
                current_budget = budget_reader()
                if isinstance(current_budget, dict):
                    self.last_budget = current_budget
                else:
                    self._event('budget_refresh_failed', reason='invalid_budget_response')
            except Exception:
                # Preserve the last known budget without leaking backend details.
                self._event('budget_refresh_failed', reason='budget_unavailable')
        result = {'status': status, 'reason': reason, 'submission': self.submission,
                  'model_usage': copy.deepcopy(self.client.usage), 'budget': self.last_budget}
        finalization_gate = getattr(self.runtime, 'last_draft_finalization_gate', None)
        if finalization_gate is not None and not finalization_gate.get('allowed'):
            result['draft_finalization_blocked_by_submission_gate'] = finalization_gate
        self._persist()
        _save(self.run_dir / 'result.json', result)
        self._event('session_finished', **result)
        return result

    @staticmethod
    def _valid_calls(calls):
        if not isinstance(calls, list) or len(calls) > 32:
            return False
        ids = set()
        for call in calls:
            if (not isinstance(call, dict) or call.get('type') != 'function'
                    or not isinstance(call.get('id'), str) or not call['id']
                    or call['id'] in ids or not isinstance(call.get('function'), dict)
                    or not isinstance(call['function'].get('name'), str)
                    or not isinstance(call['function'].get('arguments'), str)):
                return False
            ids.add(call['id'])
        return True

    def run(self):
        mission = self.runtime.mission()
        self.last_budget = mission.get('budget') or {}
        prompt = (Path(__file__).parent / 'prompts' / 'main.txt').read_text(encoding='utf-8')
        if self.ruleset == 'e4':
            prompt += ('\n\nE4时间约束：首次mission开始后总探索预算300秒，包含模型与科学工具。'
                       '调用save_draft明确保存你选择的law_id和你根据观测判断的三个claims；'
                       '保存成功后可继续改进，最后及时有效配对会在截止或提前结束时保留。'
                       '未明确给出claims的候选不会自动成为草稿。剩余时间少时优先save_draft或submit，'
                       '不要等待长篇说明。系统在最后30秒优先封存已保存配对；没有草稿时仍由你决定提交，'
                       '截止时没有有效配对会如实记为未完成。')
        elif self.ruleset == 'e5' and self.guidance_profile != 'e3':
            filename = 'e5_concise.txt' if self.guidance_profile == 'concise' else 'e5.txt'
            prompt += '\n\n' + (Path(__file__).parent / 'prompts' / filename).read_text(encoding='utf-8')
        if getattr(self.runtime, 'symbolic_enabled', False):
            prompt += '\n\n' + (Path(__file__).parent / 'prompts' / 'symbolic.txt').read_text(encoding='utf-8')
            if self.ruleset == 'e4':
                prompt += ('\n\n本场开启观测噪声和符号回归：输入仍是当前会话取得的含噪位置、速度，工具没有真值；仅依据公开噪声尺度做局部多项式平滑。'
                           '速度差分噪声随dt减小被放大，不要一味缩小采样间隔；兼顾运动变化、噪声尺度和时间覆盖。'
                           '搜索损失与观测轨迹RMSE均不是隐藏真值误差；比较简单候选与独立验证，不追逐噪声生成复杂项。'
                           '含噪提交条件是训练与未参与当前law拟合的数据验证位置RMSE都<0.1；全部已采集轨迹均参与当前law拟合时免验证但训练仍须<0.1；或余时<60秒；余时<40秒下一次动作必须submit。提前save_draft保留最佳候选，'
                           '在剩余预算内做有信息的检验，不要机械重复搜索或空等。')
        gate_reader = getattr(self.runtime, 'submission_gate_policy', None)
        gate = gate_reader() if callable(gate_reader) else {}
        if gate.get('enabled'):
            if gate.get('requires_validation_for_error_branch'):
                prompt += ('\n\n本场含噪门槛覆盖通用说明：训练与同一law未参与拟合的数据验证的位置RMSE都严格<0.1；全部已采集轨迹均参与该law拟合时免验证但训练仍须<0.1，'
                           '或余时严格<60秒即可submit；余时严格<40秒下一次动作必须submit。'
                           '务必提前保存你选择的law/claims/解释，届时可按该草稿直接提交。')
            else:
                prompt += (f"\n\n提交硬门槛：所选law的errors.overall.position_rmse有限非负且严格小于"
                           f"{gate['fit_rmse_threshold']}，或余时严格少于{gate['time_remaining_threshold_seconds']}秒。"
                           '独立验证仍提供诊断；save_draft不能绕过门槛。不要原样重复submit或空等。')
        if getattr(self.runtime, 'code_enabled', False):
            prompt = prompt.replace('不可访问任意文件、Shell、网络、源码或任意 Python。',
                                    '代码只可通过run_code在隔离环境执行，不可访问宿主文件、网络或仿真源码。')
            prompt += '\n\n' + (Path(__file__).parent / 'prompts' / 'code.txt').read_text(encoding='utf-8')
        self.messages = [{'role': 'system', 'content': prompt},
                         {'role': 'user', 'content': '当前任务与合法初始观测：\n' +
                          json.dumps(mission, ensure_ascii=False, allow_nan=False)}]
        self._persist()
        if not mission.get('ok'):
            return self._finish('error', 'mission_unavailable')
        if self.last_budget.get('session_ended') is True:
            return self._finish('incomplete', 'session_already_ended')
        no_action = 0
        # A turn cap would impose another model-call cap. Continue until submit,
        # a resource/error termination, or repeated replies without tool actions.
        turn = 0
        while True:
            if self.e4:
                remaining = self.runtime.budget()['wall_seconds_remaining']
                if remaining <= self.finish_reserve_seconds:
                    if self.runtime.last_valid_draft is not None and self.ruleset!='e4':
                        return self._finish('incomplete', 'deadline_reserve')
                    if remaining <= 0:
                        return self._finish('incomplete', 'wall_time_exhausted_without_saved_draft')
            forced_submit=self.ruleset=='e4' and self.runtime.budget()['wall_seconds_remaining']<40
            if forced_submit and self.runtime.last_valid_draft is not None:
                draft=self.runtime.last_valid_draft
                args={k:draft[k] for k in ('law_id','claims','physical_explanation')}
                result=self.runtime.execute('submit',args)
                self._event('mandatory_submit_from_model_selected_draft',arguments=args,result=result)
                if result.get('ok') and result.get('data',{}).get('status')=='accepted':
                    self.submission=result['data'];self.session_ended=True
                    return self._finish('submitted','mandatory_noisy_deadline_submit')
            tools=self.runtime.schemas
            if forced_submit:
                tools=[s for s in tools if s['function']['name']=='submit']
                self.messages.append({'role':'user','content':'含噪版余时少于40秒，下一动作必须submit，提交你基于现有证据选择的已拟合law；禁止其他动作。'})
            turn += 1
            try:
                message = self.client.complete(self.messages, tools=tools, role='main')
            except ModelError as exc:
                return self._finish('incomplete', exc.code)
            self.messages.append(copy.deepcopy(message))
            self._persist()
            content = message.get('content')
            if content:
                print(f'研究进展：{content}', flush=True)
                self._event('research_progress', turn=turn, content=content)
            finish_reason = self.client.last_finish_reason
            # A length-limited reply may end halfway through JSON: never execute it.
            if finish_reason not in ('stop', 'tool_calls'):
                if self.loop_recovery and finish_reason == 'length' and not self._length_recovered:
                    self._length_recovered = True
                    self._event('model_length_recovery', turn=turn, discarded_message=copy.deepcopy(message))
                    # Drop incomplete calls from the model conversation, keeping
                    # their original response in the audit event above.
                    self.messages[-1] = {'role': 'assistant', 'content': content or '上次回复达到输出上限，工具未执行。'}
                    self.messages.append({'role': 'user', 'content':
                        '上一轮达到单次4096 token上限，所有工具调用均未执行。仅允许这一次恢复。'
                        '请缩短说明，使用完整、简短的工具参数继续；不要续写半截JSON。'
                        '现有观测和候选仍有效，可以直接执行必要操作或保存/提交已有候选。'})
                    self._persist()
                    continue
                if self.loop_recovery and finish_reason == 'length' and self._length_recovered:
                    return self._finish('incomplete', 'model_finish_length_recovery_exhausted')
                return self._finish('incomplete', f'model_finish_{finish_reason}')
            calls = message.get('tool_calls') or []
            if not self._valid_calls(calls):
                return self._finish('error', 'malformed_tool_call_protocol')
            if not calls:
                no_action += 1
                if no_action >= self.max_no_action:
                    return self._finish('incomplete', 'consecutive_no_action_limit')
                self.messages.append({'role': 'user', 'content':
                    '尚未收到成功提交。请根据现有证据决定下一步并执行必要工具；'
                    '自然语言结束不会提交。资源不足或证据不足时请如实说明，不要编造结果。'})
                self._persist()
                continue
            no_action = 0
            loop_stop_reason = None
            recovery_feedback = []
            history_message = self.messages[-1]
            for call in calls:
                name = call['function']['name']
                print(f'工具调用：{name}', flush=True)
                key = None
                executed_or_invalid = False
                if loop_stop_reason is not None:
                    result = self._failure('loop_recovery_terminated')
                elif self.session_ended:
                    result = self._failure('session_ended')
                elif self.client.remaining_seconds() <= 0:
                    self.session_ended = True
                    result = self._failure('wall_time_exhausted')
                else:
                    raw_arguments = call['function']['arguments']
                    try:
                        args = json.loads(raw_arguments)
                        if not isinstance(args, dict):
                            raise ValueError()
                        # NaN/Infinity are accepted by Python's decoder but not our protocol.
                        json.dumps(args, allow_nan=False)
                    except (TypeError, ValueError):
                        args = None
                        result = self._failure('invalid_tool_arguments_json')
                        executed_or_invalid = True
                        if self.loop_recovery:
                            # Some tool parsers mark a truncated call as
                            # tool_calls rather than length. Qwen's next chat
                            # template parses historical argument strings as
                            # JSON, so the raw invalid string would cause HTTP
                            # 400 before the model can read our failure reply.
                            # Audit the original, do not execute it, and repair
                            # only its transport representation in history.
                            self._event('invalid_tool_arguments_history_repaired',
                                tool_call_id=call['id'], name=name,
                                original_arguments=raw_arguments, replacement_arguments='{}',
                                tool_executed=False)
                            for historical_call in history_message['tool_calls']:
                                if historical_call['id'] == call['id']:
                                    historical_call['function']['arguments'] = '{}'
                            result['error']['message'] = (
                                'Original tool arguments were invalid or incomplete JSON and were not executed. '
                                'The historical arguments were replaced with {} for transport only. '
                                'Send complete, concise JSON arguments to retry.')
                    if self.loop_recovery:
                        key = self._repeat_key(name, args, raw_arguments)
                        block_reason = self._repeat_block_reason(key)
                    else:
                        block_reason = None
                    if block_reason is not None:
                        state = self._repeat_states[key]
                        new_blocked_turn = state.get('blocked_turn') != turn
                        if new_blocked_turn:
                            state['blocked_turn'] = turn
                            state['blocked_turns'] = state.get('blocked_turns', 0) + 1
                        if state['blocked_turns'] >= 2:
                            loop_stop_reason = block_reason
                        try:
                            current_budget = self.runtime.budget()
                            if isinstance(current_budget, dict):
                                self.last_budget = current_budget
                        except Exception:
                            self._event('budget_refresh_failed', reason='loop_recovery_budget_unavailable')
                        result = self._failure(block_reason)
                        executed_or_invalid = False
                        notice = {'tool': name, 'identical_attempts_without_new_observations': state['attempts'],
                                  'blocked_model_turns': state['blocked_turns'], 'budget': copy.deepcopy(self.last_budget),
                                  'instruction': '本次重复调用未执行。请现在修改参数、改用其他必要操作，或保存/提交已有候选。'
                                  '这是一次恢复机会；下一轮仍原样重复会结束探索并保留已有有效草稿。'}
                        self._event('loop_recovery_blocked', name=name, reason=block_reason,
                                    attempts=state['attempts'], execution_skipped=True,
                                    continuation_allowed=loop_stop_reason is None, notice=notice)
                        if new_blocked_turn and loop_stop_reason is None:
                            recovery_feedback.append(json.dumps(notice, ensure_ascii=False, allow_nan=False))
                    elif args is not None:
                        executed_or_invalid = True
                        self._event('tool_started', tool_call_id=call['id'], name=name, arguments=args)
                        try:
                            review = self._symbolic_probe_plan_review(name, args)
                            if review is None:
                                review = self._symbolic_submission_review(name, args)
                            if review is not None:
                                result = review
                            else:
                                if name == 'fit_law_and_rollout' and getattr(self.runtime, 'symbolic_enabled', False):
                                    signature = self._symbolic_law_signature(args)
                                    if signature is not None:
                                        self._symbolic_fit_attempts.append(copy.deepcopy(signature))
                                result = self.runtime.execute(name, args)
                        except Exception:
                            # Tool exceptions are a failure, never a success or raw backend dump.
                            result = self._failure('tool_execution_error')
                if isinstance(result.get('budget'), dict):
                    self.last_budget = result['budget']
                if self.loop_recovery and executed_or_invalid:
                    notice = self._record_repeat(key, result)
                    if notice is not None:
                        recovery_feedback.append(notice)
                plan_review = self._symbolic_plan_review(name, result)
                if plan_review is not None:
                    recovery_feedback.append(plan_review)
                self.messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                      'content': json.dumps(result, ensure_ascii=False, allow_nan=False)})
                self._event('tool_finished', tool_call_id=call['id'], name=name, result=result)
                self._persist()
                data = result.get('data') or {}
                if (name == 'submit' and result.get('ok') and data.get('status') == 'accepted'
                        and data.get('session_ended') is True):
                    self.submission = data
                    self.session_ended = True
                elif data.get('session_ended') is True or self.last_budget.get('session_ended') is True:
                    self.session_ended = True
                print(f'工具结果：{name} {"ok" if result.get("ok") else "failed"}', flush=True)
            # All calls in an assistant reply have matching tool messages, even when
            # an earlier submit ended the session and later calls were skipped.
            if self.submission is not None:
                return self._finish('submitted', 'submission_accepted_locally')
            if self.session_ended:
                return self._finish('incomplete', 'session_ended_without_accepted_submission')
            if loop_stop_reason is not None:
                return self._finish('incomplete', loop_stop_reason)
            if recovery_feedback:
                self.messages.append({'role': 'user', 'content': '工具调用恢复反馈：\n' + '\n'.join(recovery_feedback)})
                self._persist()
