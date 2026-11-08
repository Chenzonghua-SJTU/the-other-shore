"""Three explicit environmental batches; local analysis shares the wall clock.

The vendored research package remains unchanged. An assistant reply is one batch:
all its actions are known before any of them run, and no model is called between
those actions. Console calls also check this boundary, including draft fallback.
"""
from __future__ import annotations

import copy
import inspect
import json
import time
from pathlib import Path

from ..repository.shore_agent.agent import Agent as RepositoryAgent, _save
from ..repository.shore_agent.model import ModelError
from .submission_policy import forced, usable_laws, noisy
from .reliability import normalize_probe_args
from .arguments import normalize as normalize_arguments
from .recovery import repair_messages, submission_messages, candidate_messages, envelope_repair_messages, code_repair_messages, review_messages, exploration_messages
from jsonschema import Draft202012Validator


ENVIRONMENT_TOOLS = frozenset({'observe_body', 'insert_probe', 'save_draft', 'submit'})
CONSOLE_INTERACTIONS = frozenset({'observe', 'launch', 'save_draft', 'submit'})
MAX_BATCH_ACTIONS = 8


class InteractionAgent(RepositoryAgent):
    def __init__(self, client, tool_runtime, run_dir, config, *, max_turns=3,
                 counting_policy='environment_batches'):
        if type(max_turns) is not int or not 1 <= max_turns <= 3:
            raise ValueError('max_turns must be an integer from 1 through 3; zero cannot disable the limit')
        if counting_policy != 'environment_batches':
            raise ValueError('counting_policy must be environment_batches')
        super().__init__(client, tool_runtime, run_dir, config)
        self.max_turns = max_turns
        self.counting_policy = counting_policy
        self.environment_batches = 0
        self.model_replies = 0
        self.model_calls = 0
        self.batches = []
        self._active_batch = None
        self._active_tool = None
        self._initializing_mission = False
        self._mission_console_seen = False
        self._finished = False
        self._repair_request = None
        self._repair_count = 0
        self._fit_errors = {}
        self._preflight_review = None
        self._envelope_repair_request = None
        self._envelope_repair_count = 0
        self._code_repair_request = None
        self._code_repair_count = 0
        self._review_request = None
        self._review_count = 0
        self._initial_code_attempted = False
        self._submission_handoff = False
        self._final_model_attempts = 0
        self._request_recoveries = 0
        environment = getattr(self.runtime, 'environment', None)
        if environment is not None:
            environment.interaction_guard = self._guard_console

    def _guard_console(self, method):
        if method == 'mission':
            if self._initializing_mission and not self._mission_console_seen:
                self._mission_console_seen = True
                return
            raise ValueError('interaction_guard: mission must use the cached initial mission')
        if method not in CONSOLE_INTERACTIONS:
            raise ValueError('interaction_guard: unsupported direct Console operation ' + str(method))
        batch = self._active_batch
        if (self._finished or batch is None or not batch['environment_batch']
                or batch['environment_round'] > self.max_turns):
            raise ValueError('interaction_guard: Console interaction outside an admitted environment batch')
        expected = {'observe_body': 'observe', 'insert_probe': 'launch',
                    'save_draft': 'save_draft', 'submit': 'submit'}.get(self._active_tool)
        if method != expected:
            raise ValueError('interaction_guard: Console operation differs from the current model-selected tool')
        # Entering the actual Console request consumes this reply's reservation,
        # even when the request later fails. Local failures never call this hook.
        batch['environment_batch_consumed'] = True
        batch['console_methods'].append(method)
        self._save_ledger()

    def _ledger(self):
        return {'counting_policy': self.counting_policy, 'max_turns': self.max_turns,
                'counted_turns': self.environment_batches,
                'environment_batches': self.environment_batches,
                'environment_batches_remaining': self.max_turns - self.environment_batches,
                'model_calls': self.model_calls, 'model_replies': self.model_replies,
                'batch_max_actions': MAX_BATCH_ACTIONS, 'batches': copy.deepcopy(self.batches)}

    def _save_ledger(self):
        _save(self.run_dir / 'interaction-ledger.json', self._ledger())

    def _refresh_budget(self):
        self.last_budget = self.runtime.budget()
        return self.last_budget

    def _remaining(self):
        budget = self._refresh_budget()
        if budget.get('session_ended'):
            return 0.
        return float(budget.get('wall_seconds_remaining', 0.))

    def _prompt(self):
        directory = Path(inspect.getfile(RepositoryAgent)).parent / 'prompts'
        prompt = (directory / 'main.txt').read_text(encoding='utf-8')
        if self.ruleset == 'e5' and self.guidance_profile != 'e3':
            name = 'e5_concise.txt' if self.guidance_profile == 'concise' else 'e5.txt'
            prompt += '\n\n' + (directory / name).read_text(encoding='utf-8')
        if getattr(self.runtime, 'symbolic_enabled', False):
            prompt += '\n\n' + (directory / 'symbolic.txt').read_text(encoding='utf-8')
        if getattr(self.runtime, 'code_enabled', False):
            prompt = prompt.replace('不可访问任意文件、Shell、网络、源码或任意 Python。',
                                    '代码只可通过run_code在隔离环境执行，不可访问宿主文件、网络或仿真源码。')
            prompt += '\n\n' + (directory / 'code.txt').read_text(encoding='utf-8')
            skill = Path(__file__).resolve().parents[1] / 'skills/shore-run-code/SKILL.md'
            content = skill.read_text(encoding='utf-8')
            import hashlib
            _save(self.run_dir / 'run-code-skill.json', {'path': str(skill),
                'sha256': hashlib.sha256(content.encode()).hexdigest(), 'injected': True})
            prompt += '\n\n' + content
        prompt = prompt.replace('最后30秒系统封存最后一次及时通过格式与可执行性检查的草稿；没有有效草稿或提交则如实未完成。', '必须在最后一轮真实提交；没有额外自动提交轮次。')
        prompt += '\n\n' + Path(__file__).with_name('three_rounds_prompt.txt').read_text(encoding='utf-8')
        return prompt + (
            '\n\n环境交互批次硬限制：本局最多' + str(self.max_turns) + '个环境交互批次。'
            '一次助手回复中包含observe_body、insert_probe、save_draft或submit任一项时，整份回复计1批；'
            '同批可有1至8项工具操作，按给定顺序执行，工具之间不追加模型调用。'
            '首次mission与首光不计批次，重复mission只读缓存。纯本地推理、拟合、符号回归、验证或已获取资料读取'
            '不计环境批次，但照常消耗300秒墙钟。save_draft也会访问控制台，因此计入环境批次。'
            '参数预检失败且没有环境调用时不计批次；执行过环境操作的批次不退还。任一执行失败后本批后续操作跳过。'
            '不能嵌套batch，submit必须放在整份回复最后；整批格式先检查，再执行。'
            '最多前两批用于探索，最后一批保留submit。第二批后先完成纯本地拟合与验证，不能再请求新观测。'
            '第三批之后立即结束，没有第四批或额外自动提交；保存过的有效配对仍留给控制台。'
            '请在工具参数中表达所有行动，不把多次助手回复称作同一批。'
        )

    def _needs_initial_exploration(self):
        remaining=self._remaining()
        return (self.max_turns>=2 and self.environment_batches==0 and remaining>180
                and (self._initial_code_attempted or remaining<=240))

    def _submission_phase(self):
        return forced(self.runtime) or self._submission_handoff

    def _can_submit_now(self):
        laws = usable_laws(self.runtime)
        eligible = getattr(self.runtime, 'submission_eligibility', None)
        if callable(eligible):
            return any(eligible(law).get('allowed') is True for law in laws)
        return bool(laws) and (not noisy(self.runtime) or self._remaining() <= 90)

    def _model_recovery_pause(self, reason):
        # Recoverable failures have no extra call-count quota. A small bounded
        # pause avoids busy loops while consuming the same session wall clock.
        remaining = self._remaining()
        delay = min(remaining, .1 * max(1, min(self._request_recoveries, 5)))
        if delay > 0:
            self._event('model_recovery_backoff', reason=reason,
                        delay_seconds=delay, wall_seconds_remaining=remaining,
                        environment_calls=0)
            time.sleep(delay)

    def _available_schemas(self):
        schemas=self.runtime.schemas
        if self._submission_phase():
            return [s for s in schemas if s['function']['name']=='submit']
        remaining=self._remaining()
        if self._needs_initial_exploration():
            return [s for s in schemas if s['function']['name'] in ('insert_probe','observe_body')]
        if not usable_laws(self.runtime) and remaining<=180:
            return [s for s in schemas if s['function']['name']=='fit_law_and_rollout']
        blocked={'save_draft'}
        if self.environment_batches>=self.max_turns-1 or remaining<=100:
            blocked.update(('observe_body','insert_probe'))
        if remaining<=100:blocked.add('symbolic_regression')
        if remaining<=115:blocked.add('run_code')
        return [s for s in schemas if s['function']['name'] not in blocked]

    def _envelope_error(self, calls, offered_schemas=None):
        self._preflight_review=None
        if not RepositoryAgent._valid_calls(calls):
            return 'malformed_tool_call_protocol'
        if not 1 <= len(calls) <= MAX_BATCH_ACTIONS:
            return 'batch_requires_1_through_8_actions'
        schemas={s['function']['name']:s['function']['parameters'] for s in (offered_schemas if offered_schemas is not None else self._available_schemas())}
        total_probes=0
        for index, call in enumerate(calls):
            name=call['function']['name']
            if name not in schemas:
                instruction=('Plan the first active experiment with insert_probe/observe_body before more local analysis.'
                    if set(schemas)<= {'insert_probe','observe_body'} else
                    'Select an existing candidate and submit now.' if set(schemas)=={'submit'} else
                    'Register an explicit candidate with fit_law_and_rollout before further analysis; the last environment round is reserved for submit.')
                return 'tool_unavailable_in_current_phase: '+str(name)+'. Allowed tools: '+','.join(schemas)+'. '+instruction
            if name=='submit' and index!=len(calls)-1:return 'submit_must_be_last'
            try:
                args=json.loads(call['function']['arguments'])
                if not isinstance(args,dict):raise ValueError('arguments must be an object')
                json.dumps(args,allow_nan=False)
                args,changes=normalize_arguments(args,schemas[name],(self.runtime.task or {}).get('allowed_properties',[]))
                if changes:
                    self._event('argument_normalization',tool=name,changes=changes)
                    call['function']['arguments']=json.dumps(args,ensure_ascii=False)
                if name=='insert_probe':
                    args,note=normalize_probe_args(args)
                    if note:
                        self._event('probe_preflight_normalization',original=call['function']['arguments'],normalized=args,note=note)
                        call['function']['arguments']=json.dumps(args,ensure_ascii=False)
                error=next(Draft202012Validator(schemas[name]).iter_errors(args),None)
                if error:
                    detail=error.message[:500]
                    if error.validator in ('maxItems','minItems'):
                        detail=f'{error.validator}={error.validator_value}, actual_items={len(error.instance)}'
                    raise ValueError(str(list(error.path))+': '+detail)
                if name=='insert_probe':total_probes+=len(args['probes'])
            except (ValueError,TypeError) as exc:
                return 'preflight_arguments: '+name+': '+str(exc)[:600]
        energy=self.runtime.budget().get('energy_remaining')
        if isinstance(energy,(int,float)) and total_probes>energy:
            return 'preflight_energy: requested '+str(total_probes)+' probes, remaining '+str(energy)
        # Restore the original Agent's one-time research checks BEFORE any
        # environment operation; a review is local feedback, never a spent round.
        for call in calls:
            name=call['function']['name'];args=json.loads(call['function']['arguments'])
            review=None
            if name=='insert_probe':review=self._symbolic_probe_plan_review(name,args)
            if name=='submit':
                args=copy.deepcopy(args)
                if args.get('law_id')=='last_fit':
                    laws=usable_laws(self.runtime)
                    if laws:args['law_id']=laws[-1]['law_id']
                review=self._symbolic_submission_review(name,args)
            if review is not None:
                self._preflight_review={'tool':name,'result':review}
                return 'scientific_review_needed'
        return None

    @staticmethod
    def _call_names(calls):
        if not isinstance(calls, list):
            return []
        return [c.get('function', {}).get('name') for c in calls
                if isinstance(c, dict) and isinstance(c.get('function'), dict)]

    def _new_batch(self, names, *, environment_batch, automatic=False, error=None):
        if environment_batch:
            if self.environment_batches >= self.max_turns:
                raise ValueError('environment_batch_limit_reached')
            self.environment_batches += 1
        record = {'index': len(self.batches) + 1, 'model_reply': self.model_replies,
                  'environment_batch': environment_batch,
                  'declared_environment_batch': environment_batch,
                  'environment_batch_reserved': environment_batch,
                  'environment_batch_consumed': False,
                  'reservation_released': False, 'console_methods': [],
                  'reserved_environment_round': self.environment_batches if environment_batch else None,
                  'environment_round': self.environment_batches if environment_batch else None,
                  'automatic': automatic, 'tools': list(names), 'preflight_error': error,
                  'results': [], 'completed_actions': 0, 'skipped_actions': 0}
        self.batches.append(record)
        self._save_ledger()
        self._event('interaction_batch_started', **copy.deepcopy(record))
        return record

    def _settle_batch_reservation(self, batch):
        if not batch.get('environment_batch_reserved'):
            return
        batch['environment_batch_reserved'] = False
        if not batch['environment_batch_consumed']:
            # There is exactly one active reply; never refund a request merely
            # because its result was an error or its submitted law was rejected.
            self.environment_batches -= 1
            batch['reservation_released'] = True
            batch['environment_batch'] = False
            batch['environment_round'] = None
            self._event('environment_batch_reservation_released',
                        batch_index=batch['index'],
                        reserved_environment_round=batch['reserved_environment_round'],
                        environment_batches=self.environment_batches,
                        reason='No Console interaction started in this reply')
        self._save_ledger()

    @staticmethod
    def _failed(result):
        if not isinstance(result, dict) or result.get('ok') is not True:
            return True
        data = result.get('data') or {}
        return (isinstance(data, dict) and
                (data.get('status') == 'rejected' or data.get('fit_status') == 'failed'))

    def _record_result(self, record, name, result, *, skipped=False):
        if isinstance(result.get('budget'), dict):
            self.last_budget = result['budget']
        record['results'].append({'tool': name, 'ok': not self._failed(result),
                                  'error': copy.deepcopy(result.get('error')), 'skipped': skipped})
        record['skipped_actions' if skipped else 'completed_actions'] += 1
        data = result.get('data') or {}
        if (name == 'submit' and result.get('ok') is True and data.get('status') == 'accepted'
                and data.get('session_ended') is True):
            self.submission = copy.deepcopy(data)
            self.session_ended = True
        elif data.get('session_ended') or self.last_budget.get('session_ended'):
            self.session_ended = True
        self._save_ledger()

    def _runtime_hook(self, name, record):
        hook = getattr(self.runtime, name, None)
        if callable(hook):
            hook(copy.deepcopy(record))

    def _finish(self, status, reason):
        # Deliberately do not call RepositoryAgent._finish: its finalize_draft
        # may submit outside the action loop. A fallback is its own charged batch.
        if (not self._finished and self.submission is None and not self.session_ended
                and self.environment_batches < self.max_turns
                and getattr(self.runtime, 'last_valid_draft', None) is not None
                and self._remaining() > 0):
            draft = self.runtime.last_valid_draft
            args = {key: copy.deepcopy(draft[key])
                    for key in ('law_id', 'claims', 'physical_explanation')}
            batch = self._new_batch(['submit'], environment_batch=True, automatic=True)
            self._active_batch = batch
            try:
                self._runtime_hook('begin_batch', batch)
                self._active_tool = 'submit'
                result = self.runtime.execute('submit', args)
                if not isinstance(result, dict):
                    result = self._failure('invalid_tool_result')
            except Exception:
                result = self._failure('automatic_draft_submit_failed')
            finally:
                self._settle_batch_reservation(batch)
                self._active_tool = None
                self._active_batch = None
            self._runtime_hook('end_batch', batch)
            self._record_result(batch, 'submit', result)
            self._event('automatic_selected_draft_submit', arguments=args, result=result)
            if self.submission is not None:
                status, reason = 'submitted', 'saved_draft_after_' + reason
        self._finished = True
        self._active_batch = None
        try:
            self._refresh_budget()
        except Exception:
            pass
        result = {'status': status, 'reason': reason, 'submission': self.submission,
                  'model_usage': copy.deepcopy(self.client.usage), 'budget': self.last_budget,
                  'saved_draft_available': getattr(self.runtime, 'last_valid_draft', None) is not None,
                  'interaction_ledger': self._ledger()}
        self._persist()
        self._save_ledger()
        _save(self.run_dir / 'result.json', result)
        self._event('session_finished', **result)
        return result

    def run(self):
        if self._finished or self.messages:
            raise RuntimeError('An InteractionAgent instance cannot restart or reset its session budget')
        self._initializing_mission = True
        try:
            mission = self.runtime.mission()
        finally:
            self._initializing_mission = False
        self.last_budget = mission.get('budget') or {}
        self.messages = [{'role': 'system', 'content': self._prompt()},
                         {'role': 'user', 'content': '当前任务与合法初始观测：\n' +
                          json.dumps(mission, ensure_ascii=False, allow_nan=False)}]
        self._persist()
        self._save_ledger()
        if not mission.get('ok'):
            return self._finish('error', 'mission_unavailable')
        no_action = 0
        while True:
            if self.environment_batches >= self.max_turns:
                return self._finish('incomplete', 'environment_batch_limit_reached')
            remaining = self._remaining()
            if remaining <= 0:
                return self._finish('incomplete', 'wall_time_exhausted')
            if ((self._submission_phase() or remaining <= self.finish_reserve_seconds)
                    and getattr(self.runtime, 'last_valid_draft', None) is not None):
                return self._finish('incomplete', 'deadline_reserve')
            left = self.max_turns - self.environment_batches
            state = {'environment_batches_used': self.environment_batches,
                     'environment_batches_remaining': left, 'model_replies': self.model_replies,
                     'wall_seconds_remaining': remaining, 'batch_max_actions': MAX_BATCH_ACTIONS,
                     'usable_candidate_ids':[r.get('law_id') for r in usable_laws(self.runtime)],
                     'candidate_registration_required':not usable_laws(self.runtime) and remaining<=180,
                     'instruction': ('下一次含环境工具的回复是最后一批：把submit放在最后，所有必要本地准备可先完成。'
                                     if left == 1 else '可先本地分析；环境工具在同一回复内批量安排，submit只能最后。')}
            self.messages.append({'role': 'user', 'content': '交互预算：' + json.dumps(state, ensure_ascii=False)})
            self.model_calls += 1
            self._save_ledger()
            try:
                tools = self._available_schemas()
                exploring=bool(tools) and all(s['function']['name'] in ('insert_probe','observe_body') for s in tools)
                if len(tools)==1 and tools[0]['function']['name']=='fit_law_and_rollout':
                    self.messages.append({'role':'user','content':'现在必须先注册第一个可提交候选。请根据已有证据明确给出law_spec、parameter_spec和train_probe_ids，调用fit_law_and_rollout；不能继续只用run_code或口头分析。候选允许不完美，随后仍可验证、修改和比较。不得猜世界答案。'})
                if self._submission_phase():
                    self.messages.append({'role':'user','content':'现在进入最终提交阶段；请选择已有候选的law_id，明确claims和解释，并立即submit。'})
                model_messages=self.messages
                if exploring:
                    model_messages=exploration_messages(self.runtime,self.messages)
                    self._event('initial_exploration_handoff',environment_calls=0)
                if not usable_laws(self.runtime) and remaining<=180:
                    model_messages=candidate_messages(self.runtime,self.messages)
                    self._event('candidate_context_handoff',environment_calls=0)
                if self._submission_phase():
                    model_messages=submission_messages(self.runtime,self.messages)
                    self._event("submission_context_handoff",candidate_ids=state["usable_candidate_ids"],environment_calls=0)
                if self._review_request is not None and usable_laws(self.runtime) and remaining>120 and self._review_count<2 and not self._submission_phase() and not exploring:
                    model_messages=review_messages(self.runtime,self.messages,self._review_request)
                    self._review_count+=1
                    self._event('evidence_review_handoff',review_number=self._review_count,trigger=self._review_request,environment_calls=0)
                    self._review_request=None
                if self._code_repair_request is not None and remaining>180 and self._code_repair_count<3 and any(s['function']['name']=='run_code' for s in tools):
                    tools=[s for s in tools if s['function']['name']=='run_code']
                    model_messages=code_repair_messages(self._code_repair_request,self.runtime)
                    self._code_repair_count+=1
                    self._event('code_repair_started',repair_number=self._code_repair_count,environment_calls=0)
                    self._code_repair_request=None
                if self._envelope_repair_request is not None and not self._submission_phase() and remaining>180 and self._envelope_repair_count<3:
                    model_messages=envelope_repair_messages(self._envelope_repair_request,self.runtime)
                    self._envelope_repair_count+=1
                    self._event('envelope_repair_started',repair_number=self._envelope_repair_count,environment_calls=0)
                    self._envelope_repair_request=None
                if self._repair_request is not None and not self._submission_phase() and remaining>35 and self._repair_count<6 and not exploring:
                    tools=[s for s in self.runtime.schemas if s['function']['name']=='fit_law_and_rollout']
                    self._repair_request['repair_attempt']=self._repair_count+1
                    model_messages=repair_messages(self._repair_request,self.runtime)
                    self._repair_count+=1
                    self._event('syntax_repair_started',repair_number=self._repair_count,request=self._repair_request,environment_calls=0)
                    self._repair_request=None
                final_request = self._submission_phase()
                if final_request:
                    # Audit attempts without introducing a model-decision cap.
                    self._final_model_attempts += 1
                message = self.client.complete(model_messages, tools=tools, role='main')
            except ModelError as exc:
                remaining_after_error = self._remaining()
                recoverable = exc.code in {
                    'submission_reserve_reached', 'wall_time_exhausted', 'api_timeout',
                    'api_transport_error', 'api_stream_error', 'incomplete_api_stream',
                    'invalid_api_stream', 'invalid_api_json', 'invalid_api_message',
                    'http_status_408', 'http_status_429', 'http_status_500',
                    'http_status_502', 'http_status_503', 'http_status_504',
                }
                if recoverable and remaining_after_error > 0:
                    if exc.code == 'submission_reserve_reached':
                        self._submission_handoff = True
                    elif self._can_submit_now():
                        self._submission_handoff = True
                    # This counter is telemetry, never a replacement for the
                    # actual session clock or an extra research/final-call cap.
                    self._request_recoveries += 1
                    self._event('model_request_recovery', reason=exc.code,
                        remaining_seconds=remaining_after_error,
                        final_submission_phase=self._submission_phase(),
                        final_attempts=self._final_model_attempts,
                        environment_calls=0, partial_tools_executed=False)
                    self.messages.append({'role':'user','content':
                        '上次模型请求未完整返回，所有未完成工具调用均未执行，原时钟和用量仍有效。'
                        + ('现在只给一个完整submit调用，选择已有law_id并明确claims；不要继续研究。'
                           if self._submission_phase() else
                           '请缩短回复，依据已有证据完成必要操作，必须预留最后60秒提交。')})
                    self._persist()
                    self._model_recovery_pause(exc.code)
                    continue
                return self._finish('incomplete', exc.code)
            self.model_replies += 1
            self.messages.append(copy.deepcopy(message))
            self._persist()
            if self._remaining() <= 0:
                return self._finish('incomplete', 'late_model_response_rejected')
            calls = message.get('tool_calls') or []
            if not calls:
                no_action += 1
                self._event('model_reply_without_actions', model_reply=self.model_replies)
                if no_action >= 2 and not usable_laws(self.runtime) and self._repair_count < 6:
                    # A prose-only candidate is not an action. Ask the same
                    # model to serialize its stated hypothesis, preserving the
                    # remaining time instead of ending a recoverable session.
                    self._repair_request={'original_arguments':{},
                        'error':{'code':'no_tool_call','message':'Your previous replies described a candidate but did not execute a tool. Express that same hypothesis as fit_law_and_rollout arguments now.'},
                        'previous_explanation':message.get('content') or ''}
                    no_action=0
                    self._event('no_action_recovery_scheduled',environment_calls=0)
                if no_action >= self.max_no_action and not self._submission_phase():
                    return self._finish('incomplete', 'consecutive_no_action_limit')
                self.messages.append({'role': 'user', 'content': '自然语言不会执行工具或提交；请按现有证据给出完整工具调用。'})
                self._persist()
                if self._submission_phase():
                    self._model_recovery_pause('final_reply_without_action')
                continue
            no_action = 0
            names = self._call_names(calls)
            # The reply may have crossed the 60-second boundary in flight.
            current_tools = self._available_schemas() if self._submission_phase() else tools
            error = self._envelope_error(calls, offered_schemas=current_tools)
            if getattr(self.client, 'last_finish_reason', 'tool_calls') not in ('stop', 'tool_calls'):
                error = 'incomplete_model_reply_not_executed'
            if error and (error.startswith('preflight_') or error=='batch_requires_1_through_8_actions'):
                self._envelope_repair_request={'rejected_calls':copy.deepcopy(calls),'error':error}
            if error:
                # The deployed chat template parses historical argument JSON.
                # Keep rejected raw text in an explicit wrapper for history,
                # while original calls/events remain unchanged and unexecuted.
                for previous in self.messages[-1].get('tool_calls',[]):
                    function=previous.get('function',{});raw=function.get('arguments')
                    try:
                        value=json.loads(raw)
                        if not isinstance(value,dict):raise ValueError('not an object')
                        json.dumps(value,allow_nan=False)
                    except (ValueError,TypeError):
                        replacement=json.dumps({'rejected_argument_text':raw},ensure_ascii=False)
                        function['arguments']=replacement
                        self._event('invalid_tool_arguments_history_repaired',tool_call_id=previous.get('id'),original_arguments=raw,replacement_arguments=replacement,tool_executed=False)
            # Rejected preflight has made NO environment call, so consumes no round.
            is_environment = error is None and bool(set(names) & ENVIRONMENT_TOOLS)
            batch = self._new_batch(names, environment_batch=is_environment, error=error)
            aborted = error
            self._active_batch = batch
            history_message = self.messages[-1]
            science_feedback=[]
            try:
                if error is None:
                    self._runtime_hook('begin_batch', batch)
                if not isinstance(calls, list):
                    calls = []
                for call in calls:
                    if (not isinstance(call, dict) or not isinstance(call.get('id'), str)
                            or not isinstance(call.get('function'), dict)):
                        continue
                    name = call['function'].get('name')
                    skipped = aborted is not None or self.session_ended
                    if skipped:
                        result = self._failure(error or 'batch_aborted_after_failure')
                        if self._preflight_review and name==self._preflight_review['tool']:
                            result=copy.deepcopy(self._preflight_review['result'])
                    elif self._remaining() <= 0:
                        result = self._failure('wall_time_exhausted')
                    elif self._submission_phase() and name != 'submit':
                        result = self._failure('submission_required_now')
                    else:
                        raw = call['function'].get('arguments')
                        try:
                            args = json.loads(raw)
                            if not isinstance(args, dict):
                                raise ValueError('Tool arguments must be an object')
                            json.dumps(args, allow_nan=False)
                        except (TypeError, ValueError):
                            result = self._failure('invalid_tool_arguments_json')
                            # Qwen parses historical argument JSON. Preserve raw
                            # bytes in the event, repair only transport history.
                            self._event('invalid_tool_arguments_history_repaired', tool_call_id=call['id'],
                                        original_arguments=raw, replacement_arguments='{}', tool_executed=False)
                            for previous in history_message.get('tool_calls', []):
                                if previous.get('id') == call['id']:
                                    previous['function']['arguments'] = '{}'
                        else:
                            self._event('tool_started', tool_call_id=call['id'], name=name, arguments=args)
                            try:
                                self._active_tool = name
                                key=self._repeat_key(name,args,raw)
                                blocked=self._repeat_block_reason(key) if self.loop_recovery else None
                                if blocked:
                                    result=self._failure(blocked)
                                    self._review_request={'reason':'repeated_call_without_progress','tool':name,'arguments':copy.deepcopy(args),'error':blocked}
                                    if name=='fit_law_and_rollout' and key in self._fit_errors:
                                        self._repair_request=copy.deepcopy(self._fit_errors[key])
                                    result['error']['message']='Identical call already failed or returned unchanged data twice. Change arguments using actual record IDs and DSL syntax, or submit an existing candidate.'
                                else:
                                    if name=='fit_law_and_rollout' and getattr(self.runtime,'symbolic_enabled',False):
                                        signature=self._symbolic_law_signature(args)
                                        if signature is not None:self._symbolic_fit_attempts.append(copy.deepcopy(signature))
                                    if name=='run_code' and self.environment_batches==0:self._initial_code_attempted=True
                                    result = self.runtime.execute(name, args)
                                    if (not isinstance(result, dict) or type(result.get('ok')) is not bool
                                            or result.get('data') is not None and not isinstance(result.get('data'), dict)
                                            or result.get('error') is not None and not isinstance(result.get('error'), dict)):
                                        result = self._failure('invalid_tool_result')
                                    if name=='run_code' and not result.get('ok') and (result.get('error') or {}).get('code')=='code_execution_error':
                                        self._code_repair_request={'original_arguments':copy.deepcopy(args),'execution_error':copy.deepcopy((result.get('data') or {}).get('error') or result.get('error'))}
                                    if name=='fit_law_and_rollout' and not result.get('ok') and (result.get('error') or {}).get('code') in ('invalid_expression','invalid_law'):
                                        self._repair_request={'original_arguments':copy.deepcopy(args),'error':copy.deepcopy(result['error']),'previous_explanation':history_message.get('content') or ''}
                                        self._fit_errors[key]=copy.deepcopy(self._repair_request)
                                    if self.loop_recovery:
                                        notice=self._record_repeat(key,result)
                                        if notice:result['recovery_instruction']=notice
                                if not isinstance(result, dict):
                                    result = self._failure('invalid_tool_result')
                            except Exception:
                                result = self._failure('tool_execution_error')
                            finally:
                                self._active_tool = None
                    review=self._symbolic_plan_review(name,result)
                    if review is not None:science_feedback.append(review)
                    self._record_result(batch, name, result, skipped=skipped)
                    self.messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                          'content': json.dumps(result, ensure_ascii=False, allow_nan=False)})
                    self._event('tool_finished', tool_call_id=call['id'], name=name, result=result)
                    self._persist()
                    if self._failed(result):
                        aborted = 'batch_aborted_after_failure'
                if error is None:
                    self._active_tool = None
                    self._runtime_hook('end_batch', batch)
            finally:
                self._settle_batch_reservation(batch)
                self._active_tool = None
                self._active_batch = None
            for feedback in science_feedback:
                self.messages.append({'role':'user','content':feedback})
            self._event('interaction_batch_finished', **copy.deepcopy(batch))
            if self.submission is not None:
                return self._finish('submitted', 'submission_accepted_locally')
            if self.session_ended:
                return self._finish('incomplete', 'session_ended_without_accepted_submission')
            if error:
                for call in history_message.get('tool_calls',[]):
                    try:json.loads(call['function']['arguments'])
                    except (ValueError,TypeError):call['function']['arguments']='{}'
            if error=='scientific_review_needed':
                self.messages.append({'role':'user','content':'以上是一次本地科研自检，不是提交误差门槛，也未消耗环境轮次。请考虑具体证据；如认为已充分，可以再次submit。没有第四轮环境交互。'})
                self._persist()
            elif error or aborted:
                records=getattr(getattr(self.runtime,'store',None),'records',{})
                self.messages.append({'role':'user','content':'纠错信息：已有law_id='+json.dumps(list(records.get('law',{})))+'。不要猜编号；用实际编号或last_fit。公式里的公开属性使用q(\"属性名\")，例如q(\"mass\")，不是裸变量mass/charge。请阅读失败信息并改变调用，不要重复同一错误。'})
                self.messages.append({'role': 'user', 'content': '本批已结束；失败后余下工具未执行。预检拒绝不消耗环境轮次；实际已执行的环境批次不退还。请按具体错误修正。'})
                self._persist()
                if self._submission_phase():
                    self._model_recovery_pause(error or aborted)


Agent = InteractionAgent
