"""Compatibility for complete named-tool SSE replies from the deployed vLLM."""
import json
import time
import os
import signal
import threading
from contextlib import contextmanager
from ..repository.shore_agent.model import DeepSeekClient as OriginalClient, ModelError, _positive
from ..repository.shore_agent.store import append_event


@contextmanager
def _request_deadline(deadline, *, deadline_error=None):
    """Interrupt blocking I/O without extending an already installed alarm.

    The official entrypoint executes model calls on the Linux main thread.
    An inherited earlier/periodic alarm is dispatched to its original handler;
    only this request's own deadline becomes a ModelError.
    """
    if (threading.current_thread() is not threading.main_thread()
            or not hasattr(signal, 'setitimer')):
        raise ModelError('model_deadline_requires_main_thread')
    deadline_error = deadline_error or (lambda: ModelError('wall_time_exhausted'))
    started = time.monotonic()
    if deadline <= started:
        raise deadline_error()
    original_handler = signal.getsignal(signal.SIGALRM)
    original_delay, original_interval = signal.getitimer(signal.ITIMER_REAL)
    inherited_deadline = started + original_delay if original_delay > 0 else None

    def arm():
        next_deadline = min(deadline, inherited_deadline) if inherited_deadline is not None else deadline
        signal.setitimer(signal.ITIMER_REAL, max(.000001, next_deadline - time.monotonic()))

    def alarm(signum, frame):
        nonlocal inherited_deadline
        now = time.monotonic()
        if inherited_deadline is not None and now >= inherited_deadline:
            if original_interval > 0:
                elapsed_intervals = max(1, int((now - inherited_deadline) / original_interval) + 1)
                inherited_deadline += elapsed_intervals * original_interval
            else:
                inherited_deadline = None
            if callable(original_handler):
                original_handler(signum, frame)
            elif original_handler == signal.SIG_DFL:
                # Preserve the inherited alarm's default process action.
                signal.signal(signal.SIGALRM, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGALRM)
            # SIG_IGN is deliberately ignored, while our deadline stays armed.
        if time.monotonic() >= deadline:
            raise deadline_error()
        arm()

    signal.signal(signal.SIGALRM, alarm)
    try:
        arm()
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, original_handler)
        if inherited_deadline is not None:
            signal.setitimer(signal.ITIMER_REAL,
                max(.000001, inherited_deadline - time.monotonic()), original_interval)


class DeepSeekClient(OriginalClient):
    def _reasoning_parameters(self):
        parameters=super()._reasoning_parameters()
        # Explicit evaluation sampling settings override checkpoint defaults.
        # They are included before prompt accounting and request audit logging.
        for key in ('temperature','top_p','top_k'):
            if key in self.config:parameters[key]=self.config[key]
        return parameters

    def complete(self, messages, tools=None, role='main'):
        self._named_tool = tools[0]['function']['name'] if tools and len(tools)==1 else None
        submit_only = self._named_tool == 'submit'
        original_remaining = self.remaining_seconds
        original_tokens = self.max_tokens_per_call
        original_retries = self.max_retries
        original_http = self._http
        reserve = 5. if submit_only else 60.
        actual_remaining = max(0., original_remaining())
        allowed = max(0., actual_remaining - reserve)
        if submit_only:
            allowed = min(25., allowed)
        else:
            allowed = min(_positive(self.config, f'{role}_timeout_seconds', self.timeout), allowed)
        # One absolute deadline spans all transport attempts/backoff. Neither a
        # retry nor a stale budget callback can renew this request's allowance.
        deadline = time.monotonic() + allowed
        def remaining():
            return max(0., min(original_remaining() - reserve, deadline - time.monotonic()))
        self.remaining_seconds = remaining
        if submit_only:
            self.max_retries = 0
        owner=self
        class NamedToolHTTP:
            def __getattr__(self,name):return getattr(original_http,name)
            def stream(self,*args,**kwargs):
                if owner._named_tool:
                    # This deployed vLLM named-tool stream truncates quoted DSL strings.
                    # Expose only the phase's tool but use its reliable auto parser.
                    kwargs['json']['tool_choice']='auto'
                    append_event(owner.run_dir/'stream-compatibility.jsonl',
                        {'event':'single_tool_auto_transport','available_tool':owner._named_tool})
                return original_http.stream(*args,**kwargs)
        self._http=NamedToolHTTP()
        try:
            append_event(self.run_dir/'request-budget-guard.jsonl', {
                'event':'model_request_budget', 'phase':'submit' if submit_only else 'research',
                'actual_remaining_seconds':actual_remaining, 'reserve_seconds':reserve,
                'request_budget_seconds':allowed, 'max_tokens_per_call':self.max_tokens_per_call,
                'max_retries':self.max_retries,
            })
            with _request_deadline(deadline):
                message = super().complete(messages, tools=tools, role=role)
            if remaining() <= 0:
                # Even a complete response arriving after its allowance cannot
                # authorize a late/partial action. The caller re-enters its phase.
                raise ModelError('wall_time_exhausted')
            return message
        except ModelError as exc:
            self.last_finish_reason = None
            deadline_errors = {'wall_time_exhausted', 'api_timeout', 'api_transport_error',
                               'incomplete_api_stream', 'api_stream_error'}
            if (not submit_only and exc.code in deadline_errors and remaining() <= 0
                    and original_remaining() <= 60.5):
                append_event(self.run_dir/'request-budget-guard.jsonl', {
                    'event':'submission_reserve_reached', 'original_error':exc.code,
                    'actual_remaining_seconds':max(0., original_remaining()),
                })
                raise ModelError('submission_reserve_reached') from None
            raise
        finally:
            self._http = original_http
            self.remaining_seconds = original_remaining
            self.max_tokens_per_call = original_tokens
            self.max_retries = original_retries

    def _read_stream(self, response, deadline):
        owner=self
        class CompatibleResponse:
            def iter_lines(self):
                seen_tool=False
                for line in response.iter_lines():
                    if line.startswith('data:') and line[5:].strip()!='[DONE]':
                        try:chunk=json.loads(line[5:])
                        except ValueError:yield line;continue
                        if isinstance(chunk,dict):
                            for choice in chunk.get('choices') or []:
                                if not isinstance(choice,dict):continue
                                delta=choice.get('delta') or {}
                                if isinstance(delta,dict) and delta.get('tool_calls'):seen_tool=True
                                if (getattr(owner,'_named_tool',None) and seen_tool
                                        and choice.get('finish_reason')=='stop'):
                                    choice['finish_reason']='tool_calls'
                                    append_event(owner.run_dir/'stream-compatibility.jsonl',
                                        {'event':'named_tool_finish_reason','requested_tool':owner._named_tool,'original':'stop',
                                         'normalized':'tool_calls','call_number':owner.usage['calls']})
                                    line='data: '+json.dumps(chunk)
                    yield line
        result=super()._read_stream(CompatibleResponse(),deadline)
        # Argument syntax belongs to the Agent's atomic preflight. A complete
        # but malformed tool reply must remain recoverable without executing it.
        return result
