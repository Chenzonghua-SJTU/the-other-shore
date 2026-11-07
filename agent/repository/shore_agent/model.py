"""Small HTTP adapter with bounded requests for the single research Agent."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import ipaddress
import json
import math
import os
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from shore_rules import apply_model_rules

from .store import append_event


class ModelError(RuntimeError):
    """A deliberately redacted error; never includes HTTP headers or bodies."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _positive(config, key, default, *, integer=False):
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{key} must be positive and finite')
    if integer and int(value) != value:
        raise ValueError(f'{key} must be an integer')
    return int(value) if integer else float(value)


def _model_base_url(config):
    value = config.get('base_url', 'https://api.deepseek.com')
    if not isinstance(value, str):
        raise ValueError('base_url must be an HTTP(S) URL')
    value = value.rstrip('/')
    parsed = urlsplit(value)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('base_url must be an HTTP(S) URL without credentials/query/fragment')
    return value, parsed


def model_auth_config(config):
    """Return (environment-variable name, key required) for the model section.

    Unauthenticated requests require an explicit local deployment and loopback
    URL. Remote local servers can use a key or an SSH tunnel to a loopback port.
    This helper validates configuration without reading or returning a secret.
    """
    _, parsed = _model_base_url(config)
    deployment = config.get('deployment', 'cloud')
    if deployment not in ('cloud', 'local'):
        raise ValueError('deployment must be cloud or local')
    default_key_env = 'DASHSCOPE_API_KEY' if config.get('reasoning_protocol') == 'qwen' else 'DEEPSEEK_API_KEY'
    env_name = config.get('api_key_env', default_key_env)
    if not isinstance(env_name, str) or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', env_name) is None:
        raise ValueError('api_key_env must be an environment-variable name')
    required = config.get('api_key_required', True)
    if not isinstance(required, bool):
        raise ValueError('api_key_required must be boolean')
    if not required:
        loopback = parsed.hostname == 'localhost'
        try:
            loopback = loopback or ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            pass
        if deployment != 'local' or not loopback:
            raise ValueError('api_key_required=false requires deployment=local and a loopback base_url')
    return env_name, required


class DeepSeekClient:
    """Every HTTP attempt (including retries) consumes the research Agent budget.

    Current official protocol requires reasoning_content to survive *all*
    assistant turns in a tool-enabled conversation, even turns without calls.
    We therefore forward full message dictionaries without rebuilding them.
    """

    def __init__(self, config: dict, run_dir: Path, remaining_seconds, *, transport=None):
        config = apply_model_rules(config)
        self.config = dict(config)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.remaining_seconds = remaining_seconds
        self.model = config.get('model', 'deepseek-flash')
        self.base_url, _ = _model_base_url(config)
        self.api_key_env, self.api_key_required = model_auth_config(config)
        self.deployment = config.get('deployment', 'cloud')
        self.reasoning_protocol = config.get('reasoning_protocol', 'deepseek')
        if self.reasoning_protocol not in ('deepseek', 'chat_template', 'qwen'):
            raise ValueError('reasoning_protocol must be deepseek, chat_template or qwen')
        if self.reasoning_protocol == 'chat_template' and self.deployment != 'local':
            raise ValueError('chat_template requires deployment=local')
        self.timeout = _positive(config, 'timeout_seconds', 90)
        # Call counts remain auditable but no longer terminate a session. Ignore
        # obsolete caps in older configurations, including the per-role cap.
        self.max_calls = None
        self.max_tokens_per_call = _positive(config, 'max_tokens_per_call', 4096, integer=True)
        # Explicit null disables the cumulative quota, not the per-call cap or
        # accounting. Historical configurations with a number keep their limit.
        self.max_total_tokens = (None if config.get('max_total_tokens', 150000) is None
                                 else _positive(config, 'max_total_tokens', 150000, integer=True))
        self.role_limits = {'main': None}
        self.max_retries = config.get('max_retries', 1)
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or not 0 <= self.max_retries <= 5:
            raise ValueError('max_retries must be an integer between 0 and 5')
        self.reasoning_effort = config.get('reasoning_effort', 'high')
        if self.reasoning_effort not in ('low', 'medium', 'high', 'max'):
            raise ValueError('Invalid reasoning_effort')
        if self.deployment == 'local' and self.reasoning_effort == 'medium':
            raise ValueError('Local reasoning_effort must be low, high or max; aliases differ by server')
        self.thinking = config.get('thinking', True)
        if not isinstance(self.thinking, bool):
            raise ValueError('thinking must be boolean')
        self.sampling_seed = config.get('sampling_seed')
        if self.sampling_seed is not None and (
                type(self.sampling_seed) is not int or not 0 <= self.sampling_seed < 2**32
                or self.deployment != 'local' or self.reasoning_protocol != 'qwen'):
            raise ValueError('sampling_seed requires a local Qwen integer in [0, 2**32)')
        self.thinking_budget = (None if config.get('thinking_budget') is None
                                else _positive(config, 'thinking_budget', None, integer=True))
        self.usage = {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0,
                      'conservative_reserved_tokens': 0, 'elapsed_seconds': 0.0,
                      'roles': {role: {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0,
                                       'total_tokens': 0} for role in self.role_limits}}
        self.last_finish_reason = None
        self._prompt_reference = None
        self._progress = None
        self._progress_started = None
        self._progress_last_write = None
        self._http = httpx.Client(transport=transport, follow_redirects=False)
        append_event(self.run_dir / 'model_events.jsonl', {
            'event': 'configuration', 'model': self.model, 'base_url': self.base_url,
            'thinking': self.thinking,
            'reasoning_effort': self.reasoning_effort,
            **self._request_metadata(),
            'deployment': self.deployment, 'api_key_env': self.api_key_env,
            'api_key_required': self.api_key_required,
            'timeout_seconds': self.timeout,
            'max_calls': self.max_calls, 'max_tokens_per_call': self.max_tokens_per_call,
            'max_total_tokens': self.max_total_tokens, 'role_limits': self.role_limits,
            'max_retries': self.max_retries,
            'model_alias_note': 'Model names are provider-managed aliases; weights are not pinned.'})

    def close(self):
        self._http.close()

    def _reasoning_parameters(self):
        if self.reasoning_protocol == 'qwen':
            if self.deployment == 'local':
                return {'chat_template_kwargs': {'enable_thinking': self.thinking}}
            # DashScope Qwen uses a boolean switch, not DeepSeek effort levels.
            parameters = {'enable_thinking': self.thinking}
            if self.thinking and self.thinking_budget is not None:
                parameters['thinking_budget'] = self.thinking_budget
            return parameters
        if self.reasoning_protocol == 'chat_template':
            # DeepSeek-V4's local tokenizer consumes these explicit template
            # arguments. Server/checkpoint support must be verified at deploy
            # time; an HTTP 200 response alone does not demonstrate effectiveness.
            kwargs = {'thinking': self.thinking}
            parameters = {'chat_template_kwargs': kwargs}
            if self.thinking:
                kwargs['reasoning_effort'] = self.reasoning_effort
                # SGLang's V4 adapter reads the top-level field, whereas the
                # vLLM tokenizer reads kwargs. Keep the two values identical.
                parameters['reasoning_effort'] = self.reasoning_effort
            return parameters
        parameters = {'thinking': {'type': 'enabled' if self.thinking else 'disabled'}}
        if self.thinking:
            parameters['reasoning_effort'] = self.reasoning_effort
        return parameters

    def _request_metadata(self):
        return {'reasoning_protocol': self.reasoning_protocol,
                'requested_reasoning_effort': (None if self.reasoning_protocol == 'qwen'
                                               else self.reasoning_effort),
                'reasoning_parameters': self._reasoning_parameters()}

    def _write_progress(self, *, force=False):
        """Publish counters and phases, never model text or request credentials."""
        if self._progress is None:
            return
        now = time.monotonic()
        if not force and self._progress_last_write is not None and now - self._progress_last_write < 2:
            return
        self._progress['elapsed_seconds'] = max(0., now - self._progress_started)
        self._progress['updated_at'] = datetime.now(timezone.utc).isoformat()
        path = self.run_dir / 'model_progress.json'
        temporary = self.run_dir / '.model_progress.json.tmp'
        try:
            temporary.write_text(json.dumps(self._progress, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            temporary.replace(path)
            self._progress_last_write = now
        except OSError:
            # Display telemetry must not alter the scientific turn or retry it.
            pass
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _start_progress(self, started, event):
        self._progress_started = started
        self._progress_last_write = None
        self._progress = {
            'schema_version': 1, 'status': 'running', 'phase': 'waiting_first_byte',
            'model': self.model, 'call_number': event['call_number'], 'attempt': event['attempt'],
            'started_at': datetime.now(timezone.utc).isoformat(), 'updated_at': None,
            'last_received_at': None, 'elapsed_seconds': 0., 'timeout_seconds': event['timeout_seconds'],
            'received_chunks': 0, 'received_characters': 0, 'reasoning_characters': 0,
            'content_characters': 0, 'tool_call_characters': 0,
            'http_status': None, 'finish_reason': None, 'error': None,
        }
        self._write_progress(force=True)

    def _stream_progress(self, delta=None):
        if self._progress is None:
            return
        delta = delta or {}
        progress = self._progress
        old_phase = progress['phase']
        progress['received_chunks'] += 1
        progress['last_received_at'] = datetime.now(timezone.utc).isoformat()
        counts = {'reasoning_characters': len(delta.get('reasoning_content') or ''),
                  'content_characters': len(delta.get('content') or '') + len(delta.get('refusal') or ''),
                  'tool_call_characters': sum(len((item.get('function') or {}).get('arguments') or '')
                                              for item in (delta.get('tool_calls') or []))}
        for key, count in counts.items():
            progress[key] += count
            progress['received_characters'] += count
        if delta.get('tool_calls'):
            progress['phase'] = 'tool_calls'
        elif counts['content_characters']:
            progress['phase'] = 'content'
        elif counts['reasoning_characters']:
            progress['phase'] = 'thinking'
        self._write_progress(force=progress['phase'] != old_phase)

    def _finish_progress(self, event, *, retryable):
        if self._progress is None:
            return
        error = event.get('error')
        self._progress.update(
            status='retrying' if error and retryable else 'error' if error else 'completed',
            phase='error' if error else 'completed',
            http_status=event.get('http_status'), finish_reason=event.get('finish_reason'), error=error)
        self._write_progress(force=True)

    def _read_stream(self, response, deadline):
        """Aggregate a complete Qwen SSE turn before exposing any tool calls.

        The final usage chunk can follow the finish chunk. Both an explicit
        finish reason and the SSE [DONE] sentinel are required; a connection
        closing mid-turn never supplies a partial action to the Agent.
        """
        message = {'role': 'assistant', 'content': None}
        calls, metadata, pending = {}, {}, []
        finished = None
        done = False
        received_bytes = 0

        def consume(payload):
            nonlocal finished, done
            if payload == '[DONE]':
                if finished is None:
                    raise ModelError('incomplete_api_stream')
                done = True
                return
            try:
                chunk = json.loads(payload)
            except ValueError:
                raise ModelError('invalid_api_stream') from None
            if not isinstance(chunk, dict):
                raise ModelError('invalid_api_stream')
            if chunk.get('error') is not None:
                raise ModelError('api_stream_error')
            for field in ('model', 'system_fingerprint', 'usage'):
                if chunk.get(field) is not None:
                    metadata[field] = chunk[field]
            choices = chunk.get('choices', [])
            if not isinstance(choices, list) or len(choices) > 1:
                raise ModelError('invalid_api_stream')
            if not choices:
                self._stream_progress()
                return  # Usage-only terminal chunk.
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get('index') != 0:
                raise ModelError('invalid_api_stream')
            delta = choice.get('delta', {})
            if not isinstance(delta, dict) or finished is not None:
                raise ModelError('invalid_api_stream')
            if delta.get('role') not in (None, 'assistant'):
                raise ModelError('invalid_api_stream')
            for field in ('content', 'reasoning_content', 'refusal'):
                value = delta.get(field)
                if value is not None:
                    if not isinstance(value, str):
                        raise ModelError('invalid_api_stream')
                    message[field] = (message.get(field) or '') + value
            fragments = delta.get('tool_calls')
            if fragments is not None:
                if not isinstance(fragments, list):
                    raise ModelError('invalid_api_stream')
                for fragment in fragments:
                    if not isinstance(fragment, dict):
                        raise ModelError('invalid_api_stream')
                    index = fragment.get('index')
                    if type(index) is not int or not 0 <= index < 1024:
                        raise ModelError('invalid_api_stream')
                    call = calls.setdefault(index, {'id': '', 'type': 'function',
                                                    'function': {'name': '', 'arguments': ''}})
                    if fragment.get('type') not in (None, 'function'):
                        raise ModelError('invalid_api_stream')
                    value = fragment.get('id')
                    if value is not None:
                        if not isinstance(value, str):
                            raise ModelError('invalid_api_stream')
                        call['id'] += value
                    function = fragment.get('function')
                    if function is not None:
                        if not isinstance(function, dict):
                            raise ModelError('invalid_api_stream')
                        for field in ('name', 'arguments'):
                            value = function.get(field)
                            if value is not None:
                                if not isinstance(value, str):
                                    raise ModelError('invalid_api_stream')
                                call['function'][field] += value
            reason = choice.get('finish_reason')
            if reason is not None:
                if reason not in ('stop', 'tool_calls', 'length', 'content_filter'):
                    raise ModelError('invalid_api_stream')
                finished = reason
            self._stream_progress(delta)

        for line in response.iter_lines():
            if self.remaining_seconds() <= 0:
                raise ModelError('wall_time_exhausted')
            if time.monotonic() >= deadline:
                raise httpx.ReadTimeout('Streaming model request deadline exceeded')
            received_bytes += len(line.encode('utf-8'))
            if received_bytes > 64 * 1024 * 1024:
                raise ModelError('api_stream_too_large')
            if not line:
                if pending:
                    consume('\n'.join(pending))
                    pending = []
                if done:
                    break
            elif line.startswith('data:'):
                pending.append(line[5:].lstrip(' '))
            elif not line.startswith((':', 'event:', 'id:', 'retry:')):
                raise ModelError('invalid_api_stream')
        if pending and not done:
            consume('\n'.join(pending))
        if not done or finished is None:
            raise ModelError('incomplete_api_stream')
        if calls:
            if sorted(calls) != list(range(len(calls))):
                raise ModelError('invalid_api_stream')
            ordered = [calls[index] for index in sorted(calls)]
            if any(not item['id'] or not item['function']['name'] for item in ordered):
                raise ModelError('invalid_api_stream')
            if len({item['id'] for item in ordered}) != len(ordered):
                raise ModelError('invalid_api_stream')
            message['tool_calls'] = ordered
            if finished == 'stop':
                raise ModelError('invalid_api_stream')
        elif finished == 'tool_calls':
            raise ModelError('invalid_api_stream')
        return {**metadata, 'choices': [{'message': message, 'finish_reason': finished}]}

    def _quota(self, role):
        if role not in self.role_limits:
            raise ModelError('invalid_model_role')
        if self.remaining_seconds() <= 0:
            raise ModelError('wall_time_exhausted')
        if (self.max_total_tokens is not None and
                self.usage['total_tokens'] + self.usage['conservative_reserved_tokens'] >= self.max_total_tokens):
            self.usage['request_budget'] = {'status': 'blocked', 'reason': 'total_budget_exhausted',
                'available_tokens': max(0, self.max_total_tokens - self.usage['total_tokens']
                                         - self.usage['conservative_reserved_tokens'])}
            raise ModelError('model_token_budget_exhausted')

    @staticmethod
    def _prompt_context(body):
        return {k: v for k, v in body.items() if k not in ('messages', 'max_tokens')}

    def _prompt_estimate(self, body):
        """Reuse measured tokens only for an identical, unedited message prefix.

        The suffix is still bounded by UTF-8 bytes, with extra framing allowance.
        Changing old messages, tools or model settings invalidates the anchor.
        Never remove history/reasoning or treat provider cache hits as free tokens.
        """
        upper = len(json.dumps(body, ensure_ascii=False).encode('utf-8')) + 512
        ref = self._prompt_reference
        messages = body['messages']
        if (ref is not None and self._prompt_context(body) == ref['context']
                and len(messages) >= len(ref['messages'])
                and messages[:len(ref['messages'])] == ref['messages']):
            suffix = messages[len(ref['messages']):]
            added = len(json.dumps(suffix, ensure_ascii=False).encode('utf-8')) if suffix else 0
            return {'method': 'reported_prefix_plus_utf8_suffix',
                    'estimated_prompt_tokens': ref['prompt_tokens'] + added + 512,
                    'reference_prompt_tokens': ref['prompt_tokens'],
                    'added_message_bytes': added, 'utf8_upper_bound': upper}
        return {'method': 'utf8_upper_bound', 'estimated_prompt_tokens': upper,
                'utf8_upper_bound': upper}

    def complete(self, messages, tools=None, role='main') -> dict:
        self._quota(role)
        key = os.environ.get(self.api_key_env)
        if not key and self.api_key_required:
            code = 'missing_deepseek_api_key' if self.api_key_env == 'DEEPSEEK_API_KEY' else 'missing_model_api_key'
            raise ModelError(code)
        headers = {'Authorization': f'Bearer {key}'} if key else {}
        requested_max = self.max_tokens_per_call
        body = {'model': self.model, 'messages': copy.deepcopy(messages),
                'stream': self.reasoning_protocol == 'qwen',
                **self._reasoning_parameters()}
        if self.sampling_seed is not None:
            body['seed'] = self.sampling_seed
        if body['stream']:
            body['stream_options'] = {'include_usage': True}
        if tools:
            body.update(tools=copy.deepcopy(tools), tool_choice=({'type':'function','function':{'name':'submit'}}
                        if len(tools)==1 and tools[0]['function']['name']=='submit' else 'auto'))
        estimate = self._prompt_estimate(body)
        prompt_allowance = estimate['estimated_prompt_tokens']
        for attempt in range(self.max_retries + 1):
            self._quota(role)
            available = (None if self.max_total_tokens is None else
                         self.max_total_tokens - self.usage['total_tokens'] - self.usage['conservative_reserved_tokens'])
            output_limit = requested_max if available is None else min(requested_max, available - prompt_allowance)
            self.usage['request_budget'] = dict(estimate, available_tokens=available,
                total_quota_enabled=self.max_total_tokens is not None,
                requested_output_tokens=requested_max, output_token_limit=max(0, output_limit),
                minimum_output_tokens=64, status='blocked' if output_limit < 64 else 'admitted')
            if output_limit < 64:
                append_event(self.run_dir / 'model_events.jsonl', {
                    'event': 'budget_blocked', 'role': role,
                    'budget_usage': copy.deepcopy(self.usage)})
                raise ModelError('model_token_budget_exhausted')
            body['max_tokens'] = int(output_limit)
            timeout = min(_positive(self.config, f'{role}_timeout_seconds', self.timeout), self.remaining_seconds())
            if timeout <= 0:
                raise ModelError('wall_time_exhausted')
            self.usage['calls'] += 1
            self.usage['roles'][role]['calls'] += 1
            reservation = prompt_allowance + output_limit
            self.usage['conservative_reserved_tokens'] += reservation
            started = time.monotonic()
            event = {'event': 'request', 'role': role, 'attempt': attempt + 1,
                     'call_number': self.usage['calls'], 'max_tokens': output_limit,
                     'timeout_seconds': timeout, **self._request_metadata()}
            self._start_progress(started, event)
            append_event(self.run_dir / 'model_events.jsonl', {
                **event, 'event': 'request_started', 'started_at': self._progress['started_at'],
                'budget_usage': copy.deepcopy(self.usage)})
            retryable = False
            response_data = None
            try:
                if body['stream']:
                    with self._http.stream('POST', self.base_url + '/chat/completions', json=body,
                                           headers=headers, timeout=timeout) as response:
                        event['http_status'] = response.status_code
                        self._progress['http_status'] = response.status_code
                        self._write_progress(force=True)
                        if response.status_code != 200:
                            retryable = response.status_code in (408, 429) or response.status_code >= 500
                            raise ModelError(f'http_status_{response.status_code}')
                        response_data = self._read_stream(response, started + timeout)
                else:
                    response = self._http.post(self.base_url + '/chat/completions', json=body,
                                               headers=headers, timeout=timeout)
                    event['http_status'] = response.status_code
                    self._progress['http_status'] = response.status_code
                    self._write_progress(force=True)
                    if response.status_code != 200:
                        retryable = response.status_code in (408, 429) or response.status_code >= 500
                        raise ModelError(f'http_status_{response.status_code}')
                    try:
                        response_data = response.json()
                    except ValueError:
                        raise ModelError('invalid_api_json') from None
                reported = response_data.get('usage', {}) if isinstance(response_data, dict) else {}
                if (isinstance(reported, dict) and all(isinstance(reported.get(k), int)
                        and not isinstance(reported[k], bool) and reported[k] >= 0
                        for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
                        and reported['total_tokens'] >= reported['prompt_tokens'] + reported['completion_tokens']):
                    self.usage['conservative_reserved_tokens'] -= reservation
                    for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                        self.usage[k] += reported[k]
                        self.usage['roles'][role][k] += reported[k]
                    event['usage'] = reported
                    self.usage['request_budget']['reported_prompt_tokens'] = reported['prompt_tokens']
                    self._prompt_reference = {'context': self._prompt_context(body),
                                              'messages': body['messages'],
                                              'prompt_tokens': reported['prompt_tokens']}
                else:
                    event['usage_warning'] = 'missing_or_invalid_usage; full local allowance reserved'
                try:
                    choice = response_data['choices'][0]
                    message = choice['message']
                    finish = choice['finish_reason']
                    if not isinstance(message, dict) or message.get('role') != 'assistant':
                        raise ValueError()
                except (KeyError, IndexError, TypeError, ValueError):
                    raise ModelError('invalid_api_message') from None
                self.last_finish_reason = finish
                event['response_model'] = response_data.get('model')
                event['system_fingerprint'] = response_data.get('system_fingerprint')
                event['finish_reason'] = finish
                event['status'] = 'response'
                # Preserve the full request and response for local protocol auditing.
                append_event(self.run_dir / 'model_messages.jsonl', {
                    'role': role, 'call_number': self.usage['calls'],
                    'messages': messages, 'assistant': message, 'finish_reason': finish})
                if self.remaining_seconds() <= 0:
                    raise ModelError('wall_time_exhausted')
                return copy.deepcopy(message)
            except httpx.TimeoutException:
                retryable = True
                event['error'] = 'api_timeout'
            except httpx.TransportError:
                retryable = True
                event['error'] = 'api_transport_error'
            except ModelError as exc:
                event['error'] = exc.code
            finally:
                elapsed = time.monotonic() - started
                event['elapsed_seconds'] = elapsed
                self.usage['elapsed_seconds'] += elapsed
                event['budget_usage'] = copy.deepcopy(self.usage)
                self._finish_progress(event, retryable=retryable and attempt < self.max_retries)
                append_event(self.run_dir / 'model_events.jsonl', event)
            if not retryable or attempt >= self.max_retries:
                raise ModelError(event.get('error', 'api_error')) from None
            # Tiny bounded backoff; every retry also checks the wall clock and quotas.
            pause = min(0.5 * (attempt + 1), max(0, self.remaining_seconds()))
            if pause:
                time.sleep(pause)
        raise ModelError('api_error')
