"""Local web jobs: bounded inputs, isolated workers, and a public audit projection.

The web server never imports the simulator or executes a user-supplied command.
Only named scientific records are exposed; raw model history stays on disk.
"""
from __future__ import annotations

import ctypes
import errno
import hashlib
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import uuid
from shore_rules import apply_rules, initial_energy, rules_for

from shore_worlds import (DEFAULT_WORLD_ID, CURRENT_PHYSICS_VERSION, V15_PHYSICS_VERSION,
                          validate_world_id, world_name, validate_world_set, world_set_name)
from .model import model_auth_config
from .observations import estimate_body_trajectories, estimate_static_bodies

MODELS = ('deepseek-flash', 'deepseek-v4-pro')
QWEN_MODELS = ('qwen3.8-27b', 'qwen3.6-35b-a3b', 'qwen3.5-122b-a10b')
QWEN_OUTPUT_LIMITS = {'qwen3.8-27b': 131072, 'qwen3.6-35b-a3b': 65536, 'qwen3.5-122b-a10b': 65536}
REASONING_EFFORTS = ('low', 'high', 'max')
RUN_ID = re.compile(r'[A-Za-z0-9_-]{1,120}')
SECRET = re.compile(r'\bsk-[A-Za-z0-9_.-]{8,}|\bBearer\s+\S+', re.I)
PRIVATE_FIELDS = {'api_key', 'deepseek_api_key', 'authorization', 'access_token',
                  'reasoning_content', 'reasoning', 'messages', 'assistant', 'headers',
                  'raw_response', 'raw_messages', 'model_messages', 'password', 'secret'}
RECORD_KINDS = {'probes': 'probe', 'laws': 'law', 'validations': 'validation',
                'summaries': 'data_summary', 'submissions': 'submission',
                'body_observations': 'body_observation', 'code_executions': 'code_execution',
                'diagnostics': 'diagnostic', 'symbolic_searches': 'symbolic_search'}
MAX_FILE_BYTES = 32 * 1024 * 1024


class ValidationError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class ConflictError(RuntimeError):
    """Safe conflict code for the local HTTP API; contains no filesystem detail."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _move_directory_no_replace(source, destination):
    """Atomically rename a directory, without replacing even an empty destination.

    macOS and Linux provide exclusive rename flags; fail closed elsewhere rather
    than silently falling back to POSIX rename's overwrite behavior.
    """
    if os.path.lexists(destination):
        raise ConflictError('run_already_exists')
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes, destination_bytes = os.fsencode(source), os.fsencode(destination)
    if sys.platform == 'darwin' and hasattr(libc, 'renamex_np'):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 4)  # RENAME_EXCL
    elif sys.platform.startswith('linux') and hasattr(libc, 'renameat2'):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise ConflictError('exclusive_move_unavailable')
    if result:
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise ConflictError('run_already_exists')
        if code == errno.ENOENT:
            raise FileNotFoundError('unknown_run')
        raise OSError(code, 'run_move_failed')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    """An atomic replacement, with unique temporaries for server/worker races."""
    path = Path(path)
    temp = path.with_name('.' + path.name + '-' + uuid.uuid4().hex + '.tmp')
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _public(value, depth=0):
    if depth > 32:
        return None
    if isinstance(value, dict):
        return {k: _public(v, depth + 1) for k, v in value.items()
                if isinstance(k, str) and k.lower() not in PRIVATE_FIELDS
                and not k.lower().endswith('_api_key')}
    if isinstance(value, list):
        return [_public(v, depth + 1) for v in value]
    if isinstance(value, str):
        return SECRET.sub('[redacted]', value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _safe_file(path, root):
    # Do not follow even in-tree symlinks: files on the web API have a fixed origin.
    try:
        relative = path.relative_to(root)
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                return False
        return path.is_file() and path.stat().st_size <= MAX_FILE_BYTES
    except (OSError, ValueError):
        return False


def _read(path, root, default=None):
    if not _safe_file(path, root):
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError, UnicodeError):
        return default


def _lines(path, root):
    if not _safe_file(path, root):
        return []
    result = []
    try:
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        result.append(item)
                except ValueError:
                    pass  # A concurrent writer can leave one partial final line.
    except (OSError, UnicodeError):
        pass
    return result


def _validation_completion_order(validations, events, *, tool_log=False):
    """Associate immutable validations with successful tool completions."""
    known = {item.get('validation_id'): item for item in validations
             if isinstance(item.get('validation_id'), str)}
    order = {}
    for index, item in enumerate(events):
        if not tool_log and item.get('event') != 'tool_finished':
            continue
        if item.get('tool' if tool_log else 'name') != 'validate_law':
            continue
        response = item.get('result')
        if not isinstance(response, dict) or response.get('ok') is not True:
            continue
        data = response.get('data')
        if not isinstance(data, dict) or not isinstance(data.get('validation_id'), str):
            continue
        record = known.get(data['validation_id'])
        if record is not None and data.get('law_id', record.get('law_id')) == record.get('law_id'):
            order[data['validation_id']] = index
    return order


def accepted_submission(directory):
    root = Path(directory)
    folder = root / 'records' / 'submission'
    if folder.is_symlink():
        return None
    for path in sorted(folder.glob('*.json')):
        item = _read(path, root)
        if isinstance(item, dict) and item.get('status') == 'accepted' and item.get('session_ended') is True:
            return item
    return None


def _result_status(result, fallback='incomplete'):
    if not isinstance(result, dict):
        return fallback
    value = result.get('status')
    return value if value in {'submitted', 'incomplete', 'error', 'stopped', 'initialized_only', 'analyst_checked'} else fallback


def _external_worker_running(directory, metadata):
    """Read-only recognition of a runner worker; never grants process control."""
    if metadata.get('runner_owned') is not True or metadata.get('status') not in ('starting', 'running', 'evaluating'):
        return False
    pid = metadata.get('worker_pid')
    if type(pid) is not int or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        result = subprocess.run(['ps', '-ww', '-p', str(pid), '-o', 'command='],
                                capture_output=True, text=True, timeout=1, check=False)
        command = shlex.split(result.stdout.strip())
        index = command.index('--run-dir')
        return (result.returncode == 0 and 'shore_agent.web_worker' in command
                and index + 1 < len(command) and Path(command[index + 1]).resolve() == directory.resolve())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False


def _model_progress_projection(directory, running):
    """Only expose typed counters and lifecycle metadata, never stream text."""
    value = _read(directory / 'model_progress.json', directory)
    if not isinstance(value, dict) or value.get('schema_version') != 1:
        return None
    statuses = {'running', 'completed', 'error', 'retrying'}
    phases = {'waiting_first_byte', 'thinking', 'tool_calls', 'content', 'completed', 'error'}
    if value.get('status') not in statuses or value.get('phase') not in phases:
        return None
    projected = {'schema_version': 1, 'status': value['status'], 'phase': value['phase'],
                 'active': running and value['status'] in ('running', 'retrying')}
    for key in ('call_number', 'attempt', 'elapsed_seconds', 'timeout_seconds', 'received_chunks',
                'received_characters', 'reasoning_characters', 'content_characters', 'tool_call_characters'):
        item = value.get(key)
        if type(item) in (int, float) and math.isfinite(item) and 0 <= item <= 1e15:
            projected[key] = item
    now = datetime.now(timezone.utc)
    parsed = {}
    for key in ('started_at', 'updated_at', 'last_received_at'):
        try:
            item = datetime.fromisoformat(value.get(key))
            if item.tzinfo is None:
                continue
            parsed[key] = item
            projected[key] = item.astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError):
            continue
    if projected['active'] and 'started_at' in parsed:
        projected['elapsed_seconds'] = max(0, (now - parsed['started_at']).total_seconds())
    projected['seconds_since_last_received'] = (max(0, (now - parsed['last_received_at']).total_seconds())
                                                if 'last_received_at' in parsed else None)
    return projected


def _evaluation_projection(directory, status, metadata, configuration, mission=None, evaluation=None, running=False):
    """Read post-session scores; this web projection never imports hidden truth.

    Historical score sidecars are tied to the exact original evaluation bytes.
    Scientific/lifecycle records remain unchanged when a score is displayed.
    """
    def projection(state, score=None, reason=None):
        return {'evaluation_state': state, 'evaluation_score': score, 'evaluation_error': reason}

    submission = accepted_submission(directory)
    accepted = submission is not None or status in ('submitted', 'accepted', 'evaluating')
    if not accepted:
        return projection('not_applicable')
    if evaluation is None:
        if (directory / 'evaluation.json').exists():
            return projection('error', reason='invalid_evaluation_record')
        if (running and status == 'evaluating') or metadata.get('evaluation_status') in ('running', 'evaluating'):
            return projection('running')
        if metadata.get('evaluation_status') == 'failed' or metadata.get('evaluation_interrupted'):
            return projection('error', reason='evaluation_failed_or_interrupted')
        return projection('pending')
    if not isinstance(evaluation, dict):
        return projection('error', reason='invalid_evaluation_record')
    task = (mission or {}).get('task') or {}
    environment = configuration.get('environment') or {}
    expected = {'world_id': environment.get('world_id', task.get('world_id', DEFAULT_WORLD_ID)),
                'physics_version': environment.get('physics_version', task.get('physics_version', 'v1'))}
    if submission:
        expected.update({key: submission.get(key) for key in ('law_id', 'submission_id')})
    if any(evaluation.get(key) is not None and value is not None and evaluation[key] != value
           for key, value in expected.items()):
        return projection('error', reason='evaluation_identity_mismatch')
    score = evaluation.get('score')
    if score is None:
        sidecar = _read(directory / 'evaluation_score.json', directory)
        if isinstance(sidecar, dict):
            path = directory / 'evaluation.json'
            if (not _safe_file(path, directory) or sidecar.get('evaluation_sha256') !=
                    hashlib.sha256(path.read_bytes()).hexdigest()):
                return projection('unavailable', reason='stale_score_record')
            score = sidecar.get('score')
    if not isinstance(score, dict):
        return projection('error' if evaluation.get('complete') is False else 'unavailable',
                          reason='score_not_available')
    rules = rules_for(environment)
    if (score.get('formula_version') != rules['score_formula_version']
            or score.get('status') not in ('scored', 'unavailable')
            or (rules['ruleset'] in ('e4', 'e5') and score.get('energy_budget') != initial_energy(mission, environment))):
        return projection('unavailable', reason='invalid_score_record')
    if score['status'] == 'unavailable':
        return projection('unavailable', dict(score, total=None), score.get('reason'))

    def finite_score(value, maximum):
        return (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum)

    limits = {'total': 100, 'prediction': 80, 'law': 15, 'conservation': 5}
    if (score.get('maximum') != 100 or any(not finite_score(score.get(key), limit) for key, limit in limits.items())
            or abs(score['total'] - score['prediction'] - score['law'] - score['conservation']) > 1e-8):
        return projection('unavailable', reason='invalid_score_record')
    count = evaluation.get('n_trajectories')
    if type(count) is not int or count < 20:
        return projection('unavailable', reason='insufficient_hidden_probes')
    failures = evaluation.get('failures', [])
    if not isinstance(failures, list):
        return projection('error', reason='invalid_evaluation_failures')
    if failures or evaluation.get('complete') is not True or evaluation.get('n_successful') != count:
        if not (score.get('reason') == 'candidate_prediction_invalid' and score['prediction'] == 0
                and score['conservation'] == 0 and isinstance(failures, list) and failures
                and all(isinstance(item, dict) and item.get('stage') == 'candidate_rollout' for item in failures)):
            return projection('error', reason='incomplete_evaluation')
    return projection('scored', score)


def _archive_created_at(directory):
    # Adding post-session score sidecars must not move old CLI archives to the
    # top of the history list. Directory mtime changes whenever a file is added.
    stat = directory.stat()
    timestamp = getattr(stat, 'st_birthtime', None)
    if timestamp is None:
        timestamp = next((path.stat().st_mtime for path in
                          (directory / 'mission.json', directory / 'config.json', directory / 'result.json')
                          if _safe_file(path, directory)), stat.st_mtime)
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _world_fields(configuration=None, metadata=None):
    """Project only a neutral world label; old archives default to world 1."""
    configuration = configuration if isinstance(configuration, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    environment = configuration.get('environment')
    environment = environment if isinstance(environment, dict) else {}
    try:
        world_set = validate_world_set(environment.get('world_set', metadata.get('world_set',
            'v15' if environment.get('physics_version') == V15_PHYSICS_VERSION else 'legacy')))
    except ValueError:
        world_set = 'legacy'
    try:
        world_id = validate_world_id(environment.get('world_id', metadata.get('world_id', DEFAULT_WORLD_ID)), world_set)
    except ValueError:
        world_id = DEFAULT_WORLD_ID
    return {'world_id': world_id, 'world_name': world_name(world_id, world_set),
            'world_set': world_set, 'world_set_name': world_set_name(world_set)}


class RunManager:
    def __init__(self, project_root, runs_dir=None, config_path=None):
        self.project_root = Path(project_root).resolve()
        self.config_path = Path(config_path).resolve() if config_path is not None else None
        self.runs_dir = Path(runs_dir or self.project_root / 'runs').resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._owned = {}
        self._closed = False

    def _server_config(self):
        config = json.loads((self.project_root / 'config.example.json').read_text(encoding='utf-8'))
        if self.config_path is not None:
            override = json.loads(self.config_path.read_text(encoding='utf-8'))
            if not isinstance(override, dict):
                raise ValueError('invalid_server_configuration')
            for section, value in override.items():
                if isinstance(value, dict) and isinstance(config.get(section), dict):
                    config[section].update(value)
                else:
                    config[section] = value
        # Keep the three actual DeepSeek effort levels visible in the console.
        aliases = {'minimal': 'low', 'medium': 'high', 'xhigh': 'high', 'ultra': 'max'}
        effort = config['model'].get('reasoning_effort', 'low')
        effort = aliases.get(effort, effort) if isinstance(effort, str) else None
        if effort not in REASONING_EFFORTS:
            raise ValueError('invalid_server_reasoning_effort')
        config['model']['reasoning_effort'] = effort
        model_auth_config(config['model'])
        return apply_rules(config)

    def _model_options(self, config):
        model = config['model']
        cap = rules_for(config.get('environment'))['max_tokens_per_call']
        name = model['model']
        if model.get('reasoning_protocol') == 'qwen' and name in QWEN_MODELS:
            labels = ('Qwen3.8-27B', 'Qwen3.6-35B-A3B', 'Qwen3.5-122B-A10B')
            endpoints = model.get('local_model_endpoints') if model.get('deployment') == 'local' else None
            return [{'id': item, 'name': label + ('（本机未部署）' if endpoints is not None and item not in endpoints else ''),
                     'available': endpoints is None or item in endpoints,
                     'max_output_tokens': min(QWEN_OUTPUT_LIMITS[item], cap) if cap else QWEN_OUTPUT_LIMITS[item]}
                    for item, label in zip(QWEN_MODELS, labels)]
        # A custom endpoint/model is selected only by the trusted startup config.
        cloud_defaults = (self.config_path is None and model.get('deployment', 'cloud') != 'local'
                          and model.get('base_url', '').rstrip('/') == 'https://api.deepseek.com'
                          and name in MODELS)
        names = MODELS if cloud_defaults else (name,)
        labels = {'deepseek-flash': 'DeepSeek Flash', 'deepseek-v4-pro': 'DeepSeek V4 Pro'}
        return [{'id': item, 'name': labels.get(item, item), **({'max_output_tokens': cap} if cap else {})} for item in names]

    def client_configuration(self):
        from .code_execution import sandbox_info
        config = self._server_config()
        model, environment = config['model'], config['environment']
        rules = rules_for(environment)
        key_env, key_required = model_auth_config(model)
        return {
            'models': self._model_options(config),
            'api_key_configured': bool(os.environ.get(key_env, '').strip()),
            'api_key_required': key_required,
            'api_key_env': key_env,
            'model_protocol': model.get('reasoning_protocol', 'deepseek'),
            'reasoning_effort_supported': model.get('reasoning_protocol') != 'qwen',
            'max_output_tokens_cap': rules['max_tokens_per_call'],
            'code_sandbox': sandbox_info(config.get('code_execution')),
            'defaults': {
                'world_set': environment.get('world_set', 'v15'),
                'ruleset': rules['ruleset'], 'energy_budget': rules['energy_budget'],
                'wall_seconds': environment.get('wall_seconds', rules['wall_seconds']),
                **{name: environment[name] for name in ('noise', 'seed', 'world_id')},
                **{name: model[name] for name in ('model', 'max_total_tokens', 'max_tokens_per_call',
                                                  'thinking', 'reasoning_effort')},
                'max_calls': None,
                'code_enabled': config.get('code_execution', {}).get('enabled', False),
            },
        }

    def _directory(self, run_id):
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise FileNotFoundError('unknown_run')
        directory = self.runs_dir / run_id
        if directory.is_symlink() or not directory.is_dir() or directory.resolve().parent != self.runs_dir:
            raise FileNotFoundError('unknown_run')
        return directory

    def _config(self, payload):
        allowed = {'model', 'noise', 'seed', 'max_calls', 'max_total_tokens', 'max_tokens_per_call',
                   'thinking', 'reasoning_effort', 'api_key', 'world_id', 'world_set', 'code_enabled'}
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise ValidationError('invalid_run_configuration')
        config = self._server_config()
        rules = rules_for(config.get('environment'))
        if rules['ruleset'] in ('e4', 'e5'):
            if payload.get('thinking') is True:
                raise ValidationError(rules['ruleset'] + '_requires_no_thinking')
            if rules['ruleset'] == 'e4' and payload.get('noise') is False:
                raise ValidationError('e4_requires_noise')
            if rules['ruleset'] == 'e5' and payload.get('noise') is True:
                raise ValidationError('e5_requires_no_noise')
            if type(payload.get('max_tokens_per_call')) is int and payload['max_tokens_per_call'] > 4096:
                raise ValidationError('invalid_output_budget')
        try:
            world_set = validate_world_set(payload.get('world_set', config['environment'].get('world_set',
                'v15' if config['environment'].get('physics_version') == V15_PHYSICS_VERSION else 'legacy')))
        except ValueError:
            raise ValidationError('invalid_world_set') from None
        if rules['ruleset'] in ('e4', 'e5') and world_set != 'v15':
            raise ValidationError(rules['ruleset'] + '_requires_v15')
        try:
            config['environment']['world_id'] = validate_world_id(payload.get('world_id', config['environment']['world_id']), world_set)
        except ValueError:
            raise ValidationError('invalid_world_id') from None
        if 'world_set' in payload or 'world_set' in config['environment'] or world_set == 'v15':
            config['environment']['world_set'] = world_set
        if world_set == 'v15':
            config['environment']['physics_version'] = V15_PHYSICS_VERSION
            # Physical domains come from the selected v15 world catalog.
            for field in ('reference_length', 'position_limit', 'velocity_limit', 'time_max'):
                config['environment'].pop(field, None)
        elif config['environment'].get('physics_version') == V15_PHYSICS_VERSION:
            config['environment']['physics_version'] = CURRENT_PHYSICS_VERSION
        for name in ('noise', 'thinking', 'code_enabled'):
            if name in payload and type(payload[name]) is not bool:
                raise ValidationError('invalid_run_configuration')
        if 'model' in payload:
            if payload['model'] not in [item['id'] for item in self._model_options(config) if item.get('available', True)]:
                raise ValidationError('invalid_model')
            config['model']['model'] = payload['model']
        if config['model'].get('deployment') == 'local' and config['model'].get('local_model_endpoints'):
            config['model']['base_url'] = config['model']['local_model_endpoints'][config['model']['model']]
        if 'reasoning_effort' in payload:
            if payload['reasoning_effort'] not in REASONING_EFFORTS:
                raise ValidationError('invalid_reasoning_effort')
            config['model']['reasoning_effort'] = payload['reasoning_effort']
        for name, lower, upper in [('seed', 0, 2147483647),
                                    ('max_total_tokens', 1000, 10000000), ('max_tokens_per_call', 64, 393216)]:
            if name in payload:
                value = payload[name]
                if name == 'max_total_tokens' and value is None:
                    config['model'][name] = None
                    continue
                if type(value) is not int or not lower <= value <= upper:
                    raise ValidationError('invalid_output_budget' if name == 'max_tokens_per_call'
                                          else 'invalid_seed' if name == 'seed' else 'invalid_budget')
                config['environment' if name == 'seed' else 'model'][name] = value
        if 'noise' in payload:
            config['environment']['noise'] = payload['noise']
        if (config['model'].get('reasoning_protocol') == 'qwen'
                and config['model']['model'] in QWEN_OUTPUT_LIMITS
                and config['model']['max_tokens_per_call'] > QWEN_OUTPUT_LIMITS[config['model']['model']]):
            raise ValidationError('invalid_output_budget')
        if 'thinking' in payload:
            config['model']['thinking'] = payload['thinking']
        if 'code_enabled' in payload:
            config.setdefault('code_execution', {})['enabled'] = payload['code_enabled']
        if config.get('code_execution', {}).get('enabled'):
            from .code_execution import sandbox_info
            if not sandbox_info(config.get('code_execution'))['available']:
                raise ValidationError('code_sandbox_unavailable')
        # Old browser tabs can still send max_calls; it no longer constrains new
        # sessions. Keep explicit nulls so snapshots distinguish the new policy.
        config['model']['max_calls'] = None
        config['model']['main_max_calls'] = None
        for name in ('max_turns', 'max_rounds'):
            config.get('agent', {}).pop(name, None)
        key = payload.get('api_key', '')
        if not isinstance(key, str) or len(key) > 512 or any(ord(char) < 32 or ord(char) == 127 for char in key):
            raise ValidationError('invalid_api_key')
        key_env, key_required = model_auth_config(config['model'])
        key = key.strip() or os.environ.get(key_env, '').strip()
        if not key and key_required:
            raise ValidationError('missing_api_key')
        return apply_rules(config), key

    def start(self, payload):
        with self._lock:
            if self._closed:
                raise RuntimeError('manager_closed')
            self._refresh()
            if self._owned:
                raise RuntimeError('run_already_active')
            config, key = self._config(payload)
            run_id = 'web-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
            directory = self.runs_dir / run_id
            directory.mkdir(mode=0o700)
            _write(directory / 'config.json', config)
            _write(directory / 'web.json', {'id': run_id, 'status': 'running', 'mode': 'autonomous_discovery',
                                           'created_at': _now(), 'model': config['model']['model'],
                                           **_world_fields(config)})
            # New keys exist only in this call and in the child. Do not retain them
            # in a manager field, argv, a log, or a config file.
            env = dict(os.environ)
            env.pop('DEEPSEEK_API_KEY', None)
            env.pop(model_auth_config(config['model'])[0], None)
            env['PYTHONPATH'] = str(self.project_root)
            process = None
            try:
                process = subprocess.Popen([sys.executable, '-u', '-m', 'shore_agent.web_worker',
                                            '--run-dir', str(directory)], cwd=self.project_root, env=env,
                                           stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, text=True, encoding='utf-8',
                                           start_new_session=True)
                self._owned[run_id] = process
                process.stdin.write(json.dumps({'api_key': key}, ensure_ascii=False) + '\n')
                process.stdin.flush()
                process.stdin.close()
            except (OSError, subprocess.SubprocessError):
                if process is not None:
                    self._terminate(process)
                self._owned.pop(run_id, None)
                self._update_metadata(directory, status='error', error='worker_start_failed')
                raise ValueError('worker_start_failed') from None
            finally:
                key = None
            return run_id

    @staticmethod
    def _terminate(process):
        if process.stdin is not None and not process.stdin.closed:
            try:
                process.stdin.close()
            except OSError:
                pass
        # Kill the session's group even if its leader exited while a simulator
        # descendant was still running. All processes were launched in this group.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)

    @staticmethod
    def _update_metadata(directory, **updates):
        value = _read(directory / 'web.json', directory, {})
        value.update(updates)
        _write(directory / 'web.json', value)

    def _refresh(self):
        for run_id, process in list(self._owned.items()):
            if process.poll() is None:
                continue
            process.wait()
            # Clean up any surviving simulator/evaluator from abnormal exits.
            self._terminate(process)
            directory = self._directory(run_id)
            result = _read(directory / 'result.json', directory)
            metadata = _read(directory / 'web.json', directory, {})
            if metadata.get('status') in ('running', 'evaluating'):
                status = 'submitted' if accepted_submission(directory) else _result_status(result, 'error')
                self._update_metadata(directory, status=status, finished_at=_now())
            self._owned.pop(run_id)

    def stop(self, run_id):
        with self._lock:
            directory = self._directory(run_id)
            self._refresh()
            process = self._owned.get(run_id)
            if process is None:
                if _external_worker_running(directory, _read(directory / 'web.json', directory, {})):
                    raise ConflictError('run_managed_externally')
                return self.snapshot(run_id)
            self._terminate(process)
            self._owned.pop(run_id, None)
            submission = accepted_submission(directory)
            status = 'submitted' if submission else 'stopped'
            # If cancellation landed after persisted acceptance but before
            # Agent._finish, retain that real scientific result accurately.
            result = _read(directory / 'result.json', directory)
            if submission and not (isinstance(result, dict) and result.get('status') == 'submitted'):
                _write(directory / 'result.json', {'status': 'submitted', 'reason': 'accepted_before_stop',
                                                   'submission': submission})
            elif not result:
                _write(directory / 'result.json', {'status': 'stopped', 'reason': 'stopped_by_user', 'submission': None})
            self._update_metadata(directory, status=status, finished_at=_now(),
                                  stop_requested=True, evaluation_interrupted=bool(submission and not (directory / 'evaluation.json').exists()))
            return self.snapshot(run_id)

    def _trash_directory(self, *, create=False):
        trash = self.runs_dir / '.trash'
        if create and not os.path.lexists(trash):
            try:
                trash.mkdir(mode=0o700)
            except FileExistsError:
                pass
        if not os.path.lexists(trash):
            raise FileNotFoundError('unknown_run')
        if trash.is_symlink() or not trash.is_dir() or trash.resolve().parent != self.runs_dir:
            raise ConflictError('unsafe_trash_directory')
        return trash

    @staticmethod
    def _trash_metadata_directory(trash, *, create=False):
        metadata = trash / '.metadata'
        if create and not os.path.lexists(metadata):
            try:
                metadata.mkdir(mode=0o700)
            except FileExistsError:
                pass
        if os.path.lexists(metadata) and (metadata.is_symlink() or not metadata.is_dir()
                                          or metadata.resolve().parent != trash):
            raise ConflictError('unsafe_trash_directory')
        return metadata

    def delete(self, run_id):
        """Move a finished run into the recoverable local trash, never erase it."""
        with self._lock:
            directory = self._directory(run_id)
            if not any(_safe_file(directory / name, directory) for name in ('web.json', 'result.json', 'mission.json')):
                raise FileNotFoundError('unknown_run')
            self._refresh()
            if run_id in self._owned or _external_worker_running(directory, _read(directory / 'web.json', directory, {})):
                raise ConflictError('run_is_active')
            trash = self._trash_directory(create=True)
            destination = trash / run_id
            if os.path.lexists(destination):
                raise ConflictError('run_already_exists')
            metadata_directory = self._trash_metadata_directory(trash, create=True)
            metadata_path = metadata_directory / (run_id + '.json')
            snapshot = self.snapshot(run_id)
            metadata = {key: snapshot[key] for key in ('id', 'created_at', 'mode', 'status', 'world_id', 'world_name',
                                                       'world_set', 'world_set_name')}
            metadata['deleted_at'] = _now()
            # Keep the original run byte-for-byte unchanged. Trash bookkeeping
            # lives next to it, and exclusive create protects preexisting data.
            try:
                with metadata_path.open('x', encoding='utf-8') as stream:
                    json.dump(metadata, stream, ensure_ascii=False, allow_nan=False)
            except FileExistsError:
                raise ConflictError('run_already_exists') from None
            try:
                _move_directory_no_replace(directory, destination)
            except Exception:
                metadata_path.unlink()
                raise
            return {'deleted_run_id': run_id}

    def restore(self, run_id):
        """Restore a deleted run without overwriting a run or following a symlink."""
        with self._lock:
            if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
                raise FileNotFoundError('unknown_run')
            trash = self._trash_directory()
            source = trash / run_id
            if source.is_symlink() or not source.is_dir() or source.resolve().parent != trash:
                raise FileNotFoundError('unknown_run')
            destination = self.runs_dir / run_id
            if os.path.lexists(destination):
                raise ConflictError('run_already_exists')
            metadata_directory = self._trash_metadata_directory(trash)
            metadata_path = metadata_directory / (run_id + '.json')
            if metadata_path.is_symlink() or (os.path.lexists(metadata_path) and not metadata_path.is_file()):
                raise ConflictError('unsafe_trash_directory')
            _move_directory_no_replace(source, destination)
            metadata_path.unlink(missing_ok=True)
            return {'restored_run_id': run_id}

    def _deleted_runs(self):
        try:
            trash = self._trash_directory()
            metadata_directory = self._trash_metadata_directory(trash)
        except (FileNotFoundError, ConflictError):
            return []
        rows = []
        for directory in trash.iterdir():
            if not RUN_ID.fullmatch(directory.name) or directory.is_symlink() or not directory.is_dir():
                continue
            metadata = _read(metadata_directory / (directory.name + '.json'), trash, {})
            if not isinstance(metadata, dict):
                metadata = {}
            result = _read(directory / 'result.json', directory, {})
            if not isinstance(result, dict):
                result = {}
            configuration = _read(directory / 'config.json', directory, {})
            web_metadata = _read(directory / 'web.json', directory, {})
            world_metadata = dict(web_metadata if isinstance(web_metadata, dict) else {}, **metadata)
            fallback_time = _archive_created_at(directory)
            status = metadata.get('status') or _result_status(result)
            evaluation = _read(directory / 'evaluation.json', directory)
            rows.append({'id': directory.name,
                         'created_at': metadata.get('created_at') or fallback_time,
                         'deleted_at': metadata.get('deleted_at') or fallback_time,
                         'mode': metadata.get('mode') or result.get('mode') or 'historical_unverified',
                         'status': status,
                         **_evaluation_projection(directory, status, web_metadata or {}, configuration or {},
                                                  _read(directory / 'mission.json', directory, {}), evaluation),
                         **_world_fields(configuration, world_metadata)})
        rows.sort(key=lambda row: str(row['deleted_at']), reverse=True)
        return _public(rows)

    def code_artifact(self, run_id, execution_id, filename):
        from .code_execution import ARTIFACT_NAME, MIMES
        directory = self._directory(run_id)
        if not re.fullmatch(r'code_[0-9a-f]{12}', execution_id) or not ARTIFACT_NAME.fullmatch(filename):
            raise FileNotFoundError('unknown_artifact')
        record = _read(directory / 'records' / 'code_execution' / (execution_id + '.json'), directory, {})
        meta = next((a for a in record.get('artifacts', []) if a.get('name') == filename), None)
        path = directory / 'code_executions' / execution_id / 'artifacts' / filename
        if not meta or not _safe_file(path, directory) or path.suffix not in MIMES:
            raise FileNotFoundError('unknown_artifact')
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != meta.get('sha256'):
            raise FileNotFoundError('artifact_changed')
        return content, MIMES[path.suffix]

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for run_id in list(self._owned):
                self.stop(run_id)

    def list_runs(self):
        with self._lock:
            self._refresh()
            rows = []
            for directory in self.runs_dir.iterdir():
                if not directory.is_dir() or directory.is_symlink() or not RUN_ID.fullmatch(directory.name):
                    continue
                if not any((directory / name).is_file() for name in ('web.json', 'result.json', 'mission.json')):
                    continue
                snapshot = self.snapshot(directory.name)
                result = snapshot.get('result') or {}
                submission = result.get('submission') or next(iter(snapshot['records']['submissions']), {})
                rows.append({k: snapshot[k] for k in ('id', 'status', 'created_at', 'mode', 'world_id', 'world_name',
                                                    'world_set', 'world_set_name', 'model_settings', 'experiment_id', 'ruleset', 'energy_budget',
                                                    'running', 'can_stop', 'managed_externally',
                                                    'evaluation_state', 'evaluation_score', 'evaluation_error')} | {
                    'energy_remaining': snapshot['budget'].get('energy_remaining'), 'law_id': submission.get('law_id')})
            rows.sort(key=lambda item: item['created_at'], reverse=True)
            return {'runs': rows, 'active_run_id': next(iter(self._owned), None),
                    'deleted_runs': self._deleted_runs()}

    def snapshot(self, run_id):
        with self._lock:
            self._refresh()
            directory = self._directory(run_id)
            result = _read(directory / 'result.json', directory)
            metadata = _read(directory / 'web.json', directory, {})
            mission = _read(directory / 'mission.json', directory, {})
            configuration = _read(directory / 'config.json', directory, {})
            model_config = configuration.get('model') or {}
            model_events = _lines(directory / 'model_events.jsonl', directory)
            raw_events = _lines(directory / 'events.jsonl', directory)
            tool_events = _lines(directory / 'tools.jsonl', directory)
            events, budget, usage = [], metadata.get('initial_budget') or {}, {}
            for item in raw_events:
                event_type = item.get('event')
                if event_type == 'session_started':
                    budget = item.get('budget') or budget
                    events.append({'kind': 'status', 'content': 'autonomous_session_started'})
                elif event_type == 'research_progress':
                    events.append({'kind': 'progress', 'content': item.get('content', '')})
                elif event_type == 'tool_started':
                    events.append({'kind': 'tool_start', 'tool': item.get('name'), 'arguments': item.get('arguments')})
                elif event_type == 'tool_finished':
                    response = item.get('result') or {}
                    events.append({'kind': 'tool_result', 'tool': item.get('name'), **{
                        k: response.get(k) for k in ('ok', 'data', 'error')}})
                    budget = response.get('budget') or budget
                elif event_type == 'session_finished':
                    events.append({'kind': 'status', 'content': item.get('reason', ''), 'data': {'status': item.get('status')}})
                    budget = item.get('budget') or budget
            if not raw_events:
                for item in tool_events:
                    response = item.get('result') or {}
                    events.extend([{'kind': 'tool_start', 'tool': item.get('tool'), 'arguments': item.get('arguments')},
                                   {'kind': 'tool_result', 'tool': item.get('tool'), **{k: response.get(k) for k in ('ok', 'data', 'error')}}])
                    budget = response.get('budget') or budget
            for item in model_events:
                usage = item.get('budget_usage') or usage
            if isinstance(result, dict):
                budget = result.get('budget') or budget
                usage = result.get('model_usage') or usage
            owned = run_id in self._owned
            external = metadata.get('runner_owned') is True
            running = owned or _external_worker_running(directory, metadata)
            if running and not budget.get('session_ended') and metadata.get('session_started_at'):
                try:
                    started = datetime.fromisoformat(metadata['session_started_at'])
                    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                    initial_wall = metadata.get('initial_budget', {}).get('wall_seconds_remaining')
                    if isinstance(initial_wall, (int, float)):
                        budget = dict(budget, wall_seconds_remaining=max(0, initial_wall - max(0, elapsed)))
                except (TypeError, ValueError):
                    pass
            status = metadata.get('status') or _result_status(result)
            if not running and status in ('running', 'evaluating'):
                status = 'submitted' if accepted_submission(directory) else _result_status(result, 'interrupted')
            mode = metadata.get('mode') or (result or {}).get('mode')
            if not mode:
                if (result or {}).get('autonomous_discovery') is False or run_id.startswith('offline-'):
                    mode = 'manual_engineering_smoke'
                elif any(item.get('role') == 'main' and item.get('event') == 'request' for item in model_events):
                    mode = 'autonomous_discovery'
                elif any(item.get('role') == 'analyst' for item in model_events):
                    mode = 'standalone_analyst_diagnostic'
                else:
                    mode = 'historical_unverified'
            records = {}
            for public_name, kind in RECORD_KINDS.items():
                folder = directory / 'records' / kind
                records[public_name] = []
                if not folder.is_symlink():
                    for path in sorted(folder.glob('*.json')):
                        value = _read(path, directory)
                        if isinstance(value, dict):
                            records[public_name].append(value)
            validations = records['validations']
            validation_order = _validation_completion_order(validations, raw_events)
            validation_source = 'events.jsonl'
            if validations and not validation_order:
                # Legacy archives can retain only status events plus tools.jsonl.
                # This detail-only fallback must not add log reads to history.
                fallback = tool_events if not raw_events else _lines(directory / 'tools.jsonl', directory)
                validation_order = _validation_completion_order(validations, fallback, tool_log=True)
                validation_source = 'tools.jsonl'
            unknown_validations = [item.get('validation_id') for item in validations
                                   if not isinstance(item.get('validation_id'), str)
                                   or item['validation_id'] not in validation_order]
            # Unmatched records have unknown chronology. Keep their stable file
            # order before logged records without inventing timestamps for them.
            records['validations'] = sorted(validations, key=lambda item: validation_order.get(
                item.get('validation_id') if isinstance(item.get('validation_id'), str) else None, -1))
            validation_order_info = {
                'source': validation_source if validation_order else 'stable_record_order',
                'chronology_complete': not unknown_validations,
                'unmatched_validation_ids': unknown_validations,
                'unmatched_placement': 'stable_record_order_before_logged_records',
            }
            # Display the same acquired information available to the Agent.
            # Old archives may keep first-light observations only in mission.json.
            observations = records['body_observations']
            first_light = (mission.get('first_light') or {}).get('body_observation')
            if isinstance(first_light, dict) and not any(
                    item.get('observation_id') == first_light.get('observation_id') for item in observations):
                observations.insert(0, first_light)
            body_estimate, bodies, body_source = None, [], 'no_observations'
            task = mission.get('task') or {}
            body_protocol = task.get('body_observation_protocol', 'static')
            if body_protocol == 'time_series':
                try:
                    body_estimate = estimate_body_trajectories(
                        observations, task.get('body_initial_states'), task.get('position_noise_sigma'))
                    bodies, body_source = body_estimate['bodies'], 'observed_trajectories'
                except (KeyError, TypeError, ValueError):
                    body_source = 'invalid_observations'
            elif observations:
                try:
                    body_estimate = estimate_static_bodies(observations, task.get('position_noise_sigma'))
                    bodies, body_source = body_estimate['bodies'], 'observed_estimate'
                except (KeyError, TypeError, ValueError):
                    # Never recover a malformed observation from accurate task coordinates.
                    body_source = 'invalid_observations'
            elif task.get('visible_bodies') and not mission.get('body_position_estimate'):
                bodies, body_source = task['visible_bodies'], 'legacy_public_coordinates'
            created_at = metadata.get('created_at') or _archive_created_at(directory)
            for seq, event in enumerate(events, 1):
                event['seq'] = seq
            evaluation = _read(directory / 'evaluation.json', directory)
            environment_config = configuration.get('environment') or {}
            rules = rules_for(environment_config)
            return _public({'id': run_id, 'status': status, 'running': running, 'events': events,
                            'ruleset': rules['ruleset'], 'energy_budget': initial_energy(mission, environment_config),
                            'wall_seconds': environment_config.get('wall_seconds', rules['wall_seconds']),
                            'can_stop': owned, 'managed_externally': external,
                            'model_progress': _model_progress_projection(directory, running),
                            **_world_fields(configuration, metadata),
                            'limits': {k: model_config.get(k) for k in ('max_calls', 'max_total_tokens', 'max_tokens_per_call')},
                            'model_settings': {k: model_config.get(k) for k in
                                               ('model', 'thinking', 'reasoning_effort', 'reasoning_protocol', 'deployment')},
                            'code_enabled': configuration.get('code_execution', {}).get('enabled', False),
                            'experiment_id': metadata.get('experiment_id'), 'experiment_arm': metadata.get('arm'),
                            'budget': budget, 'model_usage': usage, 'records': records, 'result': result,
                            'validation_order': validation_order_info,
                            'evaluation': evaluation,
                            **_evaluation_projection(directory, status, metadata, configuration, mission, evaluation, running),
                            'evaluation_status': metadata.get('evaluation_status'), 'finished_at': metadata.get('finished_at'),
                            'mode': mode, 'created_at': created_at,
                            'bodies': bodies, 'body_position_estimate': body_estimate,
                            'body_position_source': body_source,
                            'body_observation_protocol': body_protocol})
