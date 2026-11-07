"""Docker-only execution. Never falls back to running model Python on the host."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from jsonschema import Draft202012Validator

IMAGE = 'shore-code:e2-r04'
MAX_RESPONSE = 14 * 1024 * 1024
MAX_INPUT = 24 * 1024 * 1024
ARTIFACT_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}')
MIMES = {'.png': 'image/png', '.txt': 'text/plain; charset=utf-8',
         '.json': 'application/json', '.csv': 'text/csv; charset=utf-8'}
SCHEMA_KEYS = {'type', 'properties', 'required', 'items', 'additionalProperties', 'enum',
               'minimum', 'maximum', 'minItems', 'maxItems', 'minLength', 'maxLength', 'description'}


def validate_output_schema(schema):
    if not isinstance(schema, dict) or len(json.dumps(schema)) > 16000:
        raise ValueError('output_schema must be an object of at most 16000 characters')

    def check(node, depth=0):
        if depth > 12 or not isinstance(node, (dict, bool)):
            raise ValueError('output_schema too deep or invalid')
        if isinstance(node, bool):
            return
        if set(node) - SCHEMA_KEYS:
            raise ValueError('unsupported output_schema keyword; references/patterns are not allowed')
        for item in node.get('properties', {}).values():
            check(item, depth + 1)
        for key in ('items', 'additionalProperties'):
            if key in node:
                check(node[key], depth + 1)

    Draft202012Validator.check_schema(schema)
    check(schema)


def docker_command(config):
    docker = shutil.which('docker')
    if not docker:
        raise ValueError('code_sandbox_unavailable: Docker CLI missing')
    context = config.get('docker_context')
    if context is None and sys.platform == 'darwin' and (Path.home() / '.colima/shore-code/docker.sock').exists():
        context = 'colima-shore-code'
    return [docker] + (['--context', context] if context else [])


def docker_env():
    # Host Docker transport needs only these settings; nothing is passed into the container.
    return {key: value for key, value in os.environ.items()
            if key in ('PATH', 'HOME', 'DOCKER_CONFIG', 'DOCKER_HOST', 'DOCKER_CONTEXT',
                       'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH', 'TMPDIR')}


def sandbox_info(config=None):
    config = config or {}
    try:
        command = docker_command(config)
        image = config.get('image', IMAGE)
        reply = subprocess.run(command + ['image', 'inspect', image, '--format', '{{.Id}}'],
                               capture_output=True, timeout=8, text=True, env=docker_env())
        if reply.returncode or not re.fullmatch(r'sha256:[0-9a-f]{64}', reply.stdout.strip()):
            return {'available': False, 'reason': '请启动 Docker 并构建 code_sandbox 镜像。'}
        return {'available': True, 'backend': 'docker', 'image_id': reply.stdout.strip()}
    except (ValueError, OSError, subprocess.SubprocessError):
        return {'available': False, 'reason': '代码执行环境未就绪；请启动 Docker。'}


def _decode(response, output_dir, schema):
    if not isinstance(response, dict) or response.get('status') not in ('success', 'error'):
        raise ValueError('invalid sandbox response')
    record = {key: response.get(key) for key in ('status', 'result', 'error', 'stdout', 'stderr',
              'compute_seconds', 'cpu_seconds_reported', 'peak_rss_kib_reported')}
    warnings = response.get('warnings', [])
    if not isinstance(warnings, list) or len(warnings) > 10 or any(not isinstance(w,str) or len(w)>2000 for w in warnings):
        raise ValueError('invalid sandbox warnings')
    record['warnings'] = warnings
    for key in ('stdout', 'stderr'):
        if not isinstance(record[key], str):
            raise ValueError('invalid sandbox log')
        record[key] = record[key][:18000]
    for key in ('compute_seconds', 'cpu_seconds_reported', 'peak_rss_kib_reported'):
        value = record[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid sandbox measurement')
    if len(json.dumps(record['result'], allow_nan=False).encode()) > 2 * 1024 * 1024:
        raise ValueError('sandbox result too large')
    if record['status'] == 'success':
        errors = list(Draft202012Validator(schema).iter_errors(record['result']))
        if errors:
            record.update(status='schema_error', error={'type': 'OutputSchemaError',
                          'message': errors[0].message[:2000], 'path': list(errors[0].path)})
    items = response.get('artifacts')
    if not isinstance(items, list) or len(items) > 10:
        raise ValueError('invalid artifacts')
    artifacts, total, names = [], 0, set()
    for item in items:
        name = item.get('name', '') if isinstance(item, dict) else ''
        suffix = Path(name).suffix
        if not ARTIFACT_NAME.fullmatch(name) or suffix not in MIMES or name in names:
            raise ValueError('invalid artifact name')
        content = base64.b64decode(item['base64'], validate=True)
        total += len(content)
        if total > 8 * 1024 * 1024:
            raise ValueError('artifacts too large')
        if suffix == '.png' and not content.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('invalid PNG artifact')
        names.add(name)
        artifacts.append({'name': name, 'size_bytes': len(content), 'mime': MIMES[suffix],
                          'sha256': hashlib.sha256(content).hexdigest()})
    # Validate the entire manifest before writing any files; host never extracts a tar.
    output_dir.mkdir(parents=True, exist_ok=False)
    for item in items:
        (output_dir / item['name']).write_bytes(base64.b64decode(item['base64'], validate=True))
    record['artifacts'] = artifacts
    return record


def execute_code(code, inputs, schema, artifact_dir, config, timeout_seconds, files=None):
    started = time.monotonic()
    name = 'shore-code-' + uuid.uuid4().hex
    command, env = docker_command(config), docker_env()
    info = sandbox_info(config)
    if not info['available']:
        raise ValueError('code_sandbox_unavailable')
    timeout = min(float(config.get('timeout_seconds', 90)), float(timeout_seconds))
    if timeout < 1:
        raise ValueError('code_execution_time_exhausted')
    memory = int(config.get('memory_mb', 1024))
    if not 128 <= memory <= 4096 or not 1 <= timeout <= 300:
        raise ValueError('invalid_code_execution_limits')
    payload = json.dumps({'code': code, 'inputs': inputs, 'files': files or {}}, allow_nan=False, ensure_ascii=False).encode()
    if len(payload) > MAX_INPUT:
        raise ValueError('code_input_too_large')
    # No bind mounts, host directories, sockets, credentials, ports or environment inheritance.
    args = command + ['run', '--name', name, '--pull=never', '--network=none', '--read-only',
        '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit=64',
        '--memory', f'{memory}m', '--memory-swap', f'{memory}m', '--cpus=1',
        '--user=65534:65534', '--ipc=private', '--shm-size=16m',
        '--tmpfs', '/work:rw,noexec,nosuid,nodev,size=128m,mode=1777',
        '--tmpfs', '/tmp:rw,noexec,nosuid,nodev,size=32m,mode=1777',
        '--log-driver=none', *[item for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
                                              'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy')
                              for item in ('--env', key + '=')],
        '-i', info['image_id'], str(math.ceil(timeout))]
    status, returncode, response = 'protocol_error', None, None
    process = None
    try:
        with tempfile.TemporaryFile() as input_file, tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            input_file.write(payload)
            input_file.seek(0)
            process = subprocess.Popen(args, stdin=input_file, stdout=stdout, stderr=stderr, env=env)
            deadline = started + timeout + 8  # bounded container startup/transport allowance
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    status = 'timeout'
                    break
                if os.fstat(stdout.fileno()).st_size + os.fstat(stderr.fileno()).st_size > MAX_RESPONSE:
                    status = 'output_limit'
                    break
                time.sleep(0.05)
            if process.poll() is None:
                subprocess.run(command + ['kill', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=8, env=env)
                process.kill()
            returncode = process.wait(timeout=8)
            stdout.seek(0)
            stderr.seek(0)
            raw, raw_error = stdout.read(MAX_RESPONSE + 1), stderr.read(18000).decode('utf-8', errors='replace')
            if status not in ('timeout', 'output_limit'):
                if len(raw) > MAX_RESPONSE:
                    status = 'output_limit'
                elif returncode == 0:
                    response = _decode(json.loads(raw), Path(artifact_dir), schema)
                    status = response['status']
                else:
                    inspect = subprocess.run(command + ['inspect', '--format', '{{.State.OOMKilled}}', name],
                                             capture_output=True, text=True, timeout=8, env=env)
                    status = 'memory_limit' if inspect.stdout.strip() == 'true' else 'timeout' if returncode == 142 else 'error'
            if response is None:
                response = {'status': status, 'result': None, 'stdout': '', 'stderr': raw_error,
                            'error': {'type': status, 'message': 'Execution did not return a valid result.'}, 'artifacts': []}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        response = {'status': 'protocol_error', 'result': None, 'stdout': '', 'stderr': '',
                    'error': {'type': type(exc).__name__, 'message': str(exc)[:2000]}, 'artifacts': []}
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=8)
        try:
            subprocess.run(command + ['rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=8, env=env)
        except (OSError, subprocess.SubprocessError):
            pass
    response.update(elapsed_seconds=time.monotonic() - started, exit_code=returncode,
                    sandbox={**info, 'memory_mb': memory, 'timeout_seconds': timeout, 'cpus': 1,
                             'network': 'none', 'host_mounts': False})
    return response
