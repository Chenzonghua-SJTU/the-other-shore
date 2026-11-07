"""Observation adapter; communicates with the backend only over JSON stdio.

No simulator imports, source reads, or arbitrary method execution exist here.
"""
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import threading


class Environment:
    def __init__(self, config=None, *, secret_env_names=()):
        config = dict(config or {})
        from shore_rules import rules_for
        self.ruleset = rules_for(config)['ruleset']
        from shore_worlds import V15_PHYSICS_VERSION, validate_world_set
        world_set = validate_world_set(config.pop('world_set',
            'v15' if config.get('physics_version') == V15_PHYSICS_VERSION else 'legacy'))
        worker = 'shore_v15.worker' if world_set == 'v15' else 'shore_sim.worker'
        if world_set == 'legacy' and config.get('physics_version') == V15_PHYSICS_VERSION:
            raise ValueError('world_set and physics_version disagree')
        if world_set == 'legacy':
            if self.ruleset in ('e4', 'e5'):
                raise ValueError(self.ruleset.upper() + ' requires v15 worlds')
            config.pop('ruleset', None)
            config.pop('energy_budget', None)
        if world_set == 'v15':
            # Legacy config templates carry rectangular-domain defaults. The
            # public v15 catalog owns these ranges and noise reference scales.
            for field in ('reference_length', 'position_limit', 'velocity_limit', 'time_max'):
                config.pop(field, None)
        env = dict(os.environ)
        env.pop("DEEPSEEK_API_KEY", None)
        for name in secret_env_names:
            env.pop(name, None)
        project = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = project
        self._process = subprocess.Popen(
            [sys.executable, "-u", "-m", worker, json.dumps(config, allow_nan=False)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", env=env, cwd=project, bufsize=1)
        self._lock = threading.Lock()
        self._last_budget = None
        self._closed = False
        self._rpc_timeout = 30.0

    def request(self, method, **kwargs):
        with self._lock:
            if self._closed or self._process.poll() is not None:
                return self._failure("environment_closed", "The observation process is unavailable")
            try:
                payload = json.dumps({"method": method, "kwargs": kwargs}, allow_nan=False)
                self._process.stdin.write(payload + "\n")
                self._process.stdin.flush()
                with selectors.DefaultSelector() as selector:
                    selector.register(self._process.stdout, selectors.EVENT_READ)
                    ready = selector.select(timeout=self._rpc_timeout)
                if not ready:
                    self._process.kill()
                    self._process.wait(timeout=5)
                    return self._failure("environment_timeout", "Observation process exceeded its response timeout")
                line = self._process.stdout.readline()
                if not line:
                    return self._failure("environment_closed", "The observation process returned no response")
                response = json.loads(line)
                if not isinstance(response, dict) or set(response) != {"ok", "data", "error", "budget"}:
                    return self._failure("environment_protocol", "The observation process returned an invalid envelope")
                self._last_budget = response["budget"]
                return response
            except (OSError, ValueError, TypeError):
                return self._failure("environment_protocol", "The observation request could not be exchanged")

    def _failure(self, code, message):
        return {"ok": False, "data": None, "error": {"code": code, "message": message}, "budget": self._last_budget}

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._process.stdin:
            self._process.stdin.close()
        try:
            self._process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        if self._process.stdout:
            self._process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
