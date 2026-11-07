"""One autonomous discovery session per OS process; key arrives over stdin only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from .web_jobs import _now, _read, _write, accepted_submission


class ExplorationDeadline(BaseException):
    """Bypass model/tool Exception handlers, so an expired session cannot continue."""


@contextmanager
def exploration_watchdog(seconds, *, enabled):
    if not enabled:
        yield
        return
    def expired(_signum, _frame):
        raise ExplorationDeadline()
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run(directory):
    directory = Path(directory)
    environment = client = None
    key_env = 'DEEPSEEK_API_KEY'
    try:
        # The server bounds the field; bound the local protocol as defense in depth.
        envelope = json.loads(sys.stdin.readline(8192))
        key = envelope.pop('api_key', None)
        config = _read(directory / 'config.json', directory, {})
        from shore_rules import apply_rules
        config = apply_rules(config)
        e4 = config['environment'].get('ruleset') in ('e4', 'e5')
        from .model import DeepSeekClient, model_auth_config
        key_env, key_required = model_auth_config(config.get('model', {}))
        if key_required and (not isinstance(key, str) or not key):
            _write(directory / 'result.json', {'status': 'incomplete', 'reason': 'missing_api_key', 'submission': None})
            return 2
        if isinstance(key, str) and key:
            os.environ[key_env] = key
        else:
            os.environ.pop(key_env, None)
        del key, envelope
        from .agent import Agent
        from .environment import Environment
        from .runtime import ToolRuntime
        from .store import append_event
        environment = Environment(config['environment'], secret_env_names=(key_env,))
        runtime = ToolRuntime(environment, directory, config=config.get('numerics'), code_config=config.get('code_execution'))
        started = time.monotonic()
        started_at = datetime.now(timezone.utc)
        mission = runtime.mission()
        remaining = (mission.get('budget') or {}).get('wall_seconds_remaining', 300)
        deadline = started + min(300, remaining) if e4 else None
        _metadata(directory, session_started_at=started_at.isoformat(), initial_budget=mission.get('budget') or {})
        if e4:
            runtime.exploration_started_monotonic = started
            runtime.exploration_deadline_monotonic = deadline
            _metadata(directory, exploration_deadline_at=(started_at + timedelta(seconds=deadline-started)).isoformat())
        append_event(directory / 'events.jsonl', {'event': 'session_started', 'budget': mission.get('budget')})
        client = DeepSeekClient(config['model'], directory,
            remaining_seconds=lambda: max(0, runtime.budget()['wall_seconds_remaining'] - (5 if e4 else 0)))
        agent = Agent(client, runtime, directory, config.get('agent', {}))
        try:
            with exploration_watchdog(deadline-time.monotonic() if e4 else 0, enabled=e4):
                result = agent.run()
        except ExplorationDeadline:
            result = agent._finish('incomplete', 'wall_time_exhausted_without_saved_draft')
        finished_exploration = time.monotonic()
        submission = result.get('submission') or {}
        _metadata(directory, exploration_finished_at=_now(),
                  exploration_wall_seconds=finished_exploration-started,
                  submission_deadline_met=submission.get('submission_deadline_met'),
                  submission_elapsed_seconds=submission.get('pair_validated_elapsed_seconds'))
        client.close()
        client = None
        environment.close()
        environment = None
        os.environ.pop('DEEPSEEK_API_KEY', None)
        os.environ.pop(key_env, None)
        # The main Agent has returned and its simulator has closed. Only now may
        # a separate, key-free evaluator access hidden initial conditions.
        if result.get('status') == 'submitted' and accepted_submission(directory):
            evaluation_started = time.monotonic()
            _metadata(directory, status='evaluating', evaluation_started_at=_now())
            project = Path(__file__).resolve().parents[1]
            try:
                evaluation = subprocess.run([sys.executable, str(project / 'evaluate.py'), '--run-dir', str(directory)],
                                            cwd=project, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL, timeout=180)
                _metadata(directory, evaluation_status='complete' if evaluation.returncode == 0 else 'failed')
            finally:
                _metadata(directory, evaluation_finished_at=_now(),
                          evaluation_wall_seconds=time.monotonic()-evaluation_started)
        _metadata(directory, status=result.get('status', 'incomplete'), finished_at=_now())
        return 0
    except Exception:
        # No raw exception/traceback can disclose request headers or credentials.
        if not accepted_submission(directory):
            _write(directory / 'result.json', {'status': 'error', 'reason': 'worker_failed', 'submission': None})
            _metadata(directory, status='error', finished_at=_now())
        else:
            _metadata(directory, status='submitted', evaluation_status='failed', finished_at=_now())
        return 1
    finally:
        os.environ.pop('DEEPSEEK_API_KEY', None)
        os.environ.pop(key_env, None)
        for resource in (client, environment):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    pass


def _metadata(directory, **updates):
    value = _read(directory / 'web.json', directory, {})
    value.update(updates)
    _write(directory / 'web.json', value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    raise SystemExit(run(Path(args.run_dir)))


if __name__ == '__main__':
    main()
