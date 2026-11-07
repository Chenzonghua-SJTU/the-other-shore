"""Human entry point. No extra model-facing tools."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from .environment import Environment
from .runtime import ToolRuntime
from .model import model_auth_config
from shore_worlds import (DEFAULT_WORLD_ID, WORLD_IDS, WORLD_SETS, CURRENT_PHYSICS_VERSION,
                          V15_PHYSICS_VERSION, validate_world_id, validate_world_set)


def main():
    parser = argparse.ArgumentParser(description='Local physical-law discovery prototype')
    parser.add_argument('command', choices=['run', 'inspect', 'init'])
    parser.add_argument('--config', default='config.example.json')
    parser.add_argument('--run-dir')
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--world', choices=WORLD_IDS, help='World to explore; defaults to the configured world or world_1')
    parser.add_argument('--world-set', choices=WORLD_SETS, help='legacy: original seven worlds; v15: six development worlds')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    environment = config.setdefault('environment', {})
    if args.world_set is not None:
        environment['world_set'] = args.world_set
        environment['physics_version'] = (V15_PHYSICS_VERSION if args.world_set == 'v15'
                                          else CURRENT_PHYSICS_VERSION)
    if args.world is not None:
        config.setdefault('environment', {})['world_id'] = args.world
    try:
        world_set = validate_world_set(environment.get('world_set',
            'v15' if environment.get('physics_version') == V15_PHYSICS_VERSION else 'legacy'))
        validate_world_id(environment.get('world_id', DEFAULT_WORLD_ID), world_set)
    except ValueError:
        parser.error('Unknown environment.world_id; choose ' + ', '.join(WORLD_IDS))
    if world_set == 'v15':
        environment['world_set'] = world_set
        environment['physics_version'] = V15_PHYSICS_VERSION
        for field in ('reference_length', 'position_limit', 'velocity_limit', 'time_max'):
            environment.pop(field, None)
    if args.noise:
        config['environment']['noise'] = True
    from shore_rules import apply_rules
    config = apply_rules(config)
    if config['environment'].get('ruleset') in ('e4', 'e5') and world_set != 'v15':
        parser.error(config['environment']['ruleset'].upper() + ' requires v15 worlds')
    if args.command == 'inspect':
        if not args.run_dir:
            parser.error('inspect requires --run-dir')
        directory = Path(args.run_dir)
        for name in ('result.json', 'evaluation.json'):
            if (directory / name).exists():
                print((directory / name).read_text(encoding='utf-8'))
        print('Records:', {p.name: len(list(p.glob('*.json'))) for p in (directory / 'records').glob('*') if p.is_dir()})
        return
    directory = Path(args.run_dir or ('runs/live-' + datetime.now().strftime('%Y%m%d-%H%M%S')))
    if directory.exists() and any(directory.iterdir()):
        parser.error('Use a new empty run directory; sessions cannot silently reset or resume')
    directory.mkdir(parents=True, exist_ok=True)
    # Persist the effective settings even when starting from an older config.
    config.setdefault('model', {}).update(max_calls=None, main_max_calls=None)
    config.setdefault('agent', {}).pop('max_turns', None)
    config['agent'].pop('max_rounds', None)
    (directory / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    key_env, key_required = model_auth_config(config['model'])
    if args.command == 'run' and config['environment'].get('ruleset') in ('e4', 'e5'):
        # Reuse the same process watchdog, draft sealing and separate evaluator
        # as web/batch E4/E5. No parent-side mission call can start a second clock.
        child_env = dict(os.environ)
        for name in ('DEEPSEEK_API_KEY', 'DASHSCOPE_API_KEY', 'OPENAI_API_KEY', key_env):
            child_env.pop(name, None)
        process = subprocess.run([sys.executable, '-m', 'shore_agent.web_worker',
                                  '--run-dir', str(directory.resolve())],
            input=json.dumps({'api_key': os.environ.get(key_env, '')}), text=True,
            cwd=Path(__file__).resolve().parents[1], env=child_env)
        print('Run directory:', directory.resolve())
        if process.returncode:
            raise SystemExit(process.returncode)
        return
    env = Environment(config['environment'], secret_env_names=(key_env,))
    try:
        runtime = ToolRuntime(env, directory, config=config.get('numerics'), code_config=config.get('code_execution'))
        mission = runtime.mission()
        if not mission['ok']:
            raise RuntimeError(mission['error'])
        if args.command == 'init':
            print(json.dumps(mission, ensure_ascii=False, indent=2))
            result = {'status': 'initialized_only', 'reason': 'No model called; first-light records saved.'}
        elif key_required and not os.environ.get(key_env):
            result = {'status': 'incomplete', 'reason': key_env + ' is absent; real API run not verified.',
                      'real_api_attempted': False, 'submission': None}
            print(result['reason'])
        else:
            from .model import DeepSeekClient
            from .agent import Agent
            client = DeepSeekClient(config['model'], directory,
                                    remaining_seconds=lambda: runtime.budget()['wall_seconds_remaining'])
            try:
                result = Agent(client, runtime, directory, config.get('agent', {})).run()
            finally:
                client.close()
        (directory / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Run directory:', directory.resolve())
    finally:
        env.close()


if __name__ == '__main__':
    main()
