"""E16 controller: freeze, warm two CPU workers, execute every registered pair."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .compare_sr_filter_fixed import save, sha

A = Path(__file__).resolve().parent
DRIVERS = ('compare_sr_filter_fixed.py', 'evaluate_sr_filter_fixed.py', 'run_sr_filter_fixed.py')


def freeze(output):
    expected = json.loads((A/'experiments/E16-fixed-sr/scientific-source.json').read_text())
    for name, digest in expected['files'].items():
        if sha(A/name) != digest:raise ValueError('Scientific source drift: '+name)
    destination = output/'submission/agent';destination.mkdir(parents=True)
    for folder in ['repository', 'pipeline_bridge']:
        shutil.copytree(A/folder, destination/folder, ignore=shutil.ignore_patterns('__pycache__', '.*', '*.pyc'))
    for name in ['__init__.py', 'repository-source.json', 'repository-config.json', *DRIVERS]:
        shutil.copy2(A/name, destination/name)
    # Runtime assets already installed by the environment, not redistributed or
    # installed by this experiment. Preserve their exact existing bundle path.
    (destination/'vendor_runtime').symlink_to((A/'vendor_runtime').resolve(), target_is_directory=True)
    hashes = {str(p.relative_to(destination)): sha(p) for p in destination.rglob('*') if p.is_file() and 'vendor_runtime' not in p.parts}
    save(output/'frozen-source-hashes.json', hashes)
    save(output/'scientific-source.json', expected)
    return destination, hashes


def summarize(output, rows):
    def metrics(row, name='selected'):
        evaluation = row.get('evaluation', {})
        chosen = evaluation.get(name, {})
        return {'score': chosen.get('score', {}).get('total', 0.),
                'prediction': chosen.get('score', {}).get('prediction', 0.),
                'correct': bool(chosen.get('recovery', {}).get('correct', False))}
    pairs = []
    for world in range(1, 7):
        for seed in [42, 43, 44]:
            found = {r['arm']: r for r in rows if r['world'] == world and r['seed'] == seed}
            if len(found) != 2:continue
            m = {arm: metrics(row) for arm, row in found.items()}
            pairs.append({'world': world, 'seed': seed, 'legacy': m['legacy'], 'joint': m['joint'],
                          'score_delta': m['joint']['score']-m['legacy']['score']})
    arms = {}
    for arm in ['legacy', 'joint']:
        part = [r for r in rows if r['arm'] == arm]
        values = [metrics(r) for r in part]
        arms[arm] = {'attempts': len(part), 'completed': sum(r['status'] == 'completed' for r in part),
                     'mean_total': sum(v['score'] for v in values)/len(part) if part else None,
                     'mean_prediction': sum(v['prediction'] for v in values)/len(part) if part else None,
                     'raw_training_selected_recovered': sum(v['correct'] for v in values),
                     'mean_original_sr_rank_one_total': sum(metrics(r, 'sr_rank_one')['score'] for r in part)/len(part) if part else None,
                     'mean_original_sr_rank_one_prediction': sum(metrics(r, 'sr_rank_one')['prediction'] for r in part)/len(part) if part else None,
                     'original_sr_rank_one_recovered': sum(metrics(r, 'sr_rank_one')['correct'] for r in part),
                     'top_three_coverage': sum(r.get('evaluation', {}).get('any_top_three_recovered', False) for r in part)}
    worlds = {}
    for world in range(1, 7):
        part = [p for p in pairs if p['world'] == world]
        if part:
            worlds[str(world)] = {'pairs': len(part),
                'legacy_mean_total': sum(p['legacy']['score'] for p in part)/len(part),
                'joint_mean_total': sum(p['joint']['score'] for p in part)/len(part),
                'mean_delta': sum(p['score_delta'] for p in part)/len(part),
                'legacy_recovered': sum(p['legacy']['correct'] for p in part),
                'joint_recovered': sum(p['joint']['correct'] for p in part)}
    report = {'experiment': 'E16-fixed-sr', 'completed_attempts': len(rows), 'registered_attempts': 36,
              'worlds': worlds,
              'arms': arms, 'paired_results': pairs, 'trials': rows,
              'scope': 'Fixed-observation SR ablation. No model calls; fixed original claims/energy; not autonomous discovery.'}
    save(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True);args = parser.parse_args()
    output = Path(args.output).resolve();output.mkdir(parents=True, exist_ok=False)
    registration = json.loads((A/'experiments/E16-fixed-sr/registration.json').read_text())
    registration.update(status='registered', baseline_absolute=str(Path(args.baseline).resolve()),
                        registered_at_unix=time.time())
    save(output/'registration.json', registration)
    frozen, hashes = freeze(output)
    from .benchmark_official import fixed_hashes
    official_before = fixed_hashes();save(output/'official-files-before.json', official_before)
    rows = []
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       PYTHONPATH=str(frozen.parent)+os.pathsep+str(A.parent)+os.pathsep+os.environ.get('PYTHONPATH', ''))
    # Clear inherited session credentials; trials replay records without Console.
    environment.pop('RIFT_SESSION_TOKEN', None);environment.pop('RIFT_CONSOLE_URL', None)
    def pair(world, seed, pair_index, endpoint):
        found = []
        arms = ['legacy', 'joint'] if pair_index % 2 == 0 else ['joint', 'legacy']
        for arm in arms:
            directory = output/f'world-{world}'/f'seed-{seed}'/arm
            log = output/'logs'/f'world-{world}-seed-{seed}-{arm}.log';log.parent.mkdir(exist_ok=True)
            command = [sys.executable, '-B', '-m', 'agent.compare_sr_filter_fixed', '--baseline', args.baseline,
                       '--output', str(directory), '--endpoint', endpoint, '--world', str(world), '--seed', str(seed), '--arm', arm]
            print('START', world, seed, arm, flush=True)
            try:
                with log.open('w') as stream:
                    process = subprocess.run(command, cwd=frozen.parent, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
                row = json.loads((directory/'trial.json').read_text())
                row['process_exit_code'] = process.returncode
            except Exception as exc:
                row = {'world': world, 'seed': seed, 'arm': arm, 'status': 'failed', 'error_type': type(exc).__name__}
            found.append(row)
            score = row.get('evaluation', {}).get('selected', {}).get('score', {}).get('total')
            print('DONE', world, seed, arm, row['status'], score, flush=True)
        return found
    from .pipeline_bridge.service import symbolic_service
    try:
        with symbolic_service(output, size=2, timeout=900., trajectory_filter=True) as endpoint:
            save(output/'pool.json', {'endpoint': endpoint, 'size': 2})
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(pair, world, seed, (world-1)*3+i, endpoint)
                           for world in range(1, 7) for i, seed in enumerate([42, 43, 44])]
                for future in as_completed(futures):
                    rows.extend(future.result());summarize(output, rows)
    finally:
        report = summarize(output, rows)
        integrity = {'scientific_source_unchanged': all(sha(frozen/p) == digest for p, digest in hashes.items()),
                     'official_files_unchanged': fixed_hashes() == official_before,
                     'actual_attempts': len(rows), 'registered_attempts': 36}
        save(output/'integrity.json', integrity)
        print('FINAL', json.dumps({'arms': report['arms'], 'integrity': integrity}), flush=True)


if __name__ == '__main__':main()
