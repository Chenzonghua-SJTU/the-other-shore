"""Summarize registered trials and verified filtering, without dropping failures."""
from pathlib import Path
import collections
import json
import sys


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def summarize(directory):
    directory = Path(directory)
    report = json.loads((directory/'report.json').read_text())
    rows = []
    for row in report['summary_rows']:
        output = directory/row['repeat']/('world-'+str(row['world']))/'agent-output'
        tools = records(output/'tools.jsonl')
        receipts = records(output/'filter-consumption.jsonl')
        automatic = records(output/'filter-automatic.jsonl')
        artifacts = [json.loads(p.read_text()) for p in (output/'trajectory-filters').glob('*.json')]
        per_artifact = {a['preprocessing_id']: sum(p['status'] == 'filtered' for p in a['per_probe']) for a in artifacts}
        actual = sum(per_artifact.values())
        counts = collections.Counter(t['tool'] for t in tools)
        rows.append({**{k: v for k, v in row.items() if k != 'directory'},
                     'filter_calls': counts['denoise_trajectories'],
                     'automatic_calls': len(automatic),
                     'automatic_filter_calls': sum(a.get('filter_calls',0) for a in automatic),
                     'automatic_preprocessing_seconds': sum(a.get('preprocessing_seconds',0.) for a in automatic),
                     'automatic_failed_calls': sum(a.get('status') == 'failed' for a in automatic),
                     'symbolic_calls': counts['symbolic_regression'],
                     'filtered_probe_artifacts': actual,
                     'verified_searches': sum(r.get('worker_consumption_verified') is True for r in receipts),
                     'unverified_searches': sum(r.get('worker_consumption_verified') is not True for r in receipts),
                     'effective_filtered_searches': sum(r.get('worker_consumption_verified') is True and per_artifact.get(r['preprocessing_id'], 0) > 0 for r in receipts),
                     'tool_error_codes': dict(collections.Counter(str((t.get('result', {}).get('error') or {}).get('code')) for t in tools if not t.get('result', {}).get('ok')))})
    result = {k: report[k] for k in ['baseline_complete', 'source_integrity', 'official_files_unchanged', 'overall_mean', 'per_world']}
    result.update(experiment=directory.name, rows=rows,
                  success_count=sum(r['success'] for r in rows),
                  filter_calls=sum(r['filter_calls'] for r in rows),
                  automatic_calls=sum(r['automatic_calls'] for r in rows),
                  automatic_filter_calls=sum(r['automatic_filter_calls'] for r in rows),
                  automatic_failed_calls=sum(r['automatic_failed_calls'] for r in rows),
                  automatic_preprocessing_seconds=sum(r['automatic_preprocessing_seconds'] for r in rows),
                  verified_searches=sum(r['verified_searches'] for r in rows),
                  unverified_searches=sum(r['unverified_searches'] for r in rows),
                  effective_filtered_searches=sum(r['effective_filtered_searches'] for r in rows))
    (directory/'filter-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


if __name__ == '__main__':
    result = summarize(sys.argv[1])
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'per_world')}, ensure_ascii=False))
