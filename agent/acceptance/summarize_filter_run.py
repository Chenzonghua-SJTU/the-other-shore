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
        artifacts = [json.loads(p.read_text()) for p in (output/'trajectory-filters').glob('*.json')]
        actual = sum(p['status'] == 'filtered' for a in artifacts for p in a['per_probe'])
        counts = collections.Counter(t['tool'] for t in tools)
        rows.append({**{k: v for k, v in row.items() if k != 'directory'},
                     'filter_calls': counts['denoise_trajectories'],
                     'symbolic_calls': counts['symbolic_regression'],
                     'filtered_probe_artifacts': actual,
                     'verified_searches': sum(r.get('worker_consumption_verified') is True for r in receipts),
                     'unverified_searches': sum(r.get('worker_consumption_verified') is not True for r in receipts),
                     'tool_error_codes': dict(collections.Counter(str((t.get('result', {}).get('error') or {}).get('code')) for t in tools if not t.get('result', {}).get('ok')))})
    result = {k: report[k] for k in ['baseline_complete', 'source_integrity', 'official_files_unchanged', 'overall_mean', 'per_world']}
    result.update(experiment=directory.name, rows=rows,
                  success_count=sum(r['success'] for r in rows),
                  filter_calls=sum(r['filter_calls'] for r in rows),
                  verified_searches=sum(r['verified_searches'] for r in rows),
                  unverified_searches=sum(r['unverified_searches'] for r in rows))
    (directory/'filter-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


if __name__ == '__main__':
    result = summarize(sys.argv[1])
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'per_world')}, ensure_ascii=False))
