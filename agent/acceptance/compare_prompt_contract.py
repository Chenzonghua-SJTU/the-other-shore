"""Compare actual first requests and schemas against a registered BF2 round."""
from pathlib import Path
import hashlib
import json
import sys


def read_contract(output):
    with (output/'model_messages.jsonl').open() as stream:
        first = json.loads(next(stream))
    prompt = next(m['content'] for m in first['messages'] if m['role'] == 'system')
    provenance = json.loads((output/'runtime-provenance.json').read_text())
    return {'system_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'schemas_sha256': provenance['tool_schemas_sha256']}


def compare(experiment, baseline):
    rows = []
    for folder in sorted((experiment/'R01').glob('world-*')):
        actual = read_contract(folder/'agent-output')
        reference = read_contract(baseline/'R07'/folder.name/'agent-output')
        rows.append({'world': folder.name, 'actual': actual, 'baseline': reference,
                     'system_matches': actual['system_sha256'] == reference['system_sha256'],
                     'schemas_match': actual['schemas_sha256'] == reference['schemas_sha256']})
    result = {'rows': rows, 'all_six_match': len(rows) == 6 and all(r['system_matches'] and r['schemas_match'] for r in rows),
              'scope': 'Actual first-request system message and runtime schemas; later trajectories and tool results may differ.'}
    (experiment/'prompt-contract-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(compare(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
