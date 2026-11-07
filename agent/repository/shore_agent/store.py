"""Append-only scientific records. Public workspace paths never become arbitrary host paths."""
import copy
import json
import re
import uuid
from pathlib import Path

TYPES = ('body_observation', 'probe', 'law', 'validation', 'residual', 'submission', 'code_execution', 'symbolic_search')


def preview(value):
    if isinstance(value, dict):
        return {k: preview(v) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > 6:
            return {'preview': [preview(v) for v in value[:3]], 'total_items': len(value),
                    'truncated': True, 'retrieve': 'read_file(path=the returned workspace path, offset=0)'}
        return [preview(v) for v in value]
    return value


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records = {kind: {} for kind in TYPES}

    def add(self, kind, data, record_id=None):
        if kind not in TYPES:
            raise ValueError('Unknown record type')
        rid = record_id or f'{kind}_{uuid.uuid4().hex[:12]}'
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', rid):
            raise ValueError('Invalid record identifier')
        if rid in self.records[kind]:
            if self.records[kind][rid] != data:
                raise ValueError('Records are immutable')
            return rid
        path = self.directory / kind
        path.mkdir(exist_ok=True)
        saved = copy.deepcopy(data)
        with (path / f'{rid}.json').open('x', encoding='utf-8') as f:
            json.dump(saved, f, ensure_ascii=False, indent=2, allow_nan=False)
        self.records[kind][rid] = saved
        return rid

    def get(self, kind, rid):
        if kind not in TYPES or rid not in self.records[kind]:
            raise ValueError(f'No acquired {kind} record: {rid}')
        return copy.deepcopy(self.records[kind][rid])

    def query(self, record_type='all', record_ids=None, detail='summary'):
        kinds = TYPES if record_type == 'all' else (record_type,)
        if any(k not in TYPES for k in kinds) or detail not in ('full', 'summary'):
            raise ValueError('Invalid archive query')
        if detail == 'full' and not record_ids:
            raise ValueError('Full records require explicit record_ids')
        found = []
        for kind in kinds:
            for rid, value in self.records[kind].items():
                if not record_ids or rid in record_ids:
                    found.append({'record_type': kind, 'record_id': rid,
                                  'content': copy.deepcopy(value) if detail == 'full' else preview(value)})
        if record_ids and set(record_ids) - {r['record_id'] for r in found}:
            raise ValueError('Some requested records have not been acquired')
        return found


def append_event(path, event):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
