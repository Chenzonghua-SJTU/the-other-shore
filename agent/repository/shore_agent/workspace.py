"""Current-session public files. Paths are registry keys, never host paths."""
import hashlib
import json
import re
from pathlib import Path

FOLDERS = {'probe': 'observations', 'body_observation': 'observations', 'law': 'laws',
           'validation': 'validations', 'residual': 'residuals', 'submission': 'submissions',
           'symbolic_search': 'analyses'}


def path_key(path, directory=False):
    if not isinstance(path, str) or len(path) > 300 or path.startswith('/') or '\\' in path:
        raise ValueError('Invalid workspace path')
    if directory:
        path = path.rstrip('/')
    if not path and directory:
        return ''
    if not path or any(p in ('', '.', '..') or not re.fullmatch(r'[A-Za-z0-9_.-]+', p) for p in path.split('/')):
        raise ValueError('Use a relative current-session workspace path; no absolute paths or traversal')
    return path


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf-8')


class SessionFiles:
    def __init__(self, store, run_dir):
        self.store, self.run_dir = store, Path(run_dir)

    def entries(self):
        entries = {}
        for kind, folder in FOLDERS.items():
            for rid, value in self.store.records[kind].items():
                entries[f'{folder}/{rid}.json'] = ('json', value)
        for rid, record in self.store.records['code_execution'].items():
            prefix = f'analyses/{rid}/'
            entries[prefix + 'record.json'] = ('json', record)
            entries[prefix + 'result.json'] = ('json', record.get('result'))
            for filename, field in [('analysis.py', 'code'), ('stdout.txt', 'stdout'), ('stderr.txt', 'stderr')]:
                entries[prefix + filename] = ('text', record.get(field) or '')
            for artifact in record.get('artifacts', []):
                name = artifact['name']
                if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', name):
                    raise ValueError('Invalid archived artifact name')
                entries[prefix + 'artifacts/' + name] = ('artifact', (rid, name, artifact['sha256']))
        return entries

    def read_bytes(self, path):
        key = path_key(path)
        item = self.entries().get(key)
        if item is None:
            raise ValueError(f'No acquired file {key!r} in this session; use list_files')
        kind, value = item
        if kind == 'json':
            return json_bytes(value)
        if kind == 'text':
            return value.encode('utf-8')
        rid, name, expected = value
        path = self.run_dir / 'code_executions' / rid / 'artifacts' / name
        # Only allowlisted output artifacts, never arbitrary files in the run folder.
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
            raise ValueError('Artifact missing or symlinked')
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError('Artifact too large')
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('Artifact integrity check failed')
        return content

    def list_files(self, path='', offset=0, limit=100):
        directory = path_key(path, directory=True)
        prefix = directory + '/' if directory else ''
        children = {}
        for key in self.entries():
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            first, sep, _ = rest.partition('/')
            child = prefix + first + ('/' if sep else '')
            children[child] = {'path': child, 'type': 'directory' if sep else 'file'}
        if directory and not children:
            raise ValueError('No acquired directory at this path')
        ordered = [children[key] for key in sorted(children)]
        end = min(offset + limit, len(ordered))
        return {'path': directory, 'entries': ordered[offset:end], 'total': len(ordered),
                'next_offset': end if end < len(ordered) else None}

    def read_file(self, path, offset=0, limit=12000):
        raw = self.read_bytes(path)
        try:
            content = raw.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError('Binary file: pass this path to run_code(files=[...]) and use read_bytes(path)')
        end = min(offset + limit, len(content))
        return {'path': path, 'content': content[offset:end], 'offset': offset,
                'total_chars': len(content), 'next_offset': end if end < len(content) else None,
                'truncated': offset > 0 or end < len(content),
                'sha256': hashlib.sha256(raw).hexdigest()}
