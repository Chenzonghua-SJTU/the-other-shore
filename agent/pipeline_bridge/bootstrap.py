from pathlib import Path
import sys,hashlib,json

def repository():
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'repository'))
    manifest=json.loads((root/'repository-source.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((root/'repository/shore_agent'/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Repository source integrity mismatch: '+name)
    for name,digest in manifest.get('support_files',{}).items():
        if hashlib.sha256((root/'repository'/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Repository support integrity mismatch: '+name)
    from ..repository import shore_agent
    return shore_agent
