"""Bounded model-facing views; full scientific records are kept separately."""
import json


def input_hints(store, kinds, ids=None):
    """Actual acquired record layouts, grouped to avoid repeating identical schemas."""
    import numpy as np
    groups = {}
    for kind in kinds:
        for rid, record in store.records[kind].items():
            if ids is not None and rid not in ids:
                continue
            fields = {}
            for key, value in record.items():
                info = {'type': type(value).__name__}
                if isinstance(value, list):
                    try:
                        info['shape'] = list(np.asarray(value).shape)
                    except ValueError:
                        info.update(shape=None, ragged=True)
                elif isinstance(value, dict):
                    info['keys'] = list(value)
                fields[key] = info
            signature = json.dumps([kind, fields], sort_keys=True)
            group = groups.setdefault(signature, {'record_type': kind, 'record_ids': [], 'fields': fields})
            group['record_ids'].append(rid)
            from .workspace import FOLDERS
            group.setdefault('paths', []).append(f'{FOLDERS[kind]}/{rid}.json')
    return list(groups.values())


def size(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False))


def compact(value, budget=8000):
    view, cut = _compact(value, budget)
    if size(view) > budget:
        return {'truncated': True, 'type': type(value).__name__,
                'total_items':len(value) if isinstance(value, (dict,list,str)) else None}, True
    return view, cut


def _compact(value, budget):
    if size(value) <= budget:
        return value, False
    if isinstance(value, dict):
        shown, omitted, left = {}, [], max(0, budget-400)
        for key, item in value.items():
            if left < 150:
                omitted.append(key)
                continue
            view, shortened = compact(item, min(left, max(300, budget//3)))
            cost = size({key: view})
            if cost > left:
                omitted.append(key)
            else:
                shown[key] = view
                left -= cost
        return {'preview': shown, 'total_keys': len(value), 'omitted_keys': omitted[:30],
                'omitted_key_count': len(omitted), 'truncated': True}, True
    if isinstance(value, list):
        width = max(1, min(3, budget//500))
        each = max(80, (budget-250)//(2*width))
        return {'first': [compact(v, each)[0] for v in value[:width]],
                'last': [compact(v, each)[0] for v in value[-width:]] if len(value)>width else [],
                'total_items': len(value), 'truncated': True}, True
    if isinstance(value, str):
        return {'preview_text': value[:max(0,budget-150)], 'total_chars': len(value), 'truncated': True}, True
    return {'truncated': True, 'type': type(value).__name__}, True


def execution_preview(record):
    """No code/schema/provenance echo. Small comparison tables remain complete."""
    result, truncated = compact(record.get('result'))
    error = record.get('error')
    if error:
        message = error.get('message') or (error.get('traceback') or '').splitlines()[-1:]
        if isinstance(message,list):
            message = message[0] if message else ''
        layout, cut = compact(error.get('input_layout', {}), 1500)
        error = {'type':str(error.get('type', 'error'))[:100], 'message':str(message)[:1000],
                 'line':error.get('line'), 'source_line':str(error.get('source_line') or '')[:500],
                 'input_layout':layout}
        truncated |= cut
    data = {key: record.get(key) for key in ('execution_id', 'status', 'elapsed_seconds', 'artifacts')}
    data.update(result=result, error=error)
    data['warnings'], cut = compact(record.get('warnings', []), 1000)
    truncated |= cut
    for field in ('stdout','stderr'):
        data[field], cut = compact(record.get(field, ''), 1500)
        truncated |= cut
    data['truncated'] = truncated
    data['retrieve'] = {'tool':'read_file', 'path':f'analyses/{record["execution_id"]}/result.json', 'offset':0}
    data['meaning'] = 'Computed from acquired data; execution success is not a scientific correctness verdict.'
    return data


def select_field(record, path, offset=0, limit=20):
    value = record
    for part in path:
        if isinstance(value, dict) and isinstance(part, str) and part in value:
            value = value[part]
        elif isinstance(value, list) and type(part) is int and 0 <= part < len(value):
            value = value[part]
        else:
            raise ValueError('Unknown field_path; use existing dictionary keys or valid list indexes')
    if isinstance(value, (list, str)):
        total = len(value)
        chunk, cut = compact(value[offset:offset+limit])
        return {'value':chunk,'total_items':total,'offset':offset,'limit':limit,
                'truncated':cut or offset > 0 or offset+limit < total}
    if offset:
        raise ValueError('offset is only supported for lists and strings')
    chunk, cut = compact(value)
    return {'value':chunk,'truncated':cut}
