"""Separate model code errors, resource limits and backend availability."""
from pathlib import Path
import json,collections,sys
E=Path(sys.argv[1]);rows=[]
for p in sorted(E.glob('R*/world-*/result.json')):
    result=json.loads(p.read_text());a=p.parent/'agent-output'
    tools=[json.loads(s) for s in (a/'tools.jsonl').read_text().splitlines()] if (a/'tools.jsonl').exists() else []
    events=[json.loads(s) for s in (a/'model_events.jsonl').read_text().splitlines()] if (a/'model_events.jsonl').exists() else []
    messages=[json.loads(s) for s in (a/'model_messages.jsonl').read_text().splitlines()] if (a/'model_messages.jsonl').exists() else []
    code_turns={m['call_number'] for m in messages if any(t.get('function',{}).get('name')=='run_code' for t in (m.get('assistant') or {}).get('tool_calls',[]))}
    requests=[e for e in events if e.get('event')=='request']
    calls=[r for r in tools if r['tool']=='run_code'];statuses=collections.Counter();elapsed=0.;errors=collections.Counter()
    for r in calls:
        reply=r.get('result') or {};data=reply.get('data') or {}
        statuses[data.get('status','argument_error' if not reply.get('ok') else 'unknown')]+=1
        elapsed+=data.get('elapsed_seconds') or 0.
        error=data.get('error') or reply.get('error') or {};errors[str(error.get('type',error.get('code','none')))]+=1
    rows.append({'round':p.parent.parent.name,'world':result['world'],'total':(result.get('score') or {}).get('total'),
        'calls':len(calls),'code_bearing_model_requests':len(code_turns),'code_bearing_model_seconds':sum(e.get('elapsed_seconds',0.) for e in requests if e.get('call_number') in code_turns),'all_model_seconds':sum(e.get('elapsed_seconds',0.) for e in requests),'statuses':dict(statuses),'error_types':dict(errors),'elapsed_seconds':elapsed,
        'code_backend_receipt':(a/'code-backend-receipt.json').exists(),
        'backend_unavailable':statuses['unavailable']+statuses['protocol_error']})
allstatuses=collections.Counter()
for r in rows:allstatuses.update(r['statuses'])
report={'rows':rows,'completed':len(rows),'trials_using_code':sum(r['calls']>0 for r in rows),
        'trials_with_success':sum(r['statuses'].get('success',0)>0 for r in rows),'calls':sum(r['calls'] for r in rows),
        'code_bearing_model_seconds':sum(r['code_bearing_model_seconds'] for r in rows),'all_model_seconds':sum(r['all_model_seconds'] for r in rows),'statuses':dict(allstatuses),'elapsed_seconds':sum(r['elapsed_seconds'] for r in rows),
        'backend_unavailable_calls':sum(r['backend_unavailable'] for r in rows),
        'missing_backend_receipts':sum(not r['code_backend_receipt'] for r in rows)}
report['backend_valid_for_interpretation']=len(rows)==18 and not report['backend_unavailable_calls'] and not report['missing_backend_receipts']
(E/'code-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False))
