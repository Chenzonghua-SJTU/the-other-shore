"""Read-only runtime audit; writes summary only inside experiment /agent tree."""
from pathlib import Path
from collections import Counter
import json,sys
E=Path(sys.argv[1])
def lines(p):
 if not p.exists():return []
 out=[]
 for s in p.read_text().splitlines():
  try:out.append(json.loads(s))
  except json.JSONDecodeError:pass
 return out
rows=[]
for p in sorted(E.glob('R*/world-*/result.json')):
 r=json.loads(p.read_text());a=p.parent/'agent-output';model=lines(a/'model_events.jsonl');tools=lines(a/'tools.jsonl');aud=lines(a/'submission-selection-audit.jsonl')
 selected=[x for x in aud if x.get('accepted')]
 last=selected[-1] if selected else {}
 requests=[x for x in model if x.get('event')=='request'];config=next((x for x in model if x.get('event')=='configuration'),{})
 score=r.get('score') or {}
 row={'round':p.parent.parent.name,'world':r['world'],'total':score.get('total'),'accepted':r['has_valid_submission'],'end_reason':r['end_reason'],'exit_code':r['agent_exit_code'],'model_requests':len(requests),'tools':dict(Counter(x['tool'] for x in tools)),'tool_error_codes':dict(Counter(str((x.get('result',{}).get('error') or {}).get('code')) for x in tools if not x.get('result',{}).get('ok'))),'finish_reasons':dict(Counter(str(x.get('finish_reason')) for x in requests)),'thinking':config.get('thinking'),'max_tokens':config.get('max_tokens_per_call'),'pair_matches':bool(last) and last.get('law_sha256')==r.get('law_sha256') and last.get('claims_sha256')==r.get('claims_sha256'),'invalid_cases':score.get('invalid_cases',[]),'traceback_in_agent_log':'Traceback (most recent call last)' in (p.parent/'agent.log').read_text(),'directory':str(p.parent)}
 rows.append(row)
x={'completed':len(rows),'rows':rows,'accepted':sum(x['accepted'] for x in rows),'pair_mismatches':[x for x in rows if x['accepted'] and not x['pair_matches']],'traceback_cases':[{'round':x['round'],'world':x['world']} for x in rows if x['traceback_in_agent_log']]}
(E/'runtime-audit.json').write_text(json.dumps(x,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in x.items() if k!='rows'},ensure_ascii=False));print([(x['round'],x['world'],round(x['total'] or 0,2),x['model_requests']) for x in rows])
