"""Read observed evidence only; no reference force law or simulator imports."""
from pathlib import Path
import json,math,sys,collections
from ..pipeline_bridge.reliability import coverage

def load(p):return json.loads(p.read_text())
def logs(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []

def main():
 E=Path(sys.argv[1]);rows=[]
 for p in sorted(E.glob('R*/world-*/result.json')):
  r=load(p);A=p.parent/'agent-output';task=load(A/'mission.json')['task'];selected=[x for x in logs(A/'submission-selection-audit.jsonl') if x.get('accepted')]
  last=selected[-1] if selected else {};lid=last.get('law_id');law=load(A/'records/law'/f'{lid}.json') if lid else {}
  probes=[load(x) for x in (A/'records/probe').glob('*.json')];cov=coverage(probes,task['allowed_properties']);val=[load(x) for x in (A/'records/validation').glob('*.json')];val=[x for x in val if x.get('law_id')==lid]
  errors=law.get('errors',{}).get('overall',{});sv=task.get('velocity_noise_sigma',0);ratio=errors.get('velocity_rmse',0)/(math.sqrt(2)*sv) if sv else None
  tools=logs(A/'tools.jsonl');row={'round':p.parent.parent.name,'world':r['world'],'score':r.get('score',{}).get('total'),'selected_law':lid,'law_spec':law.get('law_spec'),'converged':law.get('converged'),'training_errors':errors,'velocity_noise_ratio':ratio,'coverage':cov,'validation_errors':[x['errors']['overall'] for x in val],'tool_counts':dict(collections.Counter(x['tool'] for x in tools))}
  rows.append(row)
 low=[r for r in rows if (r['score'] or 0)<90]
 summary={'completed':len(rows),'low_score_cases':len(low),'low_with_training_velocity_gt5sigma':sum(r['velocity_noise_ratio'] is not None and r['velocity_noise_ratio']>5 for r in low),'low_without_selected_law_validation':sum(not r['validation_errors'] for r in low),'low_with_unisolated_properties':sum(bool(r['coverage']['unresolved_property_controls']) for r in low),'low_fit_not_converged':sum(r['converged'] is False for r in low),'low_without_symbolic_search_tool_call':sum(not r['tool_counts'].get('symbolic_regression') for r in low)}
 out={'summary':summary,'rows':rows,'scope':'Observable failure indicators, not causal attribution or independent confidence bounds.'}
 (E/'scientific-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
