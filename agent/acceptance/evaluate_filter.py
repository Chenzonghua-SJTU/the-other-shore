"""Independent analytic evaluation; no model/PySR/official simulation calls."""
from pathlib import Path
import collections,copy,hashlib,json,statistics,time,sys
import numpy as np
from .analytic_fixtures import core_suite,rms
from ..pipeline_bridge import trajectory_filter as tf
from ..repository.shore_agent.symbolic_quality import derivatives as legacy


def main():
    out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=True)
    source=Path(tf.__file__);before=hashlib.sha256(source.read_bytes()).hexdigest()
    rows=[]
    for item in core_suite():
        p=item['probe'];snapshot=copy.deepcopy(p);t,r,v=[np.asarray(p[k]) for k in ('times','r_obs','v_obs')]
        old=legacy(p,t,r,v);beg=time.monotonic();result=tf.filter_probe(p,deadline=beg+2.)
        assert p==snapshot,'filter mutated original input'
        truth=[item['truth'][k][1:-1] for k in ('r','v','a')]
        raw=[r[1:-1],v[1:-1],np.gradient(v,t,axis=0)[1:-1]]
        filtered=[np.asarray(result['arrays'][k])[1:-1] for k in ('r','v','a')] if result['status']=='filtered' else old[:3]
        row={k:item[k] for k in ['family','seed','noise','irregular']}
        row.update(status=result['status'],reason=result.get('reason'),elapsed=time.monotonic()-beg,selected_lambda=result.get('lambda'),errors={})
        for method,values in [('raw',raw),('legacy',old[:3]),('joint',filtered)]:
            row['errors'][method]={k:rms(x-y) for k,x,y in zip(('r','v','a'),values,truth)}
        rows.append(row)
    groups={}
    for fam in sorted({r['family'] for r in rows}):
        subset=[r for r in rows if r['family']==fam]
        groups[fam]={m:{k:float(np.sqrt(np.mean([r['errors'][m][k]**2 for r in subset]))) for k in ('r','v','a')} for m in ('raw','legacy','joint')}
    report={'source_sha256':before,'source_unchanged_during_test':before==hashlib.sha256(source.read_bytes()).hexdigest(),'cases':len(rows),'statuses':dict(collections.Counter(r['status'] for r in rows)),'seconds_total':sum(r['elapsed'] for r in rows),'seconds_max':max(r['elapsed'] for r in rows),'grouped_rmse':groups,'rows':rows,'scope':'Synthetic analytic diagnostics only; not six-world scores or hidden-physics-informed training.'}
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
