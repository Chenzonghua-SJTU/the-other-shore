"""Post-hoc denoising ablation on fixed acquired public-development observations.

The evaluator regenerates clean reference arrays. Neither reference values nor
world identifiers enter filter inputs. This module is never imported by Agent.
No filter parameters are selected from these results.
"""
from pathlib import Path
import copy,hashlib,json,sys,time,collections
import numpy as np
from development_worlds import catalog
from development_worlds.worlds import DevelopmentSimulator
from ..pipeline_bridge import trajectory_filter as tf
from ..pipeline_bridge.compiler import property_mapping
from ..repository.shore_agent.symbolic_quality import derivatives as legacy
from .analytic_fixtures import rms


def main():
 root=Path(sys.argv[1]);out=Path(sys.argv[2]);rows=[];sha=hashlib.sha256(Path(tf.__file__).read_bytes()).hexdigest()
 for w in range(1,7):
  A=root/'R07'/f'world-{w}'/'agent-output';task=json.loads((A/'mission.json').read_text())['task'];sim=DevelopmentSimulator(str(w));cfg=catalog.config(str(w))
  for path in sorted((A/'records/probe').glob('*.json')):
   probe=json.loads(path.read_text());probe['observation_noise']={'position_sigma':task['position_noise_sigma'],'velocity_sigma':task['velocity_noise_sigma']};snapshot=copy.deepcopy(probe)
   t,r,v=[np.asarray(probe[k],dtype=float) for k in ('times','r_obs','v_obs')];initial=probe['initial_conditions']
   # Public alias mapping used by the official bridge; no force law information.
   pp={k:val['default'] for k,val in cfg['probe_properties'].items()};sp={k:val['default'] for k,val in cfg['source_properties'].items()}
   mapping=property_mapping({'controls':cfg})
   for alias,value in initial['properties'].items():
    scope,key=mapping[alias];(pp if scope=='probe' else sp)[key]=value
   clean=sim.launch([{'pos':initial['r0'],'vel':initial['v0'],'properties':pp}],initial['t0'],(t-initial['t0']).tolist(),source_properties=sp)
   assert np.allclose(t,clean['times'],rtol=0,atol=1e-10)
   cr,cv=clean['positions'][:,0],clean['velocities'][:,0]
   ca=np.array([sim.acceleration(float(ti),pos[None,:],source_properties=sp,probe_properties=[pp])[0] for ti,pos in zip(t,cr)])
   before=time.monotonic();filtered=tf.filter_probe(probe,deadline=before+2.);elapsed=time.monotonic()-before
   assert snapshot==probe
   old=legacy(probe,t,r,v);joint=[np.asarray(filtered['arrays'][k])[1:-1] for k in ('r','v','a')] if filtered['status']=='filtered' else old[:3]
   raw=[r[1:-1],v[1:-1],np.gradient(v,t,axis=0)[1:-1]];truth=[cr[1:-1],cv[1:-1],ca[1:-1]]
   row={'world':str(w),'probe_id':probe['probe_id'],'samples':len(t),'duration':float(t[-1]-t[0]),'status':filtered['status'],'reason':filtered.get('reason'),'lambda':filtered.get('lambda'),'elapsed':elapsed,'errors':{}}
   for method,values in [('raw',raw),('legacy',old[:3]),('joint',joint)]:row['errors'][method]={k:rms(x-y) for k,x,y in zip(('r','v','a'),values,truth)}
   rows.append(row)
 groups={}
 for w in range(1,7):
  part=[r for r in rows if r['world']==str(w)]
  groups[str(w)]={'probes':len(part),'filtered':sum(r['status']=='filtered' for r in part),'rmse':{m:{k:float(np.sqrt(np.mean([r['errors'][m][k]**2 for r in part]))) for k in ('r','v','a')} for m in ('raw','legacy','joint')}}
 report={'source_sha256':sha,'source_unchanged':sha==hashlib.sha256(Path(tf.__file__).read_bytes()).hexdigest(),'baseline':'E10-20261107-224311/R07','groups':groups,'rows':rows,'scope':'Post-hoc fixed-observation input-quality ablation, not PySR or Agent score improvement. Clean references only in evaluator; no algorithm tuning.'}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({'groups':groups,'probes':len(rows)},indent=2))
if __name__=='__main__':main()
