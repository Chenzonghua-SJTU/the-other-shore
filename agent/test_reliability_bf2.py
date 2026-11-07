"""BF2 regression: actual Console, protocol recovery, noise diagnostics, coupled RHS."""
import copy,json,os,tempfile,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rift import Console,load_law,rollout
from development_worlds.console import DevelopmentConsole
from development_worlds.service import DevelopmentSession,serve_session
from development_worlds import catalog
from .pipeline_bridge.bootstrap import repository
repository()
from .pipeline_bridge.environment import Environment
from .pipeline_bridge.backend import install,Backend
from .pipeline_bridge.reliability import configure,coverage,normalize_probe_args,ProtocolClient,diagnostics
from .repository.shore_agent.runtime import ToolRuntime
from .repository.shore_agent import numerics,expressions

def main():
 A=Path(__file__).parent;out=Path(tempfile.mkdtemp(prefix='BF2-',dir=A/'_tests'));os.environ['RIFT_OUTPUT_DIR']=str(out)
 session=DevelopmentSession(DevelopmentConsole('2',catalog.noise_sigma('2'),42,catalog.config('2')),out/'session')
 with serve_session(session) as credentials:
  env=Environment(Console(credentials['endpoint'],credentials['token']));env.ruleset='e5';backend=install(env,out/'rt');rt=ToolRuntime(env,out/'rt',config={'symbolic_regression_enabled':True});backend.attach(rt);configure(rt)
  rt.configure_submission_gate({'submission_fit_rmse_threshold':1e-5});assert not rt.submission_gate_policy()['enabled']
  mission=rt.mission();assert 'bf2_diagnostics' in mission['data']
  args={'probes':[{'r0':[1.5,0],'v0':[0,.3],'properties':{'mass':2,'charge':2},'t0':0}],'dt':.05,'n_samples':10}
  good=rt.execute('insert_probe',args);assert good['ok'],good;assert good['protocol_normalization'];assert env.budget()['energy_used']==1
  detail=diagnostics(rt);assert not detail['coverage']['single_property_control_pairs']['mass'];assert not detail['coverage']['single_property_control_pairs']['charge']
  bad=copy.deepcopy(args);bad['t0']=1;before=env.budget()['energy_used'];rejected=rt.execute('insert_probe',bad);assert not rejected['ok'] and env.budget()['energy_used']==before
  args['probes'][0]['properties']={'mass':2,'charge':1};assert rt.execute('insert_probe',args)['ok'];detail=diagnostics(rt);assert all(detail['coverage']['single_property_control_pairs'].values())
  later={'probes':[{'r0':[1.5,0],'v0':[0,.3],'properties':{'mass':1,'charge':1}}],'t0':3,'dt':.1,'n_samples':4}
  later_reply=rt.execute('insert_probe',later);assert later_reply['ok'],later_reply
  late_id=later_reply['data']['probes'][0]['probe_id']
  beyond=rt.execute('fit_law_and_rollout',{'law_spec':{'ax':'0*sum_b(bx-x)','ay':'0*sum_b(by-y)'},'parameter_spec':[],'train_probe_ids':[late_id]})
  assert beyond['ok'] and beyond['data']['fit_status']!='failed',beyond
  fit=rt.execute('fit_law_and_rollout',{'law_spec':{'ax':'0','ay':'0'},'parameter_spec':[],'train_probe_ids':['probe_0001']});assert fit['ok'],fit
  assert 'velocity' in fit['data']['bf2_diagnostics']['noise_scaled_residuals']
  lid=fit['data']['law_id'];claims={'dynamics_order':2,'hidden_matter':False,'time_dependence':False};submission={'law_id':lid,'claims':claims,'physical_explanation':'协议回归测试，不是科学结果。'}
  assert rt.execute('save_draft',submission)['ok']
  first=rt.execute('submit',submission);assert not first['ok'] and first['error']['code']=='scientific_review_needed'
  second=rt.execute('submit',submission);assert second['ok'] and second['data']['status']=='accepted',second
 class Capture:
  def complete(self,messages,**kw):return messages[0]['content']
 prompt=ProtocolClient(Capture()).complete([{'role':'system','content':'提交硬门槛：所选规律训练位置RMSE严格<1e-5，或剩余时间严格<40秒；否则继续研究。save_draft也不得绕过。'}]);assert '提交硬门槛' not in prompt and 'BF2正式接口' in prompt
 # Noncircular moving body under shared linear RHS. Independent analytic reference.
 mission={'controls':{'probe_properties':{},'source_properties':{}},'source_property_target_id':None,'background_initial_state':{'ids':['anchor','mover'],'positions':[[0.,0.],[2.,0.]],'velocities':[[0.,0.],[0.,.4]]},'noise':{'position_sigma':0.}}
 k=.09;omega=math.sqrt(k);t=np.arange(41)*.15
 bg=np.stack([2*np.cos(omega*t),.4/omega*np.sin(omega*t)],axis=1)
 context={'body_ids':['anchor','mover'],'times':t.tolist(),'positions':np.stack([np.zeros_like(bg),bg],axis=1).tolist()}
 spec={'ax':'k*sum_b(bx-x)','ay':'k*sum_b(by-y)'};pars=[{'name':'k','initial':k}];ex=expressions.Expression(spec,pars)
 backend=Backend(None,out/'coupled',numerics,expressions)
 props,evidence=backend.background_model(ex,mission,context);assert props['mover']['bridge_motion_mode']==3
 source=backend.source(ex,{'k':k},mission,context);path=out/'coupled.py';path.write_text(source);law=load_law(path)
 times=np.arange(81)*.1;state=mission['background_initial_state'];case={'schema_version':'1.0.0','t_launch':0.,'readout':times.tolist(),'background':[{'id':i,'pos':p,'vel':v,'properties':{}} for i,p,v in zip(state['ids'],state['positions'],state['velocities'])],'probes':[{'id':'probe','pos':[3.,1.],'vel':[.1,0.],'properties':{}}]}
 pred=rollout(law,case);w2=math.sqrt(2*k);bgtest=np.stack([2*np.cos(omega*times),.4/omega*np.sin(omega*times)],axis=1)
 analytical=np.array([1.,1.])*np.cos(w2*times)[:,None]+np.array([.1,-.4])/w2*np.sin(w2*times)[:,None]+bgtest
 err=float(np.max(np.abs(np.asarray(pred['positions'])[:,0]-analytical)));assert err<2e-5,err
 # Sampling recommendation does not mutate tiny-dt requests.
 tiny={'probes':[{'r0':[1,0],'v0':[0,0],'properties':{}}],'t0':0,'dt':.005,'n_samples':50};assert normalize_probe_args(tiny)[0]==tiny
 report={'protocol_and_gate':True,'isolated_controls':True,'velocity_noise_diagnostic':True,'review_one_time_only':True,'shared_background_analytic_error':err,'official_rollout_only':True,'no_hard_min_dt':True,'autonomous_rollout_beyond_acquired_background_times':True}
 (out/'report.json').write_text(json.dumps(report,indent=2));print('PASS',out,json.dumps(report),flush=True)
if __name__=='__main__':main()
