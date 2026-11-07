"""Synthetic regression: moving-source fit and export use only official rollout."""
import copy,json,tempfile
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from rift import load_law,rollout
from .pipeline_bridge.bootstrap import repository
repository()
from .repository.shore_agent import numerics,expressions
from .pipeline_bridge.compiler import compile_law
from .pipeline_bridge.backend import Backend

def main():
    A=Path(__file__).resolve().parent;out=Path(tempfile.mkdtemp(prefix='BF1-',dir=A/'_tests'))
    mission={'controls':{'probe_properties':{'mass':{'default':1.}},'source_properties':{}},'source_property_target_id':None,
      'background_initial_state':{'ids':['anchor','orbiter'],'positions':[[.2,-.3],[2.2,-.3]],'velocities':[[0.,0.],[0.,.4]]},'noise':{'position_sigma':0.}}
    props={'orbiter':{'bridge_motion_mode':2.,'bridge_center_x':.2,'bridge_center_y':-.3}}
    spec={'ax':'k*sum_b(bx-x)','ay':'k*sum_b(by-y)'};parameters=[{'name':'k','initial':.03}]
    src=compile_law(spec,{'ax':'0','ay':'0'},parameters,mission,background_props=props);(out/'reference.py').write_text(src);law=load_law(out/'reference.py')
    initial=mission['background_initial_state'];background=[{'id':ident,'pos':p,'vel':v,'properties':{}} for ident,p,v in zip(initial['ids'],initial['positions'],initial['velocities'])]
    t_bg=(np.arange(41)*.15).tolist()
    queries=copy.deepcopy(background)
    for q in queries:q['properties'].update(props.get(q['id'],{}))
    def bg_only(t,state,p):return law.acceleration(t,tuple({**b,'kind':'background'} for b in state),p)
    bg_law=replace(law,body_props={},acceleration=bg_only)
    bg_prediction=rollout(bg_law,{'schema_version':'1.0.0','t_launch':0.,'readout':t_bg,'background':[],'probes':queries})
    context={'kind':'observed_body_trajectories','body_ids':initial['ids'],'times':t_bg,'positions':np.asarray(bg_prediction['positions']).tolist()}
    offsets=(np.arange(41)*.05).tolist();case={'schema_version':'1.0.0','t_launch':2.,'readout':offsets,'background':background,'probes':[{'id':'P','pos':[3.,1.],'vel':[.1,0.],'properties':{'mass':1.}}]}
    expected=rollout(law,case);times=(2+np.asarray(offsets)).tolist()
    record={'probes':[{'pos':[3.,1.],'vel':[.1,0.],'properties':{'mass':1.}}],'t_launch':2.,'times':times,'readout':offsets,'source_properties':{},'positions':np.asarray(expected['positions']).tolist(),'velocities':np.asarray(expected['velocities']).tolist()}
    probe={'probe_id':'p','initial_conditions':{'r0':[3.,1.],'v0':[.1,0.],'t0':2.,'properties':{'mass':1.}},'times':times,'r_obs':np.asarray(expected['positions'])[:,0].tolist(),'v_obs':np.asarray(expected['velocities'])[:,0].tolist()}
    env=SimpleNamespace(public_mission=mission,acquired={'p':{'_official_record':record}})
    backend=Backend(env,out/'runtime',numerics,expressions);expression=expressions.Expression(spec,parameters)
    got=backend.integrate(expression,{'k':.03},probe,context)
    position_error=float(np.max(np.abs(np.asarray(got['r_pred'])-np.asarray(expected['positions'])[:,0])))
    assert position_error<2e-8,position_error
    # Current parameters must be read from LawModel, never stale compiled defaults.
    other=backend.integrate(expression,{'k':.05},probe,context)
    restored=backend.integrate(expression,{'k':.03},probe,context)
    assert len(list((out/'runtime/bridge-cache').glob('*.py')))==1
    assert np.max(np.abs(np.asarray(other['r_pred'])-got['r_pred']))>1e-3
    assert np.array_equal(np.asarray(restored['r_pred']),np.asarray(got['r_pred']))
    exported=backend.source(expression,{'k':.03},mission,context);(out/'exported.py').write_text(exported);candidate=load_law(out/'exported.py')
    heldout=copy.deepcopy(case);heldout['t_launch']=10.
    future=rollout(candidate,heldout);truth=rollout(law,heldout)
    future_error=float(np.max(np.abs(np.asarray(future['positions'])-truth['positions'])))
    assert future_error<2e-7,future_error
    assert len(candidate.params)==1 and candidate.body_props
    # A noncircular/nonlinear observed background is not silently replayed.
    bad=copy.deepcopy(context)
    for i,t in enumerate(t_bg):bad['positions'][i][1]=[2.2+.4*t,.2*t*t-.3]
    rejected=False
    from .pipeline_bridge.background import infer,BackgroundModelError
    try:infer(mission,bad,0.)
    except BackgroundModelError:rejected=True
    fallback,evidence=backend.background_model(expression,mission,bad)
    assert fallback['orbiter']['bridge_motion_mode']==3
    assert all(e.get('status')=='candidate_hypothesis_not_established' for e in evidence)
    assert rejected
    # Infer a translated center without any stationary reference object.
    from .pipeline_bridge.background import infer
    unanchored=copy.deepcopy(mission)
    unanchored['background_initial_state']={k:[v[1]] for k,v in initial.items()}
    solo_context={**context,'body_ids':['orbiter'],'positions':np.asarray(context['positions'])[:,1:2,:].tolist()}
    learned,_=infer(unanchored,solo_context,0.)
    assert abs(learned['orbiter']['bridge_center_x']-.2)<1e-7
    assert abs(learned['orbiter']['bridge_center_y']+.3)<1e-7
    noisy=copy.deepcopy(context);rng=np.random.default_rng(17)
    noisy['positions']=(np.asarray(noisy['positions'])+rng.normal(0,.02,np.asarray(noisy['positions']).shape)).tolist()
    noisy['positions'][0]=copy.deepcopy(initial['positions'])
    noisy_props,_=infer(mission,noisy,.02)
    assert noisy_props['orbiter']['bridge_motion_mode']==2.
    assert infer({**mission,'background_initial_state':{'ids':[],'positions':[],'velocities':[]}},None,0.)==({},[])
    report={'position_error':position_error,'unobserved_future_error':future_error,'parameter_cache_no_stale_values':True,'compiled_templates_for_three_parameter_settings':1,'global_parameter_count':len(candidate.params),'body_props':candidate.body_props,'unsupported_kinematic_model_rejected':rejected,'shared_rhs_fallback_explicit':True,'official_rollout_only':True,'unanchored_center_recovered':True,'noisy_geometry_supported':True,'empty_background_supported':True}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',out,json.dumps(report))
if __name__=='__main__':main()
