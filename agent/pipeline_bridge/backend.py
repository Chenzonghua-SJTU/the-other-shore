"""Numerical/serialization boundary. Repository scientific code remains byte-identical."""
import ast,copy,hashlib,json,time
from pathlib import Path
from dataclasses import replace
from contextlib import contextmanager
import numpy as np
from rift import load_law,rollout
from rift.observations import case_from_observation
from .compiler import compile_law,property_mapping
from .background import explicit_background,BackgroundModelError
from .joint_expression import background_context, scoped_expression

class Backend:
    def __init__(self,environment,output,numerics,expressions):
        self.env,self.output,self.numerics,self.expressions=environment,Path(output),numerics,expressions
        self.cache={};self.source_cache={};self.background_history_cache={};self.rollout_calls=0;self.exports={};self.selection=None
        self._background_spec=None
        (self.output/'bridge-cache').mkdir(parents=True,exist_ok=True)

    def _frame(self,probe):
        if self.env is not None:
            record=copy.deepcopy(self.env.acquired[probe['probe_id']]['_official_record'])
            record['times']=probe['times']
            record['readout']=[float(t)-record['t_launch'] for t in probe['times']]
            record['positions']=np.asarray(probe['r_obs'])[:,None,:].tolist()
            record['velocities']=np.asarray(probe['v_obs'])[:,None,:].tolist()
            return self.env.public_mission,record
        # SR worker receives only acquired probe records. Pure-field screening
        # represents its already-public per-experiment controls as probe metadata.
        initial=probe['initial_conditions'];props=initial['properties']
        mission={'controls':{'source_properties':{},'probe_properties':{k:{'default':v} for k,v in props.items()}},
            'source_property_target_id':None,'background_initial_state':{'ids':[],'positions':[],'velocities':[]}}
        record={'t_launch':initial['t0'],'readout':[float(t)-initial['t0'] for t in probe['times']],
            'times':probe['times'],'probes':[{'pos':initial['r0'],'vel':initial['v0'],'properties':props}],
            'positions':np.asarray(probe['r_obs'])[:,None,:].tolist(),'velocities':np.asarray(probe['v_obs'])[:,None,:].tolist()}
        return mission,record

    @contextmanager
    def candidate(self, background_law_spec):
        """Bind only an explicit model choice, scoped to one candidate operation."""
        previous=self._background_spec
        self._background_spec=copy.deepcopy(background_law_spec)
        token=background_context.set(self._background_spec)
        try:yield
        finally:
            self._background_spec=previous
            background_context.reset(token)

    def background_model(self,expression,mission,bodies=None):
        # Observations never select a background model. They remain available to
        # the LLM, which must supply the mathematical hypothesis explicitly.
        try:return explicit_background(expression,self._background_spec,mission)
        except BackgroundModelError as exc:
            raise self.numerics.NumericalError(str(exc),exc.code) from exc

    def source(self,expression,parameters,mission,bodies=None):
        background_spec=self.background_model(expression,mission,bodies)
        specs=[{**s,'initial':float(parameters[s['name']])} for s in expression.parameter_spec]
        return compile_law(expression.law_spec,background_spec,specs,mission)

    def integrate(self,expression,parameters,probe,bodies,deadline=None):
        tool_limit=getattr(self,'tool_deadline',None)
        def check_tool_deadline():
            if tool_limit is not None and time.monotonic()>=tool_limit[0]:
                raise tool_limit[1]()
        check_tool_deadline()
        mission,record=self._frame(probe)
        background_spec=self.background_model(expression,mission,bodies)
        signature=json.dumps({'law':expression.law_spec,'spec':expression.parameter_spec,'mapping':property_mapping(mission),'target':mission.get('source_property_target_id'),'background':background_spec},sort_keys=True)
        if signature not in self.source_cache:
            initial_parameters={p['name']:p['initial'] for p in expression.parameter_spec}
            source=self.source(expression,initial_parameters,mission,bodies)
            digest=hashlib.sha256(source.encode()).hexdigest()
            path=self.output/'bridge-cache'/(digest+'.py');path.write_text(source)
            self.source_cache[signature]=digest
            self.cache[digest]=load_law(path)
            if len(self.source_cache)>96:
                old=self.source_cache.pop(next(iter(self.source_cache)))
                if old not in self.source_cache.values():self.cache.pop(old,None)
        digest=self.source_cache[signature]
        # The official immutable LawModel accepts current parameters separately.
        # Reuse mathematical code; never recompile it for each optimizer perturbation.
        law=replace(self.cache[digest],params={k:float(v) for k,v in parameters.items()})
        stop=min(time.monotonic()+self.numerics.MAX_ROLLOUT_SECONDS,deadline if deadline is not None else float('inf'),tool_limit[0] if tool_limit is not None else float('inf'))
        rhs_calls=0
        def acceleration(t,state,params):
            nonlocal rhs_calls
            rhs_calls+=1
            check_tool_deadline()
            if time.monotonic()>stop:raise self.numerics.NumericalError('Official rollout wall-time budget exhausted','numerical_timeout')
            return law.acceleration(t,state,params)
        try:
            case=case_from_observation(mission,record)
            prediction=rollout(replace(law,acceleration=acceleration),case)
            check_tool_deadline()
            self.rollout_calls+=1
            self.last_rhs_calls=rhs_calls
        except Exception as exc:
            code='numerical_timeout' if time.monotonic()>stop else 'numerical_failure'
            raise self.numerics.NumericalError('Official rift.rollout: '+str(exc)[:600],code) from exc
        r=np.asarray(prediction['positions'])[:,0,:];v=np.asarray(prediction['velocities'])[:,0,:]
        if expression.uses_bodies and case['background']:
            history_key=json.dumps({'background':case['background'],'background_law_spec':background_spec,'times':probe['times'],'law':digest,'parameters':parameters},sort_keys=True)
            if history_key not in self.background_history_cache:
                queries=[];query_ids={};occupied={b['id'] for b in case['background']}
                for i,body in enumerate(case['background']):
                    identifier='bridge_background_query_'+str(i)
                    while identifier in occupied:identifier+='_' 
                    occupied.add(identifier);query_ids[identifier]=body['id']
                    queries.append({**copy.deepcopy(body),'id':identifier})
                def background_only(t,state,p):
                    check_tool_deadline()
                    if time.monotonic()>stop:raise self.numerics.NumericalError('Background diagnostic rollout time budget exhausted','numerical_timeout')
                    # Retain all interacting background bodies. A passive copy
                    # follows each source's explicit RHS; mapping its ID makes
                    # sum_b exclude the original source and its diagnostic copy.
                    translated=tuple({**body,'kind':'background','id':query_ids[body['id']]}
                                     if body['id'] in query_ids else body for body in state)
                    return law.acceleration(t,translated,p)
                bg_law=replace(law,body_props={},hidden=[],acceleration=background_only)
                bg_case={'schema_version':'1.0.0','t_launch':0.,'readout':list(probe['times']),'background':copy.deepcopy(case['background']),'probes':queries}
                bg_prediction=rollout(bg_law,bg_case)
                self.background_history_cache[history_key]=np.asarray(bg_prediction['positions'])
                if len(self.background_history_cache)>64:self.background_history_cache.pop(next(iter(self.background_history_cache)))
            background_positions=self.background_history_cache[history_key]
        else:
            background_positions=np.repeat(np.asarray(mission['background_initial_state']['positions'],float)[None,:,:],len(r),axis=0) if mission['background_initial_state']['ids'] else np.empty((len(r),0,2))
        a=[]
        for j,(t,rp,vp) in enumerate(zip(probe['times'],r,v)):
            check_tool_deadline()
            bg=[{'body_id':bid,'position':pos} for bid,pos in zip(mission['background_initial_state']['ids'],background_positions[j])]
            a.append(expression.acceleration(float(t),rp,vp,bg,probe['initial_conditions']['properties'],parameters).tolist())
        with (self.output/'official-rollout-audit.jsonl').open('a') as stream:
            stream.write(json.dumps({'probe_id':probe['probe_id'],'source_template_sha256':digest,'parameters':{k:float(v) for k,v in parameters.items()},'calls':self.rollout_calls,'backend':'unchanged rift.rollout'})+'\n')
        return {'probe_id':probe['probe_id'],'times':probe['times'],'r_pred':r.tolist(),'v_pred':v.tolist(),'a_pred':a}

    def short_rollout(self,candidate,dataset,deadline):
        # Original terminal time, original observations, original 0.4 s check;
        # only the numerical engine is the fixed organizer rollout.
        expression=scoped_expression(candidate['law_spec'],candidate['parameter_spec'])
        parameters={p['name']:p['initial'] for p in candidate['parameter_spec']}
        stop=min(deadline,time.monotonic()+.4)
        position_errors=[];velocity_errors=[];calls=0
        for probe in dataset['probes']:
            t,r,v=(np.asarray(probe[k],dtype=float) for k in ('times','r_obs','v_obs'))
            t0=float(probe['initial_conditions'].get('t0',t[0]))
            end=min(float(t[-1]),t0+min(1.,2*float(np.median(np.diff(t)))))
            if end<=t0:raise self.numerics.NumericalError('short_rollout_invalid_interval')
            short=copy.deepcopy(probe)
            short['times']=[end]
            short['r_obs']=[[float(np.interp(end,t,r[:,i])) for i in range(2)]]
            short['v_obs']=[[float(np.interp(end,t,v[:,i])) for i in range(2)]]
            result=self.integrate(expression,parameters,short,[],deadline=stop)
            calls+=self.last_rhs_calls
            if end>=t[0]:
                position_errors.append(float(np.sum((np.asarray(result['r_pred'][0])-short['r_obs'][0])**2)))
                velocity_errors.append(float(np.sum((np.asarray(result['v_pred'][0])-short['v_obs'][0])**2)))
        return {'status':'passed','probe_count':len(dataset['probes']),'rhs_calls':calls,
            'position_rmse':float(np.sqrt(np.mean(position_errors))) if position_errors else None,
            'velocity_rmse':float(np.sqrt(np.mean(velocity_errors))) if velocity_errors else None,
            'scope':'Short acquired-data replay with official rift.rollout; not independently validated.'}

    @staticmethod
    def selection_key(law_id,claims,explanation):
        return law_id,json.dumps(claims,sort_keys=True,ensure_ascii=False),explanation

    def _record_commit(self,kind,key,directory,reply):
        with (self.output/'submission-selection-audit.jsonl').open('a') as stream:
            stream.write(json.dumps({'kind':kind,'law_id':key[0],
                'law_sha256':hashlib.sha256((directory/'law.py').read_bytes()).hexdigest(),
                'claims_sha256':hashlib.sha256((directory/'claims.json').read_bytes()).hexdigest(),
                'accepted':reply.get('accepted'),'selection_source':'original repository runtime'})+'\n')

    def commit_submission(self):
        if self.selection is None:raise ValueError('No original-Agent selected submission pair')
        directory=self.exports[self.selection]
        reply=self.env.call('submit',law=str(directory/'law.py'),claims=str(directory/'claims.json'))
        if reply.get('accepted') is not True or reply.get('submitted') is not True:raise ValueError('Official Console rejected selected submission')
        self._record_commit('submit',self.selection,directory,reply)
        return reply

    def attach(self,runtime):
        # Persist the explicit background formula before the immutable law
        # record is written; a later fit cannot alter an earlier candidate.
        original_add=runtime.store.add
        def add(kind,data,record_id=None):
            if kind=='law':
                data=copy.deepcopy(data)
                data['background_law_spec']=copy.deepcopy(self._background_spec)
                data['background_selection_source']='model_tool_argument' if self._background_spec is not None else 'unused_by_body_free_probe_law'
            return original_add(kind,data,record_id)
        runtime.store.add=add
        original_fit=runtime.fit_law_and_rollout
        def fit(law_spec,parameter_spec,train_probe_ids,background_law_spec=None):
            with self.candidate(background_law_spec):
                expression=scoped_expression(law_spec,parameter_spec)
                chosen=explicit_background(expression,background_law_spec,self.env.public_mission)
                result=original_fit(law_spec,parameter_spec,train_probe_ids)
            result['background_law_spec']=copy.deepcopy(chosen)
            return result
        runtime.fit_law_and_rollout=fit

        def for_law(operation):
            def bound(law_id,*args,**kwargs):
                law=runtime._law(law_id)
                with self.candidate(law.get('background_law_spec')):
                    return operation(law_id,*args,**kwargs)
            return bound
        runtime.validate_law=for_law(runtime.validate_law)
        runtime.analyze_residual=for_law(runtime.analyze_residual)
        # Preserve each original tool call and its scientific checks, then
        # serialize only the exact pair selected by that original runtime.
        original_save=runtime.save_draft;original_submit=runtime.submit;original_finalize=runtime.finalize_draft
        def save(law_id,claims,physical_explanation):
            previous=copy.deepcopy(runtime.last_valid_draft)
            latest=runtime.run_dir/'latest_draft.json'
            previous_bytes=latest.read_bytes() if latest.exists() else None
            result=original_save(law_id,claims,physical_explanation)
            key=self.selection_key(law_id,claims,physical_explanation);directory=self.exports[key]
            try:
                reply=self.env.call('save_draft',law=str(directory/'law.py'),claims=str(directory/'claims.json'))
                if reply.get('accepted') is not True:raise ValueError('Official Console rejected selected draft')
            except Exception:
                runtime.last_valid_draft=previous
                if previous_bytes is None:latest.unlink(missing_ok=True)
                else:latest.write_bytes(previous_bytes)
                raise
            self._record_commit('save_draft',key,directory,reply)
            result.setdefault('checks',{})['official_reply']=reply
            runtime.last_valid_draft.setdefault('checks',{})['official_reply']=reply
            latest.write_text(json.dumps(runtime.last_valid_draft,ensure_ascii=False,indent=2))
            return result
        def submit(law_id,claims,physical_explanation):
            self.selection=self.selection_key(law_id,claims,physical_explanation)
            try:return original_submit(law_id,claims,physical_explanation)
            finally:self.selection=None
        def finalize(reason):
            draft=runtime.last_valid_draft
            if draft is not None:self.selection=self.selection_key(draft['law_id'],draft['claims'],draft['physical_explanation'])
            try:return original_finalize(reason)
            finally:self.selection=None
        runtime.save_draft=save;runtime.submit=submit;runtime.finalize_draft=finalize

    def export(self,directory,law,claims,probes,physical_explanation):
        with self.candidate(law.get('background_law_spec')):
            return self._export(directory,law,claims,probes,physical_explanation)

    def _export(self,directory,law,claims,probes,physical_explanation):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        expression=scoped_expression(law['law_spec'],law['parameter_spec'])
        source=self.source(expression,law['fitted_parameters'],self.env.public_mission,law.get('body_context',law.get('known_bodies',[])))
        points=0
        for probe in probes:
            prediction=self.integrate(expression,law['fitted_parameters'],probe,law.get('body_context',law.get('known_bodies',[])))
            points+=len(prediction['times'])
        (directory/'law.py').write_text(source)
        (directory/'claims.json').write_text(json.dumps(claims,allow_nan=False))
        (directory/'law.json').write_text(json.dumps(law,allow_nan=False))
        (directory/'physical_explanation.txt').write_text(physical_explanation)
        key=self.selection_key(law['law_id'],claims,physical_explanation)
        self.exports[key]=directory
        return {'format':True,'finite_parameters':True,'executable':True,'finite_observed_state_outputs':points,
            'parameter_count':len(law['fitted_parameters']),'body_input_protocol':'official_prediction',
            'state_encoding':'formal law API 1.0.0'}


def install(environment,output):
    from .bootstrap import repository
    repository()
    from ..repository.shore_agent import numerics,expressions,submission,symbolic_regression
    from ..repository.shore_agent import runtime
    runtime.Expression=scoped_expression
    numerics.Expression=scoped_expression
    backend=Backend(environment,output,numerics,expressions)
    numerics._integrate=backend.integrate
    def autonomous_coverage(expression,bodies,probe):
        if expression.uses_bodies:
            if environment is None:
                raise numerics.NumericalError('This SR worker has no acquired background context','background_context_unavailable')
            # The LLM supplies the dynamics; the fixed integrator propagates
            # initial states. Never infer a model or replay future observations.
            backend.background_model(expression,environment.public_mission,bodies)
    numerics._check_body_coverage=autonomous_coverage
    def short_rollout(candidate,dataset,deadline):
        try:return backend.short_rollout(candidate,dataset,deadline)
        except expressions.ExpressionError as exc:
            raise symbolic_regression.SymbolicRegressionError('short_rollout_domain') from exc
        except numerics.NumericalError as exc:
            code='short_rollout_time_limit' if exc.code=='numerical_timeout' else 'short_rollout_integration_failure'
            raise symbolic_regression.SymbolicRegressionError(code) from exc
    symbolic_regression._short_rollout=short_rollout
    if environment is not None:
        submission.export_submission=backend.export
        environment.backend=backend
    return backend
