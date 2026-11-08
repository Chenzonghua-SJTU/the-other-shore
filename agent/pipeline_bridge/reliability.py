"""Observable protocol corrections. No world identities or reference laws.

The vendored scientific package stays unchanged. Tool schemas/names stay fixed;
all interventions and the actual system prompt are logged in each episode.
"""
import copy,json,math,types,time,uuid
from pathlib import Path
import numpy as np

VERSION='interaction-v1'

class _ToolDeadlineExceeded(BaseException):
    """Cancellation must escape scientific tools' broad Exception handlers."""


def coverage(probes,controls):
    rows=[p['initial_conditions'] for p in probes]
    levels={k:sorted({float(r['properties'][k]) for r in rows if k in r['properties']}) for k in controls}
    isolated={k:False for k in controls}
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if any(a[k]!=b[k] for k in ('r0','v0','t0')):continue
            keys=set(a['properties'])|set(b['properties'])
            changed=[k for k in keys if a['properties'].get(k)!=b['properties'].get(k)]
            if len(changed)==1 and changed[0] in isolated:isolated[changed[0]]=True
    spans=[float(p['times'][-1]-p['times'][0]) for p in probes if len(p.get('times',[]))>=2]
    return {'property_levels':levels,'single_property_control_pairs':isolated,
            'unresolved_property_controls':[k for k,v in isolated.items() if not v],
            'max_probe_duration':max(spans,default=0.),
            'warning':'Changing several properties together cannot establish independence; unchanged ratios can hide dependence. Missing isolation is uncertainty, not evidence of dependence.'}

def sampling_advice(task,probes):
    sigma=float(task.get('velocity_noise_sigma') or 0.)
    accelerations=[np.asarray(p.get('a_est',[]),float).reshape(-1,2) for p in probes if p.get('a_est')]
    scale=float(np.median(np.linalg.norm(np.concatenate(accelerations),axis=1))) if accelerations else 0.
    grid=float(task.get('constraints',{}).get('time_grid',.005))
    # SNR target is a design heuristic, not a certified physical bound.
    noise_dt=5*sigma/(math.sqrt(2)*max(scale,1e-8)) if scale>0 else None
    suggested=max(.05,min(.5,noise_dt or .05)) if sigma>0 else None
    if suggested is not None:suggested=math.ceil(suggested/grid)*grid
    return {'velocity_sigma':sigma,'observed_acceleration_scale':scale,
      'central_difference_sigma_formula':'sigma_v / (sqrt(2) * dt), independent Gaussian velocity errors',
      'noise_snr5_dt_estimate':noise_dt,'recommended_local_dt_floor':suggested,
      'hard_minimum_dt':None,'recommended_long_probe':'Use near 50 samples and roughly 10 seconds when legal (e.g. dt=0.205, 50 samples); compare finer resolution if dynamics are fast.',
      'note':'Advisory only. Sampling dt does not change the official integrator. Smaller dt amplifies derivative noise; larger dt can alias fast motion. Longer observation span and isolated controls are separate requirements.'}

def diagnostics(runtime,law=None):
    task=runtime.task or {};records=runtime.store.records
    ps=list(records['probe'].values());controls=task.get('allowed_properties',[])
    out={'coverage':coverage(ps,controls),'sampling':sampling_advice(task,ps)}
    if law:
        overall=(law.get('errors') or {}).get('overall',{})
        ratios={}
        for kind in ('position','velocity'):
            sigma=task.get(kind+'_noise_sigma');err=overall.get(kind+'_rmse')
            if isinstance(sigma,(int,float)) and sigma>0 and isinstance(err,(int,float)):
                ratios[kind]={'rmse':err,'two_coordinate_noise_rms':math.sqrt(2)*sigma,'ratio':err/(math.sqrt(2)*sigma)}
        out['noise_scaled_residuals']=ratios
        out['residual_interpretation']='Noise ratios are diagnostics, not confidence intervals. Large velocity residuals remain evidence against a law even when position error is near its noise floor.'
        out['validation_ids_for_this_law']=[k for k,v in records['validation'].items() if v.get('law_id')==law.get('law_id')]
    return out

def normalize_probe_args(args):
    args=copy.deepcopy(args);probes=args.get('probes');moved=[]
    if not isinstance(probes,list):return args,None
    for key in ('t0','dt','n_samples'):
        if not any(isinstance(p,dict) and key in p for p in probes):continue
        values=[p.get(key,args.get(key)) if isinstance(p,dict) else None for p in probes]
        if any(v is None for v in values) or any(v!=values[0] for v in values) or (key in args and args[key]!=values[0]):
            raise ValueError('Different/conflicting '+key+' in one insert_probe: use separate calls in the SAME assistant reply. No environment call executed.')
        args[key]=values[0]
        for p in probes:p.pop(key,None)
        moved.append(key)
    return args,('Moved identical per-probe '+','.join(moved)+' to top level; physical request unchanged.' if moved else None)

def configure(runtime):
    from .submission_policy import install as install_policy, forced
    install_policy(runtime)
    original_execute=runtime.execute
    references={'last_fit':None,'last_launch':[]}
    audit=Path(runtime.run_dir)/'bf2-protocol-audit.jsonl'
    def log(value):
        with audit.open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
    def execute_impl(name,args):
        # Check again at execution time: a model reply or earlier batch tool
        # may have crossed the deadline after the schema was offered.
        if forced(runtime) and name != 'submit':
            result={'ok':False,'data':None,'error':{'code':'submission_required_now',
                'message':'Noisy-session remaining time is <=60 seconds; only explicit submit is allowed.'},'budget':runtime.budget()}
            log({'tool':name,'result':result});return result
        normalized=copy.deepcopy(args);note=None
        try:
            if not isinstance(normalized,dict):raise ValueError('Tool arguments must be an object.')
            if 'law_id' in normalized:
                ref=normalized['law_id']
                # Leading-zero formatting aliases identify the SAME numbered law.
                import re
                if isinstance(ref,str) and ref not in runtime.store.records['law'] and re.fullmatch(r'law_0+[1-9][0-9]*',ref):
                    canonical='law_'+str(int(ref[4:]))
                    if canonical in runtime.store.records['law']:
                        normalized['law_id']=canonical;ref=canonical
                if ref=='last_fit':
                    if references['last_fit'] is None:raise ValueError('No successful current-session fit for last_fit.')
                    normalized['law_id']=references['last_fit']
                elif ref in ('saved','draft'):
                    draft=runtime.last_valid_draft
                    if draft is None:raise ValueError('No model-selected current-session draft for '+ref+'.')
                    normalized['law_id']=draft['law_id']
            for key in ('train_probe_ids','test_probe_ids','probe_ids'):
                if isinstance(normalized.get(key),list) and 'last_launch' in normalized[key]:
                    if not references['last_launch']:raise ValueError('No successful current-session launch for last_launch.')
                    expanded=[]
                    for ref in normalized[key]:
                        expanded.extend(references['last_launch'] if ref=='last_launch' else [ref])
                    normalized[key]=list(dict.fromkeys(expanded))
        except (ValueError,TypeError) as exc:
            result={'ok':False,'data':None,'error':{'code':'invalid_candidate_reference','message':str(exc)},'budget':runtime.budget()}
            log({'tool':name,'arguments':args,'result':result});return result
        if name=='insert_probe':
            try:normalized,note=normalize_probe_args(normalized)
            except (ValueError,TypeError) as exc:
                result={'ok':False,'data':None,'error':{'code':'batch_launch_time_conflict','message':str(exc)},'budget':runtime.budget()}
                log({'tool':name,'arguments':args,'result':result});return result
        # Keep a margin before the forced-submit phase for HTTP and serialization.
        if name=='symbolic_regression':
            allowed=min(40,normalized.get('time_budget_seconds',40),runtime.budget()['wall_seconds_remaining']-100)
            if allowed<1:
                return {'ok':False,'data':None,'error':{'code':'insufficient_search_time','message':'Search must finish before the 60-second submission phase.'},'budget':runtime.budget()}
            normalized['time_budget_seconds']=allowed
        old_code_timeout=runtime.code_config.get('timeout_seconds',90)
        if name=='run_code':
            allowed=min(20,runtime.budget()['wall_seconds_remaining']-90)
            if allowed<1:
                return {'ok':False,'data':None,'error':{'code':'insufficient_code_time','message':'Code plus bounded RPC recovery must leave the 60-second submission reserve.'},'budget':runtime.budget()}
            runtime.code_config['timeout_seconds']=allowed
        try:result=original_execute(name,normalized)
        finally:
            if name=='run_code':runtime.code_config['timeout_seconds']=old_code_timeout
        data=result.get('data') or {}
        if name=='run_code' and result.get('ok') and data.get('status')!='success':
            result['ok']=False
            detail=data.get('error') or {}
            result['error']={'code':'code_execution_'+str(data.get('status','unknown')),
                'message':str(detail.get('type','ExecutionError'))+': '+str(detail.get('message','execution failed'))[:800]}
            result['recovery_instruction']='Do not repeat identical failing code. Virtual paths require run_code files=[path] and read_json/read_file, never open(path). To read a previous stdout artifact, use the read_file tool directly. Register a candidate before spending more time coding.'
        if name=='run_code' and result.get('ok') and data.get('result') is None and data.get('stdout') and data.get('execution_id'):
            data['retrieve_stdout']={'tool':'read_file','arguments':{'path':'analyses/'+data['execution_id']+'/stdout.txt','offset':0,'limit':12000}}
            data['output_reading_guidance']='Printed output already exists as an artifact. Read it once with the read_file tool using retrieve_stdout, not repeated run_code calls slicing 150 characters. For subsequent calculations assign a small structured result dictionary.'
        if result.get('ok') and name=='fit_law_and_rollout' and data.get('fit_status')!='failed' and data.get('law_id'):
            references['last_fit']=data['law_id']
        if result.get('ok') and name=='insert_probe':
            references['last_launch']=[p['probe_id'] for p in data.get('probes',[])]
        if normalized!=args:
            result['resolved_arguments']=normalized
        if note:result['protocol_normalization']={'note':note,'executed_arguments':normalized}
        if result.get('ok') and name in ('mission','insert_probe','fit_law_and_rollout','validate_law','symbolic_regression','analyze_residual'):
            data=result.get('data') or {};lid=data.get('law_id');law=runtime.store.records['law'].get(lid)
            # Validation diagnostics must use validation errors, not training errors.
            if name=='validate_law' and law:law={**law,'errors':data.get('errors')}
            data['bf2_diagnostics']=diagnostics(runtime,law);result['data']=data
        if not result.get('ok'):
            result['available_law_ids']=list(runtime.store.records['law'])
            if (result.get('error') or {}).get('code')=='invalid_expression':
                result['dsl_correction']='Properties require q(\"property_name\"), not bare mass/charge. Coefficients must be declared in parameter_spec. Read mission expression_syntax; change the expression explicitly.'
        if not result.get('ok') and (result.get('error') or {}).get('message')=='Unknown tool':
            result['error']['available_tools']=[s['function']['name'] for s in runtime.schemas]
        if result.get('ok') and name=='fit_law_and_rollout' and result.get('data',{}).get('fit_status')=='failed':
            result['data']['usable_law']=False
            result['data']['next_action']='Read the failure message. A backend limitation is not a physical refutation; syntax errors do not imply unsupported operators.'
        log({'tool':name,'normalized':bool(note),'diagnostics':(result.get('data') or {}).get('bf2_diagnostics')})
        return result
    def execute(name,args):
        # No new environment permission: only the existing local tool schemas
        # receive a shorter wall-clock allowance. Submit keeps its true budget.
        local_names={s['function']['name'] for s in runtime.schemas} - {
            'submit','save_draft','observe_body','insert_probe','mission'}
        if name not in local_names:
            return execute_impl(name,args)
        from .model_transport import _request_deadline
        deadline=time.monotonic()+max(0.,runtime.budget()['wall_seconds_remaining']-60.)
        store=runtime.store
        before_references=copy.deepcopy(references)
        before={kind:set(records) for kind,records in store.records.items()}
        directory=getattr(store,'directory',None)
        before_files=(set(Path(directory).glob('*/*.json')) if directory is not None else set())
        backend=getattr(getattr(runtime,'environment',None),'backend',None)
        prior_limit=getattr(backend,'tool_deadline',None) if backend is not None else None
        if backend is not None:backend.tool_deadline=(deadline,_ToolDeadlineExceeded)
        try:
            with _request_deadline(deadline,deadline_error=_ToolDeadlineExceeded):
                result=execute_impl(name,args)
                if time.monotonic()>=deadline:raise _ToolDeadlineExceeded()
                return result
        except _ToolDeadlineExceeded:
            references.clear();references.update(before_references)
            # A cancelled call publishes no new scientific records. Preserve
            # partial files outside the Agent's acquired-record workspace for
            # audit; all records that predate this call remain untouched.
            removed=[]
            for kind,records in store.records.items():
                for rid in set(records)-before.get(kind,set()):
                    records.pop(rid,None);removed.append({'kind':kind,'record_id':rid})
            if directory is not None:
                new_files=set(Path(directory).glob('*/*.json'))-before_files
                if new_files:
                    quarantine=Path(runtime.run_dir)/'interrupted-tools'/uuid.uuid4().hex
                    for path in new_files:
                        target=quarantine/path.relative_to(directory)
                        target.parent.mkdir(parents=True,exist_ok=True)
                        path.replace(target)
            result={'ok':False,'data':None,'error':{'code':'submission_required_now',
                'message':'Local tool cancelled at the 60-second submission reserve; no unfinished result is usable.'},
                'budget':runtime.budget()}
            log({'tool':name,'event':'local_tool_deadline','discarded_records':removed,'result':result})
            return result
        finally:
            if backend is not None:backend.tool_deadline=prior_limit
    runtime.execute=execute
    runtime.schemas=copy.deepcopy(runtime.schemas)
    for schema in runtime.schemas:
        f=schema['function']
        if f['name']=='run_code':f['description']=f['description'].replace('90秒/1GiB/1CPU','最多20秒执行时间/1GiB/1CPU，启动和清理也消耗本局墙钟')
        if f['name']=='insert_probe':f['description']+=' t0 is required at batch top level; separate calls for different t0. For noisy data consult mission sampling advice; preserve long-duration probes, not only dense short windows.'
        if f['name']=='fit_law_and_rollout':
            f['parameters']['properties']['background_law_spec']={
                'type':'object','properties':{'ax':{'type':'string','maxLength':600},'ay':{'type':'string','maxLength':600}},
                'required':['ax','ay'],'additionalProperties':False}
            f['description']='由模型明确提出探针law_spec及所需的背景background_law_spec，再使用官方rift.rollout拟合参数。使用sum_b必须显式提供背景ax/ay；不会自动选择直线、圆周或共享动力学。背景与探针共用同一组最多5个PARAMS，不增加隐藏参数。背景可省略仅当探针不依赖背景。公式及背景假设随law_id冻结，验证/提交不重选。train_probe_ids可含last_launch以引用当前会话最近成功投放的全部探针。'
        if f['name']=='validate_law':
            f['description']='冻结探针公式、模型指定的背景公式和拟合参数，使用官方rift.rollout在未参与拟合的轨迹上验证；不插值或回放未来背景观测，不重新选择规律。'
        if 'law_id' in f['parameters']['properties']:
            f['parameters']['properties']['law_id']['description']='已有候选编号，或last_fit引用最近成功拟合，saved/draft引用模型明确保存的候选；claims仍须由模型显式给出。'
        for key in ('train_probe_ids','test_probe_ids','probe_ids'):
            if key in f['parameters']['properties']:
                f['parameters']['properties'][key]['description']='当前会话探针编号；last_launch展开为最近一次成功insert_probe的全部编号，可在同批后续工具中使用。'
    original_mission=runtime.mission
    def mission():
        result=original_mission()
        if result.get('ok'):
            result['data']['bf2_diagnostics']=diagnostics(runtime)
            result['data']['expression_syntax']=result['data']['expression_syntax'].replace('Linear interpolation requires observations covering the full rollout interval; no extrapolation.','Official autonomous rollout is used. For sum_b, explicitly provide background_law_spec; no background hypothesis is selected automatically and future observations are never replayed.')
        return result
    runtime.mission=mission
    return runtime

OVERRIDE='''\n\nBF2正式接口及研究质量说明（替换旧版插值与提交门槛描述）：
轨迹计算始终使用官方rift.rollout；dt仅是观测间隔。sum_b的背景必须由你在fit_law_and_rollout的background_law_spec中明确提出ax/ay，再自主积分；系统不会自动补选零加速度、圆周或共享探针动力学。背景和探针共享同组至多5个参数，两份公式的参数并集最多5个，允许某个参数仅用于背景公式。无需背景的探针公式可以省略该字段。背景推演限制不能当作物理反证。
带噪声时，只有剩余时间<=90秒，或所选同一law的训练位置RMSE与独立留出验证位置RMSE都严格<0.01，才允许submit；缺少验证不可豁免，其他law的验证不能代替。剩余时间<=60秒时只有submit可执行，拟合、验证、搜索、读文件等也禁止；必须提前登记候选，及时明确选择law_id、claims和解释。无噪声时沿用原提交许可。此规则覆盖所有旧提交门槛提示；不得自动生成候选或claims，不增加环境轮次。
同一次回复可安排多个环境工具合计一轮，但最多两轮用于新观测/探针，保留最后一轮submit。本地分析和拟合不占环境轮次。参数预检失败且未访问环境不计轮次；真正执行环境操作后该轮不可退还。先拟合已有数据，再用第二轮做必要对照和留出探针；第二轮结束后只能本地分析和最终提交。不要在第三轮继续投放然后期待第四轮提交。不要用save_draft浪费最后提交机会。
公式中的公开属性必须用q("属性名")读取（例如q("mass")、q("charge")），不能裸写mass或charge；实际可用属性以mission为准。编号使用工具返回的law_1等原文，不写law_001或自造编号。
insert_probe的t0、dt、n_samples都必须位于顶层，各probe只含r0、v0、properties。不同投放时刻或采样间隔分成多个insert_probe调用，放在同一回复内；不改变原实验请求。调用模板：{"t0":0,"dt":0.2,"n_samples":50,"probes":[{"r0":[1,0],"v0":[0,0.2],"properties":{"mass":1,"charge":1}}]}。数值仅为格式例子，实际范围和可用属性以mission为准。law_id可用last_fit，探针编号可用last_launch；多次投放后使用返回的具体编号，last_launch只指最后一次投放。

属性同时变化而轨迹相同不能证明属性无关，特别要警惕比值不变。用相同位置、速度、t0，只改变一个属性的对照；不要从世界编号猜关系。零训练误差不等于可辨识性，验证要改变当前尚未检验的控制量。尽早用至少一条合法长探针（约10秒、接近50点）和偏轴初态检验短窗候选；只改变发射t0而保持短窗口不能替代连续长轨迹。
有噪声时看bf2_diagnostics：二维RMSE纯噪声基准约sqrt(2)*每坐标sigma。位置误差接近噪声时仍必须看速度残差；速度残差显著大于噪声时继续辨别结构，不可宣布只是位置噪声。一个SR候选失败不证明另一个候选正确。fit_status=not_converged只表示优化终止状态，需综合误差和验证，不能单凭标签否定。
采样间隔建议是软约束：dt太小放大差分噪声且缩短总时长，dt太大漏掉快速变化；参考公开噪声、运动尺度与返回的建议，不机械追求最小dt。新候选在最终提交前尽量做固定law_id的留出验证，声明和解释对应实际选中的同一候选。'''

class ProtocolClient:
    def __init__(self,client):self._client=client
    def __getattr__(self,name):return getattr(self._client,name)
    def complete(self,messages,**kwargs):
        if messages and messages[0].get('role')=='system' and 'BF2正式接口及研究质量说明' not in messages[0]['content']:
            p=messages[0]['content']
            p=p.replace('新版工具按已取得的天体位置时间序列分段线性插值，禁止范围外外推；使用天体位置的候选需要观测覆盖整段推演时间，必要时补充 observe_body。','正式接口使用模型明确指定的背景动力学和官方rollout；使用sum_b时须提供background_law_spec。')
            p=p.replace('提交硬门槛：所选规律训练位置RMSE严格<1e-5，或剩余时间严格<40秒；否则继续研究。save_draft也不得绕过。','提交时依据验证与适用范围作判断；可提前save_draft，无需等待计时器。')
            messages[0]['content']=p+OVERRIDE
        return self._client.complete(messages,**kwargs)
