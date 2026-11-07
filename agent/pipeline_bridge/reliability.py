"""BF2 observable protocol corrections. No world identities or reference laws.

The vendored scientific package stays unchanged. Tool schemas/names stay fixed;
all interventions and the actual system prompt are logged in each episode.
"""
import copy,json,math,types
from pathlib import Path
import numpy as np

VERSION='BF2'

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
    args=copy.deepcopy(args);probes=args.get('probes')
    if not isinstance(probes,list):return args,None
    times=[p.get('t0',args.get('t0')) for p in probes if isinstance(p,dict)]
    if not any(isinstance(p,dict) and 't0' in p for p in probes):return args,None
    if len(times)!=len(probes) or any(t is None for t in times) or len(set(times))!=1 or ('t0' in args and args['t0']!=times[0]):
        raise ValueError('Different/conflicting launch times in one batch: call insert_probe separately for each t0. No probe executed; no energy spent.')
    args['t0']=times[0]
    for p in probes:p.pop('t0',None)
    return args,'Moved identical per-probe t0 to the batch t0; no physical request changed.'

def configure(runtime):
    original_configure=runtime.configure_submission_gate
    def configure_gate(config):
        original_configure({**config,'submission_fit_rmse_threshold':None})
        runtime.submission_fit_rmse_threshold=None
    runtime.configure_submission_gate=configure_gate
    configure_gate({})
    original_execute=runtime.execute
    reviewed=set()
    audit=Path(runtime.run_dir)/'bf2-protocol-audit.jsonl'
    def log(value):
        with audit.open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
    def execute(name,args):
        normalized=copy.deepcopy(args);note=None
        if name=='insert_probe':
            try:normalized,note=normalize_probe_args(args)
            except (ValueError,TypeError) as exc:
                result={'ok':False,'data':None,'error':{'code':'batch_launch_time_conflict','message':str(exc)},'budget':runtime.budget()}
                log({'tool':name,'arguments':args,'result':result});return result
        if name=='submit' and args.get('law_id') not in reviewed:
            law=runtime.store.records['law'].get(args.get('law_id'),{});detail=diagnostics(runtime,law);reasons=[]
            if not detail.get('validation_ids_for_this_law'):reasons.append('Selected law has no held-out validation record.')
            unresolved=detail['coverage']['unresolved_property_controls']
            if unresolved:reasons.append('No isolated same-state controls for '+','.join(unresolved)+'. Independence is not established.')
            vr=detail.get('noise_scaled_residuals',{}).get('velocity',{}).get('ratio',0)
            if vr>5:reasons.append('Velocity residual is >5 times the two-coordinate noise RMS; position noise cannot explain it.')
            reviewed.add(args.get('law_id'))
            if reasons and runtime.budget().get('wall_seconds_remaining',0)>75:
                result={'ok':False,'data':None,'error':{'code':'scientific_review_needed','message':'One-time review, not an official score threshold. '+ ' '.join(reasons)+' Save a draft now; use remaining resources for informative tests. A second submit is allowed; do not wait for a timer.', 'diagnostics':detail},'budget':runtime.budget()}
                log({'tool':name,'arguments':args,'result':result});return result
        result=original_execute(name,normalized)
        if note:result['protocol_normalization']={'note':note,'executed_arguments':normalized}
        if result.get('ok') and name in ('mission','insert_probe','fit_law_and_rollout','validate_law','symbolic_regression','analyze_residual'):
            data=result.get('data') or {};lid=data.get('law_id');law=runtime.store.records['law'].get(lid)
            # Validation diagnostics must use validation errors, not training errors.
            if name=='validate_law' and law:law={**law,'errors':data.get('errors')}
            data['bf2_diagnostics']=diagnostics(runtime,law);result['data']=data
        if not result.get('ok') and (result.get('error') or {}).get('message')=='Unknown tool':
            result['error']['available_tools']=[s['function']['name'] for s in runtime.schemas]
        if result.get('ok') and name=='fit_law_and_rollout' and result.get('data',{}).get('fit_status')=='failed':
            result['data']['usable_law']=False
            result['data']['next_action']='Read the failure message. A backend limitation is not a physical refutation; syntax errors do not imply unsupported operators.'
        log({'tool':name,'normalized':bool(note),'diagnostics':(result.get('data') or {}).get('bf2_diagnostics')})
        return result
    runtime.execute=execute
    runtime.schemas=copy.deepcopy(runtime.schemas)
    for schema in runtime.schemas:
        f=schema['function']
        if f['name']=='insert_probe':f['description']+=' t0 is required at batch top level; separate calls for different t0. For noisy data consult mission sampling advice; preserve long-duration probes, not only dense short windows.'
    original_mission=runtime.mission
    def mission():
        result=original_mission()
        if result.get('ok'):
            result['data']['bf2_diagnostics']=diagnostics(runtime)
            result['data']['expression_syntax']=result['data']['expression_syntax'].replace('Linear interpolation requires observations covering the full rollout interval; no extrapolation.','Official autonomous rollout is used. Acquired body observations constrain background hypotheses; future observations are never replayed.')
        return result
    runtime.mission=mission
    return runtime

OVERRIDE='''\n\nBF2正式接口及研究质量说明（替换旧版插值与提交门槛描述）：
轨迹计算始终使用官方rift.rollout；dt仅是观测间隔。sum_b的背景通过自主动力学推演，不是把天体观测插值作为未来输入。线性/圆周背景不成立时，适配层可以尝试同一候选力律作用于背景的共享动力学假设；这只是候选假设，不证明天体对探针施力。背景推演限制不能当作物理反证。
submit没有1e-5或剩余40秒硬门槛；save_draft可随时保存选中配对，最后30秒封存。无需等待计时器；禁止为了耗时间反复read_file/list_files。继续研究应有具体待排除假设。物理解释控制在约300中文字，留下验证时间。
属性同时变化而轨迹相同不能证明属性无关，特别要警惕比值不变。用相同位置、速度、t0，只改变一个属性的对照；不要从世界编号猜关系。零训练误差不等于可辨识性，验证要改变当前尚未检验的控制量。尽早用至少一条合法长探针（约10秒、接近50点）和偏轴初态检验短窗候选；只改变发射t0而保持短窗口不能替代连续长轨迹。
有噪声时看bf2_diagnostics：二维RMSE纯噪声基准约sqrt(2)*每坐标sigma。位置误差接近噪声时仍必须看速度残差；速度残差显著大于噪声时继续辨别结构，不可宣布只是位置噪声。一个SR候选失败不证明另一个候选正确。fit_status=not_converged只表示优化终止状态，需综合误差和验证，不能单凭标签否定。
采样间隔建议是软约束：dt太小放大差分噪声且缩短总时长，dt太大漏掉快速变化；参考公开噪声、运动尺度与返回的建议，不机械追求最小dt。新候选在最终提交前尽量做固定law_id的留出验证，声明和解释对应实际选中的同一候选。'''

class ProtocolClient:
    def __init__(self,client):self._client=client
    def __getattr__(self,name):return getattr(self._client,name)
    def complete(self,messages,**kwargs):
        if messages and messages[0].get('role')=='system' and 'BF2正式接口及研究质量说明' not in messages[0]['content']:
            p=messages[0]['content']
            p=p.replace('新版工具按已取得的天体位置时间序列分段线性插值，禁止范围外外推；使用天体位置的候选需要观测覆盖整段推演时间，必要时补充 observe_body。','正式接口使用自主背景动力学和官方rollout，背景观测用于拟合假设。')
            p=p.replace('提交硬门槛：所选规律训练位置RMSE严格<1e-5，或剩余时间严格<40秒；否则继续研究。save_draft也不得绕过。','提交时依据验证与适用范围作判断；可提前save_draft，无需等待计时器。')
            messages[0]['content']=p+OVERRIDE
        return self._client.complete(messages,**kwargs)
