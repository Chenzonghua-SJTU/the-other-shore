"""A compact same-model syntax-repair context; no hidden physics or candidate selection."""
import json
import ast
import copy

def syntax_diagnostics(arguments):
    findings=[]
    for component in ('law_spec','background_law_spec'):
        for axis,source in (arguments.get(component) or {}).items():
            try:tree=ast.parse(source,mode='eval')
            except (SyntaxError,TypeError):continue
            parents={child:parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
            for node in ast.walk(tree):
                if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub)):
                    for child in (node.left,node.right):
                        try:literal=ast.literal_eval(child)
                        except (ValueError,TypeError):continue
                        if isinstance(literal,(int,float)) and literal!=0:
                            findings.append({'field':component+'.'+axis,'error':'numeric offset must be a declared parameter','expression':ast.get_source_segment(source,node)})
                if isinstance(node,ast.BinOp) and isinstance(node.op,ast.BitXor):
                    findings.append({'field':component+'.'+axis,'error':'^ is unsupported XOR; use ** only if exponentiation was intended','expression':ast.get_source_segment(source,node)})
                if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)) and (type(node.value) is not int or node.value not in (0,1)):
                    target=node;parent=parents.get(target);value=node.value
                    if isinstance(parent,ast.UnaryOp) and isinstance(parent.op,(ast.UAdd,ast.USub)):
                        target=parent;value=ast.literal_eval(target);parent=parents.get(target)
                    power=isinstance(parent,ast.BinOp) and isinstance(parent.op,ast.Pow) and parent.right is target and type(value) is int and -3<=value<=3
                    if not power:findings.append({'field':component+'.'+axis,'error':'coefficient literal must use an explicitly declared parameter','expression':ast.get_source_segment(source,node)})
                if isinstance(node,(ast.Compare,ast.IfExp,ast.BoolOp)):
                    findings.append({'field':component+'.'+axis,'error':'unsupported '+type(node).__name__,'expression':ast.get_source_segment(source,node)})
    return findings

def repair_messages(request,runtime):
    instructions='''BF2正式接口及研究质量说明：Correct the rejected call. Fix EVERY listed invalid subexpression in BOTH ax and ay. Do not echo the previous answer. Return the corrected tool call. 当前是同一科研Agent的局部工具参数纠错步骤，仍属于同一会话，消耗原墙钟，不访问环境。
上一份由你提出的候选未通过DSL语法检查。请根据具体报错，修正同一候选的表达方式，返回一次fit_law_and_rollout。不要继续原样重复，不要推测世界答案或换成一个已知物理定律。若原候选使用了未定义的缩写，你必须明确表达你原先意指的量。
DSL标量状态只有t,x,y,vx,vy。没有隐含r、r2、rho或速度模长变量：应写出你意指的坐标表达式。公开属性用q('属性名')，仅可用给定属性；q的括号和引号必须正确闭合。bx/by只允许在sum_b(...)内部引用，不能在外部引用。所有引用bx/by的项都要在sum_b内部，例如sum_b(bx)+by仍然非法，因为by在外面；使用sum_b时须明确background_law_spec，不能把不同天体合并成一个未定义坐标。
允许sin,cos,exp,log,sqrt,abs以及基本算术。不支持比较符号<、>、<=、>=、==，不支持布尔掩码、if/else或分段表达式。系数属于parameter_spec，探针和背景公式的并集最多5个标量，允许系数只用于其中一份公式。结构常数仅0、±1；乘方运算符是**，不是^；指数结构字面量仅整数-3..3。声明参数后公式必须使用其名称，不能抄initial数字替代名称。sqrt(E)可表达E的平方根，不写E**0.5；非结构常数必须明确声明参数，不能靠数字算术规避限制。非零的加减偏移即使是1也必须声明参数，例如分母x*x+y*y+1非法，应使用你明确声明的偏移参数。
优先保留原候选的作用方向、依赖关系、已有参数和训练探针，除非这些本身是报错原因。如果原假设无法在该DSL内表达，不要原样重发，也不能谎称等价转换；依据附带已获取观测自行提出可表达的替代候选，并明确说明这是新假设，实际拟合结果会检验它；不要添加未取得的观测、假拟合结果或声明。工具会实际重新验证公式；此次输出仍由你负责。'''
    request=dict(request)
    if request.get('original_arguments'):request.pop('previous_explanation',None)
    payload={**request,'syntax_diagnostics':syntax_diagnostics(request.get('original_arguments',{})),'allowed_properties':(runtime.task or {}).get('allowed_properties',[]),
             'available_probe_ids':list(runtime.store.records.get('probe',{}))}
    if 'unsupported DSL syntax' in request.get('error',{}).get('message',''):
        payload['acquired_probes']=[{k:p[k] for k in ('probe_id','initial_conditions','times','r_obs','v_obs') if k in p} for p in runtime.store.records.get('probe',{}).values()]
        payload['public_task']=runtime.task
    return [{'role':'system','content':instructions},{'role':'user','content':json.dumps(payload,ensure_ascii=False,allow_nan=False)}]

def submission_messages(runtime, messages):
    """A deadline handoff containing real candidates/evidence, not failed actions."""
    from .submission_policy import usable_laws
    candidates=[]
    for law in usable_laws(runtime):
        candidate={k:law[k] for k in ('law_id','law_spec','background_law_spec','fitted_parameters','fit_status','train_probe_ids') if k in law}
        candidate['errors']={'overall':(law.get('errors') or {}).get('overall',{})}
        candidates.append(candidate)
    records=runtime.store.records
    validations=[{**{k:v[k] for k in ('validation_id','law_id','test_probe_ids','validation_scope','independent_of_discovery') if k in v},'errors':{'overall':(v.get('errors') or {}).get('overall',{})}} for v in records.get('validation',{}).values()]
    notes=[m['content'][-600:] for m in messages if m.get('role')=='assistant' and m.get('content')][-2:]
    probes=[{'probe_id':key,'initial_conditions':p.get('initial_conditions'),'samples':len(p.get('times',[])),'time_span':[p['times'][0],p['times'][-1]] if p.get('times') else None} for key,p in records.get('probe',{}).items()]
    payload={'candidates':candidates,'validations':validations,'acquired_experiments':probes,'your_recent_analysis':notes,'public_task':runtime.task,'budget':runtime.budget()}
    instructions='''BF2正式接口及研究质量说明：这是同一探索会话的最终提交阶段；仅改变上下文组织，不增加环境轮次或墙钟。以下候选、拟合/验证结果和分析均来自你刚才的真实会话，没有新增实验或隐藏答案。必须现在调用 submit，一次调用，不能再拟合、编程或读取文件。自行根据证据选择一个现有law_id，填写claims（dynamics_order为1或2、hidden_matter和time_dependence为布尔值）及简洁中文physical_explanation。不得创造候选或声称未知验证成功；候选不完美也应如实说明并提交。解释控制在200字左右，说明公式方向与主要依赖、证据和未验证范围，不复写大量公式、数字或旧工具调用。time_dependence指完整状态固定后仍有显式t依赖，不把天体运动误认作显式时间依赖。不要等待更多信息；本阶段唯一动作是submit；解释不超过100字，整份回复必须短而完整。'''
    return [{'role':'system','content':instructions},{'role':'user','content':json.dumps(payload,ensure_ascii=False,allow_nan=False)}]

def candidate_messages(runtime, messages):
    """Reorient stalled analysis using only this session's acquired data."""
    records=runtime.store.records
    probes=[{k:p[k] for k in ('probe_id','initial_conditions','times','r_obs','v_obs') if k in p} for p in records.get('probe',{}).values()]
    notes=[m['content'] for m in messages if m.get('role')=='assistant' and m.get('content')][-3:]
    computed=[{'execution_id':r.get('execution_id'),'result':r.get('result'),'stdout':r.get('stdout')} for r in records.get('code_execution',{}).values() if r.get('status')=='success' and (r.get('result') is not None or len(r.get('stdout') or '')>500)][-4:]
    failed=[{k:r[k] for k in ('law_id','law_spec','background_law_spec','parameter_spec','fitted_parameters','train_probe_ids','fit_status','message','errors') if k in r} for r in records.get('law',{}).values() if r.get('fit_status') in ('failed','error')]
    calls={};feedback={}
    for m in messages:
        for call in m.get('tool_calls') or []:
            function=call.get('function') or {};raw=function.get('arguments')
            try:args=json.loads(raw)
            except (ValueError,TypeError):args={'rejected_argument_text':raw}
            calls[call.get('id')]={'tool':function.get('name'),'arguments':args}
        if m.get('role')!='tool':continue
        try:r=json.loads(m.get('content') or '{}')
        except (TypeError,ValueError):continue
        if not isinstance(r,dict):continue
        data=r.get('data') or {}
        if not r.get('ok') or (isinstance(data,dict) and data.get('fit_status')=='failed'):
            call=calls.get(m.get('tool_call_id'))
            item={**(call or {}),'error':r.get('error'),'fit_failure':{k:data[k] for k in ('law_id','fit_status','message') if k in data} if isinstance(data,dict) else None}
            key=json.dumps(call or item,sort_keys=True,ensure_ascii=False)
            prior=feedback.pop(key,None)
            if prior is not None:
                prior['latest_error']=r.get('error');feedback[key]=prior
            else:feedback[key]=item
    payload={'public_task':runtime.task,'acquired_probes':probes,'acquired_body_observations':list(records.get('body_observation',{}).values()),'your_recent_analysis':notes,'your_computed_results':computed,'failed_candidates':failed,'recent_tool_failures':list(feedback.values())[-10:],'budget':runtime.budget()}
    instructions='''BF2正式接口及研究质量说明：这是同一科研会话的候选登记阶段，仍使用原墙钟和原环境批次。你已经取得下列真实观测，但尚无可提交候选。现在必须调用一次fit_law_and_rollout，明确给出你根据这些数据推断的law_spec、parameter_spec和实际train_probe_ids。不要调用run_code、符号回归或仅输出计划；已有计算结果附在后面。下方failed_candidates和recent_tool_failures记录了真实失败，不能原样再次提交这些参数。数值失败需根据具体原因修改初值/范围或重新考虑假设，不能把已有失败候选说成可提交，也不能只改参数名继续相同计算。模型由你提出，工具只拟合参数。考虑属性对照、位置/速度/时间依赖，不根据世界编号猜答案；候选允许不完美，之后仍可检验和修正。
DSL只能使用t,x,y,vx,vy、q('公开属性名')、sin/cos/exp/log/sqrt/abs以及基本算术。没有r/rho等隐含缩写；不支持比较、分段if或布尔掩码。所有物理系数和非零偏移常数都必须声明参数，探针与背景参数并集最多5个；乘方写**，不写^；字面指数只允许-3到3的整数，平方根用sqrt。已声明参数必须在公式里使用名称，不能照抄initial数字。bx/by只在sum_b(...)内有效；若依赖背景，须显式提供background_law_spec，不自行假定静止源。只返回真实拟合工具调用，不重复旧计划。'''
    return [{'role':'system','content':instructions},{'role':'user','content':json.dumps(payload,ensure_ascii=False,allow_nan=False)}]


def envelope_repair_messages(request,runtime):
    """Only restructure rejected model-authored actions; execute nothing here."""
    plans=[]
    for call in request['rejected_calls']:
        function=call.get('function',{})
        try:args=json.loads(function.get('arguments','{}'))
        except (ValueError,TypeError):args={'invalid_argument_text':function.get('arguments')}
        if function.get('name')=='observe_body' and isinstance(args.get('times'),list) and args['times']:
            times=args.pop('times')
            args['sampling_plan_to_reexpress']={'original_count':len(times),'start':times[0],'end':times[-1],'original_intervals':sorted(set(round(b-a,12) for a,b in zip(times,times[1:]))),'instruction':'Choose a new legal times array within the tool maxItems and public time grid; original oversize array is deliberately omitted to prevent copying it.'}
        plans.append({'tool':function.get('name'),'planned_parameters':args})
    payload={'planned_actions':plans,'error':request['error'],'public_task':runtime.task,'budget':runtime.budget()}
    instructions="""BF2正式接口及研究质量说明：上一批工具参数未通过预检，整批未执行且未消耗环境轮次。你仍是本局同一Agent，原墙钟继续计时。请只修正这份由你提出的计划中明确报错的参数，重新给出完整工具调用；不要重复无效调用。maxItems表示数组长度上限，actual_items表示你提供的数量。依据工具schema自行选择合法的采样时刻，保持原实验意图；不要用代码块或口头说明代替调用。每份回复最多8个调用，探针总数不得超过剩余能量；不要通过多次回复绕过环境批次限制。不要添加未经获取的数据、候选公式或假结果。工具schema仍是当前允许的工具及准确参数限制。"""
    return [{'role':'system','content':instructions},{'role':'user','content':json.dumps(payload,ensure_ascii=False,allow_nan=False)}]


def code_repair_messages(request,runtime):
    payload={**request,'budget':runtime.budget()}
    instructions="""BF2正式接口及研究质量说明：你上一次run_code因以下真实Python异常而失败，没有产生有效结论。请保留原计算意图，检查所有数组维度，修正代码后调用run_code；不要原样重发或转为物理猜测。当前仍是同一会话的本地计算，消耗原墙钟，不能获得额外观测。输入文件仍来自原调用的files；每次容器全新，用read_json/read_file读取显式列出的虚拟路径。对于形状(N,2)的对应向量逐行内积，应得到(N,)的结果，例如np.sum(a*b,axis=1)；a@b.T会生成(N,N)而不是逐行内积。一维尺度乘二维向量需明确增加列轴[:,None]。布尔索引的维度必须和被索引数组匹配；不要靠盲目flatten掩盖不正确计算。先做shape断言；不能虚构数组数值或通过删除失败步骤伪造成功。结果用一个简短result字典返回，最多20秒计算。修正仍由你生成并由真实沙箱执行。"""
    return [{'role':'system','content':instructions},{'role':'user','content':json.dumps(payload,ensure_ascii=False,allow_nan=False)}]


def review_messages(runtime,messages,trigger):
    """Preserve actual evidence when the full history stalls on a fitted candidate."""
    base=candidate_messages(runtime,messages)
    payload=json.loads(base[1]['content'])
    payload['review_trigger']=trigger
    payload['existing_candidates']=[{k:r[k] for k in ('law_id','law_spec','background_law_spec','parameter_spec','fitted_parameters','train_probe_ids','fit_status','errors') if k in r} for r in runtime.store.records.get('law',{}).values()]
    payload['existing_validations']=list(runtime.store.records.get('validation',{}).values())
    instructions="""BF2正式接口及研究质量说明：这是同一会话的一次证据复核，上一次行为触发科研检查或重复调用保护。没有新增实验、候选、时间或环境轮次；下面全部是本局实际记录。你已有可提交候选，但“可执行”不代表科学正确。不要原样重复上次失败或无新信息的调用。
先用简短文字核对当前候选的数学表达式、计算结果和解释是否一致，再执行最有信息量的合法下一步。若你在解释中承认公式与数据不符，必须通过fit_law_and_rollout实际登记修正候选；仅把解释改写成另一条规律，不会改变提交law_id对应的预测函数。不能把converged当成正确，也不能靠剔除高误差轨迹来掩盖结构失配；应比较候选对共同训练数据及未拟合整条轨迹的预测。位置误差接近噪声也须核对速度误差和控制对照。避免重复读取或打印已有结果，先利用所给数据和计算结论。你自行决定是否需新假设、重新拟合、留出验证，或已有证据足以提交；不能由世界编号猜答案。最终声明对应实际选中候选，未验证范围要如实说明。
工具仍受当前阶段schema、5参数DSL、剩余能量、3环境批次和300秒墙钟限制。依赖天体时需显式背景公式；不在本步骤注入物理默认值。简短复核后给出工具调用，不写长篇计划拖延。"""
    return [{'role':'system','content':instructions},{'role':'user','content':json.dumps(payload,ensure_ascii=False,allow_nan=False)}]


def exploration_messages(runtime,messages):
    """Let the same model plan the first active batch before analysis consumes its time."""
    payload=json.loads(candidate_messages(runtime,messages)[1]['content'])
    instructions="""BF2正式接口及研究质量说明：当前还没有执行第一批主动环境实验。首光上的本地分析应先暂停；此步骤仍使用本局原墙钟和三次环境交互预算。根据下方公开任务、已获取首光及你自己的计算结果，选择并调用第一批实验。至少安排一项合法insert_probe；如需观察背景，将observe_body放在同一助手回复。一次回复最多8项工具操作，整批探针总数不得超过剩余能量。通常使用部分能量，保留第二批对照机会和最后一批submit；具体初态、属性、t0、dt和采样点由你依当前证据选择，不提供隐藏规律或默认探针。
优先检验仍混淆的控制量：相同初态只改变一个属性或速度，或用偏轴初态检查方向依赖。遵守公开位置/速度/时间网格、采样数、属性范围和工具schema。不要在此回复编程、拟合或长篇讨论；给出完整工具调用。若先前代码报错，不需要先修好代码才能取得新实验数据，之后仍可修正计算。没有新增免费观测，也不能把多次助手回复称为同一批。"""
    return [{'role':'system','content':instructions},{'role':'user','content':json.dumps(payload,ensure_ascii=False,allow_nan=False)}]
