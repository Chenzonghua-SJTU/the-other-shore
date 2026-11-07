"""The complete model-facing action surface: current-session file tools and scientific actions."""
IDS = {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'uniqueItems': True, 'maxItems': 102}
NUM = {'type': 'number'}
VECTOR = {'type': 'array', 'items': NUM, 'minItems': 2, 'maxItems': 2}


def obj(properties, required=None):
    return {'type': 'object', 'properties': properties,
            'required': list(properties) if required is None else required, 'additionalProperties': False}


def tool(name, description, properties, required=None):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': obj(properties, required)}}


SCHEMAS = [
    tool('observe_body', '获取指定绝对物理时刻的全体可见天体位置；天体可能运动。时刻严格递增且对齐任务网格，一次最多50时刻并消耗一次观察额度。新任务的第一束光已免费提供8时刻天体位置；追加观测用于扩展或加密时间覆盖。返回观测预览，完整数组按返回路径用read_file获取。',
         {'times': {'type': 'array', 'items': NUM, 'minItems': 1, 'maxItems': 50}}),
    tool('insert_probe', '真实投放一批探针；dt是采样间隔，K包含初态。加速度仅由观测速度中心差分估计。返回档案编号和明确标注的预览；完整数组按返回路径用read_file获取。',
         {'probes': {'type': 'array', 'minItems': 1, 'maxItems': 8,
                     'items': obj({'r0': VECTOR, 'v0': VECTOR, 'properties': obj({'mass': NUM, 'charge': NUM}, [])})},
          't0': NUM, 'dt': NUM, 'n_samples': {'type': 'integer', 'minimum': 1, 'maximum': 50}}),
    tool('fit_law_and_rollout', '拟合二维加速度候选，可依赖绝对t,x,y,vx,vy、当前瞬时天体位置及公开属性。新版sum_b在每个积分时刻从已取得的天体时间序列插值，使用sum_b需观测覆盖完整训练时段；缺少覆盖时先observe_body。使用任务DSL；最多5标量参数，禁止把拟合常数写入公式。目标为每条训练轨迹位置MSE等权平均；输出冻结公式、参数和训练输入快照。局部优化不保证全局最优。',
         {'law_spec': obj({'ax': {'type': 'string', 'maxLength': 600}, 'ay': {'type': 'string', 'maxLength': 600}}),
          'parameter_spec': {'type': 'array', 'maxItems': 5, 'items': obj({'name': {'type': 'string'}, 'initial': NUM, 'lower': NUM, 'upper': NUM}, ['name', 'initial'])},
          'train_probe_ids': IDS}),
    tool('validate_law', '固定公式与参数，在未参与该版本拟合的整条轨迹上验证。新版使用当次已取得的天体时间序列作为外部状态输入，并保存来源快照；范围不足请补observe_body，绝不重拟合参数。返回相对观测误差，包含天体插值误差，不设通过阈值。曾参与搜索或诊断的数据也允许验证，但会标注为参数留出验证；完全独立的结构检验应使用新实验。',
         {'law_id': {'type': 'string'}, 'test_probe_ids': IDS}),
    tool('list_files', '列出当前探索会话的公开实验文件或目录，不访问其他会话、源码或隐藏评测。返回路径供read_file和run_code使用。',
         {'path': {'type':'string','maxLength':300}, 'offset': {'type':'integer','minimum':0},
          'limit': {'type':'integer','minimum':1,'maximum':200}}, []),
    tool('read_file', '按路径读取当前会话已取得的公开文件。offset/limit按Unicode字符分页，返回next_offset；不静默截断。二进制产物可交run_code的read_bytes读取。',
         {'path': {'type':'string','minLength':1,'maxLength':300}, 'offset': {'type':'integer','minimum':0},
          'limit': {'type':'integer','minimum':1,'maximum':20000}}, ['path']),
    tool('analyze_residual', '确定性残差诊断：预测减观测、预测加速度减差分估计。报告均值、RMSE、最大值及与t,x,y,vx,vy的相关性；未定义相关返回null及原因，相关不代表因果。',
         {'law_id': {'type': 'string'}, 'probe_ids': IDS}),
    tool('submit', '提交固定已拟合版本；本地accepted仅指格式/有限输出/可执行性检查成功，立即结束会话，不表示物理正确。阶数1表示作用决定速度、2表示作用决定加速度；hidden_matter表示位置不可见的源；time_dependence表示固定完整状态后显式时间依赖。',
         {'law_id': {'type': 'string'},
          'physical_explanation': {'type': 'string', 'minLength': 1, 'maxLength': 20000, 'pattern': r'\S',
                                   'description': '用中文解释提交公式的物理含义：各项的作用方向与状态依赖、参数含义和单位、可由公式推导的运动性质、支持解释的实验/验证编号，以及适用范围和不可辨识性。区分观测证据、数学推论与物理假说，不得只复述公式或以低误差证明机制唯一。可分段。'},
          'claims': obj({'dynamics_order': {'type': 'integer', 'enum': [1, 2]},
                                                   'hidden_matter': {'type': 'boolean'}, 'time_dependence': {'type': 'boolean'}})})
]

CODE_SCHEMA = tool('run_code',
    '在隔离Python环境运行脚本，将结果赋给result。files可省略，默认[]；需要数据时选择list_files或工具返回的当前会话路径。'
    '代码中read_json(path)/read_file(path)/read_bytes(path)读取所选完整文件的只读快照，不必抄写数据；available_files列出本次路径。'
    'probe_arrays(record,time_axis="acceleration")按估计加速度时刻对齐，默认精确匹配。'
    'numpy/scipy/sympy/matplotlib可用，output/可导出png/json/csv/txt。每次新容器、无网络/宿主目录，90秒/1GiB/1CPU。'
    'NumPy结果自动转JSON；NaN/Infinity报错。返回精简结果和产物路径，完整结果可用read_file读取。'
    '执行成功不代表科学正确，仍需正式拟合、验证和submit。',
    {'code': {'type':'string','minLength':1,'maxLength':60000},
     'files': {'type':'array','items':{'type':'string','minLength':1,'maxLength':300},'uniqueItems':True,'maxItems':128}}, ['code'])


SYMBOLIC_SCHEMA = tool('symbolic_regression',
    '在当前会话指定训练探针的完整位置和速度观测上，以速度差分估计加速度，搜索基本代数、实数幂、三角、指数、对数候选。简单表达式优先，但不排除复杂运算。'
    '只搜索t,x,y,vx,vy和这些探针共同提供的mass/charge；当前不支持天体集合位置输入。'
    '变量不变化不能证明无依赖；先用控制实验提供可辨识的数据，保留已有属性对照。时间包含在300秒内，建议一次40–50秒，两轴合计；余时不足会拒绝。'
    '返回去重排名的至多3个完整law_spec、parameter_spec、fitted_parameters和residuals，区分加速度搜索、轨迹拟合及速度积分残差；附单轴候选、来源和覆盖诊断；搜索审计保存在audit_path。'
    '返回候选是待检验假说，训练检查不能代替独立验证；没有合格候选时如实返回空列表。'
    '候选须fit_law_and_rollout登记并按位置轨迹重拟合，再用未参与当前law参数拟合的轨迹validate_law；用过的发现数据会标注为参数留出验证，不冒充完全独立的结构检验。'
    '预算结束但有候选是正常可用结果；单轴joint_fit_ready=false不能直接当完整力律提交，组合仍须符合5参数上限并验证。'
    '不可用/无好候选时按已有工具继续研究，不要机械重复相同搜索；本工具不投放探针、不自动提交。',
    {'train_probe_ids': IDS,
     'time_budget_seconds': {'type': 'number', 'minimum': 1, 'maximum': 60}})


def tool_schemas(code_enabled=False, *, ruleset='e3', symbolic_enabled=False):
    import copy
    schemas = copy.deepcopy(SCHEMAS + ([CODE_SCHEMA] if code_enabled else []))
    if ruleset in ('e4', 'e5'):
        draft = copy.deepcopy(next(s for s in schemas if s['function']['name'] == 'submit'))
        draft['function']['name'] = 'save_draft'
        draft['function']['description'] = ('保存你明确选择的已拟合law_id及你依据观测判断的claims。'
            '格式与有限输出验证成功才替换上份草稿，不结束探索。预算截止时封存最后及时有效配对；'
            '不会自动选择规律、补写claims或检查物理真值。')
        schemas.append(draft)
    if symbolic_enabled:
        if ruleset not in ('e4', 'e5'):
            raise ValueError('symbolic_regression requires ruleset=e4 or e5')
        search = copy.deepcopy(SYMBOLIC_SCHEMA)
        if ruleset == 'e4':
            search['function']['description'] += ('本场观测含噪；速度差分会放大噪声，搜索损失不代表真实加速度误差。'
                                                '候选须在含噪位置轨迹上重拟合并用保留探针验证，不以噪声波动作为新物理项。')
        schemas.append(search)
    return schemas
