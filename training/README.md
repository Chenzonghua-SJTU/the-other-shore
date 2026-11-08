# 训练接口

`DiscoveryEnv` 把现有带噪控制台与固定评测链封装为 Python 的 `reset` / `step` 接口，便于接入自己的 LLM rollout worker、策略优化器或 Agent。它处理结构化工具动作，不依赖 Gymnasium，也不包含 PPO/GRPO 等训练算法。

在 `shore-worlds-20` 根目录运行，使用已有 NumPy/SciPy 环境。

```python
from training import DiscoveryEnv

env = DiscoveryEnv(clock_mode="frozen")
observation, info = env.reset(seed=42, options={"world": "7"})
mission = observation["results"][0]["result"]
# 第1轮环境交互中顺序执行多个工具；投放参数沿用公开首光初态。
probe_request = {key: mission["first_light"][key]
                 for key in ("probes", "t_launch", "readout", "source_properties")}
observation, reward, terminated, truncated, info = env.step([
    {"method": "observe", "params": {"times": [0.0, 0.5, 1.0]}},
    {"method": "insert_probe", "params": probe_request},
    {"method": "archive", "params": {}},
])
assert observation["interaction_round"] == 1
assert observation["interaction_rounds_remaining"] == 2
# 本地推理/拟合/模型调用不消耗环境交互轮；准备好后发起下一轮。
# 可在第3轮环境交互的批次内提交：
# {"method": "submit", "params": {
#     "law_path": "/absolute/path/law.py", "claims": model_generated_claims
# }}
```

`reset` 自动获取初始 mission 和免费首光，不消耗环境交互轮。`step` 接受单个 `{method, params}` 请求、有序请求列表，或显式包装 `{"method": "batch", "params": {"actions": [...]}}`；禁止嵌套 batch。每次 `step` 是一轮与环境的交互，固定最多三轮，每轮可含最多64个环境工具请求。模型调用次数、内部推理和本地拟合/代码工具次数不计入这三轮。

支持 `mission`、`observe`、`insert_probe`、`launch`、`archive`、`submit`。`insert_probe` 与 `launch` 完全等价，均需 `probes`、`t_launch`、`readout`、`source_properties`，逐探针扣相同能量。比如同一批次的 observe + insert_probe + archive 只计一轮，第三轮仍可先 observe 再 submit。一轮内无效请求返回错误，后续请求继续执行；非法实验不额外扣能量。合法提交立即结束，提交后的同批动作不执行。

观察中的 `interaction_round` / `interaction_rounds_remaining` 是明确的环境交互计数；`turn` / `turns_remaining` 保留为同值兼容字段。`MAX_INTERACTION_ROUNDS=3`，`MAX_TURNS` 为兼容别名；终局原因 `turn_limit` 也保留旧拼写，表示环境交互轮耗尽。

三轮环境交互内未提交或真实探索期限耗尽，均为任务内的终止失败：返回 `terminated=True`、`truncated=False`、奖励0，原因分别为 `turn_limit` 与 `deadline`。成功提交同样返回 `terminated=True`，终局奖励为固定评分的 `total / 100`。所有结束状态的 `info["bootstrap_allowed"]` 为 `False`，中间状态为 `True`，价值目标不得从这些失败终局继续 bootstrap。中间奖励均为0。评分使用当前会话实际消耗的能量和同一无噪声验证集，保留80/15/5公式。终局 `info["score"]` 只含 `prediction`、`law`、`saving`、`total`、`energy_used`；不给出声明真值、参考轨迹或验证初态。源文件、世界配置或验证案例改变时，实例内的参考缓存失效。基础设施/参考生成失败直接抛出异常，避免把执行失败当成策略表现为0。候选中存在无效轨迹时，沿用固定评测的零预言分规则。

`clock_mode="realtime"` 是默认值，保留300秒真实探索期限。`clock_mode="frozen"` 是显式训练模式，不计算真实经过时间，因而相同 seed、世界、动作在同一数值环境中可重现；其余能量、观察与三轮环境交互限制保持有效。终局评分在探索截止之后进行，不计入探索时限。每个并行 rollout worker 应创建自己的环境实例，不共享会话。

默认 `anonymize_world_id=True`，把 mission/archive 中的世界编号替换为 `training`，减少按编号查表。trainer 可通过 `env.trainer_metadata` 读取真实世界编号和噪声种子；该元数据不要放入策略输入。匿名化不构成进程隔离，世界范围/背景本身仍可能暴露任务身份；训练与留出测试请使用不同的世界集合，例如分别传 `worlds=["7", "8", ...]`。这20个世界均为公开固定参数任务，仅改变噪声种子并不产生新物理定律，不能据此宣称未知机制泛化。

环境通过现有 `law.py` 装载器执行候选，适用于可信的本地训练代码。它不是不可信代码沙箱；需要隔离时，在外部进程/容器中运行整个 worker。这里没有修改标准 runner，正式风格 Agent 自测继续用 `development_worlds.runner`。

## 命令行示例

```sh
# 三轮接口演示：没有候选时终止失败（terminated），奖励0。
python -m examples.rl_episode --world 7 --seed 42 --clock-mode frozen

# 接收 Agent 已产生的候选与声明，在第3轮提交、执行真实评测。
python -m examples.rl_episode --world 7 \
  --law /absolute/path/law.py --claims /absolute/path/claims.json

# 每个世界一局，每个随机排列块覆盖全部20界，保存带噪的探索数据。
python -m training.generate --episodes 20 --seed 42 --output data/exploration-20.jsonl

# 指定训练世界；输出必须使用新路径。
python -m training.generate --worlds 7 8 9 --episodes 30 --seed 2026 \
  --output data/train-7-9.jsonl

python -m unittest discover -s tests -p 'test_training.py' -v
```

生成器使用 `frozen` 模式；每个覆盖块对选定世界集做一次由种子确定的随机排列，避免以 `episode_index % 世界数` 直接暴露世界编号，真实顺序仅保存在manifest。每局第1轮执行 observe + insert_probe + archive，第2轮 observe + archive，第3轮 archive；它不拟合也不提交候选，所以这些文件是探索轨迹，奖励为0，不是已训练的策略或带正确规律标签的数据集。JSONL逐局写入，每行含初始观察和动作/后继观察；旁边的 `.manifest.json` 单独记录世界编号、每局种子、源码哈希与生成状态。manifest 中的 `status="complete"` 才表示整批完成。重跑相同种子在相同数值环境中可重现观察。输出及manifest任一个已存在均拒绝覆写。
