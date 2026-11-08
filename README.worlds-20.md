# 服务器部署说明

本目录已更新到 20 个世界，保留服务器已有的自定义 Agent、模型配置与根 README。以下说明来自完整训练包；配套基线 Agent 保存在 training-releases 中，当前根目录运行的是原有自定义 Agent。启动自定义 Agent 请使用 --agent-dir 与 --models-file，勿使用默认 --model 启动方式。部署记录及服务器验证见 DEPLOYMENT.worlds-20.md。

# 彼岸 20 世界训练包

从 `shore-contest-starter-v15` 独立派生，用于 Agent 实验设计、规律发现、离线数据生成和 RL 环境接入。共 **20 个固定参数世界：原 1–6 + 新增 7–20**。原始 starter 目录未改动。本包属于本地研究扩展，不是官方新增测试集，也不预测正式隐藏宇宙。

## 世界目录

所有世界都是二维、二阶、独立被动探针；每条真规律最多 5 个全局标量。新增世界改变作用机制，而不只是更换原世界的常数。

| ID | 世界 | 主要考察点 |
|---|---|---|
| 1 | 原径向引力 | 距离规律、源强与惯性 |
| 2 | 原各向同性弹簧 | 线性回复、属性控制 |
| 3 | 原旋转各向异性弹簧 | 主轴与方向差异 |
| 4 | 原屏蔽引力 | 指数衰减尺度 |
| 5 | 原交叉四次势 | 非线性与交叉项 |
| 6 | 原旋转主轴场 | 绝对时间与相位 |
| 7 | 软化开普勒场 | 近场软核、远场平方反比 |
| 8 | 饱和回复场 | 强度从线性向饱和过渡 |
| 9 | 径向 Duffing 场 | 振幅依赖、径向四次势 |
| 10 | 双势阱 | 多平衡点、跨阱运动 |
| 11 | 周期晶格与束缚 | 空间周期与局部陷阱 |
| 12 | 高斯势阱 | 局域吸引、远场衰减 |
| 13 | 环形势阱 | 平衡半径、内斥外吸 |
| 14 | 可见双源场 | 多源叠加、源强辨识 |
| 15 | 偏心隐藏源 | 不可见源位置与强度 |
| 16 | 对称隐藏双源 | 隐藏多源结构 |
| 17 | 脉动弹簧 | 回复强度随时间变化 |
| 18 | 圆周驱动场 | 周期外力、方向相位 |
| 19 | 行波晶格 | 空间与时间耦合 |
| 20 | 脉动隐藏偏心源 | 隐藏源与显式时变共同辨识 |

轨迹示例见 [20 世界总览](validation/world-atlas.png)；生成脚本为 `examples/plot_worlds.py`（额外依赖 Matplotlib）。

公式、参数、合法操作范围和声明标签见 [世界定义](development_worlds/README.md)，机器可读版本见 [world-catalog.json](world-catalog.json)。这些是训练方公开真值；Agent 输入只包含操作范围、可见背景和带噪观测。`training.DiscoveryEnv` 默认隐藏世界编号，避免策略直接按 ID 查表。公开同进程 Python 环境不是进程隔离沙箱。

## 环境

推荐 Python 3.12，安装原版 CPU 依赖（训练环境本身无需 GPU、vLLM 或 Gymnasium）：

```bash
cd shore-worlds-20
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

后面的 `python` 表示已安装 NumPy/SciPy 的解释器。本机可直接使用 `../physics_discovery/.venv/bin/python`；本次实际验证版本会记录在 `validation/report.json`，不把本机版本等同于比赛固定运行环境。

## 运行现有 Agent

保留 v15 `rift.Console` HTTP 协议和 `acceleration(t, bodies, params)` 候选接口。运行器默认选择全部 20 个世界，也可使用 `--worlds 7 14 20` 选择子集。

```bash
python -m development_worlds.runner \
  --agent-dir /absolute/path/to/agent \
  --models-file /absolute/path/to/models.json \
  --model-python /absolute/path/to/model/python \
  --max-interaction-rounds 3 --seed 42 --output results/agent-20-run01
```

已有本地模型服务时可加 `--base-url http://127.0.0.1:8000/v1`。所有输出目录必须是新目录。`summary.json` 包含逐世界分数和 20 世界等权平均分；这个均分不是官方三档加权排名分。自定义客户端可使用 `--agent-command`（放在命令最后），沿用 `RIFT_CONSOLE_URL`、`RIFT_SESSION_TOKEN` 和 `RIFT_OUTPUT_DIR`。

标准会话保持：10 点能量、最多 20 次 observe、单次最多 8 探针、每探针最多 50 个采样时刻、免费首光 2 探针 × 8 时刻、首次 mission 后 300 秒。**三轮指与环境交互三轮，不是三次模型调用。** 每轮可以批量调用多个工具，包括 `observe`、`insert_probe`（v15 的 `launch`）、`submit`；工具仍各自扣除能量或观测次数。第三轮可以在批次内提交。初始 mission/免费首光是会话初始化。本地推理、代码检查、拟合和读取本地记录不单独消耗环境轮次；所有工作仍受真实时间限制。若本地动作触发远程 `save_draft`，该远程操作属于当前交互轮。

配套 Agent 按一个顶层环境动作或含环境操作的 batch 计一轮，纯本地动作/额外模型调用不增加环境轮数。RL 的每次 `step` 表示一整轮，传列表即可多工具调用；在 `step` 之间可任意进行本地推理/拟合。原 v15 HTTP 协议没有轮次分组字段，轮数由上述两个入口管理；使用原始客户端的自定义 Agent 必须自己保持批次边界与三轮上限，服务端无法从单独请求推断分组。

单独评测一个候选：

```bash
python -m development_worlds.evaluation --world 20 \
  --law /absolute/path/to/law.py --claims /absolute/path/to/claims.json \
  --energy-used 4 --output results/candidate20.json
```

评分保留原规则：预测 80 + 三项声明 15 + 节约 5。正式会话的能量取控制台记录；上面单候选命令的能量仅供离线诊断。

## RL 接入与带噪数据

`DiscoveryEnv` 是结构化工具调用环境，提供 Gym 风格五元组，但不依赖 Gymnasium，不声明拥有固定张量 action/observation space。适合生成工具调用的语言模型或自行编码观测/动作的训练程序。

```python
from training import DiscoveryEnv

env = DiscoveryEnv(clock_mode="frozen")  # 显式选择可复现离线训练，不计真实耗时
obs, info = env.reset(seed=42, options={"world": "7"})
mission = obs["results"][0]["result"]
probe_request = {key: mission["first_light"][key]
                 for key in ("probes", "t_launch", "readout", "source_properties")}
obs, reward, terminated, truncated, info = env.step([
    {"method": "observe", "params": {"times": [0.0, 0.5, 1.0]}},
    {"method": "insert_probe", "params": probe_request},
    {"method": "archive", "params": {}},
])
# 以上整个批次计1轮环境交互；本地推理/模型调用不计轮。
assert obs["interaction_round"] == 1
# 可在第3轮环境交互的批次内 submit。
# env.step({"method": "submit", "params": {
#     "law_path": "/absolute/path/to/generated/law.py",
#     "claims": claims_generated_by_your_agent}})
```

默认 `clock_mode="realtime"` 执行 300 秒限制；`frozen` 专供离线训练，保留能量、观测和环境交互轮配额。提交时才给终局奖励 `total/100`；中间步骤不返回参考轨迹或真标签。耗尽三轮环境交互未提交或真实探索超时均为任务终止失败（`terminated=True`、`truncated=False`）、奖励 0；所有终局的 `info["bootstrap_allowed"]=False`，避免价值目标跨终局 bootstrap。无噪声真值只供 evaluator 计分。同种子复现噪声；重置种子不会生成新的物理定律。世界参数目前固定，需要另留完整机制/世界作为泛化评测集，不能把相同世界的新噪声种子当作新任务。

生成合法、带噪的随机实验轨迹（用于观测/工具策略数据，不是已训练的 RL 策略）：

```bash
python -m training.generate --episodes 20 --seed 42 --output data/episodes.jsonl
python -m examples.rl_episode
```

数据与清单分开保存，清单记录种子和任务身份。实际 RL 优化器、模型训练和 GPU Agent 成绩不包含在本次交付中。

## 验证

```bash
python -m unittest discover -s tests -v
python -m development_worlds.validate_suite --output validation/new-report.json
```

验证包括：20 世界固定验证集全轨迹步长减半、独立参考 `law.py` 经真实候选积分器评分、随机合法初态有限性、原 1–6 与原 starter 数值回归，以及控制台和 RL 预算/重置/终止检查。报告中的 oracle 分数只验证模拟和评分链路，不是 Agent 发现能力。

本次实测：20/20 世界通过，共 146 组固定案例（584 探针、11,680 个位置采样）；全程步长减半最大位置差约 `1.47e-8`，独立参考规律最大轨迹 MSE 约 `1.56e-27`。原 1–6 全部固定案例、首光和背景数值精确一致，原 starter 90 个文件校验均未变化。这是基础设施正确性验收，不是 Agent 成绩。

- [36 项自动化测试日志](validation/unit-tests.txt)，包含“10 次模型调用、3 轮环境交互”的回归检查
- [完整数值报告](validation/report.json)
- [新增世界边界应力检查](validation/boundary-stress.json)
- [真实 RL 提交计分烟测](validation/rl-smoke.json)
- [每世界一局带噪样例数据](validation/noisy-episodes.jsonl) 与 [种子清单](validation/noisy-episodes.jsonl.manifest.json)
- [训练接口详解](training/README.md)

`validation/report.json` 保存本次实测结果。原 1、4 的径向单位向量在穿过原点时仍有原始不连续边界；固定验证案例避开该情况。新增世界使用光滑力场，有限测试通过也不代表全合法域收敛证明。当前 20 世界全部二阶，没有第一阶或速度依赖定律覆盖。

`rift/`、`rift_numerics/` 按原版复制；本包 `agent/run.py` 已按环境交互轮次调整，原始 starter 的 Agent 未修改；原始文件校验值在 `upstream-manifest.json`。`UPSTREAM_README.md`、`development_worlds/UPSTREAM_README.md` 和 `docs/` 保留原 v15 背景资料；新增训练包以本页及世界定义为准。
