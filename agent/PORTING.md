# 实训接口移植说明

当前固定 GitHub commit：`6f8a1599b604d4fd317176d8503c84c1fd636866`。后续不自动拉取更新。

`repository/shore_agent` 为原仓库完整包，32 个文件的 Git blob 哈希见 `github-parity.json`；运行前按 `repository-source.json` 验证 SHA256。提示词、工具 schema、原始 Agent/ToolRuntime/DeepSeekClient、候选选择、拟合目标、科研循环不重写。`shore_rules.py`、`shore_worlds.py` 同样固定。启动E10前再次检查GitHub HEAD，发现5个Agent文件更新，已完整同步这些原文件。E10全部32个包文件、2个支持文件与最新提交一致，详见 `agent-identity-proof.json` 和 `github-parity.json`。旧R06比较仅适用于旧19a83f9a快照，不能作为当前版本声明。

## 接口边界

- `run.py` 仅处理启动、模型服务、原始配置读取及生命周期；`pipeline_bridge/environment.py` 将官方 Console 的 mission/observe/launch 结果转换为原版协议，保留免费首光全部8点；原始记录仅由适配器私有缓存保存。
- `pipeline_bridge/backend.py` 将原版积分入口和 PySR 短轨迹筛选入口连接到官方 `rift.rollout`，保留原版目标函数、权重、参数范围、候选排序、预算守卫。官方固定数值实现不修改。
- `pipeline_bridge/compiler.py` 负责 DSL 到正式 law API 的表达式和属性编码。草稿与正式提交必须采用原始运行时选中的同一 law/claims 对；`submission-selection-audit.jsonl` 可与正式 result.json 的两个哈希逐一核对。
- PySR 使用原仓库服务和 worker，独立私有进程池，启动预热在任务计时前完成，不连接其他实验的8890服务。worker 使用当次冻结 Agent 的源码。
- 模型工具不能访问项目历史测评结果、参考答案或复盘文件；冻结运行包不含本目录的历史实验和测试结果。

## 当前测评设置

E10「正式测评版六宇宙」：R01带噪声、R02无噪声，各6个宇宙、每宇宙一次科学会话。四卡8000–8003并行，全部使用同一份冻结Agent和同一份配置，不对失败重跑或取最高分。Qwen3.8-27B、4096 token、thinking关闭、PySR开启、300秒、10能量、环境seed=42、模型seed未固定。

两组共同沿用原仓库R06的e5配置（训练阈值1e-5、余时40秒提交），提示词、工具、上下文/记忆管理及科研循环均为最新6f8a159原文件。配置中的environment仅为仓库参考配置；实际实验数据由SDK Console提供。实验启动器在R01使用官方默认位置和速度噪声，在R02将二者设为零，并通过同一适配器传递实际噪声尺度。未修改SDK、物理世界、积分器、测试案例、评分或预算文件。R02是用户要求的无噪声对照，不能称为默认噪声协议。

原前端http://127.0.0.1:8765直接只读读取official_test的原始记录。E10结果在正式测评版分类下显示；score来源保持官方公开开发pipeline原值，不是组织方私有榜单。

旧OT02误用e4分支已单独留档，OT03和旧版本E10预检也不混入当前E10。旧E10在正式探索开始前因最新GitHub核验取消，移至agent/_preflight保存，科学会话数为0。

## 已知语义边界

仓库原数值工具可以将已观测背景位置插值后作为外部输入；正式 API 则需要法则自行推演全体状态。当前不把观测到的未来背景轨迹注入正式预测。背景采用零加速度假设；依赖背景且其公开初速度非零的 probe-only 候选，明确报 `background_model_required`，不假装实现了完整背景动力学。这是接口可表达性限制，不能归因于模型随机性，也不能据此承诺成功率无损。

原版每次积分/拟合时间守卫保持不变。官方积分计算成本不同，可能导致原版守卫提前触发；须通过逐轮记录区分有限最优解、超时与公式本身错误。

## 结果和完整性

`official-latest.json` 指向最新实验；每次测评保存独立 submission/agent 冻结包，测评中不替换代码。`audit_official.py` 校验冻结包、agent外84个官方文件、原版选择与实际提交配对、模型调用和工具错误。测评仅为官方公开开发环境诊断，不是组织方私有榜单成绩。

`official_test` 从原始v15压缩包独立解压；此后只修改本 agent 目录。原 physics_discovery 和旧 agent 保持原状。恢复仅需替换本目录的冻结运行包；不需要更改官方文件。
