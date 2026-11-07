# 可选符号回归前滤波：设计交接

状态：2026-11-07（北京时间），首版已实现，18 项轻量测试通过，等待总工程师独立验收。仅编辑本地 official_run/agent 及本交接文件；未改冻结快照、旧 physics_discovery、模型端口或服务器文件，未启动模型／PySR 搜索，未 commit/push。

## 已核实的 BF2 链路

读取来源：`/inspire/hdd/project/26-the-other-shore/263140062/official_test/agent/_experiments/E10-20261107-224311/submission/agent`。

- `repository/shore_agent/runtime.py:430` 的 `symbolic_regression` 深拷贝指定 probes，补上 mission 公开的 position_sigma、velocity_sigma，再调用 search_remote。
- `repository/shore_agent/symbolic_regression.py:89` 数据构建调用 `symbolic_quality.derivatives`，其输出 r/v/a 才是 SR 实际消费的数据。
- `repository/shore_agent/symbolic_quality.py:8` 已有最多 9 点、最高三次的局部多项式。含噪时分别平滑 r/v，通过速度局部多项式的一阶导数得到 a；不是从位置二阶差分。
- 当前 r/v 独立平滑，因此一般不满足 d(r_hat)/dt=v_hat。窗口按点数固定，没有依据噪声和实际时间跨度自适应。
- 当前 `derivatives` 返回内部时刻 `[1:-1]`，还输出分辨率敏感度权重。该敏感度不是统计置信区间。
- BF2 的配置默认使用旧路径。`repository-config.json` 的 noise 字段不代替运行时官方 mission 的噪声尺度；滤波必须读取 mission/每条 probe 的实际公开尺度，不可据静态文件猜测。

## 首版决定

采用 **位置—速度联合平滑样条**，可选，默认关闭；旧的局部多项式路径完整保留。先做离线数值及消费链路验证，再由总工程师安排正式消融。无世界 ID 分支、无指定力律／频率／系数先验。

不使用纯二维几何 MLS：已有时间戳和速度时，几何投影会丢失动力学信息，还可能混合空间相近但时间、速度不同的状态。

### 算法

每条轨迹独立处理，每个坐标使用同一套时间基底和同一个选定平滑强度。拟合连续函数 s(t)：

```
J(s) = sum_i ||s(t_i)-r_obs_i||² / sigma_r²
     + sum_i ||s'(t_i)-v_obs_i||² / sigma_v²
     + lambda * integral ||s'''(t)||² dt
```

只对 sigma>0 的通道使用软残差项；sigma=0 的观测作为精确等式约束处理，不用任意 epsilon 冒充已知噪声。公开的受控初态 r0/v0 在其真实 t0 处作为精确约束；不能把第一个含噪读数当作精确初态。

输出同一函数的 `r_hat=s, v_hat=s', a_hat=s''`，保证运动学一致。使用分段五次 B-spline、内部结点取实际观测时间，积分惩罚由逐段高斯求积构建。五次基底用于给联合导数观测足够自由度，并非假设动力学是五次多项式。矩阵保持稀疏／带状，用确定性约束消元或稀疏 KKT 求解，避免全尺寸稠密逆矩阵。

时间先按该条轨迹持续时间 T 归一化到 [0,1]；同时按链式法则转换速度、加速度、sigma_v 和惩罚项，最终还原物理单位。绝对时间、t0 和探针属性必须原样传给 SR，不可因归一化丢失时变规律。空间两轴使用共同尺度，避免人为破坏旋转对称性；不能分别把 x/y 方差归一到 1。

### 超参数选择

- 首版不向 Agent 暴露任意窗口、阶数、lambda 等调参入口。Agent 只选择方法和时间预算。
- 在归一化、按噪声白化的目标上，固定小网格 `lambda={1e-8,1e-6,1e-4,1e-2,1,100}`；这只是首版候选范围，需要用独立解析轨迹检验数值适用性，不从六宇宙成绩挑参数。
- n>=12 时使用 3 个确定性时间分块留出集：每次同时留出相同时间的 r/v，保持真实初态约束，不外推两端。按观测尺度归一化的留出误差选 lambda；CV 分数只含 sigma>0 的通道，零噪声留出通道另报预测误差，不能除以零或用任意 epsilon 改变评分。近似并列（相对差不超过预登记的 1%）时选较小 lambda，避免偏好过度平滑。这里是预处理参数选择，所有这些点都记为结构发现已使用，绝不冒充独立物理验证集。
- 两轴合计选一个 lambda。候选误差、边界命中、粗糙度、有效样本数写入分析记录。CV 优化失败或明显不可信时返回显式失败／旧路径回退，不静默选最光滑曲线。
- 8<=n<12：仍允许固定分块留出，但报告 `limited_selection_evidence`，不声称可靠选择；通过数值测试后再决定是否首版启用。首版可以保守回退旧方法。
- n<8：默认回退 BF2 局部多项式，并记录 `short_trajectory`，不强行拟合高自由度联合样条。
- 预处理总预算默认 2 秒，上限 5 秒，计入 300 秒；超时逐轨迹回退，已完成结果可以保留，但逐条记录实际方法。耗时包括数据选择、CV、求解和序列化，不能另开不计时线程。

### 互斥与回退

`legacy_local_polynomial` 和 `joint_spline_v1` 是同一个预处理阶段的互斥分支。

- 关闭：直接走现有 derivatives；工具不注册、schema/prompt 不变；不额外创建滤波记录，不多做数值计算。
- 开启 joint：原始 probes -> 联合样条 -> SR dataset。不能把滤波 r/v 再送入旧 derivatives；尤其不能对 a_hat 再做 9 点平滑／差分。
- 回退：始终从原始 probe 调用 legacy；禁止把部分滤波数据再拿去 legacy。
- 双通道零噪声：直接 legacy 原路径，r/v 不改，a 与当前 BF2 数值行为一致，明确记录 bypass。不宣称该分支的新样条导数精确。

## 工具接口和实际消费契约

暂定新工具（只在 feature flag 开启时注册）：

```
denoise_trajectories(
    train_probe_ids: list[str],
    method: "joint_spline_v1",
    time_budget_seconds: float = 2.0
)
```

输入必须来自本会话已有记录。sigma 从实际公开 mission 读取，Agent 不可覆盖。工具不得接收 world ID、真值、评分轨迹或外部文件路径。

简短返回示例（不把全部浮点数组塞进模型上下文）：

```json
{
  "status": "completed",
  "preprocessing_id": "filter_<id>",
  "method": "joint_spline_v1",
  "probe_ids": ["probe_0001", "probe_0002"],
  "elapsed_seconds": 0.18,
  "per_probe": [{
    "probe_id": "probe_0001",
    "status": "filtered",
    "samples": 32,
    "sigma_r": 0.04,
    "sigma_v": 0.0003,
    "lambda": 0.0001,
    "position_correction_rms": 0.03,
    "velocity_correction_rms": 0.0002,
    "boundary_warning": false
  }],
  "scope": "Discovery preprocessing; not law validation."
}
```

示例数字仅说明格式，不能作为工具参数、目标或成功承诺。完整数组、CV 表、哈希在本会话分析 artifact 中保存。

SR 增加可选 `preprocessing_id`，旧两参数调用保持合法。无 ID 走 legacy；有 ID 必须验证 artifact 与当前 train_probe_ids 完全匹配、原始数据哈希匹配、属于本会话，且配置允许。不能“工具执行成功但 SR 仍读原始数据”，也不能允许陈旧 artifact 偷换数据。

血缘记录至少包含：版本、方法、原始 probe IDs/内容 SHA256、实际 t0/times/属性/噪声尺度、选定 lambda、回退原因、原始和输出数组摘要哈希、wall time、精确初态约束来源。保留每条轨迹映射；同一点被多次实验采得也不能无记录合并。

发送 search_remote 的 payload 需要带受限预处理 artifact（或携带可复现配置并由 worker 重建，但避免重复计算）；服务端验证 shape、有限性、时间对齐、来源哈希，并在 `build_dataset` 的 joint 分支直接消费 r_hat/v_hat/a_hat。服务缓存键包含预处理版本与参数／artifact hash。客户端与 worker 都必须支持协议，不能仅改 Agent 端。

`dataset['probes']` 保持原始数据；新增 search-only 派生数组。所有候选参数拟合、rollout、原始路径积分一致性、正式独立验证和评分使用原始观测以及现有官方 rift rollout 路径。禁止把去噪目标回写 probe store 或传给评分链路替代原始观测。SR 搜索权重继续逐轨迹归一，预处理 CV 残差不当成加速度方差或独立置信度。

## 边界与异常

| 情况 | 处理 |
|---|---|
| 非均匀 dt | 所有设计矩阵和积分使用真实时间；不假装均匀采样，不先插值再拟合 |
| 未排序、重复时间、NaN/Inf、非法 sigma | 显式拒绝该输入；不排序／去重掩盖记录问题 |
| 只有一个通道零噪声 | 零通道等式约束，另一通道按真实 sigma 加权；验证约束数值残差 |
| 边界 | 用公开初态锚定真实 t0；输出端点诊断，保持 SR 只使用原有内部时刻；不外推观测覆盖外的数据 |
| 突然加速／快速振荡 | 使用偏向保留变化的选择规则；记录 CV 网格边界命中、局部残差结构；不强制低频／周期／中心力假设 |
| 数据太短或时间尺度分辨不了变化 | 返回低证据诊断／回退，不能声称恢复真实 a；建议更多读数但不自动花能量 |
| 病态约束／矩阵、预算耗尽 | 从原始数据回退 legacy，完整记录原因与时间 |
| 多轨迹、不同属性、不同绝对时间 | 分别拟合，不共享邻居或平均状态；只合并已对齐且带原始属性的 SR 行 |

## 实施位置（等待初始化通知后）

只编辑本地 `/Users/zonghuachen/Desktop/创智实训题目/official_run/agent`；部署和服务器改动由总工程师统一完成。

预计新增 `trajectory_filter.py`（纯数值）、滤波 artifact/工具入口；小范围调整 runtime/schema/registry、symbolic payload/service/dataset；增加专项测试。不改 BF2 快照、正式环境、rift 或评分公式。实现前需核实新工作区 AGENTS.md 和目录结构。

## 验收测试

1. **独立解析轨迹**：直线、常加速度、双频非圆运动、变频运动、平滑局部快速脉冲；参数按预先固定范围生成，不复制开发宇宙公式。位置/速度/加速度解析给出，真值仅供测试断言，工具输入不含真值。
2. **噪声改善**：固定公开 RNG seeds，测试双通道噪声、仅位置噪声、仅速度噪声；与 noisy 原始值及 BF2 legacy 比较 r/v/a RMSE。报告逐轨迹结果与聚合结果，不要求每个种子每个量都改善，不挑最佳 seed。改善标准在运行前登记；加速度改善不能以明显损失位置／速度保真为代价。
3. **快速变化保真**：测试多种采样密度、频率和脉冲宽度；记录振幅衰减、相位偏差、a 峰值及端点误差。欠采样时应诊断，不保证恢复不存在的信息。
4. **运动学一致**：在结点和结点之间验证 s'、s'' 与输出的一致性；用数值差分交叉检查时考虑步长误差。
5. **单位／坐标变换**：时间平移、时间单位缩放、共同空间尺度变换、二维旋转后的结果满足对应变换；物理绝对时间不丢失。
6. **边界条件**：n=3/5/8/12，非均匀 dt、极不均匀 dt、大绝对时间、零通道和真初态不在首读数时刻；等式约束满足容差。
7. **输入不变／血缘**：原始字典、数组及持久记录 hash 前后相同；跨会话、错 probe IDs、陈旧 hash、非法 shape 的 artifact 被拒绝。
8. **实际消费链路**：用轻量 fake search backend 捕获 payload 和 dataset；确认 joint 数组确实进入 SR 的 X/y，legacy derivatives 不被再次调用；同时 raw rollout/拟合 spy 收到原始数据。测试不启动 PySR 或模型。
9. **关闭等价**：与冻结 BF2 相同输入下，工具列表、schema、prompt、旧 SR payload 和 dataset 数组/权重完全一致（随机字段单独规范化）；无新增调用／滤波耗时。用 golden fixtures 验证，不能只测试新实现自己。
10. **预算**：超时、部分完成和失败清晰回退；统计滤波实际耗时。中等与最大 n 做 CPU 性能基准，当前阶段只在本地执行，避免影响服务器 baseline。

## 需总工程师确认／集成事项

- 同意首版只实现 joint_spline_v1，并以单独工具 + SR 可选 preprocessing_id 接入；默认关闭。若希望实验阶段完全固定 Agent 决策、仅比较预处理，则消融使用运行配置自动选择同一后端（不新增模型调用），应与“Agent 自主选滤波工具”另立实验，不能混为一个因素。
- 初始化完成后提供新工作区可编辑通知，以及统一包的 service/worker 部署约定；实现前不改 official_run。
- 目前未发现必须查询用户才能继续的算法决策。若联合样条的约束求解或短序列 CV 在解析测试中不稳，先报告并保留 legacy 回退；不为刷六宇宙成绩偷偷改方法。

## 进度

- [x] 只读核实 BF2 入口、已有 9 点平滑、原始 probe 保留路径。
- [x] 接口、互斥路径、预算、血缘、边界与测试设计。
- [x] 收到 official_run 初始化完成通知，基线 5d9a0b3 / baseline-bf2-v1。
- [x] 在授权新目录实施并运行轻量数值／消费链路测试。
- [ ] 总工程师审查后统一部署及消融。

## 实施交付（首版完成）

文件清单（均在本地 official_run/agent）：

1. 新增 `pipeline_bridge/trajectory_filter.py`：纯数值 filter_probe、受限 SR 数组消费器、准确的搜索 target/source/warnings 标记、worker warmup 能力凭据。
2. 新增 `pipeline_bridge/filter_tool.py`：可选 denoise_trajectories、跨会话隔离的 artifact、SR preprocessing_id、客户端消费回执校验、只读 workspace artifact 注册。
3. 新增 `pipeline_bridge/filter_capability.py`：共享池 /health 能力公布与 loopback-only 版本/consumer源码SHA核对。由真实 worker ready.json 汇总能力；不能凭家长进程配置布尔值自称支持。
4. 修改 `pipeline_bridge/service.py`：可选启用 worker overlay；默认原 bootstrap 路径；可复用经过握手的外部池。
5. 修改 `run.py`：新增 `--trajectory-filter`，默认不启用；开启时记录 effective_config 与 runtime-provenance。
6. 新增 `test_trajectory_filter.py`：17 项数值及链路测试。
7. 新增 `filter-tool-delivery.json`：机器可读状态、SHA、测试命令、局限及数值收益数据。

没有修改 vendored repository 文件，无需修改 repository-source.json；源码完整性测试逐项核对 manifest 均通过。新能力由独立适配层挂接，运行时声明 overlay，不能称整个 Agent 行为与 BF2 完全相同。

### 运行／共享池

- 默认运行不加 flag：工具列表、提示词和原 SR 预处理保持 BF2。
- Agent 自主滤波工具模式：加 `--trajectory-filter`；仅当 Agent 显式调用 denoise_trajectories 并向 SR 传回 preprocessing_id 才消费联合滤波数据。工具说明包含用法，无指定规律提示。
- 池可一次用 `symbolic_service(..., size=4, trajectory_filter=True)` 预热。运行 Agent 时可同时传 `--trajectory-filter --symbolic-endpoint http://127.0.0.1:<port>`；启用前必须通过 /health 的版本与源码 SHA 核对，回执写 filter-worker-capability.json。旧池或错误 SHA 直接拒绝。
- 除握手外，每次 SR 还核对 data_quality.local_derivative_quality 中各 filtered probe 的方法、raw_sha256、output_sha256。没有正确消费凭据就拒绝把结果作为 filtered search 返回。客户端记录 filter-consumption.jsonl。
- 纯数值自动预处理消融可直接调用 filter_probe，然后将结果挂到对应深拷贝 probe 的 PAYLOAD_KEY（_joint_spline_search）再交给 installed consumer 的 prepare_dataset；这不代表 Agent 自主使用工具。

### 审查修正：分辨率敏感度

已按总工程师审查修复最初把 uncertainty 置零及 quality 全1的问题。joint 分支现在只从 **原始 r/v** 调用原有分辨率诊断，同时保留原 quality 相对权重和 raw-stencil sensitivity，供现有候选简洁性容忍区间使用。仅替换搜索使用的 r/v/a，避免滤波和重新加权收益混淆。joint r/v/a 不再被局部多项式平滑；搜索权重逐轨迹总和相同。报告明确这些权重和敏感度不是 joint-spline 后验置信度。target/acceleration_source 和 warnings 在进度及最终报告中准确反映 joint 或 mixed fallback，原科研包源码未改。

### 测试与收益结论

命令：`OPENBLAS_NUM_THREADS=1 ../physics_discovery/.venv/bin/python -B -m unittest agent.test_trajectory_filter -v`（official_run 为 cwd）。18 项通过，约 1 秒；包括独立子进程 install_consumer -> 实际 prepare_dataset、实际 runtime tool -> JSON transport payload -> SR X/y、默认关闭、旧调用原数据、跨会话／陈旧引用、worker 未消费拒绝、真实 HTTP /health 握手、单通道零噪声、单位/旋转/时间平移、短序列/零噪声回退、运动学导数基底验证及快速变化测试。没有启动 Julia/PySR、模型或正式 rollout。

数值收益与协议验收分开：最初测试错误地要求所有 r/v/a 都优于 legacy，**这项收益要求未通过**。随后改为报告真实收益并检查数值正确性，没有为通过该 fixture 修改滤波算法、lambda 网格或 seed。5 个固定 seed 的光滑双频解析族，公平比较相同内部时刻，位置/速度明显改善，加速度聚合误差略差于 legacy（精确值见 filter-tool-delivery.json）；尤其个别 seed 的 CV 选择对 a 不利。不能把正确性通过表述为所有通道收益通过。

总工程师独立 80 案例的第一轮结果也显示非单调收益：chirp、pulse、polynomial 的 a 有改善，mixed_frequency 略退步；pulse 的 v 相比原始观测稍差。保留该结论，不据六宇宙成绩调参数。

本地 CPU 性能：10 条×50点约 0.111 秒，1条×1000点约 0.026秒，默认2秒内全部 filtered。只代表本地轻量测试；服务、序列化和拥塞耗时需正式消融记录。

已知限制：超时为稀疏求解前后检查的协作式 deadline，不是进程级强制打断；端点 a 可以比内部更差；CV 依据 r/v，不能保证 a 或最终物理规律更好；短于12点／双通道零噪声保守回退。n>=12 的严重病态数据可显式回退，不保证每条轨迹能滤波。尚未进行正式 Linux worker、PySR 候选／官方 rollout 的完整运行验收，留给总工程师统一部署后验证。

### 共享池轮转边界补充

已测试原 _Worker.ready()：busy worker 只要存活且有 ready.json 就仍为 ready，因此正常忙碌不会撤销 capability。单 worker 重建／预热时，health 提供 trajectory_filter_status（本地预期 capability、已认证数、待认证数、错误凭据数），完整 capability 暂不发布。新 Agent 在显式 startup timeout 内等待待认证 worker 完成，再开始 mission；超时清晰失败。旧池缺能力或存在错误源码 SHA 则立即拒绝，不把错误凭据误当作预热。测试覆盖忙碌、None 重建占位、错误 worker receipt、HTTP pending→ready 和有界等待耗尽。该等待发生在任务开始前的启动预算内，不后台占用探索时间，也不更改现有 baseline 服务。
