---
name: shore-run-code
description: Reliable run_code use for acquired physics-world data. Apply when selecting input files, aligning probe arrays, computing numerical diagnostics, or repairing Python errors within the three-round environment budget.
---

# run_code 操作技能

仅在当前工具列表含 run_code 时使用。它用于分析已取得的数据，调用不消耗环境交互批次，但消耗本局 300 秒墙钟。先判断计算能否帮助选择或检验假说；不要为使用工具而调用。首光只有两条短轨迹，最多先做一次小诊断，优先取得第一批主动对照实验，再用代码分析丰富的数据。不要在首光上反复寻优到无法检验属性或速度依赖。

## 输入与数组

- 参数只有 `code` 和可选 `files`，不传 `inputs`、`timeout` 或 `output_schema`。直接写 Python，最后赋值 `result`，不需要 main 函数。
- 从任务包、`workspace_files` 或 `list_files` 选择真实路径，放进 `files`。用 `read_json(path)` 读取；虚拟路径不能用 `open`。不猜编号、不抄聊天截断的数据。
- 先看实际字段和形状。探针是 `times/r_obs/v_obs`，天体是 `times/body_ids/positions_obs`，后者的轴为 `(时间, 天体, xy)`。天体属性须从实际任务字段读取，不假设存在 `bodies` 字段。
- 探针优先用 `probe_arrays(p, time_axis="observed")` 得到 `t/r/v`。需要加速度时改用 `time_axis="acceleration"` 得到对齐的 `t/r/v/a`；两种时间轴长度可能不同，不拼接混用，不用截短掩盖错误。精确对齐失败时，仅在科学上合理的情况下显式使用 `interpolation="linear"`，并保留该说明。
- 计算前用 `np.asarray(values, dtype=float)` 创建浮点数组，避免整数数组原位加浮点数。对 `(N,2)` 的向量取 `axis=1`；不把 `(N,)` 标量序列当二维。多个探针沿样本轴 concatenate，先确认列数和时间轴。

## 可改写模板

将 files 设为实际探针路径；下面示例只处理一个探针，不能把天体记录交给 probe_arrays。

```python
import numpy as np
path = available_files[0]
p = read_json(path)
s = probe_arrays(p, time_axis="observed")
v = np.asarray(s["v"], dtype=float)
assert v.ndim == 2 and v.shape[1] == 2
speed = np.linalg.norm(v, axis=1)
result = {"path": path, "samples": len(s["t"]),
          "mean_speed": float(speed.mean()), "alignment": s["alignment"]}
```

## 常见 NumPy 写法

- 先转换再多维索引：`body = np.asarray(record["positions_obs"], dtype=float); xy = body[:, 0, :]`。不要在 JSON 的 Python list 上写 `[:, 0, :]`。
- 对 `(N,2)` 向量逐行除以 `(N,)` 的模长：`unit = vectors / lengths[:, None]`，并显式处理零模长。取两列用 `ux, uy = unit[:, 0], unit[:, 1]`，不要用 `ux, uy = unit` 解包 N 行。
- 逐行内积用 `(a*b).sum(axis=1)`，结果为 `(N,)`；`a@b.T` 得到 `(N,N)` 的所有行配对，不能拿来当逐行结果或布尔索引。
- 两个标量取较大值用 `max(a, b)`；`np.max(array, axis=...)` 的第二个参数是轴，不是另一个待比较数。
- `files` 必须是 JSON 数组，例如 `["observations/probe_0001.json"]`，不要把整个数组再编码成一个字符串。
- run_code 的候选和参数还没有登记成 law_id。得到一个可检验假设后，及时调用 fit_law_and_rollout 注册；不要一直编程到时间结束。

## 可选：同时检查坐标与速度依赖

若需要排查坐标交叉项或速度依赖，可先用下面的线性回归作诊断起点，避免只计算径向投影后就假定中心力。它不是普适动力学模型，也不替代正式轨迹拟合。非线性特征由你根据对照数据另行提出。不要把不满秩、共线的系数解释为唯一物理机制。

`files` 只放实际探针记录，最后一条整轨迹作为留出；若它已参与结构分析，只能称参数留出。不要把同一轨迹相邻采样点随机拆为训练和验证。

```python
import numpy as np
frames = [probe_arrays(read_json(p), time_axis="acceleration") for p in available_files]
assert len(frames) >= 2, "Need separate training and held-out trajectories"
def matrices(s):
    t = np.asarray(s["t"], dtype=float)
    r, v, a = [np.asarray(s[k], dtype=float) for k in ("r", "v", "a")]
    assert t.ndim == 1 and r.shape == v.shape == a.shape == (len(t), 2)
    X = np.column_stack([np.ones(len(t)), t, r, v])
    assert np.isfinite(X).all() and np.isfinite(a).all()
    return X, a
pairs = [matrices(s) for s in frames]
X = np.concatenate([p[0] for p in pairs[:-1]], axis=0)
Y = np.concatenate([p[1] for p in pairs[:-1]], axis=0)
Xt, Yt = pairs[-1]
coef, _, rank, singular = np.linalg.lstsq(X, Y, rcond=None)
assert coef.shape == (6, 2)
result = {"features": ["constant", "t", "x", "y", "vx", "vy"],
          "coefficient_rows_ax_ay": coef, "rank": int(rank),
          "identifiable_in_this_design": bool(rank == 6),
          "train_acceleration_rmse": float(np.sqrt(np.mean((X@coef-Y)**2))),
          "heldout_acceleration_rmse": float(np.sqrt(np.mean((Xt@coef-Yt)**2)))}
```

比较整轨迹留出误差、残差及矩阵秩，再选择下一步实验或假设。差分加速度有噪声和偏差，不能把它的RMSE直接与位置噪声比较。这份多输出回归可能超过正式五参数上限，不可直接当作已登记规律；有证据支持的简洁候选仍需由你明确写成DSL、正式拟合和验证。

## 输出和纠错

- 返回少量有限数值和诊断。NumPy 数组可自动转 JSON；SymPy 表达式转 str，SymPy 数值转 float，集合转 list，字典键用字符串。不要返回函数、优化器对象或复数。缺失数值明确标成 None，并说明原因；不把 NaN/Infinity 当有效结果。
- 只查看已有 stdout 或结果文件时，直接调用 `read_file(path=工具返回路径, offset=0, limit=12000)`；不要用几十次 run_code 反复切出 150 字片段。完整计算结果优先赋值 result，避免只 print。
- 每次执行都是新容器。重新 import 并读取 files，不依赖上次的变量。要复用前次结果，显式传入工具返回的结果路径。
- `KeyError/FileNotFoundError`：核对当前文件路径和实际 keys；`IndexError/AxisError`：检查 shape；`ValueError`：检查时间对齐、有限值和维度；`TypeError/UFuncTypeError`：检查 dtype 和输出类型。看明错误后做一次最小修正，不反复重发相同代码。后端 unavailable 不靠改代码解决。
- 批量做相关的小计算，限制优化迭代，避免大网格或无界搜索。出现连续失败或接近提交阶段时，用已有证据继续正式拟合与提交；不要耗尽时间调试。当前流水线单次执行最多 20 秒，并随剩余墙钟缩短；容器资源为 1 GiB、1 CPU。代码宜保持简短，每次只解决一个明确的数值问题。
- 执行成功只是程序完成。候选仍需 `fit_law_and_rollout` 注册、独立数据 `validate_law` 验证，再 `submit`；不把数值相关性当因果独立性，不改变最多三个环境批次和五参数的约束。
