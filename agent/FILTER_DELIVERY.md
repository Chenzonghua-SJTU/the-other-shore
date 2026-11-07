# 符号回归前滤波工具交付

默认关闭，当前为实验功能；软件正确性通过不等于提高任务分数。

## 接口与使用

官方入口仍为 agent/run.py。在 agent/repository-config.json 的 numerics 中设置 `"trajectory_filter_enabled": true` 即可启用，无需修改官方入口或评分器。缺省为 false；CLI `--trajectory-filter` / `--no-trajectory-filter` 显式覆盖配置。启用后增加可选工具及简短使用说明，旧科研包32个文件仍保持原始哈希。

调用 `denoise_trajectories(train_probe_ids=[...], time_budget_seconds=2)`，再将返回的 preprocessing_id 和完全相同的 train_probe_ids 传给 symbolic_regression。省略 ID 仍走旧局部多项式。不得把只有工具注册、没有实际消费的试验说成滤波收益。

联合位置/速度五次样条按实际时间戳工作，同一曲线导出 r/v/a；公开零噪声通道和受控初态使用等式约束。固定 lambda 网格与三组留出选择，不含世界ID或力律先验。n<12、零噪声、数值异常或时间不足时明确回退。默认2秒、最大5秒，为协作式预算，不能硬中断底层求解。

仅改变符号搜索数据，原始观测不变；参数拟合、验证、rollout、评分继续使用原始观测和官方接口。客户端/worker 校验版本与哈希，保存 artifact 血缘和 filter-consumption.jsonl，拒绝陈旧或错配数据。

## 验收与实验

- 本地与 Linux 28 项数值、边界、数据消费、配置和提示词测试通过；四组既有框架回归通过。
- 真实 PySR + 官方 Console 集成通过：4条滤波探针被 worker 确认消费，原始记录不变，13次官方 rollout，合法提交。集成冒烟的零假设不是科学成绩。
- 80个预登记解析案例全部保留；混合频率案例加速度误差略退化。固定 E10 R07 全部58条观测消融：46条实际滤波、12条短序列回退；位置六世界改善，加速度世界2/3/5退化。见 acceptance/evidence。
- E10 六宇宙各五次含噪 baseline 已完成，整体均分59.7666，≥90为12/30。E11是可选工具端到端五次消融；E12另测工具使用引导，不能合并统计。

测试命令：`python -B -m unittest agent.test_filter_guidance agent.test_filter_options agent.test_trajectory_filter -q`。需要 NumPy/SciPy。科学集成还需要现有 PySR/Julia 服务。

## 部署与回退

后续工作区为 official_run，科学代码、实验记录均在 agent 内；84个官方文件与原包逐字节一致。初始版本 tag 为 baseline-bf2-v1，commit 为5d9a0b3。关闭 trajectory_filter_enabled 可恢复 BF2 默认路径；精确复现实验使用其冻结 submission/agent 及 registration.json，不能只看当前 main。

Git 不包含模型权重、vendor_runtime、Julia/PySR 依赖、会话凭据和大日志。现服务器使用既有 agent/vendor_runtime；官方镜像的依赖布局尚待核实，不能据此声称无依赖提交包已在官方镜像验证。不要把本实验交付包直接当作已验证的官方提交包。
