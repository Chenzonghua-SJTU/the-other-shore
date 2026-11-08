# 20 世界部署记录

部署到：/inspire/hdd/project/26-the-other-shore/263140062/official_test
完整发行包：/inspire/hdd/project/26-the-other-shore/263140062/official_test/training-releases/worlds20-20261108-7b6d6b6e/shore-worlds-20.zip
原文件备份：/inspire/hdd/project/26-the-other-shore/263140062/official_test/.worlds20-backups/worlds20-20261108-7b6d6b6e

保留了原有 agent/、models.json、README.md、rift/、rift_numerics/ 及模型权重。目标自定义 Agent 已采用 environment_batches 计数。三轮是环境交互轮次，每轮可有多个工具；模型调用和本地拟合不单独计轮。

已更新 development_worlds/，增加 training/、世界目录与验证工具。说明入口为 README.worlds-20.md，机器目录为 world-catalog.json。原 starter-manifest.json 与根 FILE_LIST.txt 是原包历史记录，新增文件请看部署清单。

可用解释器：/opt/shore-test5-forge2/bin/python（Python 3.12.16、NumPy 2.1.3、SciPy 1.15.3）。

    cd /inspire/hdd/project/26-the-other-shore/263140062/official_test
    PY=/opt/shore-test5-forge2/bin/python
    "$PY" -m development_worlds.runner --agent-dir "$PWD/agent" --models-file "$PWD/models.json" --max-interaction-rounds 3 --worlds 7 14 20 --output /absolute/path/to/new-results
    "$PY" -m training.generate --episodes 20 --seed 42 --output /absolute/path/to/new-data.jsonl

第一条会实际启动模型，请只在需要运行 Agent 时执行。当前部署验证仅运行 CPU 环境测试，不启动模型或占用 GPU。

部署清单：deployment-worlds20.json。
服务器测试日志：validation/server-unit-tests-worlds20-20261108-7b6d6b6e.txt。
服务器全世界数值报告：validation/server-report-worlds20-20261108-7b6d6b6e.json。
完整发行包中的 validation/report.json 和 unit-tests.txt 保留本地构建时的验收记录。
根目录测试只验证环境、训练接口与协议，不以配套基线测试替代自定义 Agent 验证。
