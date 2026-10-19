# FindingDory：超图记忆与 ReadAgent

这个对比在同一份两段视频、192 帧结构化视觉观测上运行。ReadAgent 的分页和摘要函数从固定的官方 notebook 直接加载；回看页号遵循其官方提示词和解析方式。原 notebook 的 QuALITY 选择题回答器由项目现有的共同视觉目标帧回答器替换。实验结果是离线目标帧命中率，不是导航成功率。

官方源码需要单独检出，因为 `runtime/` 不提交到 Git：

```powershell
git clone https://github.com/read-agent/read-agent.github.io runtime/baselines/read_agent_official
git -C runtime/baselines/read_agent_official checkout 569dff3e47bb708ad084d267ac554f2534d21756
```

已有 `runtime/findingdory/` 数据与共用观测运行目录时，可从项目根目录执行：

```powershell
$env:PYTHONPATH='src;.'
python -m benchmarks.findingdory_readagent --source-run runtime/findingdory_run_20261013_graphpriority_amem_final --run-dir runtime/findingdory_run_readagent_new
```

程序检查官方源码版本、模型、问题和观测哈希；两段各构建一次 ReadAgent 记忆，再让两种方法分别答 12 题。每次 API 调用和 token 用量写入运行目录，失败后可用相同命令续跑；官方分页和摘要的模型输出会按提示哈希缓存。`.env` 只在本机读取，不进入运行配置或 Git。

本次完成的记录在 `runtime/findingdory_run_20261013_readagent_official_complete/`，包含 `REPORT.md`、`DIAGNOSTICS.md`、`summary.json`、`predictions.jsonl`、`evaluation_config.json` 和 `provenance/`。两段视频的 12 道题已经多次用于开发诊断，不能作为论文最终未见测试集。
