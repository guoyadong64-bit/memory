# 超图记忆智能体

当前项目提供通用记忆问答、SQLite 超图存储、长期与会话级短期记忆，以及 Jev 的记忆复用判断，可作为后续接入新业务场景的基础。

FindingDory 当前对比使用你的超图记忆与 Mem0 官方开源 SDK，操作见 [Mem0 对比实验说明](FINDINGDORY_MEM0_GUIDE.md)。指标是离线目标帧命中率。

已加入时间适用区间、周期窗口、空间实体及有向包含关系，支持时空约束过滤和排序。结构、字段与运行示例见 [时空超图记忆说明](SPATIOTEMPORAL_MEMORY_GUIDE.md)。

进一步加入多源观测归一化、时间对齐、来源冲突检查，以及事件阶段和状态转移。当前结构与测试示例见 [多源与过程演化记忆说明](EVOLUTION_MEMORY_GUIDE.md)。

现已加入经历超边绑定与阶段链检查、查询条件约束的邻域召回、可选向量语义检索及 SQLite 向量缓存。结构与运行方法见 [联合检索说明](JOINT_RETRIEVAL_GUIDE.md)。

在 CMD 中运行：

```bat
cd /d E:\code\smp
conda activate smp-memory
set "PYTHONPATH=E:\code\smp\src"
python -m memory_agent check
python -m memory_agent demo
```

`check` 只检查配置，`demo` 使用独立的离线演示数据。真实接口配置继续保存在 `.env`，正式记忆默认位于 `runtime/memory_agent/memory.sqlite3`。

完整命令及记忆管理方法见 [记忆使用说明](MEMORY_AGENT_GUIDE.md)。测试：

```bat
python -m pytest tests -q
```
