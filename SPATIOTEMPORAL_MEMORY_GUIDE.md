# 时空超图记忆

本文介绍基础时空扩展。当前已继续加入多源观测和过程演化，完整结构为 18 类节点、8 类超边，见 [多源与过程演化说明](EVOLUTION_MEMORY_GUIDE.md)。下文的 11 类节点与 4 类超边是基础部分。

当前实现将时空上下文加入通用经验记忆，交通只是示例。接口不依赖交通视频、GIS 或项目内部数据；调用方提交结构化上下文即可。当前智能体仍执行有证据的问答和经验复用，没有新增交通任务执行器。

## 结构

可从三个逻辑层理解：实体与上下文节点层、经验关联超边层、按任务聚合的记忆组织层。它们是解释结构的逻辑层，不是三个数据库或三层神经网络。

支持 **11 种节点类型**，实际节点数量随记录内容增长并去重：

|类型|内容|
|---|---|
|intent|任务键|
|condition|必须匹配的条件|
|answer|可复用回答|
|evidence|来源说明|
|feedback|验证与失败反馈|
|time|完整时间上下文：观察时间、适用区间、周期窗口|
|space|带命名空间的空间实体|
|spatial_relation|带源、谓词、目标的有向关系描述|
|event|事件或任务类别|
|action|历史执行步骤|
|outcome|历史结果描述|

支持 **4 种超边类型**：episode 将一条经验的节点联合连接；family 按任务键及记忆作用域组织经验；invalidation 保存替代与失败反馈；spatial_relation 连接关系描述和两端实体。关系节点明确编码源和目标，避免无序超边丢失方向。

空间关系超边可由多条经验共享。经验检索通过 family 超边扩展候选，再读取完整记录进行约束判断。当前不是训练得到的超图神经网络，也没有通用图路径学习。

## 时间、空间与检索

`record.time.event_at` 是经验观察时间；`applicable_from` / `applicable_until` 是适用区间，左闭右开；`weekdays` 用星期一=0；周期窗口采用本地分钟数和显式 UTC 偏移，可跨午夜。星期约束检查查询所在的本地日历日期。当前使用固定偏移，不处理夏令时。

`record.created_at` 是写入时间，`record.valid_until` 是存储复用期限，与业务适用时间分开。短期记忆仍按 session 隔离；长期记忆跨会话保存。短期提升保留时空上下文，不会扩大其适用范围。

查询使用 `at` 指定任务发生时刻，避免历史实验依赖运行机器的当前时间。存储期限仍按真实当前时间检查。带时间条件的经验遇到未提供 `at` 的查询会被排除。

空间实体使用 `namespace` 隔离。直接实体匹配可复用；查询明确提供的 `within` 关系允许从路段到区域的传递包含匹配。`adjacent`、`nearby` 等关系可以保存，但不自动意味着经验适用。经验要求的关系必须出现在查询中。当前关系由调用方提供，不使用历史经验中的关系自行证明当前地点的关系，也不做坐标距离或 GIS 推断。

流程：

1. 检查验证状态、失败、替代、存储期限及会话作用域。
2. 硬过滤条件、事件、时间区间、周期和空间不匹配的经验。缺少必要上下文也排除。
3. 对剩余候选进行排序：词项重合分 + 时间分 + 空间分 + 事件分；条件数量、短期作用域、写入时间用于后续排序。时间分含观察时间衰减；直接空间匹配分为 1，包含匹配为 0.5。这是透明启发式，权重尚未通过实验调优，分数不是复用概率。
4. 一次 Jev 请求包含完整新旧时空上下文，返回选择、相关度、支持度。`judgment.support` 显示 0–2 的支持度；不足 1.8 时转向综合回答。置信度和相关度仍使用现有配置阈值。
5. 输出前重新从 SQLite 读取选中经验并检查约束，防止推理期间经验被更新或失效。模型生成的回答不会自动变成已验证经验。

同任务但时空适用条件不同的事实不会互相替代。经验类型继续累积。来源字符串和 `outcome` 描述本身不等于独立验证证据，Jev 分数也不构成正确性证明。

## 在 CMD 运行示例

先激活已有环境并设置路径。使用单独数据库避免混入当前记忆；示例为人工流程数据，不代表真实交通评测。

```bat
cd /d E:\code\smp
conda activate smp-memory
set "PYTHONPATH=E:\code\smp\src"
python -m memory_agent --store runtime\memory_agent\spatiotemporal_demo.sqlite3 ingest examples\spatiotemporal_records.json
python -m memory_agent --store runtime\memory_agent\spatiotemporal_demo.sqlite3 retrieve --query-file examples\spatiotemporal_query.json
python -m memory_agent --mode mock --store runtime\memory_agent\spatiotemporal_demo.sqlite3 ask --query-file examples\spatiotemporal_query.json
python -m memory_agent --store runtime\memory_agent\spatiotemporal_demo.sqlite3 inspect
python -m memory_agent --store runtime\memory_agent\spatiotemporal_demo.sqlite3 graph
```

应只召回区域 A 的经验，区域 B 被过滤。把查询时刻改到 10:00、周末，或换成无包含关系的地点，可验证无法直接复用。`retrieve`、`inspect`、`graph` 都不调用模型。想测试真实 Jev，将 `ask` 的 `--mode mock` 改为 `--mode live`，会调用已配置 API 并产生费用。重复导入同一示例会因固定记录 ID 报重复。

数据库自动接受旧版本并升级为 schema 5（新增可选向量缓存表），图快照版本为 3。旧记录无需补齐时空字段即可使用；原有无时空字段的节点、超边与记录继续保留。新排序和邻域召回见 [联合检索说明](JOINT_RETRIEVAL_GUIDE.md)，上面的排序描述为早期版本。

## 研究验证边界

这是可运行的研究原型，尚不能凭结构命名宣称论文创新或效果提升。建议固定底座模型、经验数量和 token 预算，对比无记忆、平面时空字段检索、普通超图记忆、时空超图无 Jev、完整方法。重点测量环境变化后的错误复用率、任务成功率、决策耗时和模型调用成本，再分别移除时间、空间关系和 Jev 做消融。只有平面时空检索也难以处理的联合关系案例，才能证明超图结构的额外价值。
