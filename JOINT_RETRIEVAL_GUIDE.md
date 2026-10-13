# 时空、语义与任务联合检索

新增的实现让超边参与经历绑定检查及受约束的邻域召回，同时保留平面检索作对照。现有时空、阶段、状态与观测证据规则继续作为硬约束；相似度分数不能绕过这些规则。默认不调用新的外部服务。

## 检索过程

1. 根据可选任务键、会话与生命周期选择候选，检查时间、空间、事件、入口阶段、状态及当前观测。排除不匹配、来源冲突、证据过时或包含未来结果的经历。
2. 获取问题相关性。默认使用本地 TF-IDF 余弦，改进原先未归一化的词项计数；它仍属于词法匹配，不具备完整的同义理解。配置向量服务并在 live 模式使用 ask 后，可以使用向量余弦处理改写问题。
3. 超图模式检查同一 episode 超边是否确实绑定记录声明的时间、地点、事件、行动、结果与过程。对于 evolution，同时检查自己的 process、阶段证据与源→目标转移超边；其他经历的边不能补足缺失的绑定。
4. 语义相关性超过阈值，或明确给定任务键的记录，形成直接候选。取至多 Top-K 条直接候选作种子，只在相同任务键的经验族内扩展一跳：种子经历→查询兼容的上下文节点→另一条经历。必须共享至少两种节点类型，如空间＋事件、空间＋状态；仅共享主题或来源不扩展。空间锚点只包含查询直接或经显式 within 关系到达的实体。
5. 每条被扩展的经历仍必须独立通过全部硬约束和结构绑定检查。扩展不复制种子的回答、阶段结果或观测证据，不绕过会话、作用域或明确任务键的过滤。
6. 将候选相关性、当前条件覆盖及有界的联合关系支持合成排序，取 Top-K；Jev 接收候选及绑定、扩展路径与当前证据，一次判断复用／综合／停止。输出前仍重新读取记录并检查当前条件及图绑定，防止推理期间记忆被修改。

这里“语义”是问题含义的相关性，不是额外创建一个万能的语义节点；事件、状态、行动、结果是结构化经历内容。交通仅作为结构示例，不是算法所依赖的专用输入。

## 排序与可解释输出

`search()` 返回 `(records, report)`；原有 `retrieve()` 仍返回记录列表，默认使用超图模式和本地词法评分。`ask` 的结果新增 `retrieval`，`retrieve` 命令输出改为 `{records, retrieval}` 对象。

报告包含 semantic_backend、fallback、filtered、scores；每条候选包含 semantic、context_coverage、joint_graph_support、total、recall_route、shared_paths 和 binding。binding 显示同一经历的匹配角色、入口阶段边、转移边以及通过当前证据检查的观测 ID。

当前默认总分为 `0.55×相关性 + 0.30×条件覆盖 + 0.15×联合支持`。条件覆盖是当前查询所提供的任务、时间、空间、事件、阶段、状态、观测维度中，历史经历显式保存相应条件的比例。超图联合支持为 `0.5×条件覆盖 + 0.5×邻域支持`；邻域支持取单条有效共享路径的最大类型数／4，截断到 1，不按节点度或路径数量累加。平面模式将联合支持项替换为条件覆盖；两模式共享语义编码器、硬约束与 Top-K。分数不是复用概率，权重尚需实验校准。

原 `retrieval_scores()` 保留供旧调用方读取基本上下文项；实际新排序应读取 `search()` 的 report，不能用旧辅助函数独立重算总分。

## 向量配置

新字段已以空值追加到 `.env`，现有密钥未改变。也可参考 memory-retrieval.env.example：

```dotenv
MEMORY_RETRIEVAL_STRATEGY=hypergraph
MEMORY_SEMANTIC_MIN_SIMILARITY=0.05
MEMORY_EMBEDDING_MODEL=
MEMORY_EMBEDDING_BASE_URL=
MEMORY_EMBEDDING_API_KEY=
```

填写所选服务实际支持的向量模型名称，BASE_URL 应是包含版本前缀的 OpenAI 兼容 API 根地址，客户端会追加 `/embeddings`。URL 和密钥留空时复用已有 LLM 服务配置，但该服务不一定提供向量接口；不自动假定某个模型可用。模型名留空时使用 TF-IDF。mock 永远不调用向量接口；retrieve、compare-retrieval 永远离线。

live ask 会调用配置的向量服务。记录向量按服务地址、模型和内容哈希缓存在 SQLite 新增的 embedding_cache 表中；查询向量只用于本次检索，不持久保存。修改记忆内容或更换模型会重新编码。短期记忆被 prune 删除时，其缓存通过外键同时删除。向量返回顺序按 index 校验重排，并检查批次数、维度、有限数值和非零范数。接口失败或返回错误会降级为 TF-IDF，并在 retrieval.fallback 中标明，不输出服务响应或密钥。

数据库升级到 schema 5，图快照仍为版本 3；只新增缓存表，不改写旧记录 payload，不增加现有十八类节点或八类超边。向量缓存是辅助索引，不是新记忆层。

## 离线演示与对照

```bat
cd /d E:\code\smp
conda activate smp-memory
set "PYTHONPATH=E:\code\smp\src"
python -m memory_agent --store runtime\memory_agent\joint_demo.sqlite3 ingest examples\joint_records.json
python -m memory_agent --store runtime\memory_agent\joint_demo.sqlite3 compare-retrieval --query-file examples\joint_query.json
python -m memory_agent --mode mock --store runtime\memory_agent\joint_demo.sqlite3 ask --query-file examples\joint_query.json
```

示例故意省略查询任务键：平面模式直接召回 joint_seed，超图模式还通过共享空间、事件、状态和证据要求召回 joint_neighbor；joint_wrong_phase 不满足增长阶段条件，被两模式共同排除。示例是机制检查，没有定义完整的正确答案集合，不能将“多召回一条”直接视为性能更好。mock 仍只对精确问题演示直接复用。

当前主对比使用 Mem0 官方开源 SDK，运行方式见 [Mem0 对比实验操作](FINDINGDORY_MEM0_GUIDE.md)。固定底座模型、向量模型、输入证据、回答提示词、Top-K 与预算；另可使用关闭邻域扩展、移除阶段绑定、关闭 Jev 等消融。记录正确经历 Recall@K／Precision@K、错误复用率、下游回答或任务成功率、端到端耗时与调用成本。

分层扁平图：[结构与检索](docs/figures/hypergraph-memory-paper-flat.md)。
