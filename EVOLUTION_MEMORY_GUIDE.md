# 多源观测与过程演化超图记忆

本版本在时空记忆上增加多源观测、阶段、状态和有向状态转移。交通可以提供实验背景，记忆接口仍使用通用实体、观测、状态和过程，不依赖项目内的视频、门架数据库或 GIS 格式。

## 结构如何变化

可按四个逻辑层理解：

1. **观测与上下文层**：实体、时间、来源、归一化观测和任务条件。
2. **阶段经验层**：把某阶段的状态、动作、结果及引用的观测连接起来。
3. **过程演化层**：以明确的源阶段和目标阶段连接状态转移，再按事件实例组织整段过程。
4. **任务组织层**：沿用 family 超边组织可检索经验及长短期作用域。

四层是结构解释，不是四个独立数据库。支持 **18 种节点类型、8 种超边类型**；数量由实际内容决定，不是每条记忆固定创建这么多个节点。

在原有 11 类节点上新增 observation、source、process、stage、state、requirement、transition。新增的四种超边为：

|超边|连接的内容|
|---|---|
|observation|观测值及单位、来源、空间实体、观测时间|
|stage_evidence|过程实例、阶段、状态、时间、动作、结果、引用的观测|
|state_transition|过程实例、源阶段、目标阶段、有向转移描述|
|process|同一事件实例的各阶段，可关联多条经验|

空间关系继续支持保存任意谓词。`within` 可以传递匹配；`upstream_of`、`connected_to` 等作为显式关系前提检查，当前不会自动计算交通影响传播。阶段转移的关系为 `precedes`：表示时序先后，不能直接当成因果证据。

## 多源数据如何进入

新增 `normalize` 命令。每个来源提供一个字段映射，明确输入单位、输出单位、换算比例、实体别名、来源与数据类型。转换后的统一观测包含：

`id / source / modality / namespace / entity / metric / value / unit / observed_at / provenance / quality`

例如，将门架编号 G001 对齐为 road-1，将 vehicles/hour 除以 60 转成 vehicles/min，再与其他来源同一实体的观测匹配。事件类别使用字符串，单位可以明确为 category；数值指标使用数值和明确单位。

当前入口处理 JSON 行列表，不直接解析视频、GIS 或数据库。单位、实体别名与字段映射需要调用方配置；不会猜测单位，也不会自动判断两个来源是否具有独立性。`provenance` 保存原始来源引用，但不自动获取或验证引用内容。后续专业模型可将原始数据处理结果写成这一接口。

```bat
python -m memory_agent normalize --input examples\raw_observations.json --mapping examples\observation_mapping.json
```

可追加 `--output runtime\memory_agent\normalized_observations.json` 保存。此命令不写入记忆、不调用模型。

## 经验与查询的新增字段

**Record：**

- `observations`：历史观测，保存来源、时间、单位和实体。
- `evolution`：命名空间、事件实例 ID、按时间严格递增的阶段列表。阶段包含 phase、state、action、outcome 和 observation_ids。
- `required_state`：当前任务必须满足的状态条件。
- `evidence_requirements`：当前观测必须满足的指标、单位、类型、实体、范围或类别、来源数量、质量、新鲜度、时间对齐窗口、冲突容差。

**Query：**

- `phase`：当前阶段，不能直接用事件类别代替。
- `state`：当前状态。
- `observations`：当前任务已获得的多源观测，与历史观测分开。

一条演化经验以其首阶段为复用入口，最后阶段提供历史后续结果。要在另一个阶段复用，存储以该阶段为起点的经验片段。示例中增长→恢复与恢复→稳定属于同一历史事件，但适用入口不同。

阶段引用的观测必须存在、属于相同命名空间、发生在阶段之前或同一时刻，且在配置的对齐窗口内。时间顺序、引用关系在入库时验证。当前 phase 和 state 由调用方或上游模型提供，不会由记忆模块自动识别。

## 复用判断

在原有生命周期、条件和时空过滤之外，新增：

1. 查询阶段必须匹配经验首阶段；首阶段状态与 required_state 必须满足。
2. 历史结果必须在查询时刻之前已经发生，避免历史重放实验使用未来结果。
3. 当前观测必须来自查询空间命名空间和指定实体，满足单位、类型、质量和新鲜度。
4. 每项指标以最新相关观测为时间锚，只选择对齐窗口内每个来源、每个实体的最新观测。默认检查查询的全部实体，也可在 requirement.entity 指定其中一个实体。
5. 每个实体都必须达到来源数量要求。不同来源的同实体指标超过冲突容差时阻止直接复用，不静默平均。不同指标的观测也必须在共同时间窗口内。
6. 检查数值范围或类别条件，再对可用经验排序。当前还会核验同一经历与阶段链的超边绑定，并允许受约束的邻域扩展；相关性支持本地 TF-IDF 或可选向量模型，见 [联合检索说明](JOINT_RETRIEVAL_GUIDE.md)。排序权重尚不是训练得到的最优权重。
7. 一次 Jev 请求同时看到当前观测与历史过程，判断相关性、支持度和复用路径；输出前再次检查记忆有效性。

没有候选经验时停止回答，模型不能越过硬条件。历史观测不会被拿来证明当前状态。来源数量不等于来源独立性；quality 是调用方声明的质量，当前不会自动估计。

## 离线运行

CMD 中运行。以下示例完全为合成数据，不表示真实交通效果。使用独立数据库；重复导入同一文件会因固定 ID 报重复。

```bat
cd /d E:\code\smp
conda activate smp-memory
set "PYTHONPATH=E:\code\smp\src"
python -m memory_agent --store runtime\memory_agent\evolution_demo.sqlite3 ingest examples\evolution_records.json
python -m memory_agent --mode mock --store runtime\memory_agent\evolution_demo.sqlite3 ask --query-file examples\evolution_query.json
python -m memory_agent --mode mock --store runtime\memory_agent\evolution_demo.sqlite3 ask --query-file examples\evolution_recovery_query.json
python -m memory_agent --mode mock --store runtime\memory_agent\evolution_demo.sqlite3 ask --query-file examples\evolution_conflict_query.json
python -m memory_agent --store runtime\memory_agent\evolution_demo.sqlite3 explain --query-file examples\evolution_conflict_query.json
python -m memory_agent --store runtime\memory_agent\evolution_demo.sqlite3 trace --namespace demo-network --case-id historical-incident-1
python -m memory_agent --store runtime\memory_agent\evolution_demo.sqlite3 inspect
```

预期：增长阶段复用 evolution_growth，恢复阶段复用 evolution_recovery，冲突查询 abstain。explain 显示过滤原因及匹配观测 ID；trace 通过 process 超边读取同一事件实例的经验片段，保留来源、失败、替代与验证状态。trace 是诊断历史，不代表所有显示片段都可用于当前任务；短期片段必须提供匹配的 `--session` 且未过期。

正式记忆仍位于 SQLite。schema 自动升级到 4，图快照版本为 3，兼容旧数据。提升短期记忆只确认存储作用域，完整保留时空、观测和过程条件，不把历史观测重新标成当前观测。

## 当前实现的研究边界

已实现可运行的结构化数据归一化、显式多源对齐检查、来源冲突过滤、阶段化经验复用和有向演化存储。尚未实现从原始传感器流自动抽取事件、自动阶段识别、因果发现、路网动态传播预测或交通控制执行。

结构正确、测试通过只能证明这些机制按规则工作，不能证明任务效果已经优于其他方法。后续需要与具有相同字段和规则的平面记忆比较，再分别去掉过程关系、多源对齐、冲突检查和 Jev。当前主要检索依然通过 family 聚合后做记录约束检查，process 超边用于过程关联与追踪；还不能宣称超图关系推理本身已带来性能提升。
