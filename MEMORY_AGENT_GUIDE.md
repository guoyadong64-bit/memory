# 通用超图记忆与 Jev 原型

`src/memory_agent` 提供通用记忆问答、SQLite 超图存储与 Jev 接口，支持长短期记忆、会话历史、失效反馈与显式记忆晋升。本版本是功能原型，不能当作方法优越性或论文创新已成立的证据。

时空扩展支持时间窗口、空间实体与关系、事件、执行步骤和结果；具体结构及离线验证命令见 [时空超图记忆说明](SPATIOTEMPORAL_MEMORY_GUIDE.md)。旧记录继续兼容。

当前还支持多源观测归一化与对齐、冲突检查、事件阶段和状态转移；见 [过程演化记忆说明](EVOLUTION_MEMORY_GUIDE.md)。

## 1. 运行

在项目根目录的 PowerShell 中运行，Python 需要 3.11 或更新版本：

已创建专用环境时，先运行 `conda activate smp-memory`。也可用 `conda env create -f environment-memory.yml` 重建环境。

```powershell
python -m pip install -r requirements-memory.txt
$env:PYTHONPATH = Join-Path $PWD 'src'
python -m memory_agent check
python -m memory_agent demo
```

`demo` 强制离线 mock，创建独立的演示记忆、回答精确匹配的问题，结束后清理演示数据。它不会访问模型接口，也不会修改正式记忆。已安装依赖时无需重复安装。

使用持久化记忆：

```powershell
python -m memory_agent ingest examples/memory_records.json
python -m memory_agent ask --query-file examples/memory_query.json
python -m memory_agent inspect
```

示例记忆是人工构造的项目偏好，不是真实项目事实。导入会返回 `record_ids`；相同 key 和条件下再次导入已验证记录，会替换旧断言。默认正式存储位置是 `runtime/memory_agent/memory.sqlite3`。

如果需要试用独立的存储文件，`--store` 等全局参数放在子命令之前：

```powershell
python -m memory_agent --store runtime/trial/memory.sqlite3 ingest examples/memory_records.json
python -m memory_agent --store runtime/trial/memory.sqlite3 ask --query-file examples/memory_query.json
```

## 2. 填写密钥

根目录 `.env` 保存运行配置。填写以下项目，再将模式改成 `live`：

```dotenv
MEMORY_AGENT_MODE=live
MEMORY_LLM_API_KEY=你的生成模型密钥
MEMORY_LLM_BASE_URL=https://api.openai.com/v1
MEMORY_LLM_MODEL=你账户可用的模型名称
TYPESAFE_API_KEY=你的TypeSafe密钥
TYPESAFE_BASE_URL=https://api.typesafe.ai
TYPESAFE_MODEL=jev-1.13.0
```

生成接口使用 OpenAI 兼容的 `/chat/completions`，要求模型支持 `response_format=json_object`。其他供应商可修改地址和模型名。`MEMORY_UNDERSTANDING_MODEL` 和 `MEMORY_ANSWER_MODEL` 可分别指定理解与回答模型，留空时使用 `MEMORY_LLM_MODEL`。默认模型名只是配置默认值，需以你的供应商实际可用模型为准。

```powershell
python -m memory_agent check
python -m memory_agent ask --query-file examples/memory_query.json
```

`check` 只检查配置和密钥是否填写，不进行网络请求，不能证明密钥有效或模型可用。live 的 `ask` 会发送问题与候选记忆到配置的模型服务。`.env`、运行数据已加入 `.gitignore`；环境变量优先于 `.env`。新增模块只读取上述变量，不使用旧筹划模块的模型配置。

可以先用 `python -m memory_agent probe` 测试两个 API：只发送简单的人工演示数据，不写入记忆；每个服务最多一次请求，失败时打印状态，不显示密钥。该命令始终测试真实 API，可能产生少量费用，即使 `.env` 仍为 mock 也会调用。

如果 Jev 密钥来自 OpenRouter，使用 `TYPESAFE_BASE_URL=https://openrouter.ai/api`、`TYPESAFE_MODEL=jev-1.13`。程序会附加 `/v1/systemone`，不要在这个 base URL 后重复填写 `/v1`。见 [OpenRouter System One 文档](https://openrouter.ai/docs/guides/community/typesafe-sdk)。

CMD 中设置源码路径的命令是 `set "PYTHONPATH=E:\code\smp\src"`；上文 `$env:PYTHONPATH` 写法仅适用于 PowerShell。

## 3. 超图记忆是什么

节点分为意图、适用条件、回答、来源和反馈五类。一条 episode 超边同时连接这些节点，保留完整的条件与证据组合；同一个意图的记录通过 family 超边聚合；替换与失败反馈通过 invalidation 超边保留失效关系。

当前原型围绕**带条件、可追溯、可失效的回答证据束**组织记忆。检索先检查已验证、有效期、替换和失败状态及全部适用条件，再进行相关性召回、经历与阶段绑定检查和受约束的共享条件扩展，最后联合排序。

当前检索已升级为本地 TF-IDF／可选向量相关性、经历超边绑定检查和条件约束的单跳邻域召回；详见 [联合检索说明](JOINT_RETRIEVAL_GUIDE.md)。没有学习式超图传播或图神经网络；使用超图数据结构本身不代表检索效果已经更好。

## 4. Jev 放在哪里

三个逻辑角色由协调器串行调用：

1. **理解角色**：结合当前会话未过期的历史生成检索词；不能修改调用者给定的 key、条件或会话。历史仅供理解上下文，不是事实证据。失败时回到原问题。
2. **经验判断角色**：用 Jev 在候选记录、`synthesize`、`abstain` 中选择；对各候选同时询问 Noul 相关度和 Score 支持程度。
3. **回答角色**：满足复用门槛时直接返回证据中的回答；否则由生成模型综合候选证据，并给出来源记录 ID。

Jev 通过 [TypeSafe System One API](https://docs.typesafe.ai/api) 调用，不负责生成自由文本，也没有挂载或训练我们的超图。超图由本地程序维护，检索结果作为 Jev 的 `state` 输入。

复用要求本地有效性检查通过，Jev confidence ≥ 0.85、Noul ≥ 0.8、Score ≥ 1.8（本配置采用三档，范围 0–2）。这些是原型默认门槛，**未经过实验校准**；confidence 和 Noul 不能直接解释为真实正确率。结果记录实际模型名、usage、调用次数与耗时。

Jev 出错或复用未达门槛时尝试证据综合；Jev 明确选择 abstain 时停止回答。生成结果的来源 ID 必须来自候选记录，否则拒绝输出。来源 ID 校验不是事实核验，仍可能存在错误归因，需要后续专门评估。生成的回答不会自动写回已验证记忆。

mock 只用精确问题匹配的本地规则模拟复用，没有真实 Jev 推理或生成能力。三个角色是逻辑模块，不是三个独立自主运行的智能体。

## 5. 添加、更新和失效

导入文件仍使用 JSON，但导入后记忆存储在 SQLite 中。每条 JSON 记录至少包含 `key`、`question`、`answer`、`source`；`verified` 默认 false，只有显式标为 true 的记录能用于回答。该标记由输入者负责，程序不验证来源真实性。`conditions` 是字符串映射，`created_at` 和 `valid_until` 若填写必须包含时区。省略 ID 时自动生成，指定 ID 不得重复或使用 `synthesize`/`abstain`。

同一作用域、同一会话（短期记忆）、同一 key 和完全相同条件下的新已验证记录替换旧记录，拒绝更早时间的更新。不同条件的记录可以共存；当前没有通用语义矛盾识别。失败反馈会让记录永久退出当前检索，旧的被替换记录不会因此恢复：

```powershell
python -m memory_agent feedback 这里替换为记录ID --reason '用户反馈该记录不再适用'
```

恢复时应导入一条新的已验证记录。数据存储采用 SQLite、外键、WAL 和事务，写入增量更新，多个进程的写入由数据库串行协调，等待锁最多 10 秒。批量 ingest 在同一事务中提交；任一记录失败整批回滚。超过锁等待时间时应稍后重试。

## 6. 测试与后续验证

```powershell
python -m pytest tests/test_memory_agent.py -q
```

测试用模拟 HTTP 响应检查真实接口请求结构、错误回退及证据校验，不消耗 API 配额。真实账号的连通性、响应格式与效果，需要填写密钥后验证。

后续应固定同一问题集和生成模型，比较无记忆、扁平记忆、超图记忆、超图加 Jev；保持可用证据、提示预算和错误处理一致。测量回答正确率、错误复用率、拒答率、p50/p95 端到端延迟、调用次数与 token 成本，并单独测量更新、过期、条件不匹配和来源不足场景。不能把 mock 的精确匹配结果或一次成功复用当作 Jev 提速和超图必要性的证明。

## 7. SQLite 与长短期记忆

SQLite 来自 Python 标准库，无需新安装服务或依赖。数据库用 `records` 保存带条件的事实，`nodes`、`hyperedges`、`hyperedge_members`、`hyperedge_records` 保存超图的节点、超边和关联；`session_turns` 保存会话历史，`metadata` 保存迁移信息。数据库是权威存储，检索时查询数据库；`memory.graph` 仅返回诊断快照，修改快照不会写回。`inspect` 的记录计数包含已失效及未清理记录。

长期记忆默认 `scope=long`，跨会话可复用，仍需已验证、条件匹配且未失效。短期事实用 `scope=short`，必须绑定 `session_id`，且有 `valid_until`。命令行导入短期事实时，未填写有效期就使用当前时间加 `MEMORY_SHORT_TTL_MINUTES`，默认 60 分钟。

查询的 `session_id` 匹配时，短期事实与长期事实共同检索。同一 key、完全相同条件的有效已验证短期事实临时覆盖长期事实；它不会修改长期记录，过期或失效后恢复使用长期记录。不同条件不做通用语义冲突消解。没有 session_id 的查询只使用长期记忆。

会话历史是另一类短期数据。每次带 session_id 的 `ask` 自动保存完整结果，默认每个会话最多最近 20 轮，每轮默认 60 分钟有效；过期记录不再作为上下文。只有当前会话的未过期历史进入理解模型。程序不会把历史中的用户话语或生成回答自动标成已验证事实。mock 不理解多轮指代，应使用明确问题；live 通过历史辅助生成检索词，但事实检索仍须提供正确 key 和条件。

下面是 CMD 中的完整示例，先在根目录激活 `smp-memory` 并设置 `PYTHONPATH`。旧长期示例已经导入时跳过第一条：

```bat
python -m memory_agent ingest examples/memory_records.json
python -m memory_agent ingest examples/memory_short_records.json --ttl-minutes 60
python -m memory_agent ask --query-file examples/memory_query.json --session demo-session
python -m memory_agent ask --query-file examples/memory_query.json
python -m memory_agent history --session demo-session
python -m memory_agent inspect
```

同一会话的候选应为临时 PDF 要求，无会话查询的候选应为长期 Markdown 要求。模型可能选择综合或拒答，实际查看 `candidate_ids` 和 `judgment`。验证确定性规则可在 `ask` 前加 `--mode mock`，例如 `python -m memory_agent --mode mock ask --query-file examples/memory_query.json --session demo-session`。

有效且已验证的短期事实，只有通过显式命令才能晋升为长期事实：

```bat
python -m memory_agent promote 短期记录ID --source "用户明确确认这是长期要求"
python -m memory_agent prune
```

晋升创建新长期记录，保留条件、问题和回答，使用提供的确认来源，同时记录替换关系。`prune` 只物理删除过期短期事实、过期会话历史及相关超图关系，不删除长期事实。有效期限制的是每条记录，不会自动续期；读取历史也不会延长有效期。短期数据也落盘，所以可跨进程继续会话，它的“短期”由有效期和容量定义。

## 8. 旧 JSON 数据迁移

默认数据库现在是 `runtime/memory_agent/memory.sqlite3`。现有 `graph.json` 已迁移时保留为原始备份，不继续更新。新数据库迁移旧文件的命令如下：

```bat
python -m memory_agent --store runtime/memory_agent/memory.sqlite3 migrate-json runtime/memory_agent/graph.json
```

迁移保留原始记录 ID、超图成员、替换状态和失败反馈，不删除源文件；同一文件重复迁移会跳过。若目标已有其他记忆则拒绝合并，避免覆盖数据。`.env` 的 `MEMORY_STORE_PATH` 要指向 SQLite 文件，不能继续指向 JSON。SQLite 的 `-wal` 和 `-shm` 是运行时文件，进程正在使用数据库时不要手动删除。
