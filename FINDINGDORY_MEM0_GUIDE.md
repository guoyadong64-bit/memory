# 超图记忆与 Mem0 对比实验操作

当前入口为 `benchmarks.findingdory_mem0`，只比较 `hypergraph` 和 `mem0`。原始视频、192 张截图与题目在 `runtime/findingdory`；当前运行输出在 `runtime/findingdory_run_20261012_mem0_v9`。

新版记忆在原有主题、经验、事件阶段、观测证据结构中增加实体类别标签与属性，并用带来源的角色绑定区分被操作物体、执行者、来源和落点。实体标签、角色、事件阶段和图像证据以 SQLite 超边共同索引。单帧动作保留为不完整事件。v10 将描述中能唯一归属到实体的名词短语保存为带观测证据的标签，不按名称相似度合并实例。明确的执行者抬起动作及握持关系可以支持局部取物事实，即使最终落点未知；普通握持仍不能证明已拿起。背景提及不能成为交互锚点；来源缺失和访问顺序不完整会显式标记。未交互对象查询在完整可读取历史上计算，不在 Top-K 子集上推断。回答上下文优先保留目标及时间锚点证据，并提取有观测来源的证据子图；超大记录不会阻止后续小记录进入上下文。

Mem0 使用固定版本 `mem0ai==2.2.1` 官方 SDK 的 `add(infer=True)` 与 `search`。仅模型传输适配现有 API 和累计预算，官方事实提取、去重与检索代码保留；本地持久化使用 Qdrant 和 SQLite。SDK 默认 NLP 回退方式会记录在配置中。这是视觉观察转文本后的场景适配，不是托管 Mem0 平台。

两种方法使用同一份视觉输入、同一回答模型与提示词、相同 Top-K=20 和上下文字节上限。超图回答上下文共享实体目录并聚合重复角色绑定，保留事件与观测证据，避免大记录挤掉其他目标；同一物理实体只形成一个访问目标。对比入口不使用 Jev 门控或确定性帧回答器，以隔离记忆贡献；项目智能体中的 Jev 功能继续可用。帧数、记录粒度与实际 token 用量会单独记录。

在 CMD 中运行已经建立的独立环境：

```bat
cd /d E:\code\smp
set "PYTHONPATH=E:\code\smp\src"
runtime\mem0-env\Scripts\python.exe -m benchmarks.findingdory_mem0 prepare
runtime\mem0-env\Scripts\python.exe -m benchmarks.findingdory_mem0 caption
runtime\mem0-env\Scripts\python.exe -m benchmarks.findingdory_mem0 build
runtime\mem0-env\Scripts\python.exe -m benchmarks.findingdory_mem0 index
runtime\mem0-env\Scripts\python.exe -m benchmarks.findingdory_mem0 evaluate
```

`caption`、`index`、`evaluate` 会续跑已完成部分。当前预算保留之前已使用的累计 token，不因删除旧结果而重置；上限仍为 5,000,000 reported tokens。新调用明细在 `api_requests.jsonl`，预算在 `api_budget.json`，方法与模型配置会冻结。更改模型、提示或记忆构建代码后应使用新运行目录。

最终 `summary.json` 和 `predictions.jsonl` 分别保存汇总与逐题结果；`observations` 是共同视觉证据；`memory` 是你的 SQLite 库；`mem0`、`mem0_facts` 是官方基线的存储与提取事实。过程缓存在实验结束后清理。

评分是离线目标帧命中率，12 道已用于开发的题目可用来排错；正式论文验证需要新片段和未参与调试的问题。

## 不调用 API 的开发验证

在 CMD 中运行：

```bat
cd /d E:\code\smp
set "PYTHONPATH=E:\code\smp\src"
runtime\mem0-env\Scripts\python.exe -m benchmarks.findingdory_replay
```

该命令读取 v9 已保存的观测，在临时 SQLite 中重建当前超图，完成检索、角色证据和上下文容量检查；结束后自动删除临时库，只保留 `runtime/findingdory_replay_v10.json`。不会读取 `.env`、调用视觉／嵌入／回答 API，也不会改写 v9 的观测、记忆或成绩。离线诊断使用词项检索，与原实验的在线嵌入检索不同；输出中的目标覆盖率和解析完成数不是新的答题准确率。

开发时不必每次完整重跑。视觉提取不变时复用共同观测；记忆方法变化时重建你的记忆，重新计算受影响的检索和回答；Mem0 的输入、配置及回答请求全部不变时，可以复用其结果。回答模型或共同提示词变化则双方答案都要重新评测。模块测试和离线回放用于排错，最终准确率仍要通过真实回答评测确认。

回放会计算逐题回答请求的哈希并检查共同提示词，避免把旧答案误当成新版成绩。由于此次标签和上下文投影都有变化，当前 12 个超图回答请求全部发生变化，需要重新生成答案才能报告新版准确率；Mem0 12 个已保存回答请求在解码器和上下文层面不变，其他配置仍须核对。旧累计使用量为 4,955,938/5,000,000 tokens，本次回放使用量为 0；不会自动重置预算或启动付费复测。

## 重新调用模型评测

用户明确授权本次新实验从零计数，上限 5,000,000 tokens。入口如下，默认输出目录是 `runtime/findingdory_run_20261012_mem0_v10`：

```bat
set "PYTHONPATH=E:\code\smp\src"
runtime\mem0-env\Scripts\python.exe -m benchmarks.findingdory_retest
```

入口核对共同观测、题目、模型、嵌入端点、Mem0 适配器和回答请求，然后复用 Mem0 的 12 个结果；只重建超图、更新变化的嵌入并调用回答模型。相同目录再次执行会跳过已完成题目，不会重置该目录的用量。原有消耗保留在 `historical_usage`，不计入本次额度。改变已冻结的代码或配置须使用新目录，新增预算需要另行授权。

v10 的 12 次真实作答已完成：超图 5/12，Mem0 4/12（复用），本次 reported tokens 为 202,654。本轮准确率没有提高。以运行目录中的 `summary.json`、`predictions.jsonl` 和 `api_requests.jsonl` 为完整依据。

## 当前完成结果：v14

2026-10-13 用户明确解除累计 token 上限。当前 v14 的 `api_budget.json` 保留 1,724,691 的已用计数，并将 `max_reported_tokens` 设为 0（表示不限制累计 token）；后续从该目录衔接的实验继承该设置。请求次数、单次输出与重试仍有边界，API调用日志继续记录。下面的500万上限是已完成实验当时的历史配置。已完成视觉提取的分批缓存和测试缓存已清理，最终观测与结果保留。

当前有效输出为 `runtime/findingdory_run_20261012_mem0_v14`。超图 7/12（58.3%），第一段 5/6、第二段 2/6；Mem0 3/12（25%）。v11 重新提取第二段的 96 帧，v12 在共同新观测上为双方重新生成 24 个带原图核验的回答；v13/v14 保持共同观测、模型和回答提示不变，只按回答请求哈希决定是否重测。v14 最终 12 个超图请求均变化并重新作答，12 个 Mem0 结果复用。相同目录支持断点续跑，已完成答案不会重复调用：

```bat
set "PYTHONPATH=E:\code\smp\src"
runtime\mem0-env\Scripts\python.exe -m benchmarks.findingdory_delta_retest --source-run runtime/findingdory_run_20261012_mem0_v13 --run-dir runtime/findingdory_run_20261012_mem0_v14
```

当前额度累计 1,724,691/5,000,000 reported tokens，v14 新增 176,443。v14 有一个中断后没有返回用量的请求；记录值无法覆盖其可能产生的账单费用。历史 4,955,938 单独保留，不重置或混入当前额度。

这轮修改保留跨间隔的事件连续性与不确定区间，区分局部取物证据和完整动作，给未识别时间锚点保留候选时间线，并压缩证据子图以保留前后边界和完整候选事件。第二段多来源顺序题和硬盘时间锚点题的候选标准帧覆盖率已到 1，但仍回答错误；覆盖率不是答题准确率。两张凳子只回答了一张；最远位置题仍没有可验证的距离与当前位姿支持。v14 成绩与 v13 相同，不能声称最后的压缩修改提高了准确率。

相对 v10 的提升同时包含视觉提取、共同原图核验和记忆修改，不能全部归因于超图结构。12 题已用于开发，论文仍需固定共同感知及回答器的消融和未调试测试集。逐题结果、配置和核对记录分别见 `predictions.jsonl`、`evaluation_config.json`、`verification.json`。

2026-10-13 继续优化事件证据投影：候选事件先于零散帧分配上下文，保留每种动作阶段的前后边界；类别完全无法匹配的多目标查询保留候选动作供核验，不自动改标签或认证角色。v15 验证发现候选干扰使成绩降到6/12，收紧类别回退条件后 v16 为7/12，第二段仍2/6，没有提高总体准确率。当前结果目录 `runtime/findingdory_run_20261013_mem0_v16`，11个未变超图请求复用、1题重测，Mem0的12个请求未变并复用。累计reported tokens为1,833,605，无累计token上限；相对v14新增108,914。原始数据、共同观测和有意义的实验审计保留，过程与测试缓存已清理。

## 最新完整对比：2026-10-13

有效目录为 `runtime/findingdory_run_20261013_full_compare_verified`。两段共同观测各96帧保持不变，超图从头构建，Mem0两段均重新执行官方SDK的事实提取和索引；双方12题全部重新检索、调用相同视觉回答模型，不复用任何历史答案。结果为超图7/12（第一段5/6、第二段2/6），Mem0为5/12（第一段4/6、第二段1/6）。Mem0比此前复用成绩3/12提高；超图正确题数没有提高。

本轮有效运行132个请求：84个嵌入请求、24个Mem0事实提取请求、24个回答请求，失败0次；新增851,966 reported tokens，累计2,685,595，无累计token上限。先前启动目录另消耗24 token进行维度探测，已纳入累计计数。`verification.json`核对24条唯一新答案、评分、证据图像范围、共同观测、SQLite完整性和冻结代码哈希；`sdk_audit.json`核对官方Mem0源码未改。

恢复该完整实验的CMD命令（已完成的同轮步骤会跳过）：

```bat
set "PYTHONPATH=E:\code\smp\src"
runtime\mem0-env\Scripts\python.exe -m benchmarks.findingdory_full_compare --source-run runtime/findingdory_run_20261013_full_compare --run-dir runtime/findingdory_run_20261013_full_compare_verified
```

这里的完整实验指当前选定的12道开发题；原始视频不重新下载或提取，共同感知保持固定。正式报告优先引用本轮全新结果，不能继续用之前复用的Mem0 3/12作为最新成绩。

## 事件图像与类别召回优化

`runtime/findingdory_run_20261013_evidence_repair`固定上轮共同观测和已验证的双方索引，重新检索并生成全部24条答案。新实体召回保留未匹配类别的代表性视图，新图像分配优先覆盖每个记忆事件；双方共同检查数量、目标实体重复和引用，最多纠错一次。本轮31次回答调用（24条答案加7次纠错），总55个请求、失败0次，新增462,669 reported tokens，累计3,148,264，无累计token上限。

模型原始汇总7/12与4/12。“任意一个”题双方均返回多个目标，后来加入共同任务输出归一化：不读标准答案，仅选模型顺序中首个可引用候选，并保留原始输出。规范化后超图8/12（第一段6/6、第二段2/6），Mem0为5/12（第一段4/6、第二段1/6）。归一化没有额外模型调用。评估原始模型输出使用predictions.jsonl；评估当前统一任务输出使用normalized_predictions.jsonl和normalized_summary.json，REPORT.md并列说明两者，避免混淆。

胶带题修复；五来源顺序、两张凳子、最远距离和硬盘时间锚点仍失败。178项测试通过。共同解码器已经变化，原实验的冻结代码存档在code_snapshots；修改过代码后不能在旧目录混跑，下一轮使用新目录并从最新预算衔接。不能把当前提升全部归因于超图结构。

## 新基线：你的超图与A-Mem

入口为 `benchmarks.findingdory_amem`，比较方法固定为 `hypergraph` 和 `amem`。使用作者维护的 `https://github.com/WujiangXu/A-mem-sys`，固定提交 `f303dfc71e07bdc787f4bc135d4cea328ae30e99`。源码位于被Git忽略的 `runtime/baselines/a_mem_sys`；下载时核对该提交，实验配置和implementation_audit.json保存源码哈希及依赖版本。源码保持原样，适配器通过初始化接口接入预算化千问API、相同Qwen嵌入与独立Chroma存储，JSON对象传输后按原schema本地验证。笔记分析、增强文本、关联、演化、search_agentic保留官方方法。

相同192帧观测、相同语言和嵌入模型、共同回答器、图像上限12、Top-K=20、40KB上下文与评分；A-Mem每帧一条原始笔记，分别构建两段历史。双方全部12题重新作答，不复用旧答案。对外应称“官方A-Mem系统的视觉观测转文本场景适配”，不能称为原论文配置复现；默认MiniLM嵌入已替换为共同的Qwen API嵌入。

CMD启动：

```bat
cd /d E:\code\smp
set "PYTHONPATH=E:\code\smp\src"
runtime\amem-env\Scripts\python.exe -m benchmarks.findingdory_amem
```

当前实验为并行续跑目录，使用下面这条完整命令恢复（同一段内部顺序保持不变，已完成笔记不会重复生成）：

```bat
runtime\amem-env\Scripts\python.exe -m benchmarks.findingdory_amem --source-run runtime/findingdory_run_20261013_amem --run-dir runtime/findingdory_run_20261013_amem_parallel --reuse-amem-state
```

默认输出为 `runtime/findingdory_run_20261013_amem`；本次有效续跑目录为 `runtime/findingdory_run_20261013_amem_parallel`，预算承接最新实验且累计token不限。每条新增笔记后保存快照，恢复时保留原有Chroma索引及官方尚未整合的邻居元数据状态，复用已完成的模型调用缓存；已保存的同轮答案跳过。两段独立构建并行执行，预算更新有锁，段内按原顺序演化。API网络或结构失败会停止，避免以官方异常回退产生的空元数据作为有效基线。notes.json保存笔记和关联，api_requests.jsonl记录分阶段用量；后续以summary.json、predictions.jsonl和REPORT.md核对成绩。

隔离依赖由requirements-amem.txt固定；原Mem0环境保持独立。新增模拟测试验证自动元数据、官方关联演化、检索来源和快照恢复不重复调用；该环境整体179项通过、1项因未安装可选Mem0 SDK跳过。

本轮已完成，结果为超图8/12（6/6、2/6），A-Mem为4/12（4/6、0/6），全部24题新生成，共32次回答调用含纠错。有效目录 `runtime/findingdory_run_20261013_amem_parallel`，REPORT.md、predictions.jsonl、verification.json记录完整核对。两段笔记各96条，关联195条与202条、无无效ID；源码保持不变。默认演化整合阈值100在每段96条历史中未达到，逐条演化均执行95次。新增用量1,954,435，累计当前计数5,102,699；串行迁移时有1个请求中断未返回用量，账单可能包含它。此次为官方系统场景适配，12道开发题不能证明总体优越性。

A-Mem环境路径 `E:\code\smp\runtime\amem-env`，Python入口 `Scripts\python.exe`，新增包在 `Lib\site-packages`。它是继承基础smp-memory包的Python venv（system-site-packages=true），不是新的Conda环境。Mem0仍在 `runtime/mem0-env`。若需要在另一台机器复现，按固定提交下载官方仓库到runtime/baselines/a_mem_sys，在venv中安装requirements-amem.txt并配置现有.env；运行以上并行命令时更改输出目录以隔离新实验。
