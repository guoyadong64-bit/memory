# LoCoMo：超图记忆独立实验

本功能只新增文件；不改动 FindingDory 的构建、检索、答题、评分、环境配置或历史结果。

在 `E:\code\smp` 的 CMD 或 PowerShell 运行：

```powershell
E:\conda\envs\smp-memory\python.exe -u -m benchmarks.run_locomo_hypergraph
```

**默认调用真实模型 API**，从项目 `.env` 读取提取模型、回答模型和嵌入模型。
默认模型分别为 `MEMORY_UNDERSTANDING_MODEL`、`MEMORY_ANSWER_MODEL`（未设置时使用
`MEMORY_LLM_MODEL`）以及 `MEMORY_EMBEDDING_MODEL`。不调用 JEV。不设置累计 token 硬上限。

默认对官方 `locomo10.json` 的 10 段长对话、四类可回答问题共 **1,540 题**运行完整流程：

1. 校验官方固定版本数据及评分源码，缺文件时下载公开数据。
2. 原始对话按发言落入独立 SQLite；按会话日期分批（每批最多 20 条发言）调用模型提取事件。
3. 事件关联参与者、原始对话证据、报告时间及报告阶段；保留原文，按原有规范形成超边。
4. 复用原有 SQLite 记忆实现及嵌入缓存，以语义种子和真实超边成员关联进行有界检索；
   排除任务/全会话家族节点的无差别扩展，最多两次关联跳转，按节点度惩罚泛化参与者。
5. 每题最多 32 条证据记录、32,000 字符的证据正文送给回答模型，包含准确的来源 ID、
   日期、发言人、原文及原始数据中的图片描述；不请求图片，不调用视频模型。
6. 官方评分函数计算逐题 F1，记录证据召回、请求/token 和时间，生成报告。

已完成且已落盘的请求可在同一命令续跑时复用；单题和每批事件都有断点。
连接失败或格式修复失败会留下可续跑数据。网络超时但服务端已处理的请求可能无法
恢复其未收到的结果；HTTP 账目仅记录已返回的 token，费用以平台账单为准。
不要同时启动两个进程写同一个实验目录。

## 数据、评分和比较边界

- 官方来源：<https://github.com/snap-research/locomo>。
- 固定版本：`3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`。
- 原始发布文件共 1,986 题：多跳 282、时间推理 321、开放域 96、单跳 841、不可回答 446。
- 仅 `conversation` 中的原文、发言人、时间和 `blip_caption` 进入记忆。
  **不使用 QA、标注证据、人工事件摘要或预生成观察/摘要构建记忆。**
- 参考答案和证据仅评分器读取。检索与回答函数只接受会话 ID 和问题字符串。
- 官方 F1 源码下载后核对 SHA-256，再按 AST 加载所需函数，避免为词级 F1 加载 BERTScore。
  多跳分项、开放域分号答案和 Porter 词干化均遵循该固定版本官方代码。
- F1 是部分匹配得分，不能称为“完全答对题目的比例”。
- `gold_dialog_evidence_recall` 是实际传给回答器的原始对话 ID 对标注证据的召回率，
  无标注的题不纳入其分母；它不是 FindingDory 的候选帧覆盖率。
- 这里是**同一存储框架的文本适配实验**，不使用 FindingDory 专用 `WitnessIndex` 和
  原图选择器；也不能直接把不同模型、数据子集、输入和评分的已发表分数拼成同条件排名。
- 发言顺序仅表示报告顺序，不凭此断言现实事件发生顺序；事件时间表达与报告时间分开保留。
  UTC 和微秒仅用于现有类型的存储锚点与顺序编码，不表示实际时区或测量的发言时刻。
- 原有 `verified` 入库标志表示来源/结构符合本实验检索条件，不代表模型事件经过独立事实核验。
- 默认四类总体 F1 按题数加权；启用第五类时，其官方拒答得分单列，不混入总体 F1。

## 输出目录

默认 `E:\code\smp\runtime\experiments\hypergraph_locomo`：

| 文件/目录 | 内容 |
|---|---|
| `config.json` | 固定数据、任务、代码和模型指纹；不含 API key |
| `memory/` | 各对话的 SQLite 超图及构建清单 |
| `extractions/` | 每批已完成事件提取断点 |
| `answers/` | 逐题预测、得分、原始证据、图关联路径和请求信息 |
| `response_cache/` | 已返回的模型响应，供中断续跑复用 |
| `api_budget.json`、`api_requests.jsonl` | 累计请求账目和原始用量 |
| `activity.json` | 累计活动时间，不包含两次启动之间的停机等待 |
| `predictions.jsonl` | 本次已完成题目的统一结果 |
| `summary.json`、`REPORT.md` | 总体与分类 F1、证据召回、用量、时间和中文报告 |
| `format_failures/` | 仅在发生 JSON/来源校验失败时保留原始审计证据 |

公共数据和评分源码保存在 `E:\code\smp\runtime\locomo`。
保留断点、响应缓存和错误审计用于复现；成功写入后自动清理无用临时文件。

## 可选命令

先仅本地预检（不调用模型、不下载）：

```powershell
E:\conda\envs\smp-memory\python.exe -m benchmarks.run_locomo_hypergraph --check-only
```

先小规模运行，仍会调用真实 API；独立保存，不与正式全量结果混合：

```powershell
E:\conda\envs\smp-memory\python.exe -u -m benchmarks.run_locomo_hypergraph --max-conversations 1 --max-questions 20 --run-dir runtime\experiments\hypergraph_locomo_smoke
```

四类及不可回答题全部运行：

```powershell
E:\conda\envs\smp-memory\python.exe -u -m benchmarks.run_locomo_hypergraph --categories 1,2,3,4,5 --run-dir runtime\experiments\hypergraph_locomo_all5
```

换模型/数据/检索参数/代码之后使用新的 `--run-dir`，避免混合版本。

新机器需要安装核心项目依赖和新增的可选评分依赖；本机已将后者安装在独立的
`runtime\locomo-deps`，不修改原来的 Conda 环境。可移植安装命令：

```powershell
E:\conda\envs\smp-memory\python.exe -m pip install -r requirements-locomo.txt
E:\conda\envs\smp-memory\python.exe -m benchmarks.run_locomo_hypergraph --prepare-only
```

NLTK 只使用 PorterStemmer，无需下载语料、分词模型或视觉模型。

## 本地验证

测试通过模拟 HTTP 覆盖完整构建、图检索、答题、官方评分、重启不重复请求、
收到付费响应后写断点前中断的恢复、无效来源拒绝、配置变化拒绝、第五类评分分离。
不调用真实模型 API。
