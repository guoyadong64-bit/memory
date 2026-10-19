# 四种记忆方法的统一实验入口

## 验证集前 50 个视频的共享视觉观测

已下载的 `ep_1`～`ep_50` 共 50 个视频、2,901 道题。先在项目根目录做只读预检：

```powershell
E:\conda\envs\smp-memory\python.exe -m benchmarks.prepare_validation50_observations --check-only
```

确认模型费用后运行完整视觉解析：

```powershell
E:\conda\envs\smp-memory\python.exe -m benchmarks.prepare_validation50_observations
```

正式命令会调用 `.env` 中的视觉模型；仅使用视频帧，不把问题或标准答案发送给模型。50 个视频共 4,800 帧，每批 2 帧，正常情况下至少约 2,400 次 API 请求，异常重试可能增加请求和费用。默认不设累计 token 上限；可在首次运行时用 `--max-reported-tokens N` 设上限，或用 `--max-requests N` 调整默认 10,000 次请求上限。限制写入运行账本后不能在同一目录中途更改。

共享结果默认保存在 `runtime\experiments\findingdory_validation50_perception`，其中 `observations\ep_N.json` 是每个视频的 96 帧结构化观测，`observations_manifest.json` 固定全部观测的 SHA-256，`frames\ep_N` 保存一致的原图输入，`perception\api_budget.json` 记录模型用量。每批保存检查点，同一命令中断续跑时不重复请求已完成批次；模型、抽帧、代码或输入文件改变时，需用新的 `--shared-dir`。后续各方法须读取同一份 `observations_manifest.json` 和观测目录，并保持答题器与评分规则一致。下文现有的四方法实验入口仍固定为两视频、12 道开发题，尚不会自动改为这 50 个视频。

旧视觉流程若因模型复用错误实体/交互 ID 而停在 `ep_16`，可使用带审计记录的 V2 续跑入口。先做不调用模型的预检：

```powershell
E:\conda\envs\smp-memory\python.exe -m benchmarks.resume_validation50_observations --check-only
```

再用同一入口正式续跑：

```powershell
E:\conda\envs\smp-memory\python.exe -m benchmarks.resume_validation50_observations
```

V2 默认写入 `runtime\experiments\findingdory_validation50_perception_v2`，原目录保持不变。它复用已完成视频、成功批次和可验证的失败响应；模型只会收到后续未完成的帧。V2 对跨帧 ID 冲突分配确定性新 ID；对缺少可见证据的交互不编造阶段，而是保留描述及不确定性。归一化前后的数据和改动保存在 `perception\normalizations`，旧请求及 token 计数见 `source_reuse.json`，V2 新请求另记于 `perception\api_budget.json`。JSON 缺帧或图像请求失败仍会明确报错，防止假装生成了观测。实验报告需披露旧观测与 V2 处理规则并用同一份封存结果比较所有方法。

## 使用封存的 50 段观测测试超图方法

专用入口默认读取 V2 的 50 段观测与原图，对前 50 段的全部 2,901 道官方题作答。先检查输入，不请求模型：

```powershell
E:\conda\envs\smp-memory\python.exe -m benchmarks.run_validation50_hypergraph --check-only
```

正式运行与中断续跑使用同一命令：

```powershell
E:\conda\envs\smp-memory\python.exe -m benchmarks.run_validation50_hypergraph
```

它逐视频将已有观测转换为 SQLite 超图记忆，沿用当前超图检索、嵌入、最多 12 张原图的视觉答题器及目标帧评分规则。不会重新生成视觉观测，但嵌入和答题会调用真实模型；当前实验路径不调用 Jev。控制台显示视频、题目、正确数、请求数和输入输出合计 token，不设累计 token 硬上限。模型、方法代码、题目及观测指纹不一致时拒绝混用结果。同一结果目录不允许两个终端同时运行。

默认结果目录为 `runtime\experiments\hypergraph_validation50`，可用 `--run-dir` 更换。`memory\ep_N.sqlite3` 保存各视频记忆；`answers\行号.json` 保存逐题预测、证据、标准目标帧及评分；`api_budget.json` 与 `api_requests.jsonl` 记录本方法用量；`summary.json` 保存总体、逐视频、细分类指标；完成后还生成 `predictions.jsonl` 和中文 `REPORT.md`。中断时保留检查点和已完成题目的部分汇总；重新运行复用已完成题目与已缓存模型响应。共享视觉阶段的历史用量单独列出，避免每个方法重复计费统计。

本实验是 96 帧版本的离线目标帧评测，不等同于完整 Habitat 导航 SR/SPL。报告标注 `ep_1`、`ep_33` 已参与开发，不能将全部 50 段称为未见测试集。以下四个旧入口仍对应两视频、12 题。

前 50 段的官方标注中，140 道题含 `-1` 哨兵或空目标组，没有完整可选的有效目标帧。这些题保留在 2,901 题总体分母中，预测 `-1` 不计为正确；另外在 `summary.json` 的 `with_valid_goal_frames` 中报告其余 2,761 道题，避免把这一选帧限制与方法错误混为一谈。

## 使用同一份 50 段观测单独运行 Mem0

在项目根目录直接使用现有 Mem0 隔离环境（无需激活 conda）：

```powershell
E:\code\smp\runtime\mem0-env\Scripts\python.exe -m benchmarks.run_validation50_mem0 --check-only
E:\code\smp\runtime\mem0-env\Scripts\python.exe -m benchmarks.run_validation50_mem0
```

第一条只检查环境和数据；第二条调用真实模型。Mem0 SDK 固定为 2.2.1，复用与超图实验相同的 50 段封存观测、原图、2,901 道题、嵌入与答题模型、每题最多 12 张原图以及同一个评分器。官方 `add(infer=True)` 负责记忆提取和更新，`search(top_k=20)` 负责检索；不重新生成视频观测。若本机存在 `hypergraph_validation50/config.json`，开始前会检查模型、输入和共同答题评分代码是否一致。

结果独立保存在 `runtime\experiments\mem0_validation50`：`answers` 为逐题断点，`mem0` 为官方 SDK 的向量存储和历史数据库，`mem0_ingestion` 为逐批记忆写入断点，`mem0_facts` 为每个视频的记忆快照，`REPORT.md`、`summary.json`、`predictions.jsonl` 为最终结果。控制台实时显示记忆构建、答题、正确数、请求和 token；中断后使用第二条原命令续跑，不设累计 token 硬上限。

## 四条独立实验命令

以下命令可分别运行完整的“数据校验 → 抽帧 → 视觉观测 → 本方法记忆构建 → 检索 → 12 题统一答题 → 评分报告”。在 `E:\code\smp` 中逐条执行：

```powershell
E:\conda\envs\smp-memory\python.exe -m benchmarks.run_hypergraph
E:\conda\envs\smp-memory\python.exe -m benchmarks.run_mem0
E:\conda\envs\smp-memory\python.exe -m benchmarks.run_amem
E:\conda\envs\smp-memory\python.exe -m benchmarks.run_readagent
```

四个结果分别保存在 `runtime\experiments\hypergraph_single`、`mem0_single`、`amem_single`、`readagent_single`；每个目录都有自己的 `REPORT.md`、`summary.json`、`predictions.jsonl`、进度日志和 API 账目。重复执行**同一条命令**会从该方法的检查点继续。四条命令默认共用 `runtime\experiments\findingdory_shared_perception`：先校验并导入现有 `compare_four_methods` 实验的完整视觉观测；若没有兼容的完整来源，则由先运行的脚本提取视觉观测。其余三条复用同一份 192 帧，保证输入一致并避免重复付费。请顺序运行，不要同时启动四条命令。

使用 `--run-dir` 可指定本方法的新实验目录，使用 `--shared-perception-dir` 可指定一套新的共享视觉观测，使用 `--seed-perception-from` 可指定用于导入完整视觉观测的旧实验目录。模型、视觉提取配置、数据或凭据变更后应换新的共享目录；方法代码变更后应换新的方法实验目录。报告将共享视觉费用单独列出，汇总四个方法的总费用时只计算共享视觉阶段一次。记忆构建和答题仍会调用真实模型；不设累计 token 上限。

## 同一进程依次运行四方法

在 `E:\code\smp` 的 PowerShell 或 CMD 中运行：

```powershell
E:\conda\envs\smp-memory\python.exe -m benchmarks.run_comparison --run-dir runtime\experiments\compare_four_methods
```

脚本按顺序检查 `.env` 中的语言模型、视觉模型和嵌入模型配置，确认 Mem0、A-Mem 的隔离环境及官方源码版本，校验两段视频及 12 道题。缺少默认数据时先下载；已有视频经 SHA-256 校验后复用。随后抽取或复用本地 96 帧图片，**重新请求模型提取两段视频的视觉观测**，依次构建超图、Mem0、A-Mem、ReadAgent 的记忆，各方法检索后使用同一视觉答题器和严格目标帧评分器。Jev 不参与这组离线记忆检索对比。

控制台逐阶段显示视频与帧进度，以及方法、题数、正确数、请求数和累计 token。中断后原命令可继续；每个视觉批次、记忆阶段和答题结果都有检查点。成功的模型响应另有缓存，避免在恢复时重复请求。更换模型、凭据、代码、方法或数据时，应指定一个**新的** `--run-dir`，旧目录会拒绝混用。实验不设累计 token 上限，完整运行可能产生较多 API 费用。

进度汇总使用当前子进程上报的请求数与 token，再加上已完成阶段的账目；父进程不会持续读取正在写入的当前方法账本，以避免 Windows 文件占用错误。

网络连接被远端异常断开时，传输层最多自动重试一次；建立连接失败时最多重试两次。每次尝试分别计入请求账目。服务端是否已处理远端断开前的请求无法确定，因此可能重复计费。重试后仍失败就停在检查点，等待同一命令续跑。

Mem0 隔离环境若提示缺少 spaCy 或 fastembed，属于可选 NLP/关键词检索能力未启用的提示，不是本次中断的原因。解读 Mem0 对比结果时应注明当前安装配置；如要评测含这些可选能力的 Mem0，需另建隔离实验目录完整重跑，不能在同一批结果中途切换环境。

实验目录中的 `predictions.jsonl` 保存逐题预测、证据、检索详情和每题请求；`summary.json` 与 `REPORT.md` 保存正确率、候选目标帧覆盖率和各方法用量；`api_usage.json`、各阶段 `api_requests.jsonl` 保存请求账目。`progress.jsonl` 是过程日志。报告明确标注这些题已参与开发调试，不能当作论文未见测试结果。此入口是离线目标帧任务，非官方 Habitat 导航成功率实验。

新增方法时，在可导入的 Python 模块中继承 `benchmarks.comparison_methods.MemoryMethod`，实现 `build` 和 `retrieve`，用 `@register` 注册；`environment` 可填 `core`、`mem0`、`amem` 或一个 Python 可执行文件的绝对路径。使用 `--plugin your.module --methods hypergraph your_method` 调用。每次正式比较的方法和插件源码都冻结到配置中。

本地离线验证，不请求真实模型：

```powershell
E:\conda\envs\smp-memory\python.exe -m pytest tests\test_run_comparison.py -q
```
