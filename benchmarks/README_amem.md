# A-Mem：FindingDory 验证集前 50 段视频

在 `E:\code\smp` 运行，调用 `.env` 中配置的真实模型 API：

```bat
E:\code\smp\runtime\amem-env\Scripts\python.exe -u -m benchmarks.run_validation50_amem
```

仅预检，不请求模型：

```bat
E:\code\smp\runtime\amem-env\Scripts\python.exe -u -m benchmarks.run_validation50_amem --check-only
```

默认结果目录为 `E:\code\smp\runtime\experiments\amem_validation50`，中断后用同一命令续跑。

- 使用已封存的 50 段视频、4,800 帧共享观测及 2,901 道问题，不重新抽帧或提取视觉观测。
- 沿用固定提交 `f303dfc71e07bdc787f4bc135d4cea328ae30e99` 的官方 `WujiangXu/A-mem-sys`。预检核对提交及实际源码，每段视频独立建库，每帧观测作为一条笔记。保留官方笔记分析、关联、演化与 `search_agentic`；API 传输、嵌入服务和持久化使用现有适配器。属于视频任务适配，非原论文对话任务的完全复现。
- 检索 Top-K=20，证据最多 40KB，统一视觉答题器每题最多 12 张候选原图，评分和其他 validation50 实验一致。题目、答案不写回记忆，只有原始笔记提供其来源帧。
- 控制台显示每 8 帧建库进度、逐题答题进度、正确数、请求数和累计输入输出 token。首次导入 Chroma/Transformer 依赖可能较慢；每条笔记可能需要多次请求，A-Mem 建库耗时通常较长。不设累计 token 硬上限。
- `amem/<episode>/notes.json` 和 Chroma 数据保存记忆断点，`answers/*.json` 保存逐题结果，响应缓存复用已成功落盘的请求。网络中断时服务端已处理但本地未收到的请求无法保证不重复计费。
- `summary.json`、`REPORT.md` 保存汇总和中文报告；完整运行后生成 `predictions.jsonl`。`api_budget.json`、`api_requests.jsonl` 保存用量，共享视觉前处理用量单列。失败时也生成已有结果的汇总。
- 改变模型、输入、依赖或冻结源码后，需指定新的 `--run-dir`；不要删除已有实验目录来续跑。脚本拒绝混用其他方法的逐题结果，同一结果目录不能同时启动两次。

## JSON 截断后的续跑

2026-10-24 增加了有界 JSON 恢复：普通请求仍沿用原 4096 token 输出上限；服务端标记输出截断时，仅该次失败的重试依次使用 8192、16384 上限。格式或原始官方 schema 校验失败时给予格式反馈，最多增加两次请求。只有完整且通过原 schema 校验的结果才写入记忆，不猜测或补造字段。已经落盘的原始失败响应留作审计，恢复请求使用不同缓存键，避免反复读取同一个坏响应。

原版 validation50 A-Mem 入口及适配器的源码哈希被明确列入兼容范围；确认其他模型、输入、官方代码与评分完全一致后，首次续跑会自动备份原配置至 `compatibility/`，记录修复前后指纹，再继续已有断点。其他配置变化仍然拒绝续跑。报告标注这项运行期修复，新增请求照常计入用量。原始失败请求已经产生的费用不会被撤销。

保持原命令即可继续，无须删除记忆、原图、观测、答案或响应缓存。

本地模拟测试（不会调用真实模型）：

```bat
E:\code\smp\runtime\amem-env\Scripts\python.exe -m pytest tests/test_amem_baseline.py tests/test_amem_recovery.py tests/test_run_validation50_hypergraph.py -q
```
