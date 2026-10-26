# MemoryOS：FindingDory 验证集前 50 段视频

在 `E:\code\smp` 运行（会调用 `.env` 中配置的真实模型 API）：

```bat
E:\code\smp\runtime\memoryos-env\Scripts\python.exe -u -m benchmarks.run_validation50_memoryos
```

只检查环境、配置、官方源码和本地输入，不调用模型：

```bat
E:\code\smp\runtime\memoryos-env\Scripts\python.exe -u -m benchmarks.run_validation50_memoryos --check-only
```

结果默认写入 `E:\code\smp\runtime\experiments\memoryos_validation50`。中断后使用同一命令继续；不要删除该目录。模型、输入或代码发生变化时会拒绝混用旧实验配置，需要指定新的 `--run-dir`。

## 输入、流程和输出

- 复用 `runtime/experiments/findingdory_validation50_perception_v2` 中封存的 4800 帧视觉观测，以及对应原图；不重新抽帧或生成视觉观测。
- 每段视频独立建库：每帧共享观测映射为一条历史交互，交给 MemoryOS 官方更新器。
- 使用官方短期、中期、长期记忆，默认短期容量 10、中期容量 2000、长期知识容量 100、检索队列容量 7、热度阈值 5.0、主题匹配阈值 0.6。
- 使用与已有实验相同的 `.env` 模型、嵌入服务、统一视觉答题器、最多 12 张候选原图和评分函数。检索证据最多 20 条、40KB。
- 逐题显示建库阶段、完成数、答对数、请求数和累计 token；不设累计 token 硬上限。
- `answers/*.json` 保存逐题断点；`predictions.jsonl`、`summary.json`、`REPORT.md` 保存预测、汇总和中文报告；`api_budget.json`、`api_requests.jsonl` 记录用量；`memoryos/<episode>/state.json` 保存完整三层记忆状态；`response_cache` 保存模型响应用于续跑。
- 只有请求成功且已落盘的响应才能复用。网络中断时，服务端已处理但本地未收到响应的请求，无法保证不重复计费。

## 官方代码和适配边界

官方仓库：<https://github.com/BAI-LAB/MemoryOS>，固定提交 `587ed7755c7aed179965792830ff1b5ad9a6fa92`，使用其中 `memoryos-pypi` 实现。入口执行前检查提交和源码哈希，官方文件不做修改。

适配器执行官方更新、摘要提示、连续性判断、热度及分层检索函数；替换 SDK 传输和本地嵌入下载，使其使用当前统一 API。入库时钟采用固定起点加帧序号的逻辑时钟，避免中断等待改变热度；这不是视频真实日期。整库原子检查点替代分散文件写入，并记录原始帧来源。

每道题从相同建库快照开始检索。官方检索产生的访问计数变化仅作用于当前题，测试问题和模型答案不写回记忆。派生摘要、知识和画像可提供文本上下文，但不自动获得整段视频所有原图的访问权。只有实际召回的原始页面和短期记录提供对应帧。

因此这是 **MemoryOS 官方实现对视频记忆问答的适配实验**，不是原论文对话任务的完全复现。报告结果需与相同输入和答题配置的方法比较；已用于开发的题目不能宣称为未见测试集。

## 新机器安装

当前电脑已安装隔离环境 `runtime/memoryos-env`。新机器需要先准备主环境依赖及 Git，再克隆官方源码并安装本项目依赖：

```bat
git clone https://github.com/BAI-LAB/MemoryOS.git runtime/vendor/memoryos
git -C runtime/vendor/memoryos checkout 587ed7755c7aed179965792830ff1b5ad9a6fa92
E:\conda\envs\smp-memory\python.exe -m venv --system-site-packages --without-pip runtime/memoryos-env
E:\conda\envs\smp-memory\python.exe -m pip --python runtime/memoryos-env/Scripts/python.exe install -r requirements-memoryos.txt
```

验证测试使用 HTTP 模拟响应，不请求真实 API：

```bat
runtime\memoryos-env\Scripts\python.exe -m pytest tests/test_memoryos_adapter.py tests/test_run_validation50_hypergraph.py -q
```
