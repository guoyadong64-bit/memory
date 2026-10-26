# MemoryBank：FindingDory 前 50 个验证视频

在 `E:\code\smp` 执行（会调用真实模型 API）：

```bat
E:\code\smp\runtime\memorybank-env\Scripts\python.exe -u -m benchmarks.run_validation50_memorybank
```

默认使用封存的 `findingdory_validation50_perception_v2`，完成全部 2901 题，
不会重新解析视频。沿用 `.env` 的 LLM、答题模型、嵌入模型；不调用 Jev。
每题最多 12 张原图，证据预算 40000 UTF-8 字节，评分和分母与已有实验一致。
控制台显示视频、记忆阶段、题目进度、正确数、请求数、累计输入输出 token。
不设累计 token 硬上限。中断后重复同一命令续跑；成功的模型响应有缓存。

仅检查配置、源码、依赖、视频、4800 帧及观测完整性，不请求模型：

```bat
E:\code\smp\runtime\memorybank-env\Scripts\python.exe -u -m benchmarks.run_validation50_memorybank --check-only
```

结果保存到 `runtime/experiments/memorybank_validation50/`：

- `answers/`：逐题答案、来源帧、评分、检索诊断和调用记录。
- `memorybank/<ep>/`：历史记录、摘要及嵌入检查点。
- `response_cache/`：已完成请求的缓存。
- `api_budget.json`、`api_requests.jsonl`：本方法请求与报告用量。
- `summary.json`、`REPORT.md`：汇总和分题型结果。
- `predictions.jsonl`：全部完成后的逐题汇总。
- `config.json`：输入、模型、官方源码及适配源码的冻结指纹。

## 官方实现与适配边界

官方仓库：<https://github.com/zhongwanjun/MemoryBank-SiliconFriend>

固定提交：`cf61c4196e4cfdb0f2b7a0316249fa40312dc3a9`。

适配器直接编译并执行该提交的 `summarize_content_prompt`、
`JsonMemoryLoader`、`similarity_search_with_score_by_vector` 和 `seperate_list`。
保留官方 `CHUNK_SIZE=200`，为与其他方法保持一致将检索 top-k 设为 20，
使用同一嵌入服务的归一化向量及真实 FAISS IndexFlatL2。
摘要保留官方 400 输出 token 上限和消息模板，温度统一为 0。

每段视频映射为一个独立历史日：原始帧作为历史条目，并生成日摘要。
日期是逻辑标记 `historical_day`，不伪造视频的真实拍摄日期。
采用官方默认的非随机遗忘检索路径；不从视觉观察推断用户人格。
不运行原版心理陪伴聊天器，最终使用共同视觉答题器。这是官方记忆模块的
FindingDory 适配，不能称为原始对话论文设置的完全复现。

官方邻域函数会就地拼接文档。本适配对每题复制文档，以免上题检索污染下题；
拼接同时保留确切来源文档 ID。摘要被召回本身不会开放整段视频 96 张图，
只有被召回的原始帧文档能产生可选原图。题目和答案不写回记忆。

适配器只执行命名的官方定义，避免导入旧版 LangChain/OpenAI/Gradio；
API 通信、缓存、FAISS 对象桥接、来源帧记录属于适配层。
源码版本或文件哈希变化、模型或封存输入变化会在请求前报错。

## 在新的检出中准备环境

当前电脑已准备项目内环境。以后在新机器上可使用：

```bat
git clone https://github.com/zhongwanjun/MemoryBank-SiliconFriend.git runtime\vendor\memorybank
git -C runtime\vendor\memorybank checkout cf61c4196e4cfdb0f2b7a0316249fa40312dc3a9
E:\conda\envs\smp-memory\python.exe -m venv runtime\memorybank-env
runtime\memorybank-env\Scripts\python.exe -m pip install -r requirements-memorybank.txt
```

官方仓库及运行环境放在 Git 忽略的 `runtime/` 中，不提交 `.env` 或实验数据。
也可通过 `--plugin benchmarks.memorybank_adapter` 注册到通用方法接口；
此方法需要上述 NumPy/FAISS 环境，当前推荐使用独立的 validation50 入口。
