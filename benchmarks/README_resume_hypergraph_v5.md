# V5 剩余错误题：无效 JSON 恢复入口

已有运行在 `ep_16 row=852` 因模型返回重复且未闭合的 `goal_frames` JSON 中断。
错误响应已进入响应缓存，直接使用旧命令可能反复读取同一错误响应。

在 `E:\code\smp` 执行新的续跑命令：

```powershell
E:\conda\envs\smp-memory\python.exe -u -m benchmarks.resume_hypergraph_remaining_v5
```

默认续跑原目录 `runtime\experiments\hypergraph_validation50_remaining_v5`。
**会调用真实模型 API。** 保留已完成题目，复用原始观测、图片、记忆与兼容嵌入缓存。
如果你此前使用了自定义路径，附加与原运行相同的 `--run-dir`、`--source-run`、
`--v4-run`、`--data-dir`、`--shared-perception-dir` 参数。

仅做预检（不调用模型，不修改原配置、断点或响应缓存）：

```powershell
E:\conda\envs\smp-memory\python.exe -u -m benchmarks.resume_hypergraph_remaining_v5 --check-only
```

## 恢复行为

- 不修改旧源文件、旧 `config.json`、已有逐题答案或原始错误响应。
- 原版 V5 的模型、数据、代码、历史结果指纹必须完全匹配，不放宽兼容检查。
- 正常 JSON 响应仍走原版答题器、任务约束校验及评分；不增加重试。
- 仅无法解析为 JSON 对象时追加格式请求，每轮最多两次。模型、证据、图片、温度、
  输出 token 上限均保持原请求设置，仅追加完整 JSON 格式指令。
- 不手工闭合 JSON、不截取重复列表、不根据参考答案生成或修正答案。
- 格式请求成功后仍需通过原版证据与任务校验，可解析不等于回答正确。
- 失败响应保留在原响应缓存及 `json_recovery/requests/` 审计目录。
- 新请求和 token 纳入原 `api_budget.json`、`api_requests.jsonl` 账目，不重置用量。
- 两次格式请求均失败时保留断点并停止；同一恢复命令再次启动会使用新的有界恢复轮次，
  不会永远循环读取同一组无效缓存。已成功的格式响应会在断点写入前中断时复用。
- 网络异常沿用原客户端规则，恢复层不把超时当作 JSON 格式问题重试。

`json_recovery/protocol.json` 单独封存新增策略及其代码指纹。
`summary.json` 和 `REPORT.md` 标注额外格式恢复策略；这份开发错误子集结果不能标作
纯原版 V5 的单版本完整正确率。原有答案保持原样，输出目录和续跑进度继续沿用。

本地模拟测试覆盖坏缓存恢复、所有请求有效载荷不变项、已成功请求缓存复用、有限重试、
下一轮恢复、网络失败、任务约束验证、写断点前崩溃恢复及已有答案字节不变。
验证不调用真实模型 API。
