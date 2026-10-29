# Ollama 默认 Qwen2.5-1.5B 工具调用兼容验证

## 现象与原因

用户安装 `qwen2.5:1.5b` 后，单次命令返回 `mode=llm`、`fallback_used=true`，trace 的第一步为 `llm_validation: ModelDecisionError`。复现发现该模型没有填充 OpenAI 兼容响应中的 `tool_calls`，而是在正文输出严格的 JSON 工具调用对象，例如 `{"name":"recall_candidates","arguments":{"genres":["Sci-Fi"]}}`。运行时当时只解析结构化字段，因此没有执行模型请求的工具。

修复仅接受**完整 JSON 对象且字段恰为 `name`、`arguments`**的正文工具调用，并要求工具属于当前阶段已开放集合、参数为对象。转换后仍经过原有字段白名单、用户 ID、题材、候选 ID 和预算校验。额外字段或越权工具继续回退，不执行模型生成的任意文本。

## 真实复测

环境：Ollama 0.40.12；模型 `qwen2.5:1.5b`（Ollama ID `65ec06548149`）；MovieLens-1M 预处理目录；用户 6040；查询“想看科幻，不要恐怖”；`top_k=3`。

```powershell
python scripts/run_local_llm_agent.py --processed-dir data/processed/ml1m --user-id 6040 --query "想看科幻，不要恐怖" --top-k 3
```

修复后连续多次复测该请求，均得到 `status=ok`、`fallback_used=false` 和 `recall_candidates → rank_candidates` 的工具 trace。一次运行返回电影 ID `1527, 1356, 2699`；模型选择的 ID 可能变化，验证条件是它们都来自已排序候选。

进一步在同一 [30 条冻结任务](dialogue-ml1m-2026-10-29.md)上跑默认模型，完整逐条结果见[JSON](dialogue-ollama-default-ml1m.json)：最终结果经回退后约束与证据通过率均为 100%，但 20 条普通任务的**模型直接完成率为 0%**，30 条全部回退。19 条在召回后违反工具协议，6 条在召回前违反，另有 5 条为故障注入。单次命令成功不代表默认 1.5B 模型具备稳定的跨查询工具调用能力；本项目继续保留确定性回退，并在报告中区分模型完成率与最终任务成功率。
