# 冻结对话任务评测（MovieLens-1M）

## 协议

`configs/eval/dialogue_cases.json` 冻结 30 条任务，六类各 5 条：题材偏好、排除条件、有历史用户、冷启动、无匹配候选、排序工具故障注入。每条任务人工写定预期题材、排除项及特殊成功条件；评测器不从系统的偏好解析器生成金标准。候选与历史只来自训练期索引。

运行命令：

```powershell
python scripts/evaluate_dialogue_agent.py --mode deterministic --output reports/dialogue-deterministic-ml1m.json
$env:RECAGENT_LLM_MODEL = "recagent-qwen2.5"
python scripts/evaluate_dialogue_agent.py --mode llm --output reports/dialogue-llm-ml1m.json
```

本机 LLM 是 Ollama 0.40.12 中导入的 Qwen2.5-1.5B Q4_K_M GGUF，来源、校验和见[冒烟报告](local-llm-smoke-2026-10-29.md)。两次运行使用同一冻结任务文件和 MovieLens-1M 预处理目录。逐条结果、工具 trace、失败分类和任务文件 SHA-256 分别保存在[确定性结果](dialogue-deterministic-ml1m.json)和[本地模型结果](dialogue-llm-ml1m.json)。

## 结果

| 指标 | 确定性流程 | 本地 Qwen Agent |
| --- | ---: | ---: |
| 任务数 | 30 | 30 |
| 最终任务成功率（含回退） | 100% | 100% |
| 约束通过率 | 100% | 100% |
| 证据通过率 | 100% | 100% |
| 普通任务直接完成率（不含无匹配、故障注入） | 20/20，100% | 3/20，15% |
| 全任务回退率 | 5/30，16.7% | 27/30，90% |
| Agent 内部 P50 / P95 延迟 | 3 / 1037 ms | 1123 / 2370 ms |

确定性流程的 5 次回退均由故障注入触发。本地模型的 27 次回退中，5 次来自故障注入，17 次发生在完成召回之后，5 次发生在召回之前。模型直接完成的 3 条是 `preference_3`、`exclusion_3`、`history_3`。最终任务成功率高，是因为确定性回退接管了模型未完成的请求；不能把它写成“模型成功率 100%”。

## 失败案例与解释

- `preference_1`：模型完成召回但未完成后续工具协议，系统记录 `model_protocol_after_recall`，再用确定性流程推荐。
- `cold_start_1`：模型在召回前未给出有效协议动作，记录 `model_protocol_before_recall`，冷启动热度流程接管。
- `tool_fault_1`：排序工具被显式注入异常，触发回退；这是故障演练，不应计入模型能力失误。

评测脚本只保存错误类别与阶段，不保存模型原始回复，因此阶段分类不能进一步证明模型具体生成了什么。Agent 内部延迟不含 CLI 启动和目录加载时间；这 30 条任务没有人工偏好评分，也没有与固定候选 NDCG 共用候选协议。此报告证明的是约束与故障恢复能力，同时暴露出小模型的工具协议完成率不足。下一步应增加模型参数或更强模型对照，并在同一冻结任务集上复测。
