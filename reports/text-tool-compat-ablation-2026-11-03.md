# 正文 JSON 工具调用兼容：2×2 本机消融（2026-11-03）

## 协议与复现

因素为 Ollama 模型与正文 JSON 工具调用兼容开关。四组均从代码提交 `60b51ecc2bce3e8a4cee756981263ce391eff5e9` 运行，使用同一 `configs/eval/dialogue_cases.json`、MovieLens-1M 预处理目录、`temperature=0`、`max_tokens=512`、最多 4 次工具调用、6 轮对话和 60 秒预算。每组按相同顺序运行 30 条任务一次。运行顺序：默认模型 off → 默认模型 on → 导入模型 off → 导入模型 on。Ollama 版本 0.40.13。

| 固定输入 | SHA-256 |
| --- | --- |
| 冻结任务 | `96815cb2d1476ae7fa3fcc0f560c61e8ff641b685e4f42bf9fcaf68f74758531` |
| `items.json` | `2698b765dca3ca20c034bf0341198cdea7c369fe566b5bcd9d20e568b77279ac` |
| `interactions_train.csv` | `d2792d50139f13f8e73a2db3eacd1e0b001c1c8f6cd83fdd56929a21ea74a192` |

模型 ID：`qwen2.5:1.5b` 为 `65ec06548149`，`recagent-qwen2.5:latest` 为 `a4eb4fb15c92`。`recagent-qwen2.5` 使用本地导入 GGUF。关闭开关只禁止把**正文中的完整 JSON 工具调用对象**转换成工具调用；结构化 `tool_calls`、全部参数校验和回退逻辑保持相同。

```powershell
$env:RECAGENT_LLM_MODEL = "qwen2.5:1.5b"
python scripts/evaluate_dialogue_agent.py --mode llm --text-tool-compat off --output reports/ablation-default-off.json
python scripts/evaluate_dialogue_agent.py --mode llm --text-tool-compat on  --output reports/ablation-default-on.json
$env:RECAGENT_LLM_MODEL = "recagent-qwen2.5"
python scripts/evaluate_dialogue_agent.py --mode llm --text-tool-compat off --output reports/ablation-imported-off.json
python scripts/evaluate_dialogue_agent.py --mode llm --text-tool-compat on  --output reports/ablation-imported-on.json
```

四份 JSON 保存逐条 trace、失败类别、任务哈希、模型及兼容开关。默认运行行为仍为 `on`。

## 实测结果

普通任务为偏好、排除、历史与冷启动，共 20 条；无匹配和故障注入各 5 条不计入模型直接完成率。下表中的最终成功包含确定性回退。

| 模型 / 兼容开关 | 普通任务直接完成 | 最终成功 | 全任务回退 | 召回前协议失败 | 召回后协议失败 | P50 / P95 延迟 | 总 token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 默认 Qwen / off | 0/20 | 30/30 | 30/30 | 15 | 10 | 516 / 3951 ms | 37,808 |
| 默认 Qwen / on | 0/20 | 30/30 | 30/30 | 5 | 20 | 1234 / 2373 ms | 67,705 |
| 导入 Qwen / off | 3/20 | 30/30 | 27/30 | 5 | 17 | 1390 / 3292 ms | 70,400 |
| 导入 Qwen / on | 3/20 | 30/30 | 27/30 | 5 | 17 | 1228 / 2807 ms | 70,401 |

每组另有 5 次排序故障注入回退。四组均无 `model_unavailable`。导入模型直接完成的任务在开、关两组相同：`preference_3`、`exclusion_3`、`history_3`。

## 解释与限制

兼容开关使默认模型的失败阶段从“召回前 15 / 召回后 10”移到“召回前 5 / 召回后 20”，说明严格转换确实让更多正文 JSON 调用通过第一步工具协议；但它没有让模型在本次任务中完成最终推荐。导入模型的直接完成率与失败分类在两组相同。不能将“走到召回”写成“成功推荐”，也不能把最终 30/30 写成模型能力。

延迟包含模型调用和回退流程，不含 CLI 启动与目录加载；token 由 Ollama 返回，包含失败后触发回退的请求。每格只运行一次且次序未随机化，模型生成与缓存、预热可能影响结果；P50/P95 和 token 差异仅是这轮观测值，不作统计显著性或稳定性能结论。任务集仅覆盖 MovieLens 元数据可验证偏好，没有人工满意度与在线效果。

下一步应先分析召回后的具体协议失败类别，并在不记录原始用户文本或密钥的前提下增加安全的结构化错误计数；随后对候选裁剪、最终 JSON 格式约束做单因素实验，重复运行报告方差。
