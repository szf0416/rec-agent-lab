# 本地 Qwen 工具调用冒烟验证（2026-10-29）

## 环境与模型

- Windows，NVIDIA RTX 3060 Laptop GPU（6 GB 显存）；本次未测 GPU 利用率。
- Ollama `0.40.12`；从 [Qwen 官方 GGUF 仓库](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/main)下载 `qwen2.5-1.5b-instruct-q4_k_m.gguf`，大小 1,117,320,736 字节，SHA-256 与仓库 LFS oid `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` 一致。
- 通过 `ollama create recagent-qwen2.5` 导入，`ollama show` 报告 Q4_K_M 和 `tools` 能力。
- MovieLens-1M 已预处理目录 `data/processed/ml1m`；用户 ID 6040，查询“想看科幻，不要恐怖”，最终取 3 部。

## 运行方法

```powershell
$env:RECAGENT_LLM_MODEL = "recagent-qwen2.5"
python scripts/run_local_llm_agent.py --processed-dir data/processed/ml1m --user-id 6040 --query "想看科幻，不要恐怖" --top-k 3
```

## 观察结果

最终实现先约束模型完成召回、排序，再允许它从已排序 ID 中选择结果。手动连续运行 4 次均为 `mode=llm`、`status=ok`、`fallback_used=false`，工具 trace 均为 `recall_candidates → rank_candidates`，输出 ID 均为 `1270, 2571, 1580`。Agent 内部报告的延迟依次约为 2783、2602、2517、2548 ms；该字段不含 CLI 启动和目录加载时间，也不是吞吐或 P99 指标。

初次直接开放搜索与排序工具时，1.5B 模型会跳过召回或在召回后直接回答，触发安全回退。阶段约束和一次纠偏提示解决了这条查询的协议执行问题。4 次同一查询的冒烟验证不能代表不同用户与复杂表达的成功率，也不能证明推荐质量优于基线；后续需构建多查询对话评测集。
