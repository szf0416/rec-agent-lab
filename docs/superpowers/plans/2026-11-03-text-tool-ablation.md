# 正文 JSON 工具调用兼容开关与受控消融

依据已确认 MVP 设计的第 5–6 周消融要求。四组实验固定同一代码提交、任务文件、数据目录、解码参数；因素为本地模型（`qwen2.5:1.5b`、`recagent-qwen2.5`）与正文 JSON 工具调用兼容（on/off）。

1. 在 `tests/test_llm_runtime.py` 先写红色测试：脚本化客户端只输出正文 JSON 工具调用，兼容 on 可执行，off 回退；默认仍为 on。实现 `LLMRecommendationAgent(..., accept_text_tool_calls=True)`，只在 on 时调用现有严格转换器。
2. 在 `tests/test_dialogue_eval_cli.py` 先写红色测试：CLI 的 `--text-tool-compat on|off` 被记录到输出 `run_config`，并正确传入 Agent。实现脚本参数；确定性模式不使用该选项。
3. 单元测试、全套测试、Ruff、Mypy 通过后提交代码，记下提交 SHA。基于同一提交运行四组完整 30 条冻结任务，输出到 `reports/ablation-*.json`。
4. 比较直接完成率、最终成功率、回退阶段、P50/P95 延迟与 token；报告运行次序、模型 ID、任务哈希、代码 SHA、限制。若服务不可达，保留可复现命令，不虚构结果。
5. 更新 README，合并并推送 GitHub，核对远端 SHA。
