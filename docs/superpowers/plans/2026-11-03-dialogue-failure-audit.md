# 冻结任务失败与回退贡献审计

依据已确认 MVP 设计的第 5–6 周失败分类与消融要求。输入为三份已保存的 `frozen_dialogue_v1` 逐条 JSON，不重新调用模型。

## 文件

- `recagent/eval/audit.py`：验证任务哈希和 ID 对齐，计算普通任务直接完成、最终成功、回退原因、延迟与 token 汇总。
- `scripts/audit_dialogue_runs.py`：命令行读取三份结果，输出机器可读 JSON。
- `tests/test_dialogue_audit.py`：合成配对结果测试协议不一致、ID 不一致、回退贡献和故障任务排除。
- `reports/dialogue-failure-audit-2026-11-03.md`：实测解读、限制与下一步实验。
- `README.md`：复现入口。

## 实施与验收

1. 测试先行：构造相同哈希的三组小样本，断言标准任务分母排除 `no_match` 和 `tool_fault`；直接完成需要成功且未回退；统计原因与 token；哈希或任务 ID 不一致时报错。运行单文件测试确认因模块缺失失败。
2. 实现纯函数 `audit_runs(runs: dict[str, dict]) -> dict`，只读取现有字段；写 CLI 并使用三份实际报告产生 `reports/dialogue-failure-audit-2026-11-03.json`。单文件测试转绿。
3. 报告说明这是 trace 级的回退贡献分析：直接完成率与最终成功率不可混称；三份运行未控制模型版本与运行时版本差异，不能推断提示词或模型的因果收益。明确后续真实消融方案。
4. 运行全套 `pytest`、Ruff、Mypy 和脚本复现，检查输出与原报告一致；提交、推送并核对远端提交。
