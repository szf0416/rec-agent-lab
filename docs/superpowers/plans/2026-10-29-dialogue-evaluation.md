# Dialogue Evaluation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task by task. Each task follows test → red → implementation → green → commit.

**Goal:** 冻结至少 30 条对话推荐任务，离线评测确定性流程和真实本地 Agent 的约束、证据、成功率、回退与延迟，并留下逐条失败记录。

**Architecture:** `configs/eval/dialogue_cases.json` 只保存人工定义的任务与预期；`recagent/eval/dialogue.py` 负责加载、独立检查和汇总；`scripts/evaluate_dialogue_agent.py` 选择确定性或本地模型模式并输出 JSON。模型原始回复与密钥不入报告。

**Tech Stack:** Python 3.10 标准库、现有推荐引擎、pytest、Ollama（仅本地真实评测）。

## Global Constraints

- 保持现有训练期数据边界，不读验证或测试交互作为推荐依据。
- 每条任务有冻结的预期题材、排除项和成功条件；评测器不得调用 `parse_preferences` 生成金标准。
- 真实模型结果不作为离线 CI 通过条件；CI 使用小型合成目录。
- 单条错误需记录类型、工具 trace 和回退，不存模型原始消息或密钥。
- 真实报告须分清“任务成功率”和采样候选 NDCG，不能互相替代。

### Task 1: 冻结任务集与校验

**Files:** `configs/eval/dialogue_cases.json`, `recagent/eval/dialogue.py`, `tests/test_dialogue_eval.py`。

- [ ] 写失败测试：验证任务数量至少 30、ID 唯一、分类和预期字段合法，并拒绝错误数据。
- [ ] 运行目标测试确认失败；实现严格加载与任务结构。
- [ ] 运行目标测试、Ruff、Mypy 并提交。

### Task 2: 单条评分与汇总

**Files:** `recagent/eval/dialogue.py`, `tests/test_dialogue_eval.py`。

- [ ] 写失败测试：题材遗漏、已看电影、虚构元数据、无匹配、故障注入分别得到正确结果；P50/P95 和分组指标可复现。
- [ ] 运行目标测试确认失败；实现逐条评分、失败分类、汇总与逐条 trace 输出。
- [ ] 运行完整测试、Ruff、Mypy 并提交。

### Task 3: CLI 与真实模型报告

**Files:** `scripts/evaluate_dialogue_agent.py`, `tests/test_dialogue_eval_cli.py`, `reports/dialogue-ml1m-2026-10-29.md`, `README.md`。

- [ ] 写失败测试：确定性 CLI 在小目录上输出逐条结果和汇总；无服务时本地模式清楚标注回退。
- [ ] 实现 CLI 的 `--mode deterministic|llm`、`--cases`、`--processed-dir`、`--output`；默认模型配置沿用现有环境变量。
- [ ] 离线全套验证；用真实 MovieLens-1M 与本机模型各跑一轮，报告版本、成功率、回退、失败案例、延迟和适用边界。
- [ ] 更新 README/简历事实，提交并合入 main。
