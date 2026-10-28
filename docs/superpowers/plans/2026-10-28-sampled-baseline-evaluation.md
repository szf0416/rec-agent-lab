# Sampled Baseline Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对 Popularity 与 ItemCF 在相同的“1 个测试正样本 + 100 个固定负样本”题集上报告 Recall@10、NDCG@10 与覆盖率。

**Architecture:** 推荐器公开候选打分接口；评测模块加载冻结题集并选取每用户首个符合条件的测试正样本；脚本输出机器可读 JSON。报告同时显示纳入和跳过用户数，绝不把采样排序指标称作全库召回指标。

**Tech Stack:** Python 3.11，标准库，pytest，现有 ML-1M 处理产物。

## Global Constraints

- 所有模型使用完全相同的用户、正样本和负样本；只评价训练期已存在且该用户训练期未看过的正样本。
- 每用户按 `(timestamp, item_id)` 排序选首个符合条件的测试正样本；同分时按物品 ID 升序。
- 固定候选中发现正负重叠或训练期已看物品时直接抛错；不得静默清洗。
- 结果必须记录用户数、跳过原因、候选数量、seed 42 和数据协议 `sampled_1pos_100neg`。
- 运行不需要 GPU、网络或付费 API。

---

### Task 1: 候选打分接口

**Files:** Modify `recagent/engine/baseline.py`; test `tests/test_baseline.py`.

**Interfaces:** `score_candidates(user_id: int | None, candidate_ids: tuple[int, ...], mode: Literal["popularity", "itemcf"]) -> list[Recommendation]`，返回按分数降序的候选。

- [ ] **Step 1: Write failing tests**：合成目录中断言两个模式对同一候选都返回完整且相同的 ID 集；`itemcf` 在有历史用户上排序可不同于 `popularity`；非法模式或非训练期候选抛 `ValueError`。
- [ ] **Step 2: Verify red**：`python -m pytest tests/test_baseline.py -q`，预期接口不存在导致失败。
- [ ] **Step 3: Implement**：抽出私有 `_score(user_id, item_id, mode)`；Popularity 为 `log1p(popularity)`，ItemCF 为用户历史电影与候选的余弦相似度求和加 `0.01 * log1p(popularity)`；未知用户 ItemCF 回退热度。`recommend` 复用接口并保持原有过滤行为。
- [ ] **Step 4: Verify green**：运行目标测试、`ruff check .`、`ruff format --check .`、`mypy recagent`。
- [ ] **Step 5: Commit**：`feat: expose candidate scoring for offline evaluation`。

### Task 2: 固定题集评测和报告

**Files:** Create `recagent/eval/__init__.py`, `recagent/eval/sampled.py`, `scripts/evaluate_baseline.py`, `tests/test_sampled_eval.py`; modify `README.md`; create `reports/baseline-sampled-ml1m.md`.

**Interfaces:** `evaluate_sampled(processed_dir: Path, k: int = 10, expected_negatives: int = 100) -> dict[str, object]` 返回两个模式的指标和样本计数；CLI `python scripts/evaluate_baseline.py --processed-dir data/processed/ml1m --k 10` 输出 JSON。`expected_negatives` 供小型合成测试设置，公开数据脚本固定为 100。

- [ ] **Step 1: Write failing tests**：合成题集覆盖首个可用测试正样本、正负碰撞报错、冷启动用户、NDCG 排位公式与覆盖率去重；CLI 返回可解析 JSON。
- [ ] **Step 2: Verify red**：`python -m pytest tests/test_sampled_eval.py -q`，预期模块不存在导致失败。
- [ ] **Step 3: Implement**：使用 `csv.DictReader`、`json.loads` 加载测试交互和固定负样本；每用户选首个训练期存在且训练未看过的正样本；校验 100 个不重复负样本、每个负样本训练期存在且与用户训练历史及正样本不重叠；对两种模式排序，累计 hit@k、`1/log2(rank+1)`、推荐 ID 集合；报告跳过原因和覆盖率分母。
- [ ] **Step 4: Verify green**：运行完整测试、静态检查、真实 ML-1M CLI；将真实 JSON 指标与复现命令写入报告，说明这是采样排序而非全库检索。
- [ ] **Step 5: Commit**：`feat: evaluate recommendation baselines on frozen candidates`。
