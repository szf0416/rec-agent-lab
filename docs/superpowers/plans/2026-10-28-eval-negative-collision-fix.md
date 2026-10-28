# Evaluation Negative Collision Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 固定评测负样本不再包含该用户在训练、验证或测试中交互过的物品。

**Architecture:** 预处理入口汇总每个测试用户的全切分交互物品，仅将其用作评测负标签的排除集合。`NegativeSampler.eval_negatives` 按用户及排除集合缓存，确保同配置重复调用一致且改变排除集合不会复用错误缓存。

**Tech Stack:** Python 3.11，NumPy，pytest，现有数据管线。

## Global Constraints

- 全切分交互只用于定义评测负标签，不进入推荐模型的训练或推理特征。
- 候选池仍仅来自训练期可见物品。
- 同随机种子、同用户顺序、同排除集合必须生成同样的负样本。
- 继续支持现有不传排除集合的调用方式。

---

### Task 1: 采样器排除与缓存

**Files:** Modify `recagent/data/negative.py`; test `tests/test_negative.py`.

**Interfaces:** `eval_negatives(users: Iterable[int], *, num_neg: int = 100, exclude_by_user: Mapping[int, set[int]] | None = None) -> dict[int, tuple[int, ...]]`。

- [x] **Step 1: Write failing test**：候选池 `[1,2,3,4,5]`，用户 1 排除 `{1,2}`、`num_neg=3`，断言结果恰为 `{3,4,5}`；同参数重复调用不变；改成排除 `{3,4}` 时返回 `{1,2,5}`。
- [x] **Step 2: Verify red**：`python -m pytest tests/test_negative.py -q`，预期新测试因 `exclude_by_user` 参数不存在失败。
- [x] **Step 3: Implement**：将缓存键改为 `(num_neg, user_id, frozenset(excluded_items))`；对每个用户通过 `_draw(num_neg, exclude=set(...))` 采样，返回 dict。`reset` 仍清空缓存。
- [x] **Step 4: Verify green**：运行同一测试文件、Ruff 与 Mypy，预期通过。
- [x] **Step 5: Commit**：提交采样器与测试，信息 `fix: exclude known positives from eval negatives`。

### Task 2: 预处理接线与真实数据审计

**Files:** Modify `scripts/preprocess.py`; test `tests/test_negative.py` 或新增预处理入口测试。

**Interfaces:** 预处理在调用 `eval_negatives` 时传入三段交互合并得到的 `dict[int, set[int]]`，文件格式保持 `{user_id: [item_id...]}`。

- [x] **Step 1: Write failing test**：用一组训练、验证、测试交互构造排除映射，断言各用户集合包含三段交互的物品。
- [x] **Step 2: Verify red**：运行目标测试，预期辅助函数缺失失败。
- [x] **Step 3: Implement**：新增 `build_eval_exclusions(split: DatasetSplit) -> dict[int, set[int]]`，遍历三段交互构造映射；调用采样器时传 `exclude_by_user=...`。生成前打印每个用户可用候选不足时的已有 `ParseError`。
- [x] **Step 4: Verify green**：运行完整 `pytest -q`、`ruff check .`、`ruff format --check .`、`mypy recagent`；用现有本地 ML-1M 重新运行预处理；审计新 `eval_negatives_test.json` 与三段交互的重叠计数，预期 0；确认同 seed 重跑文件哈希一致。
- [x] **Step 5: Commit**：提交预处理和测试，信息 `fix: build collision-free evaluation set`。数据产物仍由 `.gitignore` 排除。
