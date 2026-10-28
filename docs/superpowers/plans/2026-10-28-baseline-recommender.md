# Baseline Recommender Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 ML-1M 预处理产物上提供只依赖训练期数据的 Popularity + ItemCF 电影推荐与可运行命令行演示。

**Architecture:** `store.py` 只加载和校验预处理产物；`baseline.py` 建训练期索引并计算候选分数；`scripts/recommend_baseline.py` 只解析参数和展示结果。后续 Agent 工具直接调用同一个 `BaselineRecommender.recommend`。

**Tech Stack:** Python 3.11，标准库，pytest，现有 MovieLens-1M 预处理文件。

## Global Constraints

- 只使用 `interactions_train.csv` 构建推荐索引，绝不读取验证或测试交互作为输入。
- 物品 ID 以 `items.json` 为准；输出只包含训练期出现过的物品。
- 随机性为零；同分时按 `item_id` 升序，保证可复现。
- 默认排除用户已看物品；`top_k` 必须在 1 到 50 之间。
- 未知用户使用热度与题材约束回退；未知题材返回空列表。
- 无 API 密钥、GPU、网络请求和新运行时依赖。

---

## File map

- `recagent/engine/store.py`：读取 `items.json` 与 `interactions_train.csv`，生成训练期 `Catalog`。
- `recagent/engine/baseline.py`：Popularity、ItemCF、题材过滤、去重、排序与已看过滤。
- `scripts/recommend_baseline.py`：命令行演示，输出 JSON。
- `tests/test_baseline.py`：小型合成数据下的行为与防泄漏测试。
- `README.md`：增加准确的基线演示命令与当前范围。

### Task 1: 训练期数据存储

**Files:** Create `recagent/engine/__init__.py`, `recagent/engine/store.py`, `tests/test_baseline.py`.

**Interfaces:** `Catalog.load(processed_dir: Path) -> Catalog`；字段 `movies: dict[int, MovieInfo]`、`history: dict[int, tuple[int, ...]]`、`item_users: dict[int, frozenset[int]]`、`popularity: dict[int, int]`。`MovieInfo(item_id: int, title: str, genres: tuple[str, ...])`。

- [ ] **Step 1: Write failing tests**：用 `tmp_path` 写入 `items.json` 和 `interactions_train.csv`，再额外写入 `interactions_test.csv` 含仅测试期物品；断言 `Catalog.load` 的 `history`、`popularity` 不含测试期物品，并断言缺少必需文件时报 `FileNotFoundError`。
- [ ] **Step 2: Verify red**：运行 `python -m pytest tests/test_baseline.py -q`，预期因 `recagent.engine.store` 尚不存在而失败。
- [ ] **Step 3: Implement**：`MovieInfo` 为 frozen dataclass；`Catalog.load` 用 `json.loads` 与 `csv.DictReader` 解析文件，按 `(timestamp, item_id)` 对每个用户历史稳定排序；遇到训练交互引用不存在的电影 ID 时抛 `ValueError`。`item_users` 用集合构建后冻结，`popularity` 按训练交互次数累计。
- [ ] **Step 4: Verify green**：运行 `python -m pytest tests/test_baseline.py -q`，预期全部通过；运行 `ruff check recagent/engine tests/test_baseline.py` 和 `mypy recagent/engine`。
- [ ] **Step 5: Commit**：`git add recagent/engine tests/test_baseline.py && git commit -m "feat: load train-only recommendation catalog"`。

### Task 2: Popularity 与 ItemCF 推荐

**Files:** Create `recagent/engine/baseline.py`; extend `tests/test_baseline.py`.

**Interfaces:** `BaselineRecommender(catalog: Catalog)`；`recommend(user_id: int | None, top_k: int = 10, genres: tuple[str, ...] = (), exclude_items: frozenset[int] = frozenset()) -> list[Recommendation]`；`Recommendation(item_id: int, title: str, genres: tuple[str, ...], score: float, source: str)`。

- [ ] **Step 1: Write failing tests**：小型数据中，断言已看物品被排除；共同观看者较多的候选优先；未知用户按热度回退；题材过滤和显式排除生效；同分排序稳定；`top_k=0` 与 `top_k=51` 抛 `ValueError`。
- [ ] **Step 2: Verify red**：运行 `python -m pytest tests/test_baseline.py -q`，预期 `BaselineRecommender` 缺失相关用例失败。
- [ ] **Step 3: Implement**：ItemCF 分数为用户历史每部电影与候选电影的共同观看用户数，除以两电影观看用户数乘积的平方根后求和；若用户无历史则用 `log1p(popularity)`；有历史时 `score = itemcf_score + 0.01 * log1p(popularity)`。只评分训练期出现的物品；先做排除与题材过滤，再按 `(-score, item_id)` 取前 `top_k`。`source` 为 `itemcf` 或 `popularity`。
- [ ] **Step 4: Verify green**：运行 `python -m pytest tests/test_baseline.py -q`、`ruff check recagent/engine tests/test_baseline.py`、`mypy recagent/engine`，预期通过。
- [ ] **Step 5: Commit**：`git add recagent/engine tests/test_baseline.py && git commit -m "feat: add deterministic popularity and itemcf baseline"`。

### Task 3: 命令行与使用文档

**Files:** Create `scripts/recommend_baseline.py`; modify `README.md`; extend `tests/test_baseline.py`.

**Interfaces:** `python scripts/recommend_baseline.py --processed-dir data/processed/ml1m --user-id 1 --genres Comedy --top-k 5` 输出包含 `items` 数组的 JSON，退出码为 0。

- [ ] **Step 1: Write failing test**：用 `subprocess.run` 调脚本并传入合成数据目录，断言退出码 0、输出可解析 JSON、返回条数不超 `top_k`、每项 ID 来自训练期。
- [ ] **Step 2: Verify red**：运行 `python -m pytest tests/test_baseline.py -q`，预期因脚本缺失失败。
- [ ] **Step 3: Implement**：脚本用 `argparse` 解析 `--processed-dir`、`--user-id`、可重复的 `--genres`、`--top-k`，加载 `Catalog` 后调用推荐器，使用 `json.dumps(..., ensure_ascii=False, indent=2)` 输出。README 加数据准备及两条命令行示例，并说明此阶段是确定性基线。
- [ ] **Step 4: Verify green**：运行 `python -m pytest tests/test_baseline.py -q`、`python -m pytest -q`、`ruff check .`、`ruff format --check .`、`mypy recagent`；再用实际 ML-1M 产物运行示例命令，检查输出。
- [ ] **Step 5: Commit**：`git add scripts/recommend_baseline.py README.md tests/test_baseline.py && git commit -m "feat: expose baseline recommendation demo"`。

## Review gates

每个任务完成后核对训练期隔离、候选来源、排序确定性及测试结果。Task 3 完成后，对照设计文档核对：本计划只交付第一段基线，不声称 Agent、API 或对话评测完成。随后为 Agent 工具与评测写下一份独立计划。
