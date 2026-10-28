# Agent Tools and Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 让现有推荐基线可通过受约束的只读工具调用，并提供无密钥的确定性对话流程与逐步 trace。

**Architecture:** `tools.py` 封装训练期目录、历史、双路召回和候选排序；`policy.py` 将有限的中英文题材表达解析为结构化偏好；`runtime.py` 调度工具、记录 trace、执行证据检查与回退；CLI 只负责输入和 JSON 输出。此阶段明确标注为 deterministic mode，真实 LLM 适配器另行实现。

**Tech Stack:** Python 3.11，现有 NumPy/SciPy/Pydantic 依赖，pytest，MovieLens-1M 预处理产物。

## Global Constraints

- 所有工具只读，使用 `Catalog` 的训练期数据；不读取验证或测试交互。
- `search_movies` 只查电影标题与题材；不编造剧情、情绪或演员信息。
- 任何工具输入都有明确上限：`top_k` 1–20、候选 ID 最多 50、历史最多 20、总调用最多 4。
- 结果电影 ID 必须来自工具候选；理由只引用电影标题、题材、来源和用户训练期历史。
- 无 API 密钥、GPU、网络请求；固定输入与数据产生固定推荐顺序。
- 工具错误记录在 trace，回退到同一训练期推荐引擎；空匹配是正常结果。
- 不把确定性策略称为 LLM Agent。

---

### Task 1: 受约束的推荐工具

**Files:** Create `recagent/agent/__init__.py`, `recagent/agent/tools.py`, `tests/test_agent_tools.py`.

**Interfaces:** `RecommendationTools(catalog: Catalog)` 提供 `search_movies(query: str, limit: int = 10) -> list[MovieInfo]`、`get_user_history(user_id: int, limit: int = 10) -> list[MovieInfo]`、`recall_candidates(user_id: int | None, genres: tuple[str, ...], exclude_genres: tuple[str, ...], limit: int = 20) -> list[Recommendation]`、`rank_candidates(user_id: int | None, candidate_ids: tuple[int, ...], allowed_ids: frozenset[int], top_k: int) -> list[Recommendation]`。`allowed_ids` 由运行时从召回结果生成，不能由外部请求指定。

- [x] **Step 1: Write failing tests**：合成数据断言搜索只返回训练期电影、历史按时间只返回训练期记录、召回合并 Popularity 与 ItemCF 后去重并过滤已看/排除题材、排序拒绝非召回候选和超过 50 个 ID。
- [x] **Step 2: Verify red**：运行 `python -m pytest tests/test_agent_tools.py -q`，预期模块不存在失败。
- [x] **Step 3: Implement**：每个方法入口先验证长度与数量；召回分别按 Popularity 和 ItemCF 排序，取各 `limit` 个、合并去重，用两路名次的倒数和融合（RRF，常数 60）；排序仅接收 `allowed_ids` 中的 ID，`score_candidates_batch` 负责最后排序。
- [x] **Step 4: Verify green**：运行目标测试、Ruff、Mypy；确认训练期以外的电影从不出现在工具输出。
- [x] **Step 5: Commit**：`feat: add bounded read-only recommendation tools`。

### Task 2: 确定性策略、trace 与回退

**Files:** Create `recagent/agent/policy.py`, `recagent/agent/runtime.py`, `scripts/run_agent_demo.py`, `tests/test_agent_runtime.py`; modify `README.md`.

**Interfaces:** `parse_preferences(query: str) -> Preferences`；`DeterministicAgent(tools: RecommendationTools, max_steps: int = 4).run(query: str, user_id: int | None, top_k: int = 5) -> AgentResponse`。`AgentResponse` 含 `mode`, `items`, `trace_id`, `tool_trace`, `fallback_used`, `status`, `latency_ms`。

- [x] **Step 1: Write failing tests**：验证“想看科幻，不要恐怖”解析为包含 `Sci-Fi`、排除 `Horror`；工具 trace 含名称、耗时、状态、数量；每项输出在召回候选中且理由只引用题材/来源；空匹配返回 `no_match`；工具抛错时标记回退；`max_steps` 与 `top_k` 越界拒绝；CLI 输出可解析 JSON。
- [x] **Step 2: Verify red**：运行 `python -m pytest tests/test_agent_runtime.py -q`，预期缺少策略/运行时失败。
- [x] **Step 3: Implement**：有限词典解析明确题材，不理解的约束原样放入 `unparsed`；依次调用历史、召回、排序，单步计时并记录异常；超过调用上限时终止并回退；电影理由由标题、题材和召回来源模板生成，输出前检查 ID 与证据。
- [x] **Step 4: Verify green**：运行完整 `pytest -q`、`ruff check .`、`ruff format --check .`、`mypy recagent`；在真实 ML-1M 运行 `python scripts/run_agent_demo.py --processed-dir data/processed/ml1m --user-id 6040 --query "想看科幻，不要恐怖" --top-k 3`，确认 trace 与结果。
- [x] **Step 5: Commit**：`feat: add deterministic recommendation agent demo`。

## Next plan

本计划不宣称完成 LLM 工具调用。下一份计划将实现模型供应商适配器、结构化工具调用、超时与成本记录，并用同一题集比较 deterministic 与 LLM 模式。
