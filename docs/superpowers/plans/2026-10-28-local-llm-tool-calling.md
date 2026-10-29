# Local LLM Tool Calling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 在本地 OpenAI 兼容 Chat Completions 服务上运行真实模型工具调用，并在服务缺失、非法工具参数或无依据结果时安全回退。

**Architecture:** `chat_client.py` 仅负责 HTTP 与响应协议解析；`llm_runtime.py` 维护消息、工具预算和工具执行状态；现有 `RecommendationTools` 负责只读操作，`DeterministicAgent` 负责回退。CLI 默认连接 `http://127.0.0.1:11434/v1` 的 `qwen2.5:1.5b`，也允许通过环境变量换本地服务/模型。模型只能选择已授权的电影 ID；最终理由由训练期元数据生成。

**Tech Stack:** Python 3.11 标准库 `urllib`、现有 NumPy/SciPy、pytest。运行时需用户自行启动支持工具调用的本地服务；离线 CI 用脚本化假客户端。

## Global Constraints

- 不把 API 密钥、完整环境变量或模型原始响应写入日志或 trace。
- `base_url` 只来自 CLI/环境配置，不来自用户查询或模型工具参数；默认本机回环地址。
- 模型最多执行 4 次工具调用、最多 6 轮请求、总耗时上限 60 秒；单次 HTTP 超时不超过剩余预算。
- 每次只接受一个工具调用；未知工具、非法 JSON、额外参数、越界数量、未召回 ID 都视为失败并回退。
- 只有 `recall_candidates` 和 `rank_candidates` 的结果 ID 可进入最终输出，且最终 ID 必须属于已排序结果。
- 用户 ID 由外部请求固定，模型不能读取其他用户历史。
- 模型生成的自由文本不作为事实性推荐理由；理由仍由目录标题、题材与排序来源构造。
- 无本地服务时测试仍全绿；不得把假客户端测试说成真实模型实测。

---

### Task 1: OpenAI 兼容本地客户端

**Files:** Create `recagent/agent/chat_client.py`, `tests/test_chat_client.py`.

**Interfaces:** `ToolCall(id: str, name: str, arguments: str)`、`ChatTurn(content: str | None, tool_calls: tuple[ToolCall, ...], usage: dict[str, int])`、`ChatClient.complete(messages: list[dict[str, Any]], tools: list[dict[str, Any]], timeout_s: float) -> ChatTurn`；`OpenAICompatibleClient(base_url, model, api_key, max_tokens=512)` 实现该协议。

- [x] **Step 1: Write failing tests**：用 `unittest.mock.patch` 替换 `urllib.request.urlopen`，断言请求路径为 `/v1/chat/completions`、body 有 `model/messages/tools/parallel_tool_calls=false`、超时按参数传递；解析 `tool_calls[].function.arguments` 与 `usage`；错误响应和缺失 `choices` 抛类型化异常且不泄漏密钥。
- [x] **Step 2: Verify red**：`python -m pytest tests/test_chat_client.py -q`，预期模块不存在失败。
- [x] **Step 3: Implement**：`urllib.request.Request` POST JSON；只接受 `http://127.0.0.1`、`http://localhost` 等回环地址作为默认本地模式；从响应取第一条消息，校验工具调用 ID、名称、arguments 为字符串；无服务时抛 `ModelUnavailableError`。客户端不自动重试。
- [x] **Step 4: Verify green**：目标测试、Ruff、Mypy 通过。
- [x] **Step 5: Commit**：`feat: add local chat completions client`。

### Task 2: 模型工具循环与回退

**Files:** Create `recagent/agent/llm_runtime.py`, `tests/test_llm_runtime.py`.

**Interfaces:** `LLMRecommendationAgent(tools: RecommendationTools, client: ChatClient, max_tool_calls=4, max_turns=6, total_timeout_s=60).run(query, user_id=None, top_k=5) -> AgentResponse`。沿用现有响应结构，`mode="llm"`，新增可选 `model_usage` 字段。

- [x] **Step 1: Write failing tests**：脚本化假客户端先请求召回、再请求排序、最后返回 `{"item_ids":[...]}`，断言消息回放含 assistant 的 `tool_calls` 和对应 `role=tool`；trace 记录两次工具调用；非法 JSON、伪造 ID、跨用户历史、重复/并行工具、步骤超限和模型异常均回退，且模型原始理由不能进入输出。
- [x] **Step 2: Verify red**：`python -m pytest tests/test_llm_runtime.py -q`，预期模块不存在失败。
- [x] **Step 3: Implement**：工具 schema 只暴露 `search_movies/get_user_history/recall_candidates/rank_candidates`；参数采用严格字段白名单和类型/范围验证，用户 ID 从 run 入参注入；对话循环累计 token usage 和 trace，按剩余时间请求客户端；结束前校验最终 ID 是 `rank_candidates` 结果的子集且符合显式题材约束；失败时运行确定性回退并标记 `mode="llm"`、`fallback_used=true`。
- [x] **Step 4: Verify green**：目标测试、完整 `pytest -q`、Ruff、Mypy 通过；假客户端端到端用例证明真实工具路由而非固定脚本回放。
- [x] **Step 5: Commit**：`feat: run bounded local LLM tool calls`。

### Task 3: 本地演示入口与可复现文档

**Files:** Create `scripts/run_local_llm_agent.py`, `docs/LOCAL-LLM.md`; modify `README.md`; test `tests/test_llm_runtime.py`.

**Interfaces:** `python scripts/run_local_llm_agent.py --query "想看科幻，不要恐怖" --user-id 6040 --top-k 3`；环境变量 `RECAGENT_LLM_BASE_URL`、`RECAGENT_LLM_MODEL`、可选 `RECAGENT_LLM_API_KEY`。

- [x] **Step 1: Write failing test**：使用假本地 HTTP 服务/替身客户端运行 CLI，断言 JSON 中 `mode=llm`、工具 trace 与结果 ID；无服务时清楚报告回退而非崩溃。
- [x] **Step 2: Verify red**：`python -m pytest tests/test_llm_runtime.py -q`，预期 CLI 缺失失败。
- [x] **Step 3: Implement**：默认 Ollama 回环地址和 `qwen2.5:1.5b`；文档列启动服务、拉取模型、运行命令、关闭服务以及怎样辨认“模型调用成功”和“回退”。记录本机 6GB GPU 限制，不承诺性能。
- [x] **Step 4: Verify green**：完整 CI 命令通过；若本机服务与模型实际可用，运行真实请求并记录版本、耗时与失败情况。若服务缺失，仅报告未实测，不能称模型质量已验证。
- [x] **Step 5: Commit**：`docs: add local model demo and failure modes`。

