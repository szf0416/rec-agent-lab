# FastAPI 演示入口实施计划

依据：`docs/superpowers/specs/2026-10-28-rec-agent-mvp-design.md` 的 API 数据流与第四周验收。项目已确认优先 Agent / 应用 AI 岗位，本地 Ollama 模式，4–6 周可演示版本。

## 文件职责

- `recagent/serving/api.py`：请求 schema、应用工厂、`/health`、`/recommend` 与结构化异常。
- `scripts/serve_api.py`：本机启动命令，读取模型环境变量，默认绑定 `127.0.0.1`。
- `tests/test_api.py`：使用小型训练期目录通过 ASGI 客户端验证接口契约。
- `pyproject.toml`、`.github/workflows/ci.yml`：轻量 API 依赖及 CI 安装。
- `README.md`、`docs/DECISIONS.md`：启动、请求示例、结果解释及依赖决策。

## Task 1：HTTP 契约与确定性推荐

1. 先在 `tests/test_api.py` 建临时 `items.json`、`interactions_train.csv`，写 `TestClient(create_app(tmp_path))` 的失败测试：`GET /health` 返回就绪；`POST /recommend` 接受 `query`、`user_id`、`top_k`、`mode="baseline"`，返回训练期目录电影、`trace_id`、`tool_trace`、`fallback_used=false`、`latency_ms`。
2. 运行 `D:\anaconda3\envs\llm-lab\python.exe -m pytest tests/test_api.py -q`，确认因 API 模块缺失失败。
3. 最小实现 `create_app(processed_dir: Path, client: ChatClient | None = None) -> FastAPI`。启动时只加载一次 `Catalog`；基线请求运行 `DeterministicAgent`，用 `asdict` 保留现有输出结构。
4. 同一命令确认绿色。

## Task 2：输入验证与模型故障回退

1. 增加失败测试：空白或超长查询、非法 `top_k` / `user_id` / `mode`、额外字段返回 422；本地模型客户端抛 `ModelUnavailableError` 时仍返回 200 且 `mode="llm"`、`fallback_used=true`、有错误 trace；无数据目录在创建应用时明确失败。
2. 运行单文件测试确认红色。
3. 使用 Pydantic v2 严格请求模型：`query` 1–500 且非纯空白，`user_id` 正整数或空，`top_k` 1–20，`mode` 限 `baseline|llm`。失败请求沿用 FastAPI 422 结构；服务异常返回无内部堆栈的 JSON 错误。模型客户端仅在 `llm` 请求需要时构建，本地 `/v1` 地址限制继续由已有客户端实现。
4. 运行单文件测试确认绿色。

## Task 3：入口、文档与验收

1. `scripts/serve_api.py` 启动 `uvicorn`，支持 `--processed-dir`、`--port`；默认仅本机监听。`pyproject.toml` 增加 `api` extra（FastAPI、Uvicorn、HTTPX），CI 用 `.[dev,api]`。
2. README 给出安装、启动、`Invoke-RestMethod` 请求、`fallback_used` 判读；ADR 记录轻量 API extra 的原因。
3. 运行完整 `pytest -q`、`ruff check .`、`ruff format --check .`、`mypy recagent`。再用真实 MovieLens 目录发本机 HTTP 基线请求；如 Ollama 可达，再发 `llm` 请求并如实记录是否回退。
4. 核对 diff、提交、合并并推送 GitHub；核对远端提交 ID。
