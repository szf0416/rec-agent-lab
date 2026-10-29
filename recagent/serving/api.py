"""共享训练期目录与 Agent 运行时的 FastAPI 应用。"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from recagent.agent.chat_client import ChatClient, OpenAICompatibleClient
from recagent.agent.llm_runtime import LLMRecommendationAgent
from recagent.agent.runtime import DeterministicAgent
from recagent.agent.tools import RecommendationTools
from recagent.engine.store import Catalog


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    user_id: int | None = Field(default=None, ge=1, strict=True)
    top_k: int = Field(default=5, ge=1, le=20, strict=True)
    mode: Literal["baseline", "llm"] = "baseline"

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("query 不能为空白")
        return query


def create_app(processed_dir: Path, client: ChatClient | None = None) -> FastAPI:
    """构建本机演示应用；训练期目录只在创建时加载一次。"""
    catalog = Catalog.load(processed_dir)
    tools = RecommendationTools(catalog)
    app = FastAPI(title="RecAgent Lab", version="0.1.0")

    @app.exception_handler(Exception)
    async def internal_error(_request: object, _exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "推荐服务处理失败"}},
        )

    @app.get("/health")
    def health() -> dict[str, str | int]:
        return {"status": "ok", "catalog_items": len(catalog.movies)}

    @app.post("/recommend")
    def recommend(request: RecommendRequest) -> dict[str, object]:
        agent: DeterministicAgent | LLMRecommendationAgent
        if request.mode == "baseline":
            agent = DeterministicAgent(tools)
        else:
            model_client = client or OpenAICompatibleClient(
                os.environ.get("RECAGENT_LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
                os.environ.get("RECAGENT_LLM_MODEL", "qwen2.5:1.5b"),
                os.environ.get("RECAGENT_LLM_API_KEY"),
            )
            agent = LLMRecommendationAgent(tools, model_client)
        return asdict(agent.run(request.query, request.user_id, request.top_k))

    return app
