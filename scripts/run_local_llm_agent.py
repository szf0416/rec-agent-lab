"""通过本机 OpenAI 兼容服务演示真实模型工具调用。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recagent.agent.chat_client import OpenAICompatibleClient
from recagent.agent.llm_runtime import LLMRecommendationAgent
from recagent.agent.tools import RecommendationTools
from recagent.engine.store import Catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RecAgent 本地 LLM 工具调用演示")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/ml1m"))
    parser.add_argument("--query", required=True)
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    client = OpenAICompatibleClient(
        os.environ.get("RECAGENT_LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
        os.environ.get("RECAGENT_LLM_MODEL", "qwen2.5:1.5b"),
        os.environ.get("RECAGENT_LLM_API_KEY"),
    )
    agent = LLMRecommendationAgent(RecommendationTools(Catalog.load(args.processed_dir)), client)
    sys.stdout.reconfigure(encoding="utf-8")
    print(
        json.dumps(
            asdict(agent.run(args.query, args.user_id, args.top_k)), ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
