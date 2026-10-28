"""无密钥确定性对话推荐演示；不是 LLM Agent。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recagent.agent.runtime import DeterministicAgent
from recagent.agent.tools import RecommendationTools
from recagent.engine.store import Catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RecAgent 确定性工具流程演示（无 LLM）")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/ml1m"))
    parser.add_argument("--query", required=True)
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    agent = DeterministicAgent(RecommendationTools(Catalog.load(args.processed_dir)))
    sys.stdout.reconfigure(encoding="utf-8")
    print(
        json.dumps(
            asdict(agent.run(args.query, args.user_id, args.top_k)), ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
