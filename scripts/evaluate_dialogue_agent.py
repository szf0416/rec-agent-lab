"""冻结对话任务的确定性或本地模型评测。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recagent.agent.chat_client import OpenAICompatibleClient
from recagent.agent.llm_runtime import LLMRecommendationAgent
from recagent.agent.runtime import DeterministicAgent
from recagent.agent.tools import RecommendationTools
from recagent.engine.store import Catalog
from recagent.eval.dialogue import load_cases, score_case, summarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="冻结对话推荐任务评测")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/ml1m"))
    parser.add_argument("--cases", type=Path, default=Path("configs/eval/dialogue_cases.json"))
    parser.add_argument("--mode", choices=("deterministic", "llm"), default="deterministic")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    cases = load_cases(args.cases)
    catalog = Catalog.load(args.processed_dir)
    model = os.environ.get("RECAGENT_LLM_MODEL", "qwen2.5:1.5b")
    client = (
        OpenAICompatibleClient(
            os.environ.get("RECAGENT_LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
            model,
            os.environ.get("RECAGENT_LLM_API_KEY"),
        )
        if args.mode == "llm"
        else None
    )
    rows = []
    for case in cases:
        tools = RecommendationTools(catalog)
        if case.inject_rank_failure:

            def fail_rank(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("injected rank failure")

            tools.rank_candidates = fail_rank  # type: ignore[method-assign,assignment]
        agent = (
            LLMRecommendationAgent(tools, client)
            if client is not None
            else DeterministicAgent(tools)
        )
        response = agent.run(case.query, case.user_id, case.top_k)
        rows.append(score_case(case, response, catalog))
    result = {
        "protocol": "frozen_dialogue_v1",
        "case_set_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "mode": args.mode,
        "model": model if client is not None else None,
        "summary": summarize(rows),
        "cases": rows,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
