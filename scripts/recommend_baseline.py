"""使用训练期 ML-1M 数据运行确定性推荐基线。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recagent.engine.baseline import BaselineRecommender
from recagent.engine.store import Catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练期 Popularity + ItemCF 推荐基线")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/ml1m"))
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--genres", action="append", default=[], help="可重复传入的 MovieLens 题材")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    recommender = BaselineRecommender(Catalog.load(args.processed_dir))
    items = recommender.recommend(
        user_id=args.user_id,
        top_k=args.top_k,
        genres=tuple(args.genres),
    )
    print(json.dumps({"items": [asdict(item) for item in items]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
