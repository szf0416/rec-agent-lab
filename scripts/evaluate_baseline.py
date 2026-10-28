"""输出固定候选题集上的推荐基线指标。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recagent.eval.sampled import evaluate_sampled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ML-1M 固定候选推荐评测")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/ml1m"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--expected-negatives", type=int, default=100)
    args = parser.parse_args(argv)
    print(
        json.dumps(evaluate_sampled(args.processed_dir, args.k, args.expected_negatives), indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
