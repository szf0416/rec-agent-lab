"""在本机启动 RecAgent HTTP 演示服务。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recagent.serving.api import create_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RecAgent 本机 HTTP 演示")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/ml1m"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1 到 65535 之间")
    app = create_app(args.processed_dir)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
