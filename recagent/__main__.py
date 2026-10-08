"""命令行入口占位。

S0 只提供骨架，真正的子命令在后续阶段按需接入：

    S1: recagent train-loop-check      # 训练循环自检
    S3: recagent preprocess / train / evaluate
    S5: recagent serve
    S6: recagent agent "<query>"

设计约定：__main__ 只做参数解析与分发，
业务逻辑一律放在 recagent 各子包里。
"""

from __future__ import annotations

import argparse
import sys

from recagent import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recagent",
        description="RecAgent: 从零构建推荐系统到 LLM 推荐 Agent",
    )
    parser.add_argument("--version", action="version", version=f"recagent {__version__}")
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # 后续阶段在此分发子命令
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
