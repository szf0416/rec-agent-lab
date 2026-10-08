"""下载并校验数据集。

用法:
    python scripts/download_data.py --dataset ml-1m
    python scripts/download_data.py --dataset ml-1m --local-source C:/path/to/ml-1m
    python scripts/download_data.py --dataset ml-1m --verify-only

设计说明:
- 校验规格（md5/大小）的唯一真相在 recagent/data/download.py，不从配置里读，避免两处不一致。
- --local-source 用于本地已有数据时跳过下载（国内网络访问 grouplens 常超时）。
- 目录路径可在 configs/data/*.yaml 里覆盖。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# 允许直接以脚本方式运行（python scripts/xxx.py）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from recagent.data import (
    IntegrityError,
    ensure_dataset,
    get_spec,
    verify_dataset,
)
from recagent.utils.logging import get_logger

logger = get_logger("download_data")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    # 配置带 @package 指令时顶层就是数据集字段；这里统一取 data 段
    if "data" in data and isinstance(data["data"], dict):
        return dict(data["data"])
    return dict(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="下载并校验数据集")
    parser.add_argument("--dataset", default="ml-1m", help="数据集名，默认 ml-1m")
    parser.add_argument("--config", default=None, help="数据配置 yaml 路径")
    parser.add_argument("--raw-dir", default=None, help="覆盖原始数据落地目录")
    parser.add_argument(
        "--local-source",
        default=None,
        help="本地已有数据的目录；给定时优先复制，不联网",
    )
    parser.add_argument("--verify-only", action="store_true", help="只校验，不做任何下载或复制")
    parser.add_argument("--force", action="store_true", help="忽略现有文件，强制重新获取")
    parser.add_argument("--skip-size-check", action="store_true", help="只校验 md5，不校验字节大小")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config_path = Path(args.config) if args.config else Path(f"configs/data/{args.dataset}.yaml")
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    local_source = Path(args.local_source) if args.local_source else None

    if config_path.is_file():
        config = load_yaml(config_path)
        if raw_dir is None:
            raw_dir = Path(config.get("raw_dir", f"data/raw/{args.dataset}"))
        if local_source is None:
            configured = config.get("source", {}).get("local_source")
            if configured:
                local_source = Path(configured)
        logger.info("使用配置 %s", config_path)
    elif raw_dir is None:
        raw_dir = Path(f"data/raw/{args.dataset}")
        logger.warning("未找到 %s，使用默认目录 %s", config_path, raw_dir)

    if local_source is not None and not local_source.is_dir():
        logger.error("--local-source 指向的目录不存在: %s", local_source)
        return 2

    try:
        if args.verify_only:
            ok, problems = verify_dataset(raw_dir, get_spec(args.dataset))
            if ok:
                logger.info("校验通过: %s", raw_dir)
                return 0
            logger.error("校验失败，有问题的文件: %s", problems)
            return 1

        report = ensure_dataset(
            args.dataset,
            raw_dir,
            local_source=local_source,
            check_size=not args.skip_size_check,
            force=args.force,
        )
    except IntegrityError as exc:
        logger.error("数据完整性错误: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - 顶层入口需要给出可读的失败信息
        logger.error("获取数据失败: %s", exc)
        return 1

    logger.info("完成 | %s", report.summary())
    if not report.ok:
        logger.error("仍有文件未就绪: %s", sorted(set(report.files)))
        return 1
    logger.info("下一步: python scripts/preprocess.py --config configs/data/ml1m.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
