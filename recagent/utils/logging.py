"""日志配置。

约定：项目内一律用 `get_logger(__name__)`，不要用 print。
CI 与测试环境自动降噪，避免被无关日志淹没。
"""

from __future__ import annotations

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_CONFIGURED = False


def _configure_root() -> None:
    """配置根 logger。重复调用无副作用。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("RECAGENT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger("recagent")
    root.setLevel(level)
    root.addHandler(handler)
    # 不向上传播，避免被宿主应用的日志配置重复打印一遍
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """取一个挂在 `recagent` 命名空间下的 logger。

    传 `__name__` 即可。若模块名不以 recagent 开头（如脚本直跑），
    统一挂到 `recagent.external` 下，保证日志配置生效。
    """
    _configure_root()
    if name == "recagent" or name.startswith("recagent."):
        return logging.getLogger(name)
    return logging.getLogger(f"recagent.external.{name}")
