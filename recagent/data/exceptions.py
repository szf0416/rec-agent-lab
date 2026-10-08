"""数据层异常。

统一异常层次，便于调用方精确捕获（而不是裸 except）。
"""

from __future__ import annotations


class RecAgentError(Exception):
    """本项目所有异常的基类。"""


class DataError(RecAgentError):
    """数据层异常基类。"""


class DownloadError(DataError):
    """下载失败。"""


class IntegrityError(DataError):
    """文件缺失、大小或校验值不匹配。"""


class ParseError(DataError):
    """原始数据格式不符合预期。"""


class SplitError(DataError):
    """切分失败，或切分结果违反无泄漏约束。"""


class LeakageError(SplitError):
    """检测到未来信息泄漏。这是严重错误，不允许降级处理。"""
