"""代码规范守卫测试。

这里放的是「机器能替人盯住」的规则。人工 review 会漏，
但这类测试一旦写下来就永久生效。

当前守卫：
1. logger 调用必须用 %-风格占位符（Python 标准 logging 不支持 {}）
2. logger 调用的占位符数量必须与参数数量一致
3. 禁止裸 except:
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("recagent", "scripts")

LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical"}


def iter_source_files() -> list[Path]:
    files: list[Path] = []
    for directory in SOURCE_DIRS:
        base = PROJECT_ROOT / directory
        if base.is_dir():
            files.extend(sorted(base.rglob("*.py")))
    return files


def iter_logger_calls(tree: ast.AST) -> list[ast.Call]:
    """找出所有 `logger.<level>(...)` 形式的调用。"""
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in LOG_METHODS:
            continue
        value = func.value
        # 匹配 logger / self.logger / _log 等各种写法
        if isinstance(value, ast.Name):
            is_logger = "log" in value.id.lower()
        elif isinstance(value, ast.Attribute):
            is_logger = "log" in value.attr.lower()
        else:
            is_logger = False
        if is_logger:
            calls.append(node)
    return calls


def test_source_files_are_discovered() -> None:
    """守卫本身要有效：至少得扫到文件，否则等于没检查。"""
    files = iter_source_files()
    assert len(files) >= 8, f"仅发现 {len(files)} 个源文件，扫描逻辑可能失效"


@pytest.mark.parametrize("path", iter_source_files(), ids=lambda p: p.name)
def test_logger_calls_use_percent_style_placeholders(path: Path) -> None:
    """Python 标准 logging 用 %-风格；写成 {} 会在运行时抛 TypeError。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []

    for call in iter_logger_calls(tree):
        if not call.args:
            continue
        first = call.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        if "{}" in first.value or "{:" in first.value:
            offenders.append(f"line {call.lineno}: {first.value[:70]!r}")

    assert not offenders, (
        f"{path.name} 中的 logger 调用使用了 {{}} 占位符，"
        f"应改为 %s / %d / %.1f：\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize("path", iter_source_files(), ids=lambda p: p.name)
def test_logger_placeholder_count_matches_arguments(path: Path) -> None:
    """占位符数量与参数数量不一致时，logging 会在运行时报错。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []

    for call in iter_logger_calls(tree):
        if not call.args:
            continue
        first = call.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue

        # 统计未被 %% 转义的占位符
        message = first.value.replace("%%", "")
        placeholder_count = 0
        index = 0
        while index < len(message):
            if message[index] == "%":
                placeholder_count += 1
                index += 1
                # 跳过格式修饰符
                while index < len(message) and message[index] in "-+ #0123456789.":
                    index += 1
                if index < len(message) and message[index] in "diouxXeEfFgGrsca":
                    index += 1
                continue
            index += 1

        argument_count = len(call.args) - 1
        # 唯一参数是 dict 时，logging 会用 %(name)s 形式，跳过
        if placeholder_count != argument_count and not (
            argument_count == 1 and isinstance(call.args[1], ast.Dict)
        ):
            offenders.append(
                f"line {call.lineno}: 占位符 {placeholder_count} 个但参数 {argument_count} 个"
            )

    assert not offenders, f"{path.name} 中 logger 占位符与参数不匹配：\n" + "\n".join(offenders)


@pytest.mark.parametrize("path", iter_source_files(), ids=lambda p: p.name)
def test_no_bare_except(path: Path) -> None:
    """禁止裸 except:，它会把 KeyboardInterrupt 和真实 bug 一起吞掉。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = [
        f"line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and node.type is None
    ]
    assert not offenders, f"{path.name} 存在裸 except: 于 {offenders}"
