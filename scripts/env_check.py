"""环境自检：一条命令确认环境是否满足项目要求。

用法:
    python scripts/env_check.py
    python scripts/env_check.py --verbose

设计意图：S0 的验收标准是「新机器上一条命令跑通」。
这个脚本把"跑通"定义成可检查的清单，而不是靠感觉。

退出码:
    0  全部必需项通过
    1  有必需项失败
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

# 允许直接以脚本方式运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MIN_PYTHON = (3, 10)
RECOMMENDED_PYTHON = (3, 11)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True
    hint: str = ""


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def required_failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.required and not r.ok]

    @property
    def optional_failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.required and not r.ok]


# --------------------------------------------------------------------------
# 各项检查
# --------------------------------------------------------------------------


def check_python() -> CheckResult:
    version = sys.version_info
    ok = version[:2] >= MIN_PYTHON
    detail = f"{version.major}.{version.minor}.{version.micro} ({platform.system()} {platform.machine()})"
    hint = ""
    if not ok:
        hint = f"需要 Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
    elif version[:2] != RECOMMENDED_PYTHON:
        hint = f"可用，但推荐 {RECOMMENDED_PYTHON[0]}.{RECOMMENDED_PYTHON[1]}"
    return CheckResult("Python 版本", ok, detail, hint=hint)


def check_package(module: str, label: str, *, required: bool = True) -> CheckResult:
    """检查单个包能否导入，并尽量报出版本号。"""
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        return CheckResult(
            label,
            False,
            f"未安装 ({exc})",
            required=required,
            hint=f"pip install {module}",
        )
    version = getattr(mod, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as dist_version

            version = dist_version(module)
        except Exception:  # noqa: BLE001 - 版本号取不到不影响结论
            version = "版本未知"
    return CheckResult(label, True, str(version), required=required)


def check_torch() -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        import torch
    except ImportError:
        results.append(
            CheckResult("PyTorch", False, "未安装", hint="见 environment.yml 文件头的安装说明")
        )
        return results

    results.append(CheckResult("PyTorch", True, torch.__version__))

    cuda_version = torch.version.cuda
    if cuda_version is None:
        results.append(
            CheckResult(
                "CUDA 构建",
                False,
                "安装的是 CPU 版 torch",
                hint="训练需要 GPU 版：conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia",
            )
        )
    else:
        results.append(CheckResult("CUDA 构建", True, f"cu{cuda_version}"))

    available = torch.cuda.is_available()
    if available:
        device_count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        results.append(
            CheckResult("CUDA 可用", True, f"{device_count} 张卡 | {name} | 显存 {total_gb:.1f} GB")
        )
    else:
        results.append(
            CheckResult(
                "CUDA 可用",
                False,
                "torch 装了但检测不到 GPU",
                hint="检查显卡驱动；数据管线与 CPU 测试不受影响，但训练会极慢",
            )
        )
    return results


def check_ml1m_data() -> list[CheckResult]:
    """检查 ML-1M 原始数据是否就绪且校验通过。"""
    from recagent.data import ML1M_SPEC, IntegrityError, verify_dataset

    results: list[CheckResult] = []
    raw_dir = PROJECT_ROOT / "data" / "raw" / "ml-1m"

    if not raw_dir.is_dir():
        results.append(
            CheckResult(
                "ML-1M 原始数据",
                False,
                f"目录不存在: {raw_dir}",
                hint="运行 python scripts/download_data.py --dataset ml-1m "
                "--local-source <你已有的 ml-1m 目录>",
            )
        )
        return results

    try:
        ok, problems = verify_dataset(raw_dir, ML1M_SPEC)
    except IntegrityError as exc:
        results.append(CheckResult("ML-1M 原始数据", False, str(exc)))
        return results

    if ok:
        results.append(CheckResult("ML-1M 原始数据", True, f"3 个文件校验通过 ({raw_dir})"))
    else:
        results.append(
            CheckResult(
                "ML-1M 原始数据",
                False,
                f"{raw_dir} 中校验失败: {problems}",
                hint="删除对应文件后重新下载",
            )
        )
    return results


def check_processed_data() -> CheckResult:
    """预处理产物是 S1/S3 的输入，缺失不影响 S0 验收，故标记为非必需。"""
    processed = PROJECT_ROOT / "data" / "processed" / "ml1m"
    summary = processed / "summary.json"
    if summary.is_file():
        return CheckResult("预处理产物", True, str(processed), required=False)
    return CheckResult(
        "预处理产物",
        False,
        "尚未生成",
        required=False,
        hint="python scripts/preprocess.py --config configs/data/ml1m.yaml",
    )


def check_project_layout() -> CheckResult:
    """确认关键文件都在位。缺文件说明仓库不完整或跑错了目录。"""
    required_paths = [
        "pyproject.toml",
        "environment.yml",
        "recagent/__init__.py",
        "recagent/data/__init__.py",
        "configs/data/ml1m.yaml",
        "tests/test_splits.py",
    ]
    missing = [p for p in required_paths if not (PROJECT_ROOT / p).exists()]
    if missing:
        return CheckResult(
            "项目结构",
            False,
            f"缺少: {missing}",
            hint=f"请在仓库根目录运行本脚本（当前 cwd={Path.cwd()}）",
        )
    return CheckResult("项目结构", True, f"{len(required_paths)} 个关键文件在位")


def check_import_recagent() -> CheckResult:
    try:
        import recagent

        return CheckResult("recagent 可导入", True, f"v{recagent.__version__}")
    except Exception as exc:  # noqa: BLE001 - 导入失败原因多样，直接展示
        return CheckResult(
            "recagent 可导入",
            False,
            f"{type(exc).__name__}: {exc}",
            hint='在仓库根目录执行 pip install -e ".[dev]"',
        )


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def run_all(*, verbose: bool = False) -> Report:
    report = Report()

    report.add(check_python())
    report.add(check_project_layout())
    report.add(check_import_recagent())

    # 必需：数据层当前与后续阶段都要用
    for module, label in (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("yaml", "PyYAML"),
        ("pytest", "pytest"),
    ):
        report.add(check_package(module, label))

    # 推荐：lint / 类型 / 实验追踪，缺了 CI 跑不了但训练不受影响
    for module, label in (
        ("ruff", "ruff (lint)"),
        ("mypy", "mypy (类型检查)"),
        ("mlflow", "mlflow (实验追踪)"),
    ):
        report.add(check_package(module, label, required=False))

    for result in check_torch():
        report.add(result)

    for result in check_ml1m_data():
        report.add(result)

    report.add(check_processed_data())

    if verbose:
        report.add(check_package("hydra", "hydra-core", required=False))
        report.add(check_package("pydantic", "pydantic", required=False))
        report.add(check_package("pre_commit", "pre-commit", required=False))

    return report


def print_report(report: Report) -> None:
    print()
    print("=" * 78)
    print("RecAgent 环境自检")
    print("=" * 78)

    for result in report.results:
        if result.ok:
            mark = "OK  "
        elif result.required:
            mark = "FAIL"
        else:
            mark = "WARN"
        tag = "" if result.required else " (可选)"
        print(f"[{mark}] {result.name:22}{tag:8} {result.detail}")
        if result.hint and not result.ok:
            print(f"       -> {result.hint}")

    print("-" * 78)
    required_failures = report.required_failures
    optional_failures = report.optional_failures

    if not required_failures:
        print("必需项全部通过。")
    else:
        print(f"有 {len(required_failures)} 个必需项未通过，请按上面的提示处理。")

    if optional_failures:
        print(f"另有 {len(optional_failures)} 个可选项未就绪（不阻塞开发）。")

    print("=" * 78)
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RecAgent 环境自检")
    parser.add_argument("--verbose", action="store_true", help="额外检查可选依赖")
    args = parser.parse_args(argv)

    # 让 recagent 的日志不干扰本脚本的清单输出
    os.environ.setdefault("RECAGENT_LOG_LEVEL", "WARNING")

    report = run_all(verbose=args.verbose)
    print_report(report)
    return 1 if report.required_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
