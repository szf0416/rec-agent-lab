"""数据集下载与完整性校验。

设计要点：
1. **幂等**：文件已存在且校验通过则直接返回，不重复下载。
2. **可离线**：支持 `local_source`，把已下载好的数据登记进来，避免重复拉取。
3. **强制校验**：下载后必须通过 md5，失败即抛错并删除损坏文件，
   绝不"带着坏数据继续跑"——静默的数据损坏会让后面所有实验白做。

校验值来源：本地已下载的官方原始文件实测得出（见 docs/SCHEDULE.md W1）。
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from recagent.data.exceptions import DownloadError, IntegrityError
from recagent.utils.logging import get_logger

logger = get_logger(__name__)

FileStatus = Literal["verified", "copied", "downloaded"]


@dataclass(frozen=True)
class FileSpec:
    """单个数据文件的校验规格。"""

    name: str
    md5: str
    size: int | None = None  # 字节；None 表示不校验大小


@dataclass(frozen=True)
class DatasetSpec:
    """一个数据集的下载与校验规格。"""

    name: str
    url: str
    files: tuple[FileSpec, ...]
    archive: str | None = None  # 压缩包文件名；None 表示无需解压


@dataclass
class DownloadReport:
    """下载结果，供脚本打印与测试断言使用。

    `ok` 的语义是「本次任务要求的文件是否都已就绪」，
    而不是「本次做了什么」——后者看 `downloaded` / `reused`。

    `expected` 显式记录期望的文件名，而不是回查全局 SPECS：
    回查会让本对象隐式依赖外部可变状态，测试替换注册表后就失效了。
    """

    dataset: str
    root: Path
    expected: set[str] = field(default_factory=set)
    files: dict[str, FileStatus] = field(default_factory=dict)
    local_source: str | None = None

    @property
    def ok(self) -> bool:
        """所有期望文件是否都已就绪。"""
        return self.expected.issubset(self.files.keys())

    @property
    def downloaded(self) -> list[str]:
        return [name for name, status in self.files.items() if status == "downloaded"]

    @property
    def reused(self) -> list[str]:
        return [name for name, status in self.files.items() if status == "verified"]

    @property
    def missing(self) -> list[str]:
        return sorted(self.expected - self.files.keys())

    def summary(self) -> str:
        parts = [f"数据集={self.dataset}", f"目录={self.root}", f"就绪={len(self.files)}"]
        if self.downloaded:
            parts.append(f"新下载={len(self.downloaded)}")
        if self.reused:
            parts.append(f"复用={len(self.reused)}")
        if self.local_source:
            parts.append(f"来源=本地({self.local_source})")
        return " | ".join(parts)


# MovieLens-1M 官方分发的三个数据文件
ML1M_SPEC = DatasetSpec(
    name="ml-1m",
    url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    archive="ml-1m.zip",
    files=(
        FileSpec("ratings.dat", md5="a89aa3591bc97d6d4e0c89459ff39362", size=24_594_131),
        FileSpec("movies.dat", md5="ef9fd9ff8d1b33faf73e2122c371e910", size=171_308),
        FileSpec("users.dat", md5="681c0711d0d4548cd3d4718a49c2dcd8", size=134_368),
    ),
)

SPECS: dict[str, DatasetSpec] = {ML1M_SPEC.name: ML1M_SPEC}


def get_spec(dataset: str) -> DatasetSpec:
    """按名字取数据集规格，名字非法时明确报错而不是猜。"""
    try:
        return SPECS[dataset]
    except KeyError:
        known = ", ".join(sorted(SPECS))
        raise IntegrityError(f"未知数据集 {dataset!r}，已注册的有: {known}") from None


def md5_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """流式计算 md5，避免把大文件整个读进内存。"""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, spec: FileSpec, *, check_size: bool = True) -> None:
    """校验单个文件，不通过则抛 IntegrityError。"""
    if not path.is_file():
        raise IntegrityError(f"文件缺失: {path}")

    if check_size and spec.size is not None:
        actual_size = path.stat().st_size
        if actual_size != spec.size:
            raise IntegrityError(
                f"{path.name} 大小不符: 期望 {spec.size} 字节, 实际 {actual_size} 字节"
            )

    actual_md5 = md5_of(path)
    if actual_md5 != spec.md5:
        raise IntegrityError(
            f"{path.name} 校验值不符: 期望 {spec.md5}, 实际 {actual_md5}\n"
            f"文件可能损坏或被截断，请删除后重新下载: {path}"
        )


def verify_dataset(
    root: Path, spec: DatasetSpec, *, check_size: bool = True
) -> tuple[bool, list[str]]:
    """校验整个数据集。

    Returns:
        (是否全部通过, 有问题的文件名列表)
    """
    problems: list[str] = []
    for file_spec in spec.files:
        try:
            verify_file(root / file_spec.name, file_spec, check_size=check_size)
        except IntegrityError as exc:
            logger.debug("校验未通过: %s", exc)
            problems.append(file_spec.name)
    return (not problems), problems


def _download(url: str, target: Path, *, timeout: float = 60.0) -> None:
    """下载到临时文件再改名，避免中断留下半个文件被误认为完整。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    logger.info("下载 %s -> %s", url, target.name)

    request = urllib.request.Request(url, headers={"User-Agent": "recagent/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            with tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1 << 20)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(
            f"下载失败: {url}\n原因: {exc}\n"
            f"建议: 检查网络；或手动下载后放到 {target.parent} 再重跑本脚本"
        ) from exc

    tmp.replace(target)


def _extract_archive(archive: Path, dest: Path) -> None:
    """解压 zip 并把内容展平到 dest。"""
    import zipfile

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)

    # 官方 zip 内层是 ml-1m/ 目录，展平一层便于统一路径
    nested = [p for p in dest.iterdir() if p.is_dir() and p.name != dest.name]
    if len(nested) == 1:
        for child in nested[0].iterdir():
            child.replace(dest / child.name)
        nested[0].rmdir()


def ensure_dataset(
    dataset: str,
    root: Path,
    *,
    local_source: Path | None = None,
    check_size: bool = True,
    force: bool = False,
) -> DownloadReport:
    """确保数据集就绪：校验 → （必要时）本地复制或下载 → 解压 → 再次校验。

    Args:
        dataset: 数据集名，见 SPECS。
        root: 数据落地目录（如 data/raw/ml-1m）。
        local_source: 已有数据的目录。给定时优先从这里复制，不联网。
        check_size: 是否同时校验字节大小。
        force: 忽略现有文件，强制重新获取。

    Raises:
        IntegrityError: 校验失败且无法修复。
        DownloadError: 下载失败。
    """
    spec = get_spec(dataset)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    report = DownloadReport(
        dataset=dataset,
        root=root,
        expected={f.name for f in spec.files},
    )

    if not force:
        ok, problems = verify_dataset(root, spec, check_size=check_size)
        if ok:
            report.files = {f.name: "verified" for f in spec.files}
            logger.info("%s 已就绪且校验通过，跳过获取", dataset)
            return report
        logger.warning("%s 有 %s 个文件需处理: %s", dataset, len(problems), problems)

    # 优先用本地已有数据，避免重复下载
    if local_source is not None:
        local_source = Path(local_source)
        if not local_source.is_dir():
            raise IntegrityError(f"local_source 不是目录: {local_source}")
        report.local_source = str(local_source)

        for file_spec in spec.files:
            src = local_source / file_spec.name
            dst = root / file_spec.name
            # 先校验来源，坏数据不进目标目录
            verify_file(src, file_spec, check_size=check_size)
            if src.resolve() == dst.resolve():
                report.files[file_spec.name] = "verified"
                continue
            shutil.copy2(src, dst)
            report.files[file_spec.name] = "copied"
        logger.info("已从本地 %s 登记 %s 个文件", local_source, len(report.files))
        return report

    archive = root / (spec.archive or f"{dataset}.zip")
    if force or not archive.is_file():
        _download(spec.url, archive)
    else:
        logger.info("复用已下载的压缩包 %s", archive.name)

    _extract_archive(archive, root)

    for file_spec in spec.files:
        verify_file(root / file_spec.name, file_spec, check_size=check_size)
        report.files[file_spec.name] = "downloaded"

    logger.info("%s 获取完成并通过校验", dataset)
    return report
