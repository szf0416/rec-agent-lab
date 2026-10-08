"""下载与校验模块测试。

策略：不上网。用本地临时目录伪造数据集，把校验逻辑的分支全部走一遍。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recagent.data.download import (
    ML1M_SPEC,
    DatasetSpec,
    FileSpec,
    ensure_dataset,
    get_spec,
    md5_of,
    verify_dataset,
    verify_file,
)
from recagent.data.exceptions import IntegrityError


@pytest.fixture
def fake_spec(tmp_path: Path) -> tuple[DatasetSpec, Path, bytes]:
    """构造一个只有单个文件的假数据集规格。"""
    payload = b"user::item::rating::timestamp\n1::2::3::4\n"
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.dat").write_bytes(payload)

    spec = DatasetSpec(
        name="fake",
        url="http://127.0.0.1:9/never.used",
        files=(FileSpec("sample.dat", md5=md5_of(source / "sample.dat"), size=len(payload)),),
    )
    return spec, source, payload


# --------------------------------------------------------------------------
# md5 与单文件校验
# --------------------------------------------------------------------------


def test_md5_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    target = tmp_path / "x.bin"
    payload = b"hello recagent" * 1000
    target.write_bytes(payload)
    assert md5_of(target) == hashlib.md5(payload).hexdigest()


def test_md5_handles_empty_file(tmp_path: Path) -> None:
    import hashlib

    target = tmp_path / "empty.bin"
    target.write_bytes(b"")
    assert md5_of(target) == hashlib.md5(b"").hexdigest()


def test_verify_file_accepts_good_file(tmp_path: Path) -> None:
    target = tmp_path / "ok.dat"
    target.write_bytes(b"abc")
    verify_file(target, FileSpec("ok.dat", md5=md5_of(target)))


def test_verify_file_detects_missing(tmp_path: Path) -> None:
    with pytest.raises(IntegrityError, match="缺失"):
        verify_file(tmp_path / "none.dat", FileSpec("none.dat", md5="0" * 32))


def test_verify_file_detects_corrupted_content(tmp_path: Path) -> None:
    target = tmp_path / "bad.dat"
    target.write_bytes(b"corrupted")
    with pytest.raises(IntegrityError, match="校验值不符"):
        verify_file(target, FileSpec("bad.dat", md5="0" * 32))


def test_verify_file_detects_wrong_size(tmp_path: Path) -> None:
    target = tmp_path / "size.dat"
    target.write_bytes(b"abc")
    spec = FileSpec("size.dat", md5=md5_of(target), size=999)
    with pytest.raises(IntegrityError, match="大小不符"):
        verify_file(target, spec)


def test_verify_file_can_skip_size_check(tmp_path: Path) -> None:
    target = tmp_path / "size.dat"
    target.write_bytes(b"abc")
    spec = FileSpec("size.dat", md5=md5_of(target), size=999)
    verify_file(target, spec, check_size=False)  # 不应抛错


# --------------------------------------------------------------------------
# 数据集级校验
# --------------------------------------------------------------------------


def test_verify_dataset_reports_all_problems(tmp_path: Path) -> None:
    spec = DatasetSpec(
        name="two",
        url="http://127.0.0.1:9/x",
        files=(
            FileSpec("a.dat", md5="0" * 32),
            FileSpec("b.dat", md5="0" * 32),
        ),
    )
    ok, problems = verify_dataset(tmp_path, spec)
    assert ok is False
    assert set(problems) == {"a.dat", "b.dat"}


def test_verify_dataset_ok_when_all_present(tmp_path: Path) -> None:
    (tmp_path / "a.dat").write_bytes(b"a")
    spec = DatasetSpec(
        name="one",
        url="http://127.0.0.1:9/x",
        files=(FileSpec("a.dat", md5=md5_of(tmp_path / "a.dat")),),
    )
    ok, problems = verify_dataset(tmp_path, spec)
    assert ok is True
    assert problems == []


# --------------------------------------------------------------------------
# 规格注册表
# --------------------------------------------------------------------------


def test_get_spec_known_dataset() -> None:
    assert get_spec("ml-1m") is ML1M_SPEC


def test_get_spec_unknown_lists_available() -> None:
    with pytest.raises(IntegrityError, match="ml-1m"):
        get_spec("not-a-dataset")


def test_ml1m_spec_declares_three_files() -> None:
    assert {f.name for f in ML1M_SPEC.files} == {"ratings.dat", "movies.dat", "users.dat"}
    assert all(f.size for f in ML1M_SPEC.files)
    assert all(len(f.md5) == 32 for f in ML1M_SPEC.files)


# --------------------------------------------------------------------------
# ensure_dataset：本地复制路径（不联网）
# --------------------------------------------------------------------------


def test_ensure_dataset_copies_from_local_source(fake_spec, tmp_path: Path) -> None:
    spec, source, payload = fake_spec
    target = tmp_path / "target"

    # 用 monkeypatch 把注册表临时替换掉
    import recagent.data.download as download_module

    original = dict(download_module.SPECS)
    download_module.SPECS[spec.name] = spec
    try:
        report = ensure_dataset(spec.name, target, local_source=source)
    finally:
        download_module.SPECS.clear()
        download_module.SPECS.update(original)

    assert report.ok
    assert report.files["sample.dat"] == "copied"
    assert (target / "sample.dat").read_bytes() == payload


def test_ensure_dataset_reuses_verified_files_without_source(fake_spec, tmp_path: Path) -> None:
    """目标目录已就绪时，即使不给 local_source 也应直接返回，不触发下载。"""
    spec, _source, payload = fake_spec
    target = tmp_path / "target"
    target.mkdir()
    (target / "sample.dat").write_bytes(payload)

    import recagent.data.download as download_module

    original = dict(download_module.SPECS)
    download_module.SPECS[spec.name] = spec
    try:
        report = ensure_dataset(spec.name, target)  # 不传 local_source，不应联网
    finally:
        download_module.SPECS.clear()
        download_module.SPECS.update(original)

    assert report.ok
    assert report.files["sample.dat"] == "verified"
    assert report.downloaded == []


def test_ensure_dataset_rejects_corrupt_local_source(fake_spec, tmp_path: Path) -> None:
    """来源数据损坏时必须报错，绝不把坏数据复制进目标目录。"""
    spec, source, _ = fake_spec
    (source / "sample.dat").write_bytes(b"tampered")

    import recagent.data.download as download_module

    original = dict(download_module.SPECS)
    download_module.SPECS[spec.name] = spec
    try:
        # 篡改后大小与 md5 都不符，报哪个都对；关键是必须报错
        with pytest.raises(IntegrityError, match="不符"):
            ensure_dataset(spec.name, tmp_path / "target", local_source=source)
    finally:
        download_module.SPECS.clear()
        download_module.SPECS.update(original)

    assert not (tmp_path / "target" / "sample.dat").exists()


def test_ensure_dataset_rejects_missing_local_source_dir(fake_spec, tmp_path: Path) -> None:
    spec, _source, _ = fake_spec

    import recagent.data.download as download_module

    original = dict(download_module.SPECS)
    download_module.SPECS[spec.name] = spec
    try:
        with pytest.raises(IntegrityError, match="不是目录"):
            ensure_dataset(spec.name, tmp_path / "target", local_source=tmp_path / "ghost")
    finally:
        download_module.SPECS.clear()
        download_module.SPECS.update(original)


def test_ensure_dataset_summary_is_readable(fake_spec, tmp_path: Path) -> None:
    spec, source, _ = fake_spec

    import recagent.data.download as download_module

    original = dict(download_module.SPECS)
    download_module.SPECS[spec.name] = spec
    try:
        report = ensure_dataset(spec.name, tmp_path / "target", local_source=source)
    finally:
        download_module.SPECS.clear()
        download_module.SPECS.update(original)

    summary = report.summary()
    assert "fake" in summary
    assert "就绪=1" in summary
