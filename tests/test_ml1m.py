"""MovieLens-1M 解析器测试。

包含两类：
- 合成小文件测试（始终运行，覆盖格式错误等分支）
- 真实数据测试（本地有数据时运行，验证解析器对官方文件的正确性）
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from recagent.data.exceptions import IntegrityError, ParseError
from recagent.data.ml1m import (
    Movie,
    MovieLens1M,
    load_ml1m,
    load_ml1m_cached,
    parse_movies,
    parse_ratings,
    parse_users,
    save_cached,
)

# 本地已下载的 ML-1M 原始数据（用于真实数据校验，找不到就跳过）
REAL_DATA_CANDIDATES = (
    Path("C:/Users/shizifan/Desktop/code/LightGCN/raw_data/ml-1m"),
    Path("C:/Users/shizifan/Desktop/code/rec-agent-lab/data/raw/ml-1m"),
)


def write_ratings(path: Path, rows: list[tuple[int, int, int, int]]) -> Path:
    target = path / "ratings.dat"
    target.write_text(
        "\n".join(f"{u}::{i}::{r}::{t}" for u, i, r, t in rows) + "\n", encoding="latin-1"
    )
    return target


def write_metadata(path: Path) -> None:
    (path / "movies.dat").write_text(
        "1::Toy Story (1995)::Animation|Children's|Comedy\n"
        "2::Jumanji (1995)::Adventure|Children's|Fantasy\n"
        "3::No Year Movie::Drama\n",
        encoding="latin-1",
    )
    (path / "users.dat").write_text("1::F::1::10::48067\n2::M::56::16::70072\n", encoding="latin-1")


# --------------------------------------------------------------------------
# 合成数据
# --------------------------------------------------------------------------


def test_parse_ratings_basic(tmp_path: Path) -> None:
    write_ratings(tmp_path, [(1, 10, 5, 100), (1, 20, 3, 200), (2, 10, 4, 150)])
    ratings = parse_ratings(tmp_path / "ratings.dat")

    assert len(ratings) == 3
    assert ratings[0].user_id == 1
    assert ratings[0].item_id == 10
    assert ratings[0].rating == 5
    assert ratings[0].timestamp == 100


def test_parse_ratings_reports_line_number_on_bad_field_count(tmp_path: Path) -> None:
    target = tmp_path / "ratings.dat"
    target.write_text("1::10::5::100\n2::20::4\n", encoding="latin-1")
    with pytest.raises(ParseError, match="第 2 行"):
        parse_ratings(target)


def test_parse_ratings_reports_on_non_numeric(tmp_path: Path) -> None:
    target = tmp_path / "ratings.dat"
    target.write_text("1::10::abc::100\n", encoding="latin-1")
    with pytest.raises(ParseError):
        parse_ratings(target)


def test_parse_ratings_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(IntegrityError):
        parse_ratings(tmp_path / "nope.dat")


def test_parse_ratings_empty_file_raises(tmp_path: Path) -> None:
    target = tmp_path / "ratings.dat"
    target.write_text("", encoding="latin-1")
    with pytest.raises(ParseError):
        parse_ratings(target)


def test_parse_ratings_skips_blank_lines(tmp_path: Path) -> None:
    target = tmp_path / "ratings.dat"
    target.write_text("1::10::5::100\n\n\n2::20::4::200\n", encoding="latin-1")
    assert len(parse_ratings(target)) == 2


def test_parse_movies_extracts_genres_and_year(tmp_path: Path) -> None:
    write_metadata(tmp_path)
    movies = parse_movies(tmp_path / "movies.dat")

    assert len(movies) == 3
    assert movies[1].title == "Toy Story (1995)"
    assert movies[1].genres == ("Animation", "Children's", "Comedy")
    assert movies[1].year == 1995
    assert movies[3].year is None  # 标题无年份


def test_parse_users(tmp_path: Path) -> None:
    write_metadata(tmp_path)
    users = parse_users(tmp_path / "users.dat")

    assert users[1].gender == "F"
    assert users[1].age == 1
    assert users[1].occupation == 10
    assert users[1].zipcode == "48067"
    profile = users[1].to_profile()
    assert profile["zipcode_prefix"] == "480"


def test_latin1_encoding_survives(tmp_path: Path) -> None:
    """官方文件含重音字符，用 utf-8 读会崩，本项确保 latin-1 生效。"""
    (tmp_path / "movies.dat").write_text(
        "1::M*A*S*H (1970)::Comedy\n2::Amélie (2001)::Romance\n", encoding="latin-1"
    )
    movies = parse_movies(tmp_path / "movies.dat")
    assert "Amélie" in movies[2].title


def test_load_ml1m_without_metadata_is_allowed(tmp_path: Path) -> None:
    write_ratings(tmp_path, [(1, 10, 5, 100), (2, 20, 4, 200)])
    dataset = load_ml1m(tmp_path, require_metadata=False)
    assert dataset.num_ratings == 2
    assert dataset.movies == {}
    assert dataset.users == {}


def test_load_ml1m_require_metadata_raises_when_missing(tmp_path: Path) -> None:
    write_ratings(tmp_path, [(1, 10, 5, 100)])
    with pytest.raises(IntegrityError, match="require_metadata"):
        load_ml1m(tmp_path, require_metadata=True)


def test_load_ml1m_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(IntegrityError):
        load_ml1m(tmp_path / "not-there")


def test_summary_counts_are_correct(tmp_path: Path) -> None:
    write_metadata(tmp_path)
    write_ratings(
        tmp_path,
        [(1, 10, 5, 100), (1, 20, 4, 150), (2, 10, 3, 200), (9, 20, 5, 900)],
    )
    dataset = load_ml1m(tmp_path)
    summary = dataset.summary()

    assert summary["num_ratings"] == 4
    assert summary["num_users"] == 3
    assert summary["num_items"] == 2
    assert summary["num_movies_meta"] == 3
    assert summary["num_users_meta"] == 2
    assert summary["timestamp_min"] == 100
    assert summary["timestamp_max"] == 900


def test_movie_dataclass_year_parsing() -> None:
    assert Movie(1, "Alien (1979)", ("Horror",)).year == 1979
    assert Movie(1, "Unknown", ()).year is None
    assert Movie(1, "Weird ()", ()).year is None


# --------------------------------------------------------------------------
# 缓存
# --------------------------------------------------------------------------


def test_cache_roundtrip_preserves_data(tmp_path: Path) -> None:
    write_metadata(tmp_path)
    write_ratings(tmp_path, [(1, 10, 5, 100), (2, 20, 4, 200)])
    dataset = load_ml1m(tmp_path)

    cache = tmp_path / "cache" / "ml1m_raw.pkl"
    save_cached(dataset, cache)
    restored = load_ml1m_cached(tmp_path, cache.parent)

    assert isinstance(restored, MovieLens1M)
    assert restored.summary() == dataset.summary()


def test_cache_rejected_when_content_is_wrong_type(tmp_path: Path) -> None:
    from recagent.data.ml1m import load_cached

    cache = tmp_path / "bad.pkl"
    cache.write_bytes(pickle.dumps({"not": "a dataset"}))
    with pytest.raises(ParseError, match="缓存内容类型异常"):
        load_cached(cache)


# --------------------------------------------------------------------------
# 真实数据（本地有则跑，没有则跳过）
# --------------------------------------------------------------------------


def find_real_ml1m() -> Path | None:
    for candidate in REAL_DATA_CANDIDATES:
        if (candidate / "ratings.dat").is_file():
            return candidate
    return None


real_data = find_real_ml1m()


@pytest.mark.skipif(real_data is None, reason="本地没有 ML-1M 原始数据")
def test_real_ml1m_shape_matches_official_statistics() -> None:
    """官方 ML-1M 的公认统计量，用来验证解析器没有漏读或错读。"""
    assert real_data is not None
    dataset = load_ml1m(real_data)

    assert dataset.num_ratings == 1_000_209
    assert len(dataset.movies) == 3_883
    assert len(dataset.users) == 6_040

    summary = dataset.summary()
    assert summary["num_users"] == 6_040
    # 电影表里有 3883 部，但被评过分的只有 3706 部
    assert summary["num_items"] == 3_706
    assert summary["timestamp_min"] == 956_703_932
    assert summary["timestamp_max"] == 1_046_454_590


@pytest.mark.skipif(real_data is None, reason="本地没有 ML-1M 原始数据")
def test_real_ml1m_user_gender_values_are_valid() -> None:
    assert real_data is not None
    dataset = load_ml1m(real_data, require_metadata=True)
    genders = {u.gender for u in dataset.users.values()}
    assert genders <= {"F", "M"}
