"""MovieLens-1M 原始数据解析。

原始格式（`::` 分隔，latin-1 编码）：
    ratings.dat : UserID::MovieID::Rating::Timestamp
    movies.dat  : MovieID::Title::Genres
    users.dat   : UserID::Gender::Age::Occupation::Zip-code

关于编码：官方文件是 latin-1，含法语/西班牙语字符（如 M*A*S*H 的变音符号）。
用 utf-8 读会炸，这是新手最常见的翻车点之一。

关于评级：本模块保留 rating 字段，但**主实验按隐式反馈处理**（忽略评分值，
只当作"发生过交互"）。评分值可用于构造两类额外实验：
- 把 rating >= 4 视为正反馈（更严格的隐式反馈定义）
- 排序任务的分级相关性
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recagent.data.exceptions import IntegrityError, ParseError
from recagent.utils.logging import get_logger

logger = get_logger(__name__)

RAW_FILES = ("ratings.dat", "movies.dat", "users.dat")
AGE_BUCKETS = (1, 18, 25, 35, 45, 50, 56)
GENDERS = ("F", "M")


@dataclass(frozen=True, slots=True)
class Rating:
    """一条评分记录。"""

    user_id: int
    item_id: int
    rating: int
    timestamp: int


@dataclass(frozen=True, slots=True)
class Movie:
    """一部电影。"""

    item_id: int
    title: str
    genres: tuple[str, ...]

    @property
    def year(self) -> int | None:
        """从标题中解析上映年份，如 'Toy Story (1995)' -> 1995。"""
        if "(" not in self.title or not self.title.endswith(")"):
            return None
        tail = self.title[self.title.rfind("(") + 1 : -1]
        return int(tail) if tail.isdigit() else None


@dataclass(frozen=True, slots=True)
class User:
    """一个用户的人口统计信息。S4 生成 LLM 用户画像时直接使用。"""

    user_id: int
    gender: str
    age: int
    occupation: int
    zipcode: str

    def to_profile(self) -> dict[str, Any]:
        """转成可喂给 LLM 的画像字典。"""
        return {
            "user_id": self.user_id,
            "gender": self.gender,
            "age": self.age,
            "occupation": self.occupation,
            "zipcode_prefix": self.zipcode[:3],
        }


@dataclass(frozen=True)
class MovieLens1M:
    """解析后的 ML-1M 全量数据。"""

    ratings: tuple[Rating, ...]
    movies: dict[int, Movie]
    users: dict[int, User]
    raw_dir: Path

    @property
    def num_ratings(self) -> int:
        return len(self.ratings)

    @property
    def num_users(self) -> int:
        return len({r.user_id for r in self.ratings})

    @property
    def num_items(self) -> int:
        return len({r.item_id for r in self.ratings})

    def timestamp_range(self) -> tuple[int, int]:
        if not self.ratings:
            raise ParseError("ratings 为空，无法取时间范围")
        stamps = [r.timestamp for r in self.ratings]
        return min(stamps), max(stamps)

    def summary(self) -> dict[str, Any]:
        lo, hi = self.timestamp_range()
        return {
            "num_ratings": self.num_ratings,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "num_movies_meta": len(self.movies),
            "num_users_meta": len(self.users),
            "timestamp_min": lo,
            "timestamp_max": hi,
        }


def _iter_lines(path: Path, expected_fields: int) -> Iterator[list[str]]:
    """按行读取并切分，格式不符时报出具体行号。

    报行号很重要：100 万行的文件里说"格式错误"而不说在哪，是没法调试的。
    """
    with path.open("r", encoding="latin-1") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not line:
                continue
            fields = line.split("::")
            if len(fields) != expected_fields:
                raise ParseError(
                    f"{path.name} 第 {lineno} 行字段数为 {len(fields)}，"
                    f"期望 {expected_fields}: {line[:120]!r}"
                )
            yield fields


def parse_ratings(path: Path) -> list[Rating]:
    """解析 ratings.dat。"""
    if not path.is_file():
        raise IntegrityError(f"缺少必需文件: {path}")
    ratings: list[Rating] = []
    for fields in _iter_lines(path, 4):
        try:
            ratings.append(
                Rating(
                    user_id=int(fields[0]),
                    item_id=int(fields[1]),
                    rating=int(fields[2]),
                    timestamp=int(fields[3]),
                )
            )
        except ValueError as exc:
            raise ParseError(f"{path.name} 存在非法数值: {fields}") from exc

    if not ratings:
        raise ParseError(f"{path.name} 解析结果为空")
    logger.debug("解析 %s 条评分", len(ratings))
    return ratings


def parse_movies(path: Path) -> dict[int, Movie]:
    """解析 movies.dat，返回 item_id -> Movie。"""
    movies: dict[int, Movie] = {}
    for fields in _iter_lines(path, 3):
        try:
            item_id = int(fields[0])
        except ValueError as exc:
            raise ParseError(f"{path.name} 存在非法 MovieID: {fields[0]!r}") from exc
        genres = tuple(g for g in fields[2].split("|") if g)
        movies[item_id] = Movie(item_id=item_id, title=fields[1], genres=genres)
    logger.debug("解析 %s 部电影", len(movies))
    return movies


def parse_users(path: Path) -> dict[int, User]:
    """解析 users.dat，返回 user_id -> User。"""
    users: dict[int, User] = {}
    for fields in _iter_lines(path, 5):
        try:
            user_id = int(fields[0])
            age = int(fields[2])
            occupation = int(fields[3])
        except ValueError as exc:
            raise ParseError(f"{path.name} 存在非法数值: {fields}") from exc
        users[user_id] = User(
            user_id=user_id,
            gender=fields[1],
            age=age,
            occupation=occupation,
            zipcode=fields[4],
        )
    logger.debug("解析 %s 个用户", len(users))
    return users


def load_ml1m(
    raw_dir: Path,
    *,
    require_metadata: bool = False,
) -> MovieLens1M:
    """从原始目录加载 ML-1M。

    Args:
        raw_dir: 含 ratings.dat / movies.dat / users.dat 的目录。
        require_metadata: 为 True 时 movies/users 缺失即报错；
            为 False 时允许缺失（只加载评分，便于做纯协同过滤实验）。
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise IntegrityError(f"原始数据目录不存在: {raw_dir}")

    ratings = parse_ratings(raw_dir / "ratings.dat")

    movies: dict[int, Movie] = {}
    users: dict[int, User] = {}
    for name, parser, target in (
        ("movies.dat", parse_movies, movies),
        ("users.dat", parse_users, users),
    ):
        path = raw_dir / name
        if path.is_file():
            target.update(parser(path))
        elif require_metadata:
            raise IntegrityError(f"require_metadata=True 但缺少 {name}（{raw_dir}）")
        else:
            logger.warning("缺少 %s，相关特征将不可用", name)

    dataset = MovieLens1M(
        ratings=tuple(ratings), movies=movies, users=users, raw_dir=raw_dir
    )
    logger.info("ML-1M 加载完成: %s", dataset.summary())
    return dataset


# --------------------------------------------------------------------------
# 缓存：原始数据解析一次约数秒，跨脚本复用可显著加快迭代
# --------------------------------------------------------------------------


def cache_path(processed_dir: Path, name: str = "ml1m_raw.pkl") -> Path:
    return Path(processed_dir) / name


def save_cached(dataset: MovieLens1M, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("已缓存解析结果 -> %s", path)


def load_cached(path: Path) -> MovieLens1M:
    with Path(path).open("rb") as handle:
        obj = pickle.load(handle)  # noqa: S301 - 仅读取本项目自己写的缓存
    if not isinstance(obj, MovieLens1M):
        raise ParseError(f"缓存内容类型异常: {type(obj)!r}")
    return obj


def load_ml1m_cached(
    raw_dir: Path, processed_dir: Path, *, use_cache: bool = True
) -> MovieLens1M:
    """优先读缓存，缓存不存在或原始数据更新过则重新解析。

    Args:
        raw_dir: 原始数据目录（只读，本函数不会往里写任何东西）。
        processed_dir: 缓存落地目录。
        use_cache: 为 False 时既不读也不写缓存，直接解析。
            用于 --dry-run 等不应产生副作用的场景。
    """
    raw_dir = Path(raw_dir)
    if not use_cache:
        return load_ml1m(raw_dir)

    cache = cache_path(processed_dir)

    if cache.is_file():
        cache_mtime = cache.stat().st_mtime
        newest_raw = max(
            (raw_dir / name).stat().st_mtime
            for name in RAW_FILES
            if (raw_dir / name).is_file()
        )
        if cache_mtime >= newest_raw:
            logger.info("命中缓存 %s", cache)
            return load_cached(cache)
        logger.info("缓存已过期（原始数据更新），重新解析")

    dataset = load_ml1m(raw_dir)
    save_cached(dataset, cache)
    return dataset


def export_metadata(dataset: MovieLens1M, processed_dir: Path) -> dict[str, Path]:
    """导出 items / users 元数据为 JSON，供后续特征工程与 LLM 画像使用。"""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    items_path = processed_dir / "items.json"
    users_path = processed_dir / "users.json"

    items = [
        {
            "item_id": m.item_id,
            "title": m.title,
            "genres": list(m.genres),
            "year": m.year,
        }
        for m in sorted(dataset.movies.values(), key=lambda m: m.item_id)
    ]
    users = [u.to_profile() for u in sorted(dataset.users.values(), key=lambda u: u.user_id)]

    items_path.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    users_path.write_text(json.dumps(users, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("元数据导出: %s 部电影, %s 个用户", len(items), len(users))
    return {"items": items_path, "users": users_path}
