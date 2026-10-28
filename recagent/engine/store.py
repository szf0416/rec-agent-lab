"""从预处理产物构建只包含训练期信息的推荐索引。"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MovieInfo:
    item_id: int
    title: str
    genres: tuple[str, ...]


@dataclass(frozen=True)
class Catalog:
    movies: dict[int, MovieInfo]
    history: dict[int, tuple[int, ...]]
    item_users: dict[int, frozenset[int]]
    popularity: dict[int, int]

    @classmethod
    def load(cls, processed_dir: Path) -> Catalog:
        """只读取电影元数据和训练期交互。"""
        movies_path = processed_dir / "items.json"
        train_path = processed_dir / "interactions_train.csv"
        movies_raw = json.loads(movies_path.read_text(encoding="utf-8"))
        movies = {
            int(row["item_id"]): MovieInfo(
                item_id=int(row["item_id"]),
                title=str(row["title"]),
                genres=tuple(row["genres"]),
            )
            for row in movies_raw
        }
        history_rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
        item_users: dict[int, set[int]] = defaultdict(set)
        popularity: Counter[int] = Counter()
        with train_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                user_id = int(row["user_id"])
                item_id = int(row["item_id"])
                timestamp = int(row["timestamp"])
                if item_id not in movies:
                    raise ValueError(f"训练交互引用不存在的电影 ID: {item_id}")
                history_rows[user_id].append((timestamp, item_id))
                item_users[item_id].add(user_id)
                popularity[item_id] += 1
        history = {
            user_id: tuple(item_id for _, item_id in sorted(rows))
            for user_id, rows in history_rows.items()
        }
        return cls(
            movies=movies,
            history=history,
            item_users={item_id: frozenset(users) for item_id, users in item_users.items()},
            popularity=dict(popularity),
        )
