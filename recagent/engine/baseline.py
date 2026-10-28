"""确定性的热度与物品协同过滤推荐基线。"""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p, sqrt

from recagent.engine.store import Catalog


@dataclass(frozen=True, slots=True)
class Recommendation:
    item_id: int
    title: str
    genres: tuple[str, ...]
    score: float
    source: str


class BaselineRecommender:
    """只使用 Catalog 中的训练期用户与物品索引。"""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    def recommend(
        self,
        user_id: int | None,
        top_k: int = 10,
        genres: tuple[str, ...] = (),
        exclude_items: frozenset[int] = frozenset(),
    ) -> list[Recommendation]:
        if not 1 <= top_k <= 50:
            raise ValueError("top_k 必须在 1 到 50 之间")

        history = self.catalog.history.get(user_id, ()) if user_id is not None else ()
        seen = set(history)
        wanted_genres = set(genres)
        results: list[Recommendation] = []
        for item_id, popularity in self.catalog.popularity.items():
            if item_id in seen or item_id in exclude_items:
                continue
            movie = self.catalog.movies[item_id]
            if wanted_genres and not wanted_genres.issubset(movie.genres):
                continue
            similarity = 0.0
            candidate_users = self.catalog.item_users[item_id]
            for seen_id in seen:
                seen_users = self.catalog.item_users[seen_id]
                overlap = len(candidate_users & seen_users)
                if overlap:
                    similarity += overlap / sqrt(len(candidate_users) * len(seen_users))
            score = similarity + 0.01 * log1p(popularity) if seen else log1p(popularity)
            results.append(
                Recommendation(
                    item_id=item_id,
                    title=movie.title,
                    genres=movie.genres,
                    score=score,
                    source="itemcf" if seen and similarity > 0 else "popularity",
                )
            )
        return sorted(results, key=lambda item: (-item.score, item.item_id))[:top_k]
