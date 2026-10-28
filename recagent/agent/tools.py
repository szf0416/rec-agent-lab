"""对训练期推荐索引开放只读、有限边界的工具。"""

from __future__ import annotations

from collections import defaultdict

from recagent.engine.baseline import BaselineRecommender, Recommendation
from recagent.engine.store import Catalog, MovieInfo


class RecommendationTools:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.recommender = BaselineRecommender(catalog)

    @staticmethod
    def _check_limit(limit: int) -> None:
        if not 1 <= limit <= 20:
            raise ValueError("limit 必须在 1 到 20 之间")

    def search_movies(self, query: str, limit: int = 10) -> list[MovieInfo]:
        self._check_limit(limit)
        query = query.strip().casefold()
        if not query or len(query) > 100:
            raise ValueError("搜索词长度必须在 1 到 100 之间")
        matches = [
            self.catalog.movies[item_id]
            for item_id in sorted(self.catalog.popularity)
            if query in self.catalog.movies[item_id].title.casefold()
            or any(query in genre.casefold() for genre in self.catalog.movies[item_id].genres)
        ]
        return matches[:limit]

    def get_user_history(self, user_id: int, limit: int = 10) -> list[MovieInfo]:
        self._check_limit(limit)
        if user_id < 1:
            raise ValueError("user_id 必须为正整数")
        history = self.catalog.history.get(user_id, ())[-limit:]
        return [self.catalog.movies[item_id] for item_id in history]

    def recall_candidates(
        self,
        user_id: int | None,
        genres: tuple[str, ...],
        exclude_genres: tuple[str, ...],
        limit: int = 20,
    ) -> list[Recommendation]:
        self._check_limit(limit)
        if user_id is not None and user_id < 1:
            raise ValueError("user_id 必须为正整数")
        if len(genres) > 5 or len(exclude_genres) > 5:
            raise ValueError("题材约束最多各 5 项")
        wanted = set(genres)
        excluded = set(exclude_genres)
        seen = set(self.catalog.history.get(user_id, ())) if user_id is not None else set()
        candidate_ids = tuple(
            item_id
            for item_id in self.catalog.popularity
            if item_id not in seen
            and wanted.issubset(self.catalog.movies[item_id].genres)
            and not excluded.intersection(self.catalog.movies[item_id].genres)
        )
        if not candidate_ids:
            return []
        scores: dict[int, float] = defaultdict(float)
        sources: dict[int, set[str]] = defaultdict(set)
        for mode in ("popularity", "itemcf"):
            route = self.recommender.score_candidates_batch(user_id, candidate_ids, mode=mode)[
                :limit
            ]
            for rank, item in enumerate(route, start=1):
                scores[item.item_id] += 1 / (60 + rank)
                sources[item.item_id].add(mode)
        selected = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))[:limit]
        return [
            Recommendation(
                item_id=item_id,
                title=self.catalog.movies[item_id].title,
                genres=self.catalog.movies[item_id].genres,
                score=scores[item_id],
                source=",".join(
                    route for route in ("popularity", "itemcf") if route in sources[item_id]
                ),
            )
            for item_id in selected
        ]

    def rank_candidates(
        self,
        user_id: int | None,
        candidate_ids: tuple[int, ...],
        allowed_ids: frozenset[int],
        top_k: int,
    ) -> list[Recommendation]:
        self._check_limit(top_k)
        if len(candidate_ids) > 50:
            raise ValueError("候选电影最多 50 部")
        if not set(candidate_ids).issubset(allowed_ids):
            raise ValueError("排序输入包含未召回的电影")
        if not candidate_ids:
            return []
        return self.recommender.score_candidates_batch(user_id, candidate_ids, mode="itemcf")[
            :top_k
        ]
