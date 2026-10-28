"""确定性的热度与物品协同过滤推荐基线。"""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p, sqrt
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix

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
        self._cosine: csr_matrix | None = None

    def _prepare_cosine(self) -> csr_matrix:
        if self._cosine is not None:
            return self._cosine
        user_ids = sorted(self.catalog.history)
        rows: list[int] = []
        cols: list[int] = []
        for row, user_id in enumerate(user_ids):
            for item_id in set(self.catalog.history[user_id]):
                rows.append(row)
                cols.append(item_id)
        size = max(self.catalog.popularity) + 1
        matrix = csr_matrix(
            (np.ones(len(rows), dtype=np.float64), (rows, cols)),
            shape=(len(user_ids), size),
        )
        cooccurrence = (matrix.T @ matrix).tocoo()
        frequencies = np.zeros(size, dtype=np.float64)
        for item_id, viewers in self.catalog.item_users.items():
            frequencies[item_id] = len(viewers)
        cooccurrence.data /= np.sqrt(frequencies[cooccurrence.row] * frequencies[cooccurrence.col])
        self._cosine = cooccurrence.tocsr()
        return self._cosine

    def _score(self, user_id: int | None, item_id: int, mode: str) -> float:
        popularity = self.catalog.popularity[item_id]
        history = self.catalog.history.get(user_id, ()) if user_id is not None else ()
        if mode == "popularity" or not history:
            return log1p(popularity)
        candidate_users = self.catalog.item_users[item_id]
        similarity = sum(
            len(candidate_users & self.catalog.item_users[seen_id])
            / sqrt(len(candidate_users) * len(self.catalog.item_users[seen_id]))
            for seen_id in set(history)
        )
        return similarity + 0.01 * log1p(popularity)

    def score_candidates(
        self,
        user_id: int | None,
        candidate_ids: tuple[int, ...],
        mode: Literal["popularity", "itemcf"],
    ) -> list[Recommendation]:
        """为同一冻结候选集合排序，供离线评测使用。"""
        if mode not in {"popularity", "itemcf"}:
            raise ValueError(f"未知排序模式: {mode}")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("候选电影 ID 不得重复")
        results: list[Recommendation] = []
        has_history = bool(self.catalog.history.get(user_id, ())) if user_id is not None else False
        for item_id in candidate_ids:
            if item_id not in self.catalog.popularity:
                raise ValueError(f"候选电影不在训练期: {item_id}")
            movie = self.catalog.movies[item_id]
            results.append(
                Recommendation(
                    item_id=item_id,
                    title=movie.title,
                    genres=movie.genres,
                    score=self._score(user_id, item_id, mode),
                    source="itemcf" if mode == "itemcf" and has_history else "popularity",
                )
            )
        return sorted(results, key=lambda item: (-item.score, item.item_id))

    def score_candidates_batch(
        self,
        user_id: int | None,
        candidate_ids: tuple[int, ...],
        mode: Literal["popularity", "itemcf"],
    ) -> list[Recommendation]:
        """稀疏矩阵批量打分，数值定义与 score_candidates 相同。"""
        if mode not in {"popularity", "itemcf"}:
            raise ValueError(f"未知排序模式: {mode}")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("候选电影 ID 不得重复")
        if any(item_id not in self.catalog.popularity for item_id in candidate_ids):
            raise ValueError("候选电影不在训练期")
        history = self.catalog.history.get(user_id, ()) if user_id is not None else ()
        if mode == "popularity" or not history:
            return self.score_candidates(user_id, candidate_ids, mode="popularity")
        cosine = self._prepare_cosine()
        similarities = np.asarray(
            cosine[list(set(history)), :][:, list(candidate_ids)].sum(axis=0)
        ).ravel()
        results = []
        for item_id, similarity in zip(candidate_ids, similarities, strict=True):
            movie = self.catalog.movies[item_id]
            results.append(
                Recommendation(
                    item_id=item_id,
                    title=movie.title,
                    genres=movie.genres,
                    score=float(similarity) + 0.01 * log1p(self.catalog.popularity[item_id]),
                    source="itemcf",
                )
            )
        return sorted(results, key=lambda item: (-item.score, item.item_id))

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
        candidates: list[int] = []
        for item_id in self.catalog.popularity:
            if item_id in seen or item_id in exclude_items:
                continue
            movie = self.catalog.movies[item_id]
            if wanted_genres and not wanted_genres.issubset(movie.genres):
                continue
            candidates.append(item_id)
        return self.score_candidates(user_id, tuple(candidates), mode="itemcf")[:top_k]
