"""只读推荐工具的训练期隔离、约束与参数边界。"""

import pytest

from recagent.agent.tools import RecommendationTools
from recagent.engine.store import Catalog, MovieInfo


@pytest.fixture
def tools():
    movies = {
        1: MovieInfo(1, "One", ("Sci-Fi",)),
        2: MovieInfo(2, "Two", ("Sci-Fi", "Comedy")),
        3: MovieInfo(3, "Three", ("Horror",)),
        4: MovieInfo(4, "Four", ("Sci-Fi",)),
        5: MovieInfo(5, "Five", ("Comedy",)),
        6: MovieInfo(6, "Future", ("Sci-Fi",)),
    }
    catalog = Catalog(
        movies=movies,
        history={1: (1,), 2: (1, 2), 3: (2, 4), 4: (4,), 5: (3,), 6: (5,)},
        item_users={
            1: frozenset({1, 2}),
            2: frozenset({2, 3}),
            3: frozenset({5}),
            4: frozenset({3, 4}),
            5: frozenset({6}),
        },
        popularity={1: 2, 2: 2, 3: 1, 4: 2, 5: 1},
    )
    return RecommendationTools(catalog)


def test_search_and_history_use_train_only(tools):
    assert [movie.item_id for movie in tools.search_movies("Future")] == []
    assert [movie.item_id for movie in tools.search_movies("Two")] == [2]
    assert [movie.item_id for movie in tools.get_user_history(1)] == [1]
    assert tools.get_user_history(999) == []


def test_recall_merges_routes_and_filters_seen_and_genres(tools):
    result = tools.recall_candidates(
        user_id=1, genres=("Sci-Fi",), exclude_genres=("Horror",), limit=3
    )
    assert {item.item_id for item in result} == {2, 4}
    assert len({item.item_id for item in result}) == len(result)
    assert all(item.source in {"popularity", "itemcf", "popularity,itemcf"} for item in result)


def test_rank_accepts_only_recalled_candidates(tools):
    result = tools.rank_candidates(1, (2, 4), allowed_ids=frozenset({2, 4}), top_k=2)
    assert {item.item_id for item in result} == {2, 4}
    with pytest.raises(ValueError):
        tools.rank_candidates(1, (2, 5), allowed_ids=frozenset({2, 4}), top_k=2)


@pytest.mark.parametrize("limit", [0, 21])
def test_tool_limits_are_bounded(tools, limit):
    with pytest.raises(ValueError):
        tools.search_movies("Two", limit=limit)
    with pytest.raises(ValueError):
        tools.recall_candidates(None, (), (), limit=limit)


def test_rank_rejects_too_many_candidates(tools):
    with pytest.raises(ValueError):
        tools.rank_candidates(None, tuple(range(51)), allowed_ids=frozenset(range(51)), top_k=1)
