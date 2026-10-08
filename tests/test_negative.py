"""负采样测试。

重点验证两条硬约束（见 recagent/data/negative.py 模块文档）：
1. 候选池不得包含训练期尚不存在的物品
2. 评测负样本必须固定，否则不同模型不可比
"""

from __future__ import annotations

from collections import Counter

import pytest

from recagent.data.exceptions import ParseError
from recagent.data.loading import Interaction, group_by_user
from recagent.data.negative import (
    NegativeSampler,
    build_candidate_pool,
    item_frequencies,
)


def make_train(users: int = 6, items: int | None = None, per_user: int = 4) -> list[Interaction]:
    """构造确定性训练数据。

    用户 u 交互物品 (u-1)*per_user+1 .. u*per_user，
    即**每个用户有自己专属的物品段**，互不重叠。

    这样候选池大小 = users * per_user，用户未见过的物品数 = 池大小 - per_user，
    负采样才有足够空间。若各用户物品段重叠，池会小到无法采样。
    """
    if items is not None:
        per_user = max(1, items // users)
    return [
        Interaction(user_id=u, item_id=(u - 1) * per_user + k + 1, timestamp=100 + k * 10 + u)
        for u in range(1, users + 1)
        for k in range(per_user)
    ]


# --------------------------------------------------------------------------
# 候选池：约束一
# --------------------------------------------------------------------------


def test_candidate_pool_excludes_items_that_appear_only_later() -> None:
    """物品 99 首次出现在时间戳 500，cutoff=300 时必须被排除。"""
    train = [
        Interaction(1, 1, 100),
        Interaction(1, 2, 200),
    ]
    later = [Interaction(1, 99, 500)]

    pool_all = build_candidate_pool(train + later)
    assert 99 in pool_all

    pool_filtered = build_candidate_pool(train + later, cutoff=300)
    assert 99 not in pool_filtered
    assert pool_filtered == [1, 2]


def test_candidate_pool_uses_training_set_as_source_of_truth() -> None:
    """只传训练集时，池中只能有训练集里出现过的物品。"""
    train = [Interaction(1, 1, 100), Interaction(2, 2, 200)]
    assert build_candidate_pool(train) == [1, 2]


def test_empty_pool_rejected() -> None:
    with pytest.raises(ParseError):
        NegativeSampler([])


# --------------------------------------------------------------------------
# 训练负采样：排除已交互物品
# --------------------------------------------------------------------------


def test_train_negatives_never_include_seen_items() -> None:
    train = make_train(users=6, items=10, per_user=4)
    seen_by_user = {
        user_id: {i.item_id for i in items} for user_id, items in group_by_user(train).items()
    }
    sampler = NegativeSampler(build_candidate_pool(train), seed=7)
    samples = sampler.sample_for_train(train, num_neg=3)

    assert len(samples) == len(train)
    for sample in samples:
        assert len(sample.negative_items) == 3
        assert len(set(sample.negative_items)) == 3, "同组内负样本不应重复"
        assert not (set(sample.negative_items) & seen_by_user[sample.user_id])


def test_train_negatives_are_reproducible_with_same_seed() -> None:
    train = make_train()
    pool = build_candidate_pool(train)

    first = NegativeSampler(pool, seed=123).sample_for_train(train, num_neg=4)
    second = NegativeSampler(pool, seed=123).sample_for_train(train, num_neg=4)
    assert first == second


def test_different_seed_gives_different_negatives() -> None:
    train = make_train(items=50)
    pool = build_candidate_pool(train)

    first = NegativeSampler(pool, seed=1).sample_for_train(train, num_neg=8)
    second = NegativeSampler(pool, seed=2).sample_for_train(train, num_neg=8)
    assert first != second


def test_requesting_more_negatives_than_pool_raises() -> None:
    train = [Interaction(1, 1, 100), Interaction(1, 2, 200)]
    sampler = NegativeSampler(build_candidate_pool(train), seed=1)
    with pytest.raises(ParseError):
        sampler.sample_for_train(train, num_neg=5)


@pytest.mark.parametrize("num_neg", [0, -1])
def test_invalid_num_neg_rejected(num_neg: int) -> None:
    train = make_train()
    sampler = NegativeSampler(build_candidate_pool(train), seed=1)
    with pytest.raises(ParseError):
        sampler.sample_for_train(train, num_neg=num_neg)


# --------------------------------------------------------------------------
# 评测负样本：约束二（固定性 / 可比性）
# --------------------------------------------------------------------------


def test_eval_negatives_are_identical_across_repeated_calls() -> None:
    """这是"不同模型可比"的前提：同一批用户必须拿到同一组负样本。"""
    train = make_train(items=100)
    sampler = NegativeSampler(build_candidate_pool(train), seed=42)
    users = [1, 2, 3, 4, 5, 6]

    first = sampler.eval_negatives(users, num_neg=20)
    second = sampler.eval_negatives(users, num_neg=20)
    assert first == second


def test_eval_negatives_identical_across_two_sampler_instances() -> None:
    """两个独立 sampler（模拟两个模型）用同 seed 必须得到同一题集。"""
    train = make_train(items=100)
    pool = build_candidate_pool(train)
    users = list(range(1, 7))

    model_a = NegativeSampler(pool, seed=2024).eval_negatives(users, num_neg=15)
    model_b = NegativeSampler(pool, seed=2024).eval_negatives(users, num_neg=15)
    assert model_a == model_b


def test_eval_negatives_differ_for_different_users() -> None:
    train = make_train(items=200)
    sampler = NegativeSampler(build_candidate_pool(train), seed=9)
    negatives = sampler.eval_negatives([1, 2, 3], num_neg=30)
    assert negatives[1] != negatives[2]


def test_eval_negatives_cache_extends_for_new_users() -> None:
    """缓存已命中一部分用户后，再请求新用户应正常补采，不污染已有结果。"""
    train = make_train(items=200)
    sampler = NegativeSampler(build_candidate_pool(train), seed=5)

    first = sampler.eval_negatives([1, 2], num_neg=10)
    extended = sampler.eval_negatives([1, 2, 3], num_neg=10)

    assert extended[1] == first[1]
    assert extended[2] == first[2]
    assert len(extended[3]) == 10


# --------------------------------------------------------------------------
# popularity_aware
# --------------------------------------------------------------------------


def test_popularity_aware_favours_frequent_items() -> None:
    """热门物品被采为负样本的频次应显著高于冷门物品。"""
    train: list[Interaction] = []
    # 物品 1 非常热门（200 次），物品 2 冷门（5 次）
    for k in range(200):
        train.append(Interaction(user_id=1 + k % 10, item_id=1, timestamp=1000 + k))
    for k in range(5):
        train.append(Interaction(user_id=1 + k, item_id=2, timestamp=2000 + k))
    for item in range(3, 60):
        train.append(Interaction(user_id=item % 10 + 1, item_id=item, timestamp=3000 + item))

    pool = build_candidate_pool(train)
    frequencies = item_frequencies(train)

    sampler = NegativeSampler(pool, seed=11, strategy="popularity_aware", frequencies=frequencies)
    drawn = Counter()
    for _ in range(200):
        for item in sampler._draw(10, exclude=None):
            drawn[item] += 1

    assert drawn[1] > drawn[2], "热门物品应被更频繁地采为负样本"


def test_popularity_aware_requires_frequencies() -> None:
    train = make_train()
    with pytest.raises(ParseError):
        NegativeSampler(build_candidate_pool(train), strategy="popularity_aware")


def test_unknown_strategy_rejected() -> None:
    train = make_train()
    with pytest.raises(ParseError):
        NegativeSampler(build_candidate_pool(train), strategy="whatever")


def test_item_frequencies_counts_interactions() -> None:
    train = [Interaction(1, 1, 1), Interaction(2, 1, 2), Interaction(1, 2, 3)]
    frequencies = item_frequencies(train)
    assert frequencies[1] == 2
    assert frequencies[2] == 1


# --------------------------------------------------------------------------
# 重置
# --------------------------------------------------------------------------


def test_reset_restores_identical_stream() -> None:
    train = make_train(items=100)
    sampler = NegativeSampler(build_candidate_pool(train), seed=77)

    first = sampler.eval_negatives([1, 2, 3], num_neg=10)
    sampler.reset()
    again = sampler.eval_negatives([1, 2, 3], num_neg=10)
    assert first == again
