"""切分策略测试。

本文件最重要的部分是「时间戳分组不跨切分」，这是防未来信息泄漏的核心。
"""

from __future__ import annotations

import pytest

from recagent.data.exceptions import LeakageError, ParseError, SplitError
from recagent.data.loading import Interaction
from recagent.data.splits import (
    LeaveOneOutSplitter,
    TemporalSplitter,
    available_items_before,
    build_splitter,
    leakage_report,
    validate_no_leakage,
)


def make_interactions(
    *, users: int = 5, items: int = 10, per_user: int = 12, base_ts: int = 1_000_000
) -> list[Interaction]:
    """构造确定性交互：用户 u 的第 k 条交互时间戳为 base_ts + k * users + u。

    这样不同用户在同一 k 上时间戳不同，便于构造精确的时间戳分组测试。
    """
    out: list[Interaction] = []
    for u in range(1, users + 1):
        for k in range(per_user):
            out.append(
                Interaction(user_id=u, item_id=(k % items) + 1, timestamp=base_ts + k * users + u)
            )
    return out


# --------------------------------------------------------------------------
# 时间切分的基本行为
# --------------------------------------------------------------------------


def test_temporal_split_produces_three_nonempty_parts() -> None:
    split = TemporalSplitter(0.8, 0.1).split(make_interactions())
    assert split.train and split.valid and split.test
    assert len(split.train) + len(split.valid) + len(split.test) == 60


def test_temporal_split_is_strictly_ordered_in_time() -> None:
    split = TemporalSplitter(0.8, 0.1).split(make_interactions())
    assert max(i.timestamp for i in split.train) < min(i.timestamp for i in split.valid)
    assert max(i.timestamp for i in split.valid) < min(i.timestamp for i in split.test)


def test_every_interaction_lands_in_exactly_one_split() -> None:
    source = make_interactions()
    split = TemporalSplitter(0.8, 0.1).split(source)
    combined = split.train + split.valid + split.test
    assert len(combined) == len(source)
    assert {(i.user_id, i.item_id, i.timestamp) for i in combined} == {
        (i.user_id, i.item_id, i.timestamp) for i in source
    }


# --------------------------------------------------------------------------
# 核心：相同时间戳的交互组不得被切分点劈开
# --------------------------------------------------------------------------


def test_same_timestamp_group_not_split_across_train_and_valid() -> None:
    """同一时间戳的交互组必须整体归属同一侧。

    这是最容易写出 bug 的地方：朴素实现（按索引切、或切点左移一位）
    会把同一时刻的交互分到训练和验证两边，造成未来信息泄漏。

    注意数据规模与时间戳分布要留出腾挪空间：切点必须落在真正的组边界上，
    若时间戳过于集中（如只有两个大组），切点会被挤到末尾而无法切分。
    """
    interactions = [
        Interaction(user_id=index + 1, item_id=1, timestamp=1000) for index in range(100)
    ] + [Interaction(user_id=index + 1, item_id=2, timestamp=2000) for index in range(30)]

    for train_ratio, valid_ratio in ((0.8, 0.1), (0.6, 0.2)):
        split = TemporalSplitter(train_ratio, valid_ratio).split(interactions)
        validate_no_leakage(split)

        seen_in: dict[int, set[str]] = {}
        for label, part in (("train", split.train), ("valid", split.valid), ("test", split.test)):
            for interaction in part:
                seen_in.setdefault(interaction.timestamp, set()).add(label)

        for ts, labels in seen_in.items():
            assert len(labels) == 1, f"时间戳 {ts} 被劈开，出现在 {sorted(labels)} 中"

        assert len(split.train) == 100, "时间戳 1000 的整组应全部归入训练集"


def test_all_identical_timestamps_raises_instead_of_silently_leaking() -> None:
    """所有交互同一时间戳时无法切分，必须明确报错而不是产出跨组的切分。"""
    interactions = [Interaction(user_id=i + 1, item_id=1, timestamp=5000) for i in range(30)]
    with pytest.raises(SplitError, match="找不到合法的组边界"):
        TemporalSplitter(0.8, 0.1).split(interactions)


def test_temporal_split_never_leaks_on_dense_timestamps() -> None:
    """密集重复时间戳下的随机化校验，确保切分逻辑稳健。

    数据太少或时间戳过于集中时，切分可能无解（抛 SplitError）——
    这也是可接受的结果；关键是**绝不能产出泄漏的切分**。
    """
    import random

    rng = random.Random(20250928)
    interactions = [
        Interaction(
            user_id=rng.randint(1, 200), item_id=rng.randint(1, 300), timestamp=rng.randint(0, 400)
        )
        for _ in range(3000)
    ]

    succeeded = 0
    for train_ratio, valid_ratio in ((0.7, 0.15), (0.8, 0.1), (0.5, 0.25), (0.9, 0.05)):
        try:
            split = TemporalSplitter(train_ratio, valid_ratio).split(interactions)
        except SplitError:
            continue  # 无解，可接受
        validate_no_leakage(split)  # 一旦泄漏即抛 LeakageError，测试失败
        succeeded += 1

    assert succeeded >= 3, f"仅 {succeeded}/4 组比例切分成功，切分逻辑可能过于脆弱"


# --------------------------------------------------------------------------
# 参数校验
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("train_ratio", "valid_ratio"),
    [(0.0, 0.1), (1.0, 0.0), (1.1, 0.0), (0.9, 0.1), (0.8, 0.3)],
)
def test_invalid_ratios_rejected(train_ratio: float, valid_ratio: float) -> None:
    with pytest.raises(ParseError):
        TemporalSplitter(train_ratio, valid_ratio)


def test_too_few_interactions_rejected() -> None:
    with pytest.raises(SplitError):
        TemporalSplitter(0.8, 0.1).split([Interaction(1, 1, 1), Interaction(1, 2, 2)])


# --------------------------------------------------------------------------
# 留一法
# --------------------------------------------------------------------------


def test_leave_one_out_takes_last_interaction_as_test() -> None:
    interactions = make_interactions(users=3, items=10, per_user=6)
    split = LeaveOneOutSplitter(min_interactions=3).split(interactions)

    assert len(split.test) == 3
    assert len(split.valid) == 3
    assert len(split.train) == 12

    # 每个用户的测试项必须是其时间上最后一条
    for user_id in (1, 2, 3):
        user_all = sorted(
            (i for i in interactions if i.user_id == user_id), key=lambda i: i.timestamp
        )
        user_test = next(i for i in split.test if i.user_id == user_id)
        assert user_test.timestamp == user_all[-1].timestamp
        assert user_test.item_id == user_all[-1].item_id


def test_leave_one_out_skips_users_below_minimum() -> None:
    interactions = make_interactions(users=2, items=5, per_user=2)  # 每用户仅 2 条
    with pytest.raises(SplitError):
        LeaveOneOutSplitter(min_interactions=3).split(interactions)


def test_leave_one_out_rejects_min_interactions_below_three() -> None:
    with pytest.raises(ParseError):
        LeaveOneOutSplitter(min_interactions=2)


# --------------------------------------------------------------------------
# 泄漏检测能力本身要被验证
# --------------------------------------------------------------------------


def test_leakage_detector_catches_handcrafted_leak() -> None:
    """构造一个真泄漏，确认检查函数能抓到 —— 否则"检查"只是摆设。"""
    from recagent.data.loading import DatasetSplit

    shared = Interaction(user_id=1, item_id=1, timestamp=100)
    split = DatasetSplit(
        train=[Interaction(1, 2, 50), shared],
        valid=[shared],  # 同一条记录同时出现在两个集合
        test=[Interaction(1, 3, 200)],
    )
    with pytest.raises(LeakageError):
        validate_no_leakage(split)


def test_leakage_detector_catches_reversed_time_order() -> None:
    from recagent.data.loading import DatasetSplit

    split = DatasetSplit(
        train=[Interaction(1, 1, 300)],
        valid=[Interaction(1, 2, 200)],  # 时间上早于训练集
        test=[Interaction(1, 3, 400)],
    )
    with pytest.raises(LeakageError):
        validate_no_leakage(split)


def test_repeated_user_item_interaction_is_not_a_leak() -> None:
    """隐式反馈中同一用户重复消费同一物品是合法的，不能误报为泄漏。"""
    from recagent.data.loading import DatasetSplit

    split = DatasetSplit(
        train=[Interaction(1, 7, 100)],
        valid=[Interaction(1, 7, 200)],  # 同一 user/item，但时间戳不同
        test=[Interaction(1, 7, 300)],
    )
    validate_no_leakage(split)  # 不应抛错


def test_leakage_report_reports_unseen_items() -> None:
    from recagent.data.loading import DatasetSplit

    split = DatasetSplit(
        train=[Interaction(1, 1, 10), Interaction(2, 2, 20)],
        valid=[Interaction(1, 3, 30)],  # 物品 3 训练期未见
        test=[Interaction(2, 4, 40)],  # 物品 4 训练期未见
    )
    report = leakage_report(split)
    assert report["strictly_ordered"] is True
    assert report["items_unseen_in_train"] == 2
    assert set(report["unseen_examples"]) == {3, 4}


# --------------------------------------------------------------------------
# 候选池过滤
# --------------------------------------------------------------------------


def test_available_items_before_filters_future_items() -> None:
    interactions = [
        Interaction(1, 1, 100),
        Interaction(1, 2, 200),
        Interaction(1, 3, 300),
    ]
    assert available_items_before(interactions, cutoff=250) == {1, 2}
    assert available_items_before(interactions, cutoff=100) == set()
    assert available_items_before(interactions, cutoff=1000) == {1, 2, 3}


# --------------------------------------------------------------------------
# 工厂
# --------------------------------------------------------------------------


def test_build_splitter_by_name() -> None:
    assert isinstance(build_splitter("temporal"), TemporalSplitter)
    assert isinstance(build_splitter("leave_one_out"), LeaveOneOutSplitter)
    with pytest.raises(ParseError):
        build_splitter("random_split")


def test_temporal_splitter_is_deterministic() -> None:
    source = make_interactions()
    first = TemporalSplitter(0.8, 0.1).split(source)
    second = TemporalSplitter(0.8, 0.1).split(source)
    assert first == second
