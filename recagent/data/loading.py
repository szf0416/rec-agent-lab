"""交互记录与序列构造。

这一层是「原始文件」与「模型」之间的桥梁，刻意保持无状态、纯函数，
方便测试，也方便在切分方式之间切换而不用改模型代码。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from recagent.data.exceptions import ParseError
from recagent.data.ml1m import Rating


@dataclass(frozen=True, slots=True)
class Interaction:
    """一次用户-物品交互。

    主实验按隐式反馈处理时只用 (user_id, item_id, timestamp)；
    rating 保留用于分级相关性、以及"rating>=4 才算正反馈"的对照实验。
    """

    user_id: int
    item_id: int
    timestamp: int
    rating: int | None = None


class DatasetSplit(NamedTuple):
    """切分结果。三者必须满足时间上互不交叠（见 splits.validate_no_leakage）。"""

    train: list[Interaction]
    valid: list[Interaction]
    test: list[Interaction]

    def summary(self) -> dict[str, int]:
        return {"train": len(self.train), "valid": len(self.valid), "test": len(self.test)}


def from_ratings(ratings: Iterable[Rating]) -> list[Interaction]:
    """把 Rating 序列转成 Interaction 序列（丢弃评分值）。"""
    return [
        Interaction(user_id=r.user_id, item_id=r.item_id, timestamp=r.timestamp)
        for r in ratings
    ]


def from_ratings_positive(ratings: Iterable[Rating], threshold: int = 4) -> list[Interaction]:
    """只保留 rating >= threshold 的交互作为正反馈。

    用于对照实验：隐式反馈的"正样本"定义会显著影响召回模型的表现，
    这是个值得在报告里讨论的细节。
    """
    return [
        Interaction(user_id=r.user_id, item_id=r.item_id, timestamp=r.timestamp, rating=r.rating)
        for r in ratings
        if r.rating >= threshold
    ]


def sort_by_time(interactions: Iterable[Interaction]) -> list[Interaction]:
    """按 (timestamp, user_id, item_id) 稳定排序。

    三元组排序而不是只按 timestamp：同一时间戳内也要有确定顺序，
    否则同 seed 两次跑出的序列可能不同，"可复现"就成了空话。
    """
    return sorted(interactions, key=lambda i: (i.timestamp, i.user_id, i.item_id))


def group_by_user(interactions: Iterable[Interaction]) -> dict[int, list[Interaction]]:
    """按用户分组，组内按时间排序。"""
    grouped: dict[int, list[Interaction]] = defaultdict(list)
    for interaction in interactions:
        grouped[interaction.user_id].append(interaction)
    return {uid: sort_by_time(items) for uid, items in grouped.items()}


def item_universe(interactions: Iterable[Interaction]) -> set[int]:
    """出现过的物品集合。"""
    return {i.item_id for i in interactions}


def user_universe(interactions: Iterable[Interaction]) -> set[int]:
    """出现过的用户集合。"""
    return {i.user_id for i in interactions}


def first_seen_timestamps(interactions: Iterable[Interaction]) -> dict[int, int]:
    """每个物品首次出现的时间戳。

    时间切分下的负采样必须用它过滤：只能把"切分点之前已经存在"的物品
    当作候选负样本，否则等于用未来才上架的商品做负样本，属于未来信息泄漏。
    """
    first_seen: dict[int, int] = {}
    for interaction in interactions:
        current = first_seen.get(interaction.item_id)
        if current is None or interaction.timestamp < current:
            first_seen[interaction.item_id] = interaction.timestamp
    return first_seen


def filter_known_items(interactions: Iterable[Interaction], cutoff: int) -> list[Interaction]:
    """只保留首次出现时间 < cutoff 的物品所产生的交互。"""
    seen = first_seen_timestamps(interactions)
    return [i for i in interactions if seen[i.item_id] < cutoff]


def build_sequences(
    interactions: Iterable[Interaction],
    *,
    max_len: int,
    min_len: int = 5,
) -> dict[int, list[int]]:
    """构造用户行为序列（物品 id 列表），供 SASRec 使用。

    Args:
        interactions: 单个切分内的交互（**不要传入跨切分的数据**，
            否则序列会跨越切分边界，造成泄漏）。
        max_len: 只保留最近 max_len 个物品。
        min_len: 短于该长度的序列被丢弃。

    Returns:
        user_id -> 物品 id 列表（按时间升序，已被 max_len 截断）。
    """
    if max_len <= 0:
        raise ParseError(f"max_len 必须为正数，收到 {max_len}")

    sequences: dict[int, list[int]] = {}
    for user_id, items in group_by_user(interactions).items():
        # 同一物品可能被重复交互，序列里保留重复（反映真实行为频率），
        # 但相邻重复去掉，避免序列被单一物品刷满
        deduped: list[int] = []
        for interaction in items:
            if not deduped or deduped[-1] != interaction.item_id:
                deduped.append(interaction.item_id)
        if len(deduped) >= min_len:
            sequences[user_id] = deduped[-max_len:]
    return sequences


def sequence_stats(sequences: dict[int, list[int]]) -> dict[str, float]:
    """序列长度统计，用于监控截断比例。"""
    if not sequences:
        return {"num_users": 0, "mean_len": 0.0, "min_len": 0.0, "max_len": 0.0}
    lengths = [len(s) for s in sequences.values()]
    return {
        "num_users": float(len(lengths)),
        "mean_len": sum(lengths) / len(lengths),
        "min_len": float(min(lengths)),
        "max_len": float(max(lengths)),
    }


def to_dataframe(interactions: Sequence[Interaction]) -> "object":
    """转成 pandas DataFrame，供特征工程与快速探查使用。

    延迟导入 pandas：数据层的基础功能不依赖 pandas，用到再导。
    """
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "user_id": i.user_id,
                "item_id": i.item_id,
                "timestamp": i.timestamp,
                "rating": i.rating,
            }
            for i in interactions
        ]
    )
