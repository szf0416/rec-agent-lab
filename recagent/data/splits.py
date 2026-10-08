"""数据切分策略。

本模块承载项目最关键的正确性约束：**切分不得泄漏未来信息**。

两种策略：
- `TemporalSplitter`：按时间戳比例切分（主实验，见 ADR-003）
- `LeaveOneOutSplitter`：每用户留最后一条做测试（用于与论文基线对齐）

## 时间切分的边界处理（重点）

朴素实现 `cut = int(n * ratio)` 是错的：时间戳相同的交互组可能被从中间劈开，
导致同一时刻的交互既出现在训练集又出现在测试集。

本模块的做法：先把切点放到一个时间戳组的**边界**上，且一律选择
"把整组划给更靠后的一侧"，从而保证 `max(train.timestamp) < min(valid.timestamp)`。
这样做的代价是训练集比例会略微偏小，但**绝不会有泄漏**——这个取舍是刻意的。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from recagent.data.exceptions import LeakageError, ParseError, SplitError
from recagent.data.loading import (
    DatasetSplit,
    Interaction,
    first_seen_timestamps,
    group_by_user,
    sort_by_time,
)
from recagent.utils.logging import get_logger

logger = get_logger(__name__)


class Splitter(Protocol):
    """切分策略接口。见 docs/ARCHITECTURE.md 第 2.1 节。"""

    name: str

    def split(self, interactions: Iterable[Interaction]) -> DatasetSplit: ...


# --------------------------------------------------------------------------
# 时间切分
# --------------------------------------------------------------------------


class TemporalSplitter:
    """按时间戳比例切分，保证不跨越相同时间戳的交互组。

    Args:
        train_ratio: 训练集时间跨度占比。
        valid_ratio: 验证集时间跨度占比。测试集占比 = 1 - 二者之和。
    """

    name = "temporal"

    def __init__(self, train_ratio: float = 0.8, valid_ratio: float = 0.1) -> None:
        if not 0.0 < train_ratio < 1.0:
            raise ParseError(f"train_ratio 必须在 (0,1)，收到 {train_ratio}")
        if not 0.0 <= valid_ratio < 1.0:
            raise ParseError(f"valid_ratio 必须在 [0,1)，收到 {valid_ratio}")
        if train_ratio + valid_ratio >= 1.0:
            raise ParseError(
                f"train_ratio + valid_ratio 必须 < 1，收到 {train_ratio} + {valid_ratio}"
            )
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio

    # -- 内部：切点对齐 --
    @staticmethod
    def _first_strict_boundary(timestamps: list[int], cut: int) -> int:
        """把切点吸附到**离 cut 最近**的严格边界上。

        严格边界指位置 `i` 满足 `ts[i-1] < ts[i]`（i=n 也允许，表示切在末尾）。
        只有落在严格边界上，才能保证
        `max(左.timestamp) < min(右.timestamp)`——这是无泄漏的充要条件。

        为什么找"最近"而不是"向右第一个"：向右找会累积偏移量，
        在时间戳分组较大的数据上可能一路跑到末尾，导致明明能切分却报无解。

        两个实现陷阱：
        1. 「左移一位」是错的。切点落在某个时间戳组内部时，左移一位仍在同组内。
        2. 「只向右找」会在分组大、组数少的数据上浪费掉可用切点。
        """
        n = len(timestamps)
        cut = max(min(cut, n), 0)

        if cut == 0:
            i = 1
            while i < n and timestamps[i - 1] == timestamps[i]:
                i += 1
            return i if i < n else -1
        if cut == n:
            return n if timestamps[n - 1] > timestamps[n - 2] else -1

        if timestamps[cut - 1] < timestamps[cut]:
            return cut  # cut 本身就是严格边界

        # cut 落在某个时间戳组内部：向左找组起点
        group_start = cut
        while group_start > 0 and timestamps[group_start - 1] == timestamps[cut]:
            group_start -= 1

        left = group_start if group_start >= 1 and timestamps[group_start - 1] < timestamps[cut] else -1

        # 向右找组终点
        group_end = cut
        while group_end < n and timestamps[group_end] == timestamps[cut]:
            group_end += 1
        right = group_end if group_end > 0 and group_end < n else -1

        candidates = [c for c in (left, right) if c > 0]
        if not candidates:
            return -1
        # 距离相同时取左侧，避免训练集被不必要地缩小
        return min(candidates, key=lambda c: (abs(c - cut), c))

    def split(self, interactions: Iterable[Interaction]) -> DatasetSplit:
        ordered = sort_by_time(interactions)
        n = len(ordered)
        if n < 3:
            raise SplitError(f"交互数过少（{n} 条），无法做时间切分")

        timestamps = [i.timestamp for i in ordered]
        first_cut = self._first_strict_boundary(timestamps, int(n * self.train_ratio))
        second_cut = self._first_strict_boundary(
            timestamps, int(n * (self.train_ratio + self.valid_ratio))
        )
        second_cut = max(second_cut, first_cut)

        if first_cut <= 0 or first_cut >= n:
            raise SplitError(
                f"无法在 n={n} 条交互上按 train_ratio={self.train_ratio} 切出训练集"
                f"（切点算得 {first_cut}）。\n"
                f"原因：时间戳分组过大或过于集中，找不到合法的组边界。\n"
                f"建议：换用 LeaveOneOutSplitter，或检查该数据集的时间戳分布。"
            )
        if second_cut <= 0 or second_cut >= n:
            raise SplitError(
                f"无法在 n={n} 条交互上按 {self.train_ratio}/{self.valid_ratio} "
                f"切出验证集与测试集（切点算得 {second_cut}，落到序列末尾）。\n"
                f"建议：调整比例，或换用 LeaveOneOutSplitter。"
            )

        result = DatasetSplit(
            train=ordered[:first_cut],
            valid=ordered[first_cut:second_cut],
            test=ordered[second_cut:],
        )
        if not result.valid:
            logger.warning(
                "验证集为空（两个切点重合），n=%s。建议调整比例或改用 LeaveOneOutSplitter。",
                n,
            )
        validate_no_leakage(result)
        return result


# --------------------------------------------------------------------------
# 留一法切分
# --------------------------------------------------------------------------


class LeaveOneOutSplitter:
    """每用户留最后一条做测试、倒数第二条做验证，其余为训练。

    这是 SASRec / BERT4Rec 等论文的标准协议，用于与已有工作对齐。
    注意：它在严格意义上仍有轻微泄漏（用全局数据确定了"最后一条"），
    所以主实验用 TemporalSplitter，本策略只做对照。
    """

    name = "leave_one_out"

    def __init__(self, min_interactions: int = 3) -> None:
        if min_interactions < 3:
            raise ParseError(
                f"min_interactions 至少为 3（需容纳 训练/验证/测试 各一条），"
                f"收到 {min_interactions}"
            )
        self.min_interactions = min_interactions

    def split(self, interactions: Iterable[Interaction]) -> DatasetSplit:
        grouped = group_by_user(interactions)
        train: list[Interaction] = []
        valid: list[Interaction] = []
        test: list[Interaction] = []

        for user_id in sorted(grouped):
            items = grouped[user_id]
            if len(items) < self.min_interactions:
                continue
            train.extend(items[:-2])
            valid.append(items[-2])
            test.append(items[-1])

        if not train or not valid or not test:
            raise SplitError(
                "留一法切分后存在空集合。通常是交互过少或 min_interactions 过大导致。"
            )

        result = DatasetSplit(
            train=sort_by_time(train),
            valid=sort_by_time(valid),
            test=sort_by_time(test),
        )
        validate_no_leakage(result, check_temporal=False)
        return result


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------


def check_temporal_order(split: DatasetSplit) -> None:
    """校验三个集合在时间上严格有序（仅适用于时间切分）。

    这是防未来信息泄漏的**硬约束**：一旦违反，说明切分逻辑有 bug，
    必须立即失败而不是继续训练出一个虚高的指标。
    """
    if split.valid and split.train:
        train_max = max(i.timestamp for i in split.train)
        valid_min = min(i.timestamp for i in split.valid)
        if train_max >= valid_min:
            raise LeakageError(
                f"时间泄漏: max(train)={train_max} >= min(valid)={valid_min}。"
                f"切分逻辑有 bug，不允许继续。"
            )

    if split.test and split.valid:
        valid_max = max(i.timestamp for i in split.valid)
        test_min = min(i.timestamp for i in split.test)
        if valid_max >= test_min:
            raise LeakageError(
                f"时间泄漏: max(valid)={valid_max} >= min(test)={test_min}。"
                f"切分逻辑有 bug，不允许继续。"
            )


def check_disjoint(split: DatasetSplit) -> None:
    """校验三个集合无交集。

    注意：隐式反馈里 (user, item) 重复出现是**合法**的重复消费行为，
    不能按 (user, item) 判重。这里按三元组 (user, item, timestamp) 判重：
    同一条原始记录若出现在两个集合，才是真泄漏。
    """
    def _keys(items: list[Interaction]) -> set[tuple[int, int, int]]:
        return {(i.user_id, i.item_id, i.timestamp) for i in items}

    train_keys = _keys(split.train)
    valid_keys = _keys(split.valid)
    test_keys = _keys(split.test)

    for left_name, left, right_name, right in (
        ("train", train_keys, "valid", valid_keys),
        ("train", train_keys, "test", test_keys),
        ("valid", valid_keys, "test", test_keys),
    ):
        overlap = left & right
        if overlap:
            example = sorted(overlap)[:3]
            raise LeakageError(
                f"{left_name} 与 {right_name} 存在 {len(overlap)} 条完全相同的记录"
                f"（同一 user/item/timestamp），例如 {example}。这是真泄漏，不允许继续。"
            )


def validate_no_leakage(split: DatasetSplit, *, check_temporal: bool = True) -> None:
    """完整校验。任何一项不过就直接抛错，不做降级。"""
    if not split.train:
        raise SplitError("训练集为空")
    if check_temporal:
        check_temporal_order(split)
    check_disjoint(split)


def leakage_report(split: DatasetSplit) -> dict[str, object]:
    """生成泄漏检查报告，写入实验记录以便复查。"""
    def _bounds(items: list[Interaction]) -> tuple[int | None, int | None]:
        if not items:
            return (None, None)
        stamps = [i.timestamp for i in items]
        return (min(stamps), max(stamps))

    train_min, train_max = _bounds(split.train)
    valid_min, valid_max = _bounds(split.valid)
    test_min, test_max = _bounds(split.test)

    strictly_ordered = (
        train_max is not None
        and valid_min is not None
        and test_min is not None
        and train_max < valid_min
        and (valid_max or 0) < test_min
    )

    return {
        "sizes": split.summary(),
        "train_range": (train_min, train_max),
        "valid_range": (valid_min, valid_max),
        "test_range": (test_min, test_max),
        "strictly_ordered": bool(strictly_ordered),
        # 训练期未出现的物品数量。这些物品必须从候选负样本中排除，
        # 否则等于拿"未来才上架的商品"做负样本，属于未来信息泄漏。
        "items_unseen_in_train": unseen_item_count(split),
        "unseen_examples": sample_unseen_items(split),
    }


def unseen_item_count(split: DatasetSplit) -> int:
    """统计「训练期从未出现」但出现在验证/测试集的物品数量。"""
    train_items = {i.item_id for i in split.train}
    later_items = {i.item_id for i in (*split.valid, *split.test)}
    return len(later_items - train_items)


def sample_unseen_items(split: DatasetSplit, limit: int = 10) -> list[int]:
    """给出若干「训练期未出现」的物品示例，便于人工核查。"""
    train_items = {i.item_id for i in split.train}
    later_items = sorted({i.item_id for i in (*split.valid, *split.test)} - train_items)
    return later_items[:limit]


def available_items_before(interactions: Iterable[Interaction], cutoff: int) -> set[int]:
    """返回首次出现时间 < cutoff 的物品集合。

    时间切分下做负采样时，候选物品池必须用本函数从**训练集**算出。
    """
    seen = first_seen_timestamps(interactions)
    return {item_id for item_id, ts in seen.items() if ts < cutoff}


# --------------------------------------------------------------------------
# 便捷入口
# --------------------------------------------------------------------------

SPLITTERS: dict[str, type[TemporalSplitter] | type[LeaveOneOutSplitter]] = {
    TemporalSplitter.name: TemporalSplitter,
    LeaveOneOutSplitter.name: LeaveOneOutSplitter,
}


def build_splitter(
    strategy: str,
    *,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    min_interactions: int = 3,
) -> Splitter:
    """按配置名构造切分器。名字非法时列出可用选项。"""
    if strategy == TemporalSplitter.name:
        return TemporalSplitter(train_ratio=train_ratio, valid_ratio=valid_ratio)
    if strategy == LeaveOneOutSplitter.name:
        return LeaveOneOutSplitter(min_interactions=min_interactions)
    known = ", ".join(sorted(SPLITTERS))
    raise ParseError(f"未知切分策略 {strategy!r}，可用: {known}")
