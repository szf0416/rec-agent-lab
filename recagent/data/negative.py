"""负采样。

## 两条硬约束（这是本项目最容易出错的地方）

**约束一：候选池只能包含「训练期已存在」的物品。**
若用全量物品做候选池，等于拿"未来才上架的商品"当负样本，
测试集里的新物品会被当成已存在物品的干扰项，指标被人为压低，
而且训练分布与测试分布不一致。实现上用 `available_items_before(train, cutoff)`。

**约束二：训练与评测的负样本协议必须分开。**

| 场景 | 负样本 | 理由 |
| --- | --- | --- |
| 训练 | 每正样本配 N 个，**排除该用户已交互物品** | 让模型学会区分"喜欢"和"不喜欢" |
| 评测 | 每用户固定 K 个（通常 100），**对所有用户相同** | 固定候选集才能让不同模型的 NDCG 可比 |

评测负样本若每次随机重采，不同模型实际是在不同的题集上考试，
指标差异里混入了采样噪声，结论不可信。所以评测用固定 seed 采样并缓存。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from recagent.data.exceptions import ParseError
from recagent.data.loading import Interaction, group_by_user
from recagent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class NegativeSample:
    """一个正样本及其对应的负样本。"""

    user_id: int
    positive_item: int
    negative_items: tuple[int, ...]


def build_candidate_pool(
    train: Sequence[Interaction],
    cutoff: int | None = None,
) -> list[int]:
    """构造候选负样本池（排序后的物品 id 列表）。

    Args:
        train: 训练集交互。
        cutoff: 若给定，只保留首次出现时间 < cutoff 的物品（约束一）。

    Returns:
        升序排列的候选物品 id。
    """
    if cutoff is None:
        return sorted({i.item_id for i in train})

    # 延迟导入避免循环依赖
    from recagent.data.loading import first_seen_timestamps

    seen = first_seen_timestamps(train)
    return sorted(item_id for item_id, ts in seen.items() if ts < cutoff)


def item_frequencies(train: Iterable[Interaction]) -> Counter[int]:
    """物品交互频次，用于 popularity_aware 采样。"""
    return Counter(i.item_id for i in train)


class NegativeSampler:
    """负采样器。

    Args:
        pool: 候选物品池（必须已按约束一过滤）。
        seed: 随机种子，保证可复现。
        strategy: `uniform` 均匀采样；`popularity_aware` 按频次加权
            （更接近"难负样本"，通常能提升模型判别力）。
        frequencies: popularity_aware 需要；给定时必须覆盖 pool 中所有物品。
    """

    def __init__(
        self,
        pool: Sequence[int],
        *,
        seed: int = 42,
        strategy: str = "uniform",
        frequencies: Counter[int] | None = None,
    ) -> None:
        if not pool:
            raise ParseError("候选负样本池为空，无法采样")
        if strategy not in ("uniform", "popularity_aware"):
            raise ParseError(
                f"未知采样策略 {strategy!r}，可用: uniform, popularity_aware"
            )
        if strategy == "popularity_aware" and frequencies is None:
            raise ParseError("popularity_aware 策略需要提供 frequencies")

        self.pool = np.asarray(sorted(set(pool)), dtype=np.int64)
        self.seed = seed
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)

        self._pool_index: dict[int, int] = {
            int(item): idx for idx, item in enumerate(self.pool)
        }
        self._weights = self._build_weights(frequencies) if strategy != "uniform" else None
        # 评测用固定负样本缓存：key 为 num_neg，value 为 user -> 负样本元组
        self._eval_cache: dict[int, dict[int, tuple[int, ...]]] = {}

    def _build_weights(self, frequencies: Counter[int] | None) -> np.ndarray:
        assert frequencies is not None  # 由 __init__ 保证
        # 未出现的物品给权重 1，避免权重为 0 导致永远采不到
        raw = np.array(
            [max(frequencies.get(int(item), 0), 1) for item in self.pool], dtype=np.float64
        )
        # 次线性缩放：直接用原始频次会让头部物品主导采样，模型学不到长尾区分能力
        weights = np.power(raw, 0.75)
        # 显式标注：numpy 的除法返回 Any，不加注解会触发 no-any-return
        normalized: np.ndarray = weights / weights.sum()
        return normalized

    def _draw(self, size: int, exclude: set[int] | None) -> list[int]:
        """从池中无放回抽 size 个，可排除若干物品。"""
        if not exclude:
            indices = np.arange(len(self.pool), dtype=np.int64)
        else:
            # 向量化排除，避免在百万级池上做 Python 循环
            excluded = np.fromiter(exclude, dtype=np.int64, count=len(exclude))
            mask = ~np.isin(self.pool, excluded)
            indices = np.flatnonzero(mask).astype(np.int64)

        if len(indices) < size:
            raise ParseError(
                f"候选池可用物品数 {len(indices)} 少于请求的负样本数 {size}。"
                f"请减小 negative_sampling 配置或检查数据规模。"
            )

        if self._weights is None:
            chosen = self.rng.choice(indices, size=size, replace=False)
        else:
            probs = self._weights[indices]
            probs = probs / probs.sum()
            chosen = self.rng.choice(indices, size=size, replace=False, p=probs)
        return [int(self.pool[i]) for i in chosen]

    def sample_for_train(
        self,
        train: Sequence[Interaction],
        *,
        num_neg: int = 4,
    ) -> list[NegativeSample]:
        """为训练集每个交互生成负样本（约束二：排除该用户已交互物品）。"""
        if num_neg < 1:
            raise ParseError(f"num_neg 至少为 1，收到 {num_neg}")

        seen_by_user: dict[int, set[int]] = {
            user_id: {i.item_id for i in items}
            for user_id, items in group_by_user(train).items()
        }

        samples: list[NegativeSample] = []
        for interaction in train:
            negatives = self._draw(num_neg, seen_by_user.get(interaction.user_id, set()))
            samples.append(
                NegativeSample(
                    user_id=interaction.user_id,
                    positive_item=interaction.item_id,
                    negative_items=tuple(negatives),
                )
            )
        logger.info("生成 %s 组训练负样本（每组 %s 个）", len(samples), num_neg)
        return samples

    def eval_negatives(
        self,
        users: Iterable[int],
        *,
        num_neg: int = 100,
    ) -> dict[int, tuple[int, ...]]:
        """为评测生成**固定**负样本（约束二）。

        同一个 sampler 实例对同一批用户重复调用会命中缓存，返回完全相同的负样本，
        保证不同模型在同一题集上比较。
        """
        if num_neg < 1:
            raise ParseError(f"num_neg 至少为 1，收到 {num_neg}")
        if num_neg in self._eval_cache:
            cached = self._eval_cache[num_neg]
            missing = [u for u in users if u not in cached]
            if not missing:
                return {u: cached[u] for u in users}
            logger.debug("评测负样本缓存缺少 %s 个用户，补采", len(missing))
            for user_id in missing:
                cached[user_id] = tuple(self._draw(num_neg, exclude=None))
            return {u: cached[u] for u in users}

        sampled = {user_id: tuple(self._draw(num_neg, exclude=None)) for user_id in users}
        self._eval_cache[num_neg] = sampled
        logger.info("生成 %s 个用户的固定评测负样本（每人 %s 个）", len(sampled), num_neg)
        return sampled

    def reset(self, seed: int | None = None) -> None:
        """重置随机状态与缓存。用于消融实验中确保各配置起点一致。"""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self._eval_cache.clear()
