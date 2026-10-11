"""序列召回数据集：把预处理产物变成训练循环能吃的 batch。

## 一个必须避开的陷阱

`sequences_train.npz` 里的序列是「该用户训练期最近 50 条交互」，
**末项就是下一个物品本身**。如果直接拿整个序列当输入、再预测"下一个"，
模型只要抄输入的最后一项就能拿满分 —— loss 会低得离谱，但什么都没学到。

所以构造训练样本时必须**排除末项**：
    输入 = seq[:-1]，标签 = seq[-1]
并且对序列内每个位置都这样做（位置 i 预测 i+1），扩大样本量。

## 验证集的可行性

valid 正样本的时间戳全部晚于该用户训练期最后一条（实测 23635/23635），
所以用「完整训练序列」预测 valid 物品是合法的，不存在时间泄漏。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from recagent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetStats:
    """数据集规模，用于日志与实验记录。"""

    num_train_samples: int
    num_eval_users: int
    num_candidates: int
    max_seq_len: int
    num_items: int

    def as_dict(self) -> dict[str, int]:
        return {
            "num_train_samples": self.num_train_samples,
            "num_eval_users": self.num_eval_users,
            "num_candidates": self.num_candidates,
            "max_seq_len": self.max_seq_len,
            "num_items": self.num_items,
        }


def load_sequences(npz_path: Path) -> dict[int, list[int]]:
    """读取 sequences_train.npz，还原成 user_id -> 物品列表。"""
    payload = np.load(npz_path)
    user_ids = payload["user_ids"]
    items = payload["items"]
    lengths = payload["lengths"]
    offsets = np.concatenate([[0], np.cumsum(lengths)])

    return {
        int(user_id): items[offsets[i] : offsets[i + 1]].tolist()
        for i, user_id in enumerate(user_ids)
    }


class SequenceRecallDataset(Dataset[dict[str, torch.Tensor]]):
    """「1 正 + N 负」的候选集数据集。

    batch 结构（训练循环约定的格式）::

        {"user_seq": LongTensor(B, L),       # 用户历史（已 padding）
         "candidates": LongTensor(B, C),     # 候选物品（第 0 列是正样本）
         "labels": FloatTensor(B, C),        # one-hot，正样本位为 1
         "seq_len": LongTensor(B,)}

    Args:
        sequences: user_id -> 物品序列（时间升序）。
        positives: user_id -> 正样本物品列表。
        negatives: user_id -> 固定负样本列表。训练集可传 None（动态采样）。
        num_items: 物品总数，用于 embedding 大小与负采样范围。
        max_seq_len: 序列截断长度。
        num_neg: 每个样本配多少负样本。
        negative_pool: 动态负采样的候选池（训练期已存在的物品）。
        seed: 负采样随机种子（保证可复现）。
        min_prefix: 训练样本的最短前缀，太短的序列信息量不足。
    """

    def __init__(
        self,
        *,
        sequences: dict[int, list[int]],
        positives: dict[int, list[int]],
        negatives: dict[int, list[int]] | None = None,
        num_items: int,
        max_seq_len: int = 50,
        num_neg: int = 100,
        negative_pool: Sequence[int] | None = None,
        seed: int = 42,
        min_prefix: int = 3,
        mode: str = "train",
    ) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(f"mode 必须是 train/eval，收到 {mode!r}")

        self.mode = mode
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.num_neg = num_neg
        self.min_prefix = min_prefix
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        if mode == "train":
            if negative_pool is None:
                raise ValueError("mode=train 需要提供 negative_pool")
            self.negative_pool = np.asarray(sorted(set(negative_pool)), dtype=np.int64)
            self._build_train_samples(sequences)
        else:
            if negatives is None:
                raise ValueError("mode=eval 需要提供固定 negatives")
            self._build_eval_samples(sequences, positives, negatives)

    # ------------------------------------------------------------------
    # 样本构造
    # ------------------------------------------------------------------

    def _build_train_samples(self, sequences: dict[int, list[int]]) -> None:
        """展开成 (前缀, 标签) 对。

        对序列里每个位置 i（i >= min_prefix-1）都生成一个样本：
            prefix = seq[: i]，label = seq[i]
        这样 5400 个用户能产出几十万训练样本，而不是每人只有 1 个。

        **输入里绝不含 label** —— 这是本类存在的核心原因。
        """
        prefixes: list[list[int]] = []
        labels: list[int] = []
        for items in sequences.values():
            # 从 min_prefix 个物品开始，逐步扩展前缀
            for end in range(self.min_prefix, len(items)):
                prefix = items[max(0, end - self.max_seq_len) : end]
                if len(prefix) < self.min_prefix:
                    continue
                prefixes.append(prefix)
                labels.append(items[end])

        self._prefixes = prefixes
        self._labels = np.asarray(labels, dtype=np.int64)
        logger.info(
            "训练样本 %s 对（来自 %s 个用户序列）", len(prefixes), len(sequences)
        )

    def _build_eval_samples(
        self,
        sequences: dict[int, list[int]],
        positives: dict[int, list[int]],
        negatives: dict[int, list[int]],
    ) -> None:
        """评测样本：1 个正样本 + N 个固定负样本。

        负样本由预处理阶段冻结（全模型共用），保证不同模型的指标可比。

        **关键约束：不同样本的候选集长度必须完全一致。**
        固定负样本不排除用户已交互物品，所以它可能与正样本重叠。
        若简单地把重叠项删掉，不同样本的候选数就会不同（101 / 100 混杂），
        而 `torch.stack` 要求等长 —— 实测会在 collate 阶段直接报错。
        正确做法是跳过与正样本冲突的负样本，再从后面补齐，保持长度恒定。
        """
        self._eval_user_ids: list[int] = []
        self._eval_seqs: list[list[int]] = []
        self._eval_candidates: list[np.ndarray] = []

        requested = self.num_neg

        for user_id, neg_items in negatives.items():
            sequence = sequences.get(user_id)
            if sequence is None:
                continue
            positive_items = positives.get(user_id) or []
            if not positive_items:
                continue

            # 一位用户可能有多个 valid/test 交互，每个都作为一个评测样本
            for positive in positive_items:
                positive = int(positive)
                picked: list[int] = []
                for item in neg_items:
                    item = int(item)
                    if item != positive and item not in picked:
                        picked.append(item)
                    if len(picked) == requested:
                        break

                if len(picked) != requested:
                    # 候选池不够就跳过这个样本，而不是产出长度不齐的 batch
                    logger.debug(
                        "用户 %s 的正样本 %s 可用负样本不足 %s 个，跳过",
                        user_id,
                        positive,
                        requested,
                    )
                    continue

                candidates = np.asarray([positive, *picked], dtype=np.int64)
                self._eval_user_ids.append(int(user_id))
                self._eval_seqs.append(sequence)
                self._eval_candidates.append(candidates)

        if not self._eval_seqs:
            raise ValueError("评测集为空：检查 sequences 与 negatives 的用户是否匹配")

        lengths = {len(c) for c in self._eval_candidates}
        if len(lengths) != 1:
            raise ValueError(f"评测候选集长度不一致: {sorted(lengths)}，会让 collate 失败")

        logger.info(
            "评测样本 %s 条（%s 个用户，每条 %s 个候选）",
            len(self._eval_seqs),
            len(set(self._eval_user_ids)),
            lengths.pop(),
        )

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self._prefixes)
        return len(self._eval_seqs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.mode == "train":
            return self._train_item(index)
        return self._eval_item(index)

    def _pack_sequence(self, sequence: list[int]) -> tuple[np.ndarray, int]:
        """左 padding：把序列补到 max_seq_len 并返回真实长度。

        为什么左 padding：模型取"最近若干项"，右对齐后取尾部即可，
        不必额外处理 mask 的位置对齐问题。
        """
        tail = sequence[-self.max_seq_len :]
        length = len(tail)
        padded = np.zeros(self.max_seq_len, dtype=np.int64)
        padded[-length:] = tail
        return padded, length

    def _train_item(self, index: int) -> dict[str, torch.Tensor]:
        prefix = self._prefixes[index]
        label = int(self._labels[index])

        padded, length = self._pack_sequence(prefix)

        # 动态负采样：排除标签本身，避免把正样本当负样本
        negatives: list[int] = []
        attempts = 0
        while len(negatives) < self.num_neg and attempts < self.num_neg * 10:
            candidate = int(self._rng.choice(self.negative_pool))
            attempts += 1
            if candidate != label and candidate not in negatives:
                negatives.append(candidate)

        candidates = np.asarray([label, *negatives], dtype=np.int64)
        labels = np.zeros(len(candidates), dtype=np.float32)
        labels[0] = 1.0  # 正样本固定在第 0 列

        return {
            "user_seq": torch.from_numpy(padded),
            "seq_len": torch.tensor(length, dtype=torch.long),
            "candidates": torch.from_numpy(candidates),
            "labels": torch.from_numpy(labels),
        }

    def _eval_item(self, index: int) -> dict[str, torch.Tensor]:
        padded, length = self._pack_sequence(self._eval_seqs[index])
        candidates = self._eval_candidates[index]
        labels = np.zeros(len(candidates), dtype=np.float32)
        labels[0] = 1.0

        return {
            "user_seq": torch.from_numpy(padded),
            "seq_len": torch.tensor(length, dtype=torch.long),
            "candidates": torch.from_numpy(candidates),
            "labels": torch.from_numpy(labels),
        }

    # ------------------------------------------------------------------

    def stats(self) -> DatasetStats:
        return DatasetStats(
            num_train_samples=len(self._prefixes) if self.mode == "train" else 0,
            num_eval_users=len(set(self._eval_user_ids)) if self.mode == "eval" else 0,
            num_candidates=self.num_neg + 1,
            max_seq_len=self.max_seq_len,
            num_items=self.num_items,
        )
