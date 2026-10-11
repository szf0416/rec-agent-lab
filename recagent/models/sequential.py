"""推荐模型。

本文件目前只有 S1 端到端验证所需的最小序列召回模型。
完整的召回/排序模型（SASRec、DIN、双塔）在 S3 实现。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SequentialRecallModel(nn.Module):
    """最小序列召回模型：物品 embedding 的均值池化 + 内积打分。

    **它是 SASRec 的"退化版"**：去掉了自注意力，只用平均表示用户兴趣。
    存在的意义是让 S1 的训练循环能在真实数据上跑通并给出可判断的指标，
    而不是它的效果有多好。S3 会用 SASRec 替换它，指标应有明显提升。

    结构::

        user_seq (B, L) ──▶ item_embedding ──▶ mean pool ──▶ user_vec (B, d)
                                                                  │
        candidates (B, C) ──▶ item_embedding ──▶ cand_vec (B, C, d)
                                                                  │
                                        scores = user_vec · cand_vecᵀ  (B, C)
                                                                  │
                                             BCEWithLogits(scores, labels)

    为什么用 `BCEWithLogitsLoss` 而不是 `CrossEntropyLoss`：
    评测协议是「1 正 + N 负」，每行只有一个正样本且固定在第 0 列。
    用 BCE 逐元素打分更接近真实的排序场景，也便于后续替换排序损失（如 BPR、listwise softmax）。
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        padding_idx: int = 0,
        dropout: float = 0.0,
        hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        if num_items < 2:
            raise ValueError(f"num_items 至少为 2，收到 {num_items}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout 必须在 [0,1)，收到 {dropout}")
        if hidden_dim < 0:
            raise ValueError(f"hidden_dim 必须 >= 0，收到 {hidden_dim}")

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout
        self.hidden_dim = hidden_dim

        # num_items + 1：索引 0 保留给 padding，物品 id 从 1 开始（与 ML-1M 一致）
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=padding_idx)
        # 小初始化：embedding 直接决定打分尺度，过大会让 loss 一开始就爆掉
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[padding_idx].zero_()

        # dropout 层：rate=0 时退化为恒等映射（不引入额外开销）
        #
        # 这里对**用户向量和物品向量都**做 dropout。取舍说明：
        # - 用户向量加 dropout：相当于随机丢弃部分历史行为，迫使模型不依赖个别行为
        # - 物品向量加 dropout：也是常见做法，训练时会明显变慢变难
        # S1-W3 的实验会对比"加与不加""加在哪"的差异，所以两者都提供开关。
        self.user_dropout = nn.Dropout(p=dropout)
        self.item_dropout = nn.Dropout(p=dropout)

        # 打分头：把用户向量与物品向量的拼接过一个 MLP 再输出分数。
        #
        # 为什么需要它：只用「用户向量 · 物品向量」内积时模型表达能力很弱，
        # 第 0 轮就学到极限、之后只能过拟合 —— 观察不到「先升后降」的完整曲线，
        # 也就无法判断正则化是否真的把过拟合推后了。
        # 加一层非线性显著提升容量，让过拟合发生得更晚、更可观察。
        #
        # hidden_dim=0 时退化为纯内积（保持向后兼容）。
        if hidden_dim > 0:
            self.scorer: nn.Module | None = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.scorer = None

        self.loss_fn = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------

    def encode_users(
        self, user_seq: torch.Tensor, seq_len: torch.Tensor | None = None
    ) -> torch.Tensor:
        """把用户历史序列编码成向量。

        均值池化时**必须排除 padding**：padding 是索引 0，
        虽然 `padding_idx` 让它 embedding 为 0，但只在精确等于 0 时无害——
        一旦训练把它的梯度更新成非 0（`padding_idx` 只保证初始化，不锁梯度），
        池化结果就会被污染。所以这里显式用长度做掩码。
        """
        embedded = self.item_embedding(user_seq)  # (B, L, d)

        if seq_len is None:
            # 左 padding 的约定：非零项在尾部，按是否非零判断有效位置
            mask = (user_seq != 0).unsqueeze(-1).to(embedded.dtype)  # (B, L, 1)
        else:
            positions = torch.arange(user_seq.size(1), device=user_seq.device).unsqueeze(0)
            # 左 padding：有效位置在最后 seq_len 个
            mask = (positions >= (user_seq.size(1) - seq_len.unsqueeze(1))).unsqueeze(-1)
            mask = mask.to(embedded.dtype)

        summed = (embedded * mask).sum(dim=1)  # (B, d)
        counts = mask.sum(dim=1).clamp(min=1.0)  # 避免除零
        # 显式标注：torch 的算子返回 Any，不注解会触发 no-any-return
        pooled: torch.Tensor = summed / counts
        # dropout 只在 train() 模式下生效，eval() 时自动变成恒等映射，
        # 所以验证/评测走的是完整向量 —— 这是 nn.Dropout 的内置行为，不用手写开关
        dropped: torch.Tensor = self.user_dropout(pooled)
        return dropped

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """把候选物品编码成向量，形状 (B, C, d)。"""
        encoded: torch.Tensor = self.item_dropout(self.item_embedding(item_ids))
        return encoded

    def score(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """返回 (B, C) 打分。forward 与 predict 共用，避免两处逻辑漂移。

        两种打分方式（由 `hidden_dim` 决定）：
        - 纯内积：`user · item`，参数量少、表达力弱
        - MLP 头：`MLP([user, item])`，表达力强得多

        实验里两种都要有，才能验证「过拟合是否来自模型容量」。
        """
        user_vec = self.encode_users(batch["user_seq"], batch.get("seq_len"))
        candidate_vec = self.encode_items(batch["candidates"])

        if self.scorer is None:
            # (B,d) x (B,C,d) -> (B,C)
            scores: torch.Tensor = torch.einsum("bd,bcd->bc", user_vec, candidate_vec)
            return scores

        # 把用户向量广播到每个候选，拼接过 MLP
        # (B,1,d) -> (B,C,d)
        user_expanded = user_vec.unsqueeze(1).expand(-1, candidate_vec.size(1), -1)
        paired = torch.cat([user_expanded, candidate_vec], dim=-1)  # (B,C,2d)
        head_out: torch.Tensor = self.scorer(paired)  # (B,C,1)
        return head_out.squeeze(-1)

    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """返回标量 loss。训练循环只认这个接口。"""
        scores = self.score(batch)
        loss: torch.Tensor = self.loss_fn(scores, batch["labels"])
        return loss

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """返回 (B, C) 分数。注意输出的是 logits 而非概率——

        排序指标只关心相对顺序，不需要 sigmoid，省一次计算。
        """
        return self.score(batch)
