# 固定评测负样本碰撞审计

日期：2026-10-28。数据：MovieLens-1M，`configs/data/ml1m.yaml`，时间切分，seed 42。预处理产物不提交 Git。

## 发现

旧版 `NegativeSampler.eval_negatives()` 调用 `_draw(..., exclude=None)`；预处理入口没有传入用户已交互电影。旧产物覆盖 1209 个测试用户，其中 1155 人至少有一个评测负样本与已知交互重叠。负样本与训练交互的重叠计数为 14676，与测试交互的重叠计数为 5779。这两个计数按切分分别统计，不代表不同电影的去重总数。

## 修复和复核

预处理现在汇总训练、验证、测试交互，只将全集用于评测负标签排除；模型训练和推理仍只使用训练期数据。重新处理 1000209 条评分后，1209 个测试用户的固定负样本与三段交互的重叠总数为 **0**。同配置连续运行两次，`eval_negatives_test.json` 的 SHA256 均为 `76aab72cedcaa75fa39c3ae806486ed123adf5a43635c1d8ee4ac8dd11e8e070`。

复现命令：

```bash
python scripts/preprocess.py --config configs/data/ml1m.yaml
python -m pytest tests/test_negative.py -q
```

旧版产物不能用于后续准确率报告；运行新代码重新预处理后，再执行评测。
