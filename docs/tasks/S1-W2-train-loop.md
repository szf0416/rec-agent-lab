# S1-W2 · 训练循环工程化

> 阶段：S1（深度学习基础）第 2 周
> 归属：**代码由你写**；接口契约已由架构文档定好
> 预计投入：12–16 小时
> 前置：S1-W1 已通过（梯度推导与验证能力已确认）

---

## 一、为什么要单独花一周做"训练循环"

S1-W1 你写的是**一次性脚本**：数据、模型、训练写在一个文件里，改一处就得重跑全部。

S3 要训练 SASRec 和 DIN，S4 要微调 LLM。如果每次都复制粘贴这套脚本：

- 调参时改错文件 → 结果无法对应
- 早停逻辑每份实现都不一样 → 实验不可比
- 想换 scheduler 就得改 5 个地方
- **最致命**：面试官问"你怎么保证实验可复现"，你答不上来

**所以这一周的产出是「一次写对、后面复用 7 个月」的基础设施。**

> 这也是面试常被问的「你如何组织训练代码」。能讲清"循环与模型解耦、
> 通过 Protocol 定义接口、配置驱动"的人，比只会调库的人高一个层级。

---

## 二、交付物

### 文件 1：`recagent/train/loop.py`（核心）

**通用训练循环**。它必须对"模型是什么"一无所知——只依赖接口。

```python
from typing import Protocol

class TrainableModel(Protocol):
    """训练循环只认这个接口，不关心具体模型。"""
    def forward(self, batch: dict) -> torch.Tensor:
        """返回标量 loss。"""
    def predict(self, batch: dict) -> torch.Tensor:
        """返回 (B, num_candidates) 分数，供评测用。"""
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
```

然后实现：

```python
class Trainer:
    def __init__(
        self,
        model: TrainableModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        config: TrainConfig = ...,
        callbacks: list[Callback] | None = None,
    ) -> None: ...

    def fit(self, train_loader, valid_loader) -> TrainHistory: ...
    def train_epoch(self, loader) -> dict[str, float]: ...
    def validate(self, loader) -> dict[str, float]: ...
```

**必须包含的机制**（每一项都有对应验收测试）：

| 机制 | 要点 | 为什么必须 |
| --- | --- | --- |
| `zero_grad` 时机 | 每个 step 开始时清理 | W1 实验 4 证明 `.grad` 是累加的，忘了就是学习率悄悄变大 |
| 梯度裁剪 | `clip_grad_norm_`，阈值走配置 | 推荐模型 embedding 梯度容易爆炸 |
| LR 调度 | 每 step 还是每 epoch 更新？ | 这是**最容易错**的地方，见下面「坑 1」 |
| 早停 | 基于验证指标，带 patience | 没有它，训练集过拟合会被当成"效果好" |
| checkpoint | 存最好 + 存最近，含 metadata | 只存 `state_dict` 无法复现实验 |
| 混合精度 | 可选（`amp`），默认关闭 | 6GB 显存下后面可能需要 |
| 日志与追踪 | 每 epoch 记录指标，接 MLflow | 面试要能拿出 loss 曲线 |

### 文件 2：`recagent/train/callbacks.py`

用回调（而非把逻辑塞进 `fit`）实现可插拔。**至少 3 个**：

```python
class Callback(Protocol):
    def on_epoch_start(self, epoch: int, trainer: "Trainer") -> None: ...
    def on_epoch_end(self, epoch: int, metrics: dict, trainer: "Trainer") -> None: ...
    def on_train_end(self, history: "TrainHistory") -> None: ...

class EarlyStopping(Callback): ...      # 监控指标 + patience + 恢复最优权重
class ModelCheckpoint(Callback): ...    # 存 top-k，文件名带指标值
class MLflowLogger(Callback): ...       # 记录超参 + 每 epoch 指标
```

**设计要求**：`EarlyStopping` 和 `ModelCheckpoint` 都要能被独立单元测试，
不依赖真实训练。这要求它们不直接读文件系统或全局状态。

### 文件 3：`tests/test_loop.py`（进 CI）

用**假的**模型和数据集测试循环逻辑。不需要真数据、不需要 GPU，必须秒级跑完。

**必测项**（我评审时会逐条核对）：

| # | 测试 | 验证什么 |
| --- | --- | --- |
| 1 | 3 个 epoch 后 loss 是否下降 | 循环真的在优化 |
| 2 | `zero_grad` 是否正确 | 手动累加两次梯度必须等于单次的两倍（用假模型可精确计算） |
| 3 | 梯度裁剪生效 | 构造超大梯度，确认裁剪后 norm ≤ 阈值 |
| 4 | scheduler 调用次数 | 每 step 调用的 scheduler 与每 epoch 调用的行为不同（坑 1） |
| 5 | early stop 触发 | patience=2，指标连续不涨时确实停在第 N 个 epoch |
| 6 | **early stop 恢复最优权重** | 停止后模型参数等于最优 epoch 的，不是最后 epoch 的 |
| 7 | checkpoint 往返 | 存了再读，`state_dict` 完全一致 |
| 8 | 固定 seed 可复现 | 跑两次，loss 序列完全相同 |
| 9 | 空 dataloader | 不崩，报明确错误 |
| 10 | 恢复训练 | 从 checkpoint 续训，指标连续 |

### 文件 4：`configs/model/train_default.yaml`

把 `TrainConfig` 的所有字段暴露成 yaml，附中文注释说明每项作用与推荐值。

---

## 三、两个必须避开的坑（本周重点）

### 坑 1：scheduler 的步进时机

PyTorch 的 `lr_scheduler` 分两类，**用错会让学习率曲线完全不是你想要的**：

| 类型 | 例子 | 该在哪调用 |
| --- | --- | --- |
| **per-epoch** | `StepLR`、`MultiStepLR`、`ExponentialLR` | 每个 epoch 结束 |
| **per-step** | `OneCycleLR`、`CosineAnnealingWarmRestarts` | 每个 batch 之后 |

**错误示范**：用 `OneCycleLR` 但每 epoch 才 `step()` → 学习率曲线被压缩 100 倍。

**实现要求**：用配置显式声明

```yaml
scheduler:
  name: cosine
  interval: step        # step | epoch
  warmup_steps: 100
```

**你的代码必须在训练开始时校验**：如果 `interval` 与 scheduler 类型不匹配，
给出警告或报错。不要沉默。

### 坑 2：`EarlyStopping` 必须恢复最优权重

很多实现只做"停止"，不恢复权重。后果：

```
epoch 10: NDCG 0.31  ← 最优
epoch 11: NDCG 0.30
epoch 12: NDCG 0.295 ← patience 用完，停止
```

停止时模型停在 epoch 12（0.295），**你丢掉了最好的那个模型**。

正确做法：`EarlyStopping` 内部记录最优权重快照，触发时恢复。

**验收测试 #6 专门测这个**：early stop 后模型参数必须等于最优 epoch 的。

---

## 四、验收标准

- [ ] `recagent/train/loop.py` 不 import 任何具体模型，只依赖 `TrainableModel` 协议
- [ ] `pytest tests/test_loop.py` 全绿，**10 个必测项全覆盖**
- [ ] 全部测试**不使用真实数据、不要求 GPU**，总耗时 < 10 秒
- [ ] 改一个 yaml 字段就能换 optimizer / scheduler / early stop patience，不用动代码
- [ ] `scheduler.interval` 不匹配时给出明确警告（不是静默）
- [ ] `EarlyStopping` 确实恢复最优权重（测试 #6 通过）
- [ ] checkpoint 含 metadata：epoch、指标、配置快照、git commit
- [ ] `mypy recagent` 通过（这个目录进类型检查）
- [ ] **能回答**：为什么用回调而不是把 early stop 写进 `fit`？

---

## 五、建议路径

**不要从 `Trainer` 开始写。** 推荐顺序：

**第 1 步：先写 `TrainConfig`（dataclass）。**
把"训练需要哪些参数"想清楚，后面的代码结构自然清晰。
参数超过 4 个就得用 dataclass 封装（`docs/REVIEW.md` 的规范）。

**第 2 步：写最小可跑的 `train_epoch`。**
只有 forward / backward / optimizer.step，先不加任何机制。
用假模型跑通，确认 loss 下降。

**第 3 步：加 `validate` 和 `TrainHistory`。**
先能观察，才能判断后面的机制有没有生效。

**第 4 步：加回调框架，然后一个个实现。**
每加一个机制，立刻补对应测试。

**第 5 步：最后接 MLflow。**
它是回调，天然可插拔——如果你发现接 MLflow 需要改 `Trainer`，说明抽象漏了。

**第 6 步：用真实数据（ML-1M 预处理产物）跑一次。**
验证它在真实场景下可用，写进 `reports/`。

---

## 六、卡点提示（卡住 30 分钟以上再看）

<details>
<summary>提示 1：怎么测试「zero_grad 是否正确」</summary>

不要用真实的复杂模型。构造一个最简单的可解析模型：

```python
class LinearOnlyModel:
    """y = w * x，loss = (y - target)^2，w 是唯一参数"""
    def __init__(self, w=1.0):
        self.w = torch.tensor(w, requires_grad=True)
    def forward(self, batch):
        pred = self.w * batch["x"]
        return ((pred - batch["target"]) ** 2).mean()
```

对同一个 batch 连续做两次 forward+backward **且不 zero_grad**，
`w.grad` 应该是单次的两倍（手算可验证）。
然后加上 `zero_grad` 再测，应该是单次的值。

**这就是把 W1 实验 4 的发现变成自动化测试。**
</details>

<details>
<summary>提示 2：EarlyStopping 怎么"恢复最优权重"才安全</summary>

直接 `copy.deepcopy(model.state_dict())` 可能很占显存（模型大时上百 MB）。

两种做法，各有取舍：
- **内存快照**：`deepcopy`，简单但占内存
- **存磁盘**：写临时文件，省内存但慢

**推荐**：配置项 `save_best_to: memory | disk`，默认 memory（S3 的模型不大）。
重点是你要**知道有这个取舍**并说明选择理由。

另外注意：恢复时用 `model.load_state_dict(snapshot)`，
不要 `model = deepcopy(snapshot_model)`——后者会丢掉 optimizer 的关联。
</details>

<details>
<summary>提示 3：scheduler 检测 interval 是否匹配</summary>

无法 100% 自动判断（PyTorch 没有标准接口）。可行的启发式：

```python
PER_STEP_SCHEDULERS = {"OneCycleLR", "CosineAnnealingWarmRestarts", "CyclicLR"}

name = type(scheduler).__name__
if name in PER_STEP_SCHEDULERS and interval == "epoch":
    warnings.warn(f"{name} 是 per-step 调度器，但 interval=epoch，学习率曲线会被压缩")
```

不完美，但**比沉默好**。生产代码里这种"已知易错点主动报警"是加分项。
</details>

<details>
<summary>提示 4：为什么测试必须秒级</summary>

如果 `tests/test_loop.py` 要 30 秒，你以后就不会跑它。
半年后这个文件就成了摆设——而它是唯一能保证训练循环正确的东西。

**用假模型 + 假数据，10 行代码就能构造能精确验算的测试场景。**
真实数据留给 `reports/` 里的一次性验证。
</details>

---

## 七、提交格式

```
阶段: S1-W2
文件: recagent/train/loop.py, callbacks.py, tests/test_loop.py, configs/model/train_default.yaml
自检: pytest tests/test_loop.py = ? 个测试通过, 耗时 ? 秒
     mypy recagent = 通过/失败
     真实数据跑通 = 是/否（附 loss 曲线数值）
卡点: ...
问题: ...
```

我的评审重点（提前告诉你，方便自查）：

1. `Trainer` 是否真的与模型解耦（我会试着塞一个完全不同的假模型进去）
2. **`EarlyStopping` 恢复最优权重的测试是否真的有效**（最容易写成假测试）
3. 10 个必测项是否都覆盖，有没有把测试写成"验证代码假设"而非"验证行为"
4. 回调接口是否够用（接 MLflow 是否需要改 `Trainer`）

---

## 八、和后续阶段的关系

| 后续阶段 | 复用本周的什么 |
| --- | --- |
| S3 SASRec 训练 | 直接复用 `Trainer` + `EarlyStopping` + checkpoint |
| S3 DIN 训练 | 同 `Trainer`，只换模型和特征 |
| S3 消融实验 | 靠配置驱动，一次跑多组 |
| S4 LLM 微调 | 回调框架换成 `transformers.Trainer`，但早停/checkpoint 思路一致 |
| S5 部署 | checkpoint 的 metadata 里有配置快照，便于导出 ONNX |

**这周写好了，S3 的 8 周会轻松很多。**
