# ARCHITECTURE · 系统架构与接口契约

> 本文档定义**模块边界与接口契约**。原则：接口先定死，实现可以换。
> 任何接口改动都要更新本文档，并在 `docs/DECISIONS.md` 留记录。

---

## 1. 分层总览

系统分 6 层，**依赖只能自上而下单向流动**，不允许跨层反向依赖。

```
┌─────────────────────────────────────────────────────────────┐
│ L6  Serving       FastAPI / Docker / 监控                    │
├─────────────────────────────────────────────────────────────┤
│ L5  Agent         ReAct 循环 / 工具注册 / 记忆 / 编排          │
├─────────────────────────────────────────────────────────────┤
│ L4  LLM           微调 / 推理 / prompt 管理 / 结构化输出       │
├─────────────────────────────────────────────────────────────┤
│ L3  Engine        多路召回 → 粗排 → 精排 → 重排 编排          │
├─────────────────────────────────────────────────────────────┤
│ L2  Models        Recall (Popular/ItemCF/TwoTower/SASRec)     │
│                   Rank   (DIN/DeepFM)                        │
├─────────────────────────────────────────────────────────────┤
│ L1  Data          加载 / 切分 / 特征 / Dataset                 │
└─────────────────────────────────────────────────────────────┘
```

**为什么这么分层**：L5 的 Agent 要调用 L3 的引擎，L3 要用 L2 的模型，L2 吃 L1 的数据。
LLM（L4）是**被 L3 和 L5 共同使用的工具**，不是 L3 的下游，所以单独成层。

---

## 2. 核心接口契约

### 2.1 数据层

```python
# recagent/data/base.py
@dataclass(frozen=True)
class Interaction:
    user_id: int
    item_id: int
    timestamp: int
    rating: float | None = None

class DatasetSplit(NamedTuple):
    train: list[Interaction]
    valid: list[Interaction]
    test: list[Interaction]

class Splitter(Protocol):
    """时间切分 / leave-one-out 切分都实现这个协议。"""
    def split(self, interactions: list[Interaction]) -> DatasetSplit: ...

class RecDataset(torch.utils.data.Dataset):
    """统一的 Dataset。由 DatasetBuilder 产出。"""
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]: ...

class DatasetBuilder(Protocol):
    """不同数据集（ML-1M / Amazon）实现这个协议。"""
    def build(self, config: DataConfig) -> tuple[RecDataset, RecDataset, RecDataset]: ...
    @property
    def num_users(self) -> int: ...
    @property
    def num_items(self) -> int: ...
```

### 2.2 模型层

**统一约束**：所有模型继承 `BaseModel`，实现 `forward` 和 `predict`。
`forward` 返回 loss（训练用），`predict` 返回分数（评测/推理用）。

```python
# recagent/models/base.py
class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """返回标量 loss。训练循环只认这个接口。"""

    @abstractmethod
    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """返回 shape=(B, num_candidates) 的分数。"""

    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "BaseModel": ...
```

**召回模型**额外实现向量化接口（用于 FAISS）：

```python
# recagent/models/recall/base.py
class BaseRecall(BaseModel, ABC):
    @abstractmethod
    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        """返回 (N, d) 用户向量。"""

    @abstractmethod
    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """返回 (N, d) 物品向量。"""
```

**排序模型**额外声明使用的特征：

```python
# recagent/models/rank/base.py
class BaseRanker(BaseModel, ABC):
    @property
    @abstractmethod
    def feature_columns(self) -> list[str]:
        """声明需要哪些特征，引擎据此准备数据。"""
```

### 2.3 引擎层

```python
# recagent/engine/pipeline.py
@dataclass
class RecommendRequest:
    user_id: int
    top_k: int = 10
    exclude_seen: bool = True
    context: dict[str, Any] | None = None   # Agent 传来的额外约束

@dataclass
class RecommendedItem:
    item_id: int
    score: float
    source: str                 # 哪一路召回的，用于分析和解释
    explanation: str | None = None

class RecallChannel(Protocol):
    """每个召回通道实现这个协议，引擎统一调度。"""
    name: str
    def recall(self, request: RecommendRequest, size: int) -> list[RecommendedItem]: ...

class RecommendationEngine:
    """编排：多路召回 → 去重 → 粗排 → 精排 → 重排。

    设计要点：
    - 每路召回独立失败不影响整体（降级策略）
    - 记录每个候选来自哪一路，供消融和可解释性使用
    - 全流程可注入耗时埋点
    """
    def __init__(
        self,
        channels: list[RecallChannel],
        ranker: BaseRanker | None = None,
        reranker: "BaseReranker | None" = None,
        config: EngineConfig = ...,
    ) -> None: ...

    def recommend(self, request: RecommendRequest) -> list[RecommendedItem]: ...
```

**关键设计决策**：`source` 字段必须保留到最后。
Agent 要能说"我推荐这个是因为它来自序列召回"，消融实验也要靠它统计每路贡献。

### 2.4 LLM 层

```python
# recagent/llm/assistant.py
@dataclass
class LLMResponse:
    text: str
    parsed: dict[str, Any] | None    # 结构化解出的 JSON
    usage: dict[str, int]            # token 统计
    latency_ms: float

class BaseLLM(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict | None = None,      # 传入则强制结构化输出
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> LLMResponse: ...

class LocalLLM(BaseLLM):
    """transformers / llama.cpp 后端"""
class RemoteLLM(BaseLLM):
    """vLLM / OpenAI 兼容 API 后端"""
```

**为什么用 Protocol 而不是基类**：本地模型和远程 API 生命周期完全不同
（本地要管显存、要懒加载；远程要管重试、限流），强行共用基类会写出难看的 `if` 分支。

### 2.5 Agent 层

```python
# recagent/agent/tools.py
@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict          # JSON Schema
    func: Callable[..., Any]
    timeout_s: float = 5.0

class ToolRegistry:
    def register(self, spec: ToolSpec) -> None: ...
    def to_openai_schema(self) -> list[dict]: ...
    def call(self, name: str, args: dict) -> ToolResult: ...
    def unregister(self, name: str) -> None: ...

@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0
```

```python
# recagent/agent/react.py
@dataclass
class AgentStep:
    thought: str
    action: str | None
    action_input: dict | None
    observation: str | None
    elapsed_ms: float

@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep]
    recommendations: list[RecommendedItem]
    terminated_reason: str        # "final_answer" | "max_steps" | "timeout" | "error"

class ReActAgent:
    """自研 ReAct 循环。

    安全边界（必须有）：
    - max_steps: 硬上限，防死循环
    - per-step timeout 和总 timeout
    - 连续 N 次工具调用失败则降级到直接回答
    - 工具调用去重：同参数重复调用直接返回缓存
    """
    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry,
        memory: "Memory",
        max_steps: int = 8,
        total_timeout_s: float = 60.0,
    ) -> None: ...

    def run(self, user_query: str, user_id: int | None = None) -> AgentResult: ...
```

```python
# recagent/agent/memory.py
class Memory(Protocol):
    def add_turn(self, role: str, content: str) -> None: ...
    def get_context(self, max_tokens: int) -> list[dict[str, str]]: ...
    def get_user_profile(self, user_id: int) -> dict[str, Any]: ...
    def update_profile(self, user_id: int, facts: dict[str, Any]) -> None: ...

class HybridMemory(Memory):
    """短期：滑动窗口对话
    长期：用户偏好摘要（LLM 增量更新 + 去重）
    情景：历史会话的关键事件

    关键约束：长期记忆必须有容量上限和淘汰策略，否则无限膨胀。
    """
```

---

## 3. 关键设计决策

| 决策 | 选择 | 理由 | 备选与代价 |
| --- | --- | --- | --- |
| 数据切分 | **时间切分**（主）+ leave-one-out（辅） | 避免未来信息泄漏；贴近线上真实场景；面试高频考点 | 随机切分：指标虚高，面试减分 |
| 评测指标 | **自己实现** | 面试要能推导；且可控 | RecBole：快，但说不清细节；用作交叉校验 |
| 召回架构 | **多路召回**而非单模型 | 工业界标准做法；消融实验空间大；Agent 工具更丰富 | 单路：简单但故事少 |
| Agent 框架 | **自研 ReAct 为主，LangGraph 为对照** | 自研证明理解本质；LangGraph 证明会用工具 | 只用 LangGraph：容易被质疑"套壳" |
| LLM 基座 | **Qwen2.5** 系列 | 中文强、1.5B~7B 梯度完整、社区成熟 | Llama：中文弱；ChatGLM：生态略小 |
| 微调方式 | **QLoRA** | 6GB 显存唯一可行；工业界主流 | 全参微调：显存不够 |
| 向量检索 | **FAISS** | 轻量、无服务依赖、CPU 可跑 | Milvus：更工业但要额外运维 |
| 实验追踪 | **MLflow**（本地） | 无需联网、零成本、够用 | W&B：更好看但需账号 |
| 配置管理 | **Hydra** | 组合式配置，适合多组消融 | 纯 yaml：消融实验会写得很痛苦 |

---

## 4. 目录职责边界（评审重点）

| 目录 | 放什么 | **不放什么** |
| --- | --- | --- |
| `recagent/` | 可复用的库代码、有测试 | 一次性脚本、探索性代码 |
| `scripts/` | 只管编排：读配置 → 调库 → 存结果 | **任何业务逻辑** |
| `notebooks/` | 探索、画图、临时分析 | 最终结论（结论要沉淀到 `reports/`） |
| `from_scratch/` | 手写教学实现 | 生产代码 |
| `configs/` | 声明式配置 | 代码 |
| `reports/` | 实验结论、图表、分析 | 代码（画图脚本在 `scripts/`） |
| `tests/` | 单元 + 集成测试 | 需要大数据的慢测试（标记 `@pytest.mark.slow`） |

**一条硬规则**：`scripts/` 里的文件应该短到能一眼看完，逻辑都在 `recagent/` 里。
如果发现 `scripts/train.py` 超过 100 行，说明逻辑泄漏了。

---

## 5. 数据流（一次完整请求）

```
用户: "想看点烧脑科幻，别太压抑"
  │
  ├─▶ [L5] Agent 解析意图 → 结构化约束 {genre: SciFi, mood: not_dark, ...}
  │
  ├─▶ [L5] 查记忆 → 用户历史偏好 + 上次会话上下文
  │
  ├─▶ [L5] 判断是否需要澄清 → 需要则反问，不需要则继续
  │
  ├─▶ [L5] 工具调用 [L3] engine.recommend(user_id, context=约束)
  │        │
  │        ├─▶ [L2] 多路召回并行: Popular / ItemCF / TwoTower / SASRec
  │        ├─▶ 融合去重 + 过滤已看
  │        ├─▶ [L2] DIN 精排
  │        └─▶ 返回带 source 的候选列表
  │
  ├─▶ [L4] LLM 重排 + 生成推荐理由（引用用户历史作为证据）
  │
  ├─▶ [L5] 更新记忆（本轮新偏好）
  │
  └─▶ 返回: 推荐列表 + 理由 + 完整调用链 (供 Langfuse 追踪)
```

---

## 6. 性能与规模预期

| 环节 | 目标 | 说明 |
| --- | --- | --- |
| 单路召回 | < 20ms | FAISS 索引常驻内存 |
| 完整召回+排序 | < 100ms | 不含 LLM |
| LLM 重排 | < 2s | 1.5B 量化后，本地 |
| Agent 完整链路 | < 8s | 含 2-3 次工具调用 |
| 模型规模 | ML-1M 全量可训 | 6GB 显存约束下的设计 |

---

## 7. 变更流程

修改接口契约时：

1. 先改本文档，说明**为什么**要改
2. 在 `docs/DECISIONS.md` 追加一条 ADR
3. 更新受影响的测试
4. 在 PR 描述里明确标注 `BREAKING CHANGE`

> 接口随意改动是这个项目最大的技术债来源。定契约的时候多想 10 分钟，
> 比后面改 10 个文件便宜得多。
