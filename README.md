# RecAgent Lab · 从零构建推荐系统到 LLM 推荐 Agent

> 一个项目，走通「深度学习 → Transformer → 推荐模型训练 → LLM 微调 → 量化部署 → Agent 应用 → 工程化落地」的完整链路。
>
> 目标读者：我自己（面试官 / 面试官请看 [docs/RESUME.md](docs/RESUME.md)）。

---

## 为什么做这个项目

市面上大量「Agent 项目」是套壳 RAG 问答：调 API、连向量库、拼 prompt，**展示不出模型能力和工程深度**；
而大量「推荐系统项目」停在论文复现：跑通一个 LightGCN，**没有任何线上工程和 LLM 能力**。

这个项目刻意站在两者中间：**用一个真实可跑的推荐系统作为 Agent 的"手和脚"**，
让 LLM Agent 去调度召回、排序、检索、重排等真实工具，而不是空谈。

这样做的好处是，简历上可以同时讲两层深度：

| 面试官关心 | 这个项目能给的证据 |
| --- | --- |
| 懂不懂深度学习底层 | 手写反向传播、手写 Multi-Head Attention，`from_scratch/` 下有对照实验 |
| 会不会训模型 | SASRec / DIN 从零训练，有 loss 曲线、消融表、bad case 分析 |
| 懂不懂 Transformer | 从零实现 Tiny GPT + 逐行对齐 HuggingFace 实现 |
| 有没有大模型经验 | Qwen2.5 LoRA 微调成「推荐助手」，含数据构造、训练、评测全流程 |
| 会不会部署 | 量化（int8/int4）+ vLLM/llama.cpp 推理服务 + FastAPI + Docker |
| 会不会做 Agent | 自研 ReAct 循环 + LangGraph 编排 + 工具调用 + 记忆 + 评测 |
| 工程素养 | 测试覆盖、CI、可复现实验、监控、消融实验设计 |
| 业务理解 | 多路召回 → 粗排 → 精排 → 重排 的完整工业级链路 |

---

## 五大能力线

项目用 11 个月推进，每个阶段同时推进多条能力线，**任何时刻都不存在"只学不产出"的阶段**。

```
        M1        M2        M3        M4        M5        M6        M7        M8
        ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
DL 基础  ████████
Transformer      ████████
推荐模型                  ████████████████
LLM 微调                            ████████████
部署工程                            ████████████████
Agent 应用                                    ████████████████████
工程化    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████████████████████
```

| # | 能力线 | 关键产出 | 对应阶段 |
| --- | --- | --- | --- |
| 1 | **深度学习基础** | 手写线性/逻辑回归、MLP 反向传播；SASRec/DIN 训练与调参 | S1, S3 |
| 2 | **Transformer** | 从零实现 MHA / Tiny GPT；对齐 HF 实现并可视化 attention | S2 |
| 3 | **LLM 训练与微调** | Qwen2.5-1.5B/7B LoRA 微调成推荐助手；DPO 对齐实验 | S4 |
| 4 | **部署与工程** | ONNX / int8 量化 / vLLM / llama.cpp；FastAPI 服务；Docker | S5 |
| 5 | **Agent 应用** | ReAct 循环、工具注册、会话记忆、多轮澄清、LangGraph 编排 | S6 |
| 6 | **工程化与协作** | 测试、CI、实验追踪、消融报告、代码评审流程 | 贯穿全程 |

---

## 项目形态

**一句话**：一个「对话式推荐 Agent」系统。用户用自然语言描述需求，Agent 自主调用推荐工具链完成推荐，并给出可解释的理由。

```
用户: "最近想看点烧脑的科幻，但不要太压抑，我上周刚看完《降临》"
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│                    推荐 Agent (LLM)                          │
│  意图理解 → 偏好澄清 → 工具规划 → 结果重排 → 生成理由         │
└──────────────────────────────────────────────────────────────┘
  │ 调用工具                      ▲ 候选 + 特征
  ▼                              │
┌──────────────────────────────────────────────────────────────┐
│ 工具层: search_item / get_user_profile / recall_by_vector    │
│         rank_candidates / explain_recommendation / ...       │
└──────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│ 推荐引擎: 多路召回 → 粗排 → 精排(DIN) → 重排                  │
│ 基线: Popularity / ItemCF / 双塔 / SASRec                    │
└──────────────────────────────────────────────────────────────┘
  │
  ▼
MovieLens-1M (离线, 模拟时间流) + Amazon Reviews (冷启动/迁移)
```

**核心研究问题（自问自答，也是面试抓手）**：

1. 相比纯协同过滤，LLM Agent 在**冷启动**和**长尾**场景能提升多少？
2. Agent 的"偏好澄清"轮次，对最终 NDCG 的边际收益是多少？
3. 微调过的推荐助手 vs. 直接用通用大模型 prompt，差多少？
4. 量化到 int4 后，推荐质量掉多少、延迟降多少？
5. 多轮对话里的用户画像漂移，怎么用记忆机制处理？

---

## 仓库结构

```
rec-agent-lab/
├── README.md
├── docs/
│   ├── ROADMAP.md        # 11 个月五阶段路线图与验收标准
│   ├── ARCHITECTURE.md   # 分层架构、模块接口、关键设计决策
│   ├── REVIEW.md         # 代码评审流程与协作约定
│   ├── DECISIONS.md      # 技术选型记录 (ADR)
│   └── paper-notes/      # 论文精读笔记（LLM4Rec 方向）
├── recagent/
│   ├── data/             # 数据下载、预处理、时间流切分
│   ├── features/         # 特征工程
│   ├── models/           # 召回/排序模型
│   ├── engine/           # 多路召回→排序→重排 编排
│   ├── llm/              # 微调、推理、prompt 管理
│   ├── agent/            # ReAct 循环、工具、记忆
│   ├── eval/             # 离线评测、LLM-as-judge、消融
│   └── serving/          # API 服务、监控
├── from_scratch/         # 手写实现（DL 基础 / Transformer / Tiny GPT）
├── configs/              # 实验配置 (hydra/yaml)
├── scripts/              # 训练、评测、导出、部署脚本
├── tests/                # 单元测试与集成测试
├── notebooks/            # 探索性分析（不进 CI）
└── reports/              # 消融实验报告、性能分析
```

---

## 快速开始

```bash
# 环境（详见 docs/ROADMAP.md 阶段 0）
conda env create -f environment.yml
conda activate recagent

# 数据准备
python scripts/download_data.py --dataset ml-1m
python scripts/preprocess.py --config configs/data/ml1m.yaml

# 训练基线
python scripts/train.py --config configs/model/sasrec_ml1m.yaml

# 评测
python scripts/evaluate.py --run-name sasrec_ml1m --split test

# 起服务（阶段 5 之后可用）
docker compose up -d
curl -X POST localhost:8000/recommend -H 'Content-Type: application/json' \
  -d '{"user_id": 1, "query": "想看点烧脑科幻"}'
```

---

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [docs/ROADMAP.md](docs/ROADMAP.md) | **先看这个**：11 个月五阶段计划、每周任务、验收标准 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 分层架构、模块接口契约、关键设计决策 |
| [docs/REVIEW.md](docs/REVIEW.md) | 代码评审流程、Definition of Done、协作约定 |
| [docs/DECISIONS.md](docs/DECISIONS.md) | 技术选型记录及理由 |

---

## 进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| S0 | 环境与仓库骨架 | 🚧 进行中 |
| S1 | 深度学习基础 + 手写训练循环 | ⬜ |
| S2 | Transformer 从零实现 | ⬜ |
| S3 | 推荐模型（召回+排序） | ⬜ |
| S4 | LLM 微调 | ⬜ |
| S5 | 部署与量化 | ⬜ |
| S6 | 推荐 Agent | ⬜ |
| S7 | 消融、报告、简历包装 | ⬜ |

> S0 已完成骨架搭建与首次提交，等你按 `docs/ROADMAP.md` 逐项验收。

## License

MIT
