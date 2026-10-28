# 本地模型工具调用演示

这条路径让真实本地模型选择推荐工具；召回、排序、电影标题和理由仍由本项目的训练期数据决定。默认使用 Ollama 的 OpenAI 兼容端点 `http://127.0.0.1:11434/v1` 和 `qwen2.5:1.5b`。这个模型约 1 GB 量化文件，适合先在 6 GB 显存机器上尝试，但运行速度和实际显存占用需在本机测量。

## 准备数据和服务

1. 按 [README 快速开始](../README.md#快速开始) 准备 MovieLens-1M 并运行预处理。确认 `data/processed/ml1m/items.json` 和 `interactions_train.csv` 已生成。
2. 按 [Ollama 官方安装说明](https://docs.ollama.com/quickstart) 安装并启动 Ollama。另开终端执行：

   ```powershell
   ollama pull qwen2.5:1.5b
   ollama list
   ```

3. 在仓库根目录运行：

   ```powershell
   python scripts/run_local_llm_agent.py --query "想看科幻，不要恐怖" --user-id 6040 --top-k 3
   ```

   如果 Ollama 没有自动启动，先执行 `ollama serve` 并保持该终端运行。演示结束后可在服务终端按 Ctrl+C 结束手动启动的服务。

## 使用其他本机服务

只要实现 OpenAI Chat Completions 的工具调用协议，即可通过环境变量切换：

```powershell
$env:RECAGENT_LLM_BASE_URL = "http://127.0.0.1:8000/v1"
$env:RECAGENT_LLM_MODEL = "your-local-tool-capable-model"
# 服务需要令牌时才设置：$env:RECAGENT_LLM_API_KEY = "..."
python scripts/run_local_llm_agent.py --query "想看喜剧" --top-k 3
```

出于本机演示边界，客户端只接受本机回环地址的 HTTP `/v1` 端点。代码不记录 API 密钥或模型原始响应。若要使用远程服务，需要先修改边界设计并做相应安全评审。

## 如何读输出

- `mode: "llm"`、`fallback_used: false` 且 `tool_trace` 有 `recall_candidates` 和 `rank_candidates`：模型实际参与了工具选择，最终电影 ID 通过了证据检查。
- `fallback_used: true`、`status: "fallback"`：服务不可用、模型协议不兼容、工具参数非法、预算耗尽或结果无依据，系统改用确定性推荐。`tool_trace` 的 `llm_validation` 只展示错误类别。
- `model_usage` 是服务报告的 token 数；未报告时为空字典。`latency_ms` 为这次请求耗时，不是稳定吞吐指标。
- `preferences.unparsed_constraints` 非空时，输出 `partial`。例如“不要太压抑”目前不能由训练期题材字段可靠验证。

模型最多发起 4 次工具调用、6 轮对话，总预算 60 秒。用户 ID 固定在 CLI 请求中；模型不能指定其他用户。推荐理由由真实目录元数据和排序来源生成，模型自由文本不会直接进入推荐理由。

## 已验证范围

自动测试使用脚本化客户端和模拟 HTTP 响应，覆盖工具消息回放、越权参数、虚构 ID、预算超限、无服务回退与 CLI JSON 输出。它验证的是实现逻辑，不代表 `qwen2.5:1.5b` 的实际工具调用成功率或推荐质量。完成本机实测后，应记录模型版本、服务版本、请求示例、成功率、延迟与失败案例，再用于简历。
