# PUSH · 推送到 GitHub 操作手册

> 当前状态：本地仓库已初始化，已有 4 次提交，**只差一个远端**。
> 网络诊断结论见文末「排障」一节。

---

## 推送前确认

```powershell
cd C:\Users\shizifan\Desktop\code\rec-agent-lab
git log --oneline      # 应看到 4 条提交
git status             # 应为 clean
```

当前 HEAD 应为：`42c7171 docs(s0): 新增 40 周执行日历与缓冲突冲期计划`

---

## 第 1 步：在 GitHub 网页创建空仓库

1. 打开 https://github.com/new
2. **Repository name**: `rec-agent-lab`
3. **Description**: `从零构建推荐系统到 LLM 推荐 Agent：多路召回 / SASRec / DIN / QLoRA 微调 / ReAct Agent / 量化部署`
4. 可见性：**Public**（秋招要让面试官能点开，私有仓库等于没有）
5. ❌ **不要勾选** Add a README file
6. ❌ **不要勾选** Add .gitignore
7. ❌ **不要勾选** Choose a license
   > 本地已有文件，勾选会造成 push 冲突，需要额外处理
8. 点击 **Create repository**

---

## 第 2 步：配置远端并推送

### 方案 A：SSH 直连（你已有 id_ed25519，首选）

```powershell
cd C:\Users\shizifan\Desktop\code\rec-agent-lab
git remote add origin git@github.com:szf0416/rec-agent-lab.git
git branch -M main
git push -u origin main
```

**先确认公钥已加到 GitHub**（否则报 `Permission denied (publickey)`）：

```powershell
# 打印公钥，复制全文
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
```

粘贴到 https://github.com/settings/keys → **New SSH key** → 标题填 `Windows-Desktop` → 保存。

**验证 SSH 通路**（这一步现在会卡住，见文末排障）：

```powershell
ssh -T git@github.com
# 成功时输出: Hi szf0416! You've successfully authenticated...
```

---

### 方案 B：HTTPS 走本地代理

如果你有代理工具（Clash / v2ray 等），先启动它，然后：

```powershell
# 假设代理端口是 7890，按你实际的改
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

# 验证
git ls-remote https://github.com/szf0416/rec-agent-lab.git

# 推送（首次会弹出凭据窗口）
cd C:\Users\shizifan\Desktop\code\rec-agent-lab
git remote add origin https://github.com/szf0416/rec-agent-lab.git
git branch -M main
git push -u origin main
```

> HTTPS 推送需要 **Personal Access Token** 作为密码（GitHub 已不支持账号密码）。
> 生成：https://github.com/settings/tokens → Generate new token (classic) → 勾 `repo` → 复制保存。
> 凭据会被 Windows 凭据管理器记住，只需输一次。

**用完代理后想关掉**：

```powershell
git config --global --unset http.proxy
git config --global --unset https.proxy
```

---

### 方案 C：Gitee 镜像（网络实在不通时的过渡）

```powershell
# 在 https://gitee.com/projects/new 建同名仓库后
git remote add gitee git@gitee.com:<你的Gitee用户名>/rec-agent-lab.git
git push -u gitee main
```

> ⚠️ 这只是过渡。秋招面试官看的是 GitHub，网络恢复后务必迁移过去：
> ```powershell
> git remote add origin git@github.com:szf0416/rec-agent-lab.git
> git push -u origin main
> ```

---

## 第 3 步：推送后立刻做的 3 件事

### 1. 确认 CI 跑起来了

打开 `https://github.com/szf0416/rec-agent-lab/actions`

- 应该看到 `CI` workflow 正在运行或已通过
- **如果失败，把报错贴给我**，这是 W1 的验收项之一

> CI 大概率会失败在 `pytest` 或 `mypy` 那一步 —— 因为你还没装环境、没写测试。
> 这很正常，属于 W1 要解决的问题，不用慌。

### 2. 完善仓库信息（门面）

在仓库页面右侧 **About** 齿轮里设置：

- **Description**: 同上面填的那句
- **Topics**: `recommender-system` `llm` `agent` `react-agent` `qlora` `sasrec` `pytorch` `rag`

Topics 会影响 GitHub 搜索曝光，也让面试官一眼看懂技术栈。

### 3. 检查 README 在网页上的渲染

特别注意架构图（代码块形式）和表格是否正常显示。

---

## 排障

### 现象：`ssh -T git@github.com` 卡住不动，最终 timeout

**已诊断的根因**（2025-09-28 实测）：

| 检查 | 结果 | 说明 |
| --- | --- | --- |
| TCP `github.com:443` | ✅ 通 | 网络本身没问题 |
| TCP `ssh.github.com:443` | ✅ 通 | 端口可达 |
| SSH 握手 | ❌ 卡死 | **TLS/SSH 层被干扰** |
| HTTPS `git ls-remote` | ❌ 超时 | 同上 |

TCP 通但握手过不去，是典型的**中间设备干扰**，不是你配错了。

**处理顺序**：

1. **开代理**（最有效）→ 走方案 B，或让 SSH 也走代理：
   ```powershell
   # 在 ~/.ssh/config 里追加（你的文件已有 github.com 段，补 ProxyCommand）
   # Host github.com
   #   Hostname ssh.github.com
   #   Port 443
   #   User git
   #   ProxyCommand connect -H 127.0.0.1:7890 %h %p
   ```
   > `connect` 是 Git for Windows 自带的，路径在 `C:\Program Files\Git\mingw64\bin\connect.exe`

2. **换网络**：手机热点有时能绕过，值得一试

3. **换时段**：晚高峰（20:00–23:00）最差，清晨或午后重试

4. **走镜像**：`https://gh-proxy.org/https://github.com/...`（你已有仓库在用这个）

### 现象：`remote origin already exists`

```powershell
git remote set-url origin git@github.com:szf0416/rec-agent-lab.git
```

### 现象：`failed to push some refs` / `rejected`

说明远端有本地没有的提交（多半是建仓库时勾了 README）。两个选择：

```powershell
# 选择一：强行覆盖远端（确认远端确实没有你要保留的东西）
git push -u origin main --force

# 选择二：先拉取合并
git pull --rebase origin main
git push -u origin main
```

### 现象：push 卡在 `Writing objects` 不动

仓库里有大文件。检查：

```powershell
git ls-files | ForEach-Object { [PSCustomObject]@{ F=$_; KB=[math]::Round((Get-Item $_).Length/1KB,1) } } | Sort-Object KB -Descending | Select-Object -First 5
```

应全部很小（< 10KB）。若发现大文件，说明 `.gitignore` 有漏，告诉我。

---

## 以后每次提交的标准流程

```powershell
cd C:\Users\shizifan\Desktop\code\rec-agent-lab
git status                        # 看改了什么
git add -A
git commit -m "feat(s1): 实现两层 MLP 反向传播与数值对齐测试"
git push
```

**提交信息规范**见 [REVIEW.md](REVIEW.md) 第 4 节。
type 用 `feat` / `fix` / `docs` / `refactor` / `test` / `chore` / `perf`，
禁止 `update`、`fix bug`、`修改`。

> 这是要给面试官看的提交历史 —— 它会暴露你的工作习惯。
> 规范的历史本身就是"工程素养"的证据。
