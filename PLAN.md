# CompanionAI — 项目实战计划

> 基于 LangChain + ChromaDB 的 AI 陪伴角色对话系统
> 对应简历项目：CompanionAI - 基于LLM的AI陪伴角色对话系统

---

## 总体目标

构建一个 AI 陪伴角色对话系统，核心能力：
- **长期记忆**：跨会话保持角色记忆（ChromaDB 向量数据库）
- **情绪状态机**：根据对话内容动态切换角色情绪
- **个性化对话风格**：通过 Prompt 模板与角色设定控制回复风格
- **多模态扩展入口**：预留语音交互接入点（Whisper / TTS）

---

## 阶段一：环境与基础（第 1 天）

### Step 1.1 — 项目初始化
- [ ] 创建项目目录结构与虚拟环境
- [ ] 初始化 `requirements.txt` / `pyproject.toml`
- [ ] 安装核心依赖

**依赖清单：**
```txt
langchain>=0.3.0
langchain-community
langchain-openai           # OpenAI / 兼容 API
chromadb>=0.5.0
openai                     # OpenAI SDK
fastapi                    # API 服务
uvicorn                    # ASGI 服务器
pydantic                   # 数据模型
python-dotenv              # 环境变量管理
```

### Step 1.2 — LLM 连接验证
- [ ] 配置 API Key（通过 `.env` + `python-dotenv` 加载）
- [ ] 写一个简单的 LLM 调用脚本，验证连通性
- [ ] 测试流式响应（Streaming）

**产物：** `test_llm.py` — 能跑通一次对话

---

## 阶段二：基础对话链（第 1-2 天）

### Step 2.1 — LangChain Chat 链
- [ ] 使用 `ChatOpenAI` 构建基础聊天模型
- [ ] 设计 `ChatPromptTemplate`，包含角色设定占位符
- [ ] 实现 `StrOutputParser` 解析输出

### Step 2.2 — 对话历史管理
- [ ] 使用 `ChatMessageHistory` 管理当前会话上下文
- [ ] 实现窗口滑动（只保留最近 N 轮对话）
- [ ] 将对话历史注入 Prompt

**产物：** `chat_basic.py` — 多轮对话，有上下文记忆

---

## 阶段三：长期记忆系统（第 2-3 天）

### Step 3.1 — ChromaDB 集成
- [ ] 初始化 ChromaDB 持久化客户端
- [ ] 设计 `Memory` 集合的 document schema
- [ ] 实现记忆写入：对话摘要 + 关键事件提取

### Step 3.2 — RAG 检索增强
- [ ] 每次对话前，从 ChromaDB 检索相关历史记忆
- [ ] 将检索结果注入 Prompt 作为上下文
- [ ] 控制检索数量与相似度阈值

### Step 3.3 — 记忆管理策略
- [ ] 自动摘要：当对话超过阈值时，自动总结并存入长时记忆
- [ ] 记忆优先级：重要事件 vs 日常闲聊的分级存储
- [ ] 遗忘机制：定期清理低优先级记忆

**产物：** `memory_system.py` — 完整的 RAG 记忆模块

---

## 阶段四：情绪系统（第 3-4 天）

### Step 4.1 — 情绪状态机设计
- [ ] 定义情绪状态：愉悦 / 平静 / 悲伤 / 惊讶 / 生气
- [ ] 设计状态转换规则（基于用户输入内容的情感分析）
- [ ] 实现状态持久化（同步存储到 ChromaDB）

### Step 4.2 — 情绪影响对话
- [ ] 将当前情绪注入 Prompt，影响回复语气
- [ ] 为不同情绪设计不同的回复风格指令
- [ ] 测试情绪切换的流畅性

**产物：** `emotion_system.py` — 情绪状态机模块

---

## 阶段五：Agent 框架（第 4-5 天）

### Step 5.1 — 角色行为决策
- [ ] 使用 LangChain Agent 框架
- [ ] 定义角色工具：查询记忆 / 更新情绪 / 主动发起对话
- [ ] 实现角色自主行为（如主动问候、回忆往事）

### Step 5.2 — Prompt 模板系统
- [ ] 设计多层级 Prompt 结构：
  - 系统层：角色人格设定
  - 记忆层：检索到的长期记忆
  - 情绪层：当前情绪状态
  - 对话层：最近 N 轮对话历史
  - 用户层：当前用户输入
- [ ] 支持灵活组合与热切换角色设定

**产物：** `agent_core.py` — Agent 核心框架

---

## 阶段六：API 与界面（第 5-6 天）

### Step 6.1 — FastAPI 后端
- [ ] 实现 RESTful API：`POST /chat`、`GET /history`、`POST /reset`
- [ ] WebSocket 支持流式响应
- [ ] 请求/响应 Pydantic 模型

### Step 6.2 — 简易 Web 界面
- [ ] 使用 Gradio 或 HTML + JS 构建聊天界面
- [ ] 显示角色头像、情绪状态指示器
- [ ] 展示对话历史与记忆摘要

**产物：** `api_server.py` + `web_app.py` — 可运行的完整系统

---

## 阶段七：多模态扩展（第 6-7 天，可选）

### Step 7.1 — 语音输入
- [ ] 集成 Whisper API 实现语音识别
- [ ] 音频文件 → 文字的管道

### Step 7.2 — 语音输出
- [ ] 集成 Edge-TTS / CosyVoice 实现语音合成
- [ ] 文字 → 音频的流式播放

**产物：** `voice_module.py` — 语音交互模块

---

## 项目结构（最终预期）

```
companion-ai/
├── .env                      # API Key 等环境变量
├── requirements.txt          # 依赖清单
├── PLAN.md                   # 本计划文件
├── test_llm.py               # Step 1.2: LLM 连通性验证
├── chat_basic.py             # Step 2.x: 基础对话链
├── memory_system.py          # Step 3.x: ChromaDB 记忆模块
├── emotion_system.py         # Step 4.x: 情绪状态机
├── agent_core.py             # Step 5.x: Agent 框架
├── api_server.py             # Step 6.1: FastAPI 后端
├── web_app.py                # Step 6.2: Web 界面
├── voice_module.py           # Step 7.x: 语音扩展
├── chroma_db/                # ChromaDB 持久化数据（自动生成）
└── prompts/                  # Prompt 模板目录（可选）
    ├── system_prompt.txt
    └── emotion_prompts/
```

---

## 前置准备

### 你需要准备
- OpenAI API Key（或兼容的 LLM API Key）
- 如果想完全本地运行，可以替换为 Ollama + 本地模型，到时我们再调

### 环境要求
- Python 3.10+
- 网络连通（需要访问 LLM API）

---

## 接下来做什么

从 **Step 1.1** 开始，我们先：
1. 建好项目文件夹
2. 装好依赖
3. 验证 LLM 连通性

准备好了就说一声，我们直接动手敲代码。
