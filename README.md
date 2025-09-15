# AI Agent Memory Demo (ce101)

这是[猫书](https://ce101.ifuryst.com/)里第4章记忆系统与持久化中的实践案例，展示如何为AI Agent集成记忆系统。

## Quick Start

以下两种方式任选其一。

- 依赖
  - Python 3.10+
  - 或 Docker / Docker Compose
  - OpenAI 兼容的接口与密钥（设置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`）


### 方式一：Docker Compose（推荐）

1) 准备环境变量：`cp .env.example .env`，按需修改

2) 启动：

```
docker compose up -d db
# 运行一次性应用容器并进入对话 REPL
docker compose run --rm app python main.py
```

- 数据库会启用 `pgvector` 扩展，并在首次运行时自动建表。
- 如需持久化数据，`docker-compose.yaml` 已挂载卷。

### 方式二：本地运行（Python）

1) 启动数据库（使用官方 pgvector 镜像举例）：

```
docker run --name pgvector -p 65432:5432 -e POSTGRES_USER=demo -e POSTGRES_PASSWORD=demo123 -e POSTGRES_DB=langgraph_demo -d pgvector/pgvector:pg16
```

2) 安装依赖并运行：

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env

python main.py
```

3) 交互：
- 输入内容开始对话，输入 `exit` 或 `quit` 退出。


## 项目简介

- 目标：演示如何在对话中“抽取事实 → 写入向量库 → 语义检索 → 作为上下文辅助回答”。
- 技术栈：
  - LangGraph：对话图编排与状态管理（含 Postgres 检查点）。
  - Postgres + pgvector：长期记忆存储与相似度搜索。
  - LangChain OpenAI：聊天与向量嵌入（OpenAI 兼容）。

### 目录结构与模块职责

- `main.py`：精简入口，调用 `app.app.run()`。
- `app/app.py`：装配应用（配置、DB、Embedder、服务、图、REPL）。
- `app/config.py`：读取/校验环境变量，打印启动信息；支持路径：
  - `FACT_PROMPT_PATH`（默认 `prompts/fact_extraction.prompt`）
  - `SYSTEM_PROMPT_PATH`（默认 `prompts/system.prompt`）
- `app/embedding.py`：`Embedder` 封装嵌入向量生成与 pgvector 字面量转换。
- `app/db.py`：初始化数据库与 `facts` 表；`FactStore` 负责写入/检索。
- `app/facts.py`：LLM 事实抽取与 JSON 解析、去重。
- `app/llm_node.py`：`LLMService` 实际对话节点与消息拼装；`build_graph()` 构建图。
- `app/prompts.py`：通用提示词加载工具（带缓存）。
- `prompts/`：提示词文件夹：
  - `system.prompt`：系统人设/风格/行为规范（每轮对话最前注入）。
  - `fact_extraction.prompt`：事实抽取提示词（可选；缺失时有内置兜底）。

### 对话与记忆流程

1) 用户输入文本。
2) 调用 LLM 进行“事实抽取”（结构化 JSON），过滤空值并去重，逐条写入 `facts` 表（含嵌入向量）。
3) 对用户输入做语义检索，取最相关的若干条事实（默认 top-k=3）。
4) 组织 Prompt：`system prompt` → `相关信息（可选）` → `用户消息`，再请求聊天模型。
5) 返回模型输出，继续循环。

- 注意：相关信息并不强制使用；仅当判断“确实有帮助”时才融入回答（见 `prompts/system.prompt` 约束）。

## 许可证

本项目采用 [MIT 协议](./LICENSE)。