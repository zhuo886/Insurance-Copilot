# Insurance-Copilot · 智能保险顾问与搜推决策 Agent 系统

面向智能保险咨询、多产品条款深度检索、精算保费试算与个性化方案规划场景的端到端全栈 AI 应用。系统基于 **LangGraph 混合状态编排 + 跨产品混合 RAG + 四阶段工程化搜推漏斗** 构建，完整打通「自然语言画像抽取 → 多险种预算组合推荐 → 跨产品条款依据问答 → 确定性合规护栏纠偏 → HITL 人工审批落库 → 动态表单试算与幂等投保 → 真实保单查询与理赔服务」全业务闭环。

- **技术栈**：Python 3.14 · FastAPI · LangGraph · Milvus 3.0（Dense + BM25 Sparse） · PostgreSQL 16（SQLAlchemy Async + AsyncPostgresSaver） · JWT + Argon2 · React · TypeScript · Ant Design · Vite

---

## 一、系统架构设计

```mermaid
flowchart TD
    Client["前端 React / Ant Design (AdvisorPage & ApplicationPage)"] -->|"POST /api/v1/chat (SSE 流 / Command resume) & REST API"| ChatAPI["FastAPI (JWT 鉴权 / ChatService 节点级缓冲 / Checkpointer 恢复)"]
    ChatAPI --> Route["主图 route 节点 (规则寒暄直出 + 结构化意图路由)"]

    Route -->|chitchat| Chitchat["chitchat 子图"]
    Route -->|human_handoff / fallback| Handoff["转人工上下文摘要 / 服务边界引导"]
    Route -->|claim| Claim["claim 理赔子图 (LangGraph 子图 + 确定性引用护栏 + 保单/理赔/指南查询)"]
    Route -->|recommendation_plan| AdvisorSubgraph

    subgraph AdvisorSubgraph ["核心顾问混合子图 (recommendation_plan)"]
        Extract["1. extract_profile (零延迟正则增量抽取 UserProfile，不阻断纯问答)"]
        Extract --> Agent["2. advisor_agent (ReAct 多工具协同)"]
        Agent -->|"调用工具"| Tools["搜推组合 / 并行条款RAG / 查表保费试算 / 提交方案草稿"]
        Tools --> Collect["收集 retrieved_source_ids & plan_draft (触发 guardrail_buffer)"]
        Collect --> Agent
        Agent --> Verify{"3. verify_response 确定性护栏"}
        Verify -->|"引文违规或违背硬约束 (重试=0)"| Agent
        Verify -->|"重试耗尽：fail-closed 剥离伪引文或确定性拒批替换"| Deliver["4a. deliver_answer (输出回复 + 持久化 additional_info)"]
        Verify -->|"纯问答/追问校验通过"| Deliver
        Verify -->|"合法方案草稿通过校验"| Confirm["4b. plan_confirm (LangGraph interrupt 挂起等待审批)"]
        Confirm -->|approve| SavePlan["5. save_plan (写 insurance_plans & items + 推送投保卡片)"]
        Confirm -->|"reject + reject_reason"| Agent
    end

    Tools -.-> RecSys["四阶段搜推漏斗 (12款真实产品 + 11/12 官方费率表)"]
    Tools -.-> RAG["跨产品混合 RAG (术语改写 + 并行定额 + 路径去重 + Rerank)"]
    RecSys -.-> PG["PostgreSQL 16 & Milvus 3.0"]
    RAG -.-> PG
    SavePlan -.-> PG
    Claim -.-> PG
```

---

## 二、产品库收口与工程优化要点

### 1. 真实产品目录与官方精算费率表（12 款）

- **去虚构留真实**：移除早期全部 38 款无条款 PDF 的扩展产品（`ID 13-50`），`load_builtin_catalog()` 严格收敛为 **12 款真实合同产品（`ID 1-12`）**（[catalog_data.py](agent-service/app/modules/product/catalog_data.py)）。
- **官方费率表解析接入**：通过 [parse_rate_tables.py](agent-service/scripts/parse_rate_tables.py) 解析 10 份官方精算费率表 PDF，产出结构化费率字典 [rate_tables.json](agent-service/data/parsed/rate_tables.json)，覆盖 **11/12 款真实产品（`ID 2-12`）**：
  - `mode="table"`（`ID 2-12`）：按性别、投保年龄、缴费期、保障期、吸烟/社保状态/版本档位等真实维度的查表保费；
  - `mode="estimated_formula"`（仅 `ID 1` 达尔文宝贝计划 12 号，缺失官方费率 PDF）：显式返回 `is_estimated=True` 与 `pricing_disclaimer` 免责声明，杜绝无依据静默编造（[pricing.py](agent-service/app/recsys/pricing.py)）。

### 2. LangGraph 确定性护栏与理赔子图升级

- **双层引用与承保硬约束护栏**（[guardrails.py](agent-service/app/agents/guardrails.py)、[insurance_advisor.py](agent-service/app/agents/insurance_advisor.py)）：
  - `extract_all_citation_tokens` 与 `verify_citations` 全量拦截 `[ref-xxx]`、十六进制伪编号（如 `[d64a1]`、`[ref-f3b6f8e0]`）及未触发 RAG 时擅自输出引文编号（`rag_used=False`）；
  - 重试耗尽 fail-closed 时，`sanitize_citations` 仅确定性剥离伪造或超范围编号，**绝不在文末强行拼凑无关依据脚标**；
  - 对未命中的约束方案草稿先引导检索合规替代方案，再于必要时触发确定性拒批替换。
- **理赔子图接入护栏**（[claim_agent.py](agent-service/app/agents/claim_agent.py)、[chat/service.py](agent-service/app/modules/chat/service.py)）：
  - 将 `claim_agent` 升级为完整 LangGraph 子图（`build_claim_agent_graph`），新增 `get_claim_guide` 工具并复用同套引用护栏与节点级 `guardrail_buffer`。
- **评测口径去水分**（[eval_agent.py](agent-service/scripts/eval/eval_agent.py)）：
  - 切换为异构 `qwen-max` 作为 Faithfulness 裁判（避免同模自评偏袒），并拆分「仅合规拒绝」与「合规且有效给出替代方案」指标。
- **当前实测状态**：
  > 本机 PostgreSQL (`5432`) 与 Milvus (`19530`) Docker 服务当前处于停止状态，`eval_agent.py` 尚未在 12 产品与新护栏口径下重新执行；仓库历史遗留的 [agent_eval.md](agent-service/reports/agent_eval.md) 为重构前旧快照，待启动 Docker 完成联机同步与评测后刷新。

### 3. 跨产品混合 RAG 检索增强（拒绝无依据兜底）

- **检索链路**（[query_rewriter.py](agent-service/app/rag/query_rewriter.py)、[retriever.py](agent-service/app/rag/retriever.py)）：
  - 保险领域词典零延迟同义词改写；
  - 多产品对比按产品 `asyncio.gather` 并行定额检索并 Round-Robin 交错合并，结合 `(clause_name, section_path)` 语义路径去重与 `gte-rerank-v2` 重排；
  - **彻底删除无依据条款兜底生成逻辑**：检索无结果直接返回空列表并明确提示未检索到对应条款依据，不再让大模型根据商品摘要字段编造条款细节。
- **单变量消融框架与实测状态**：
  - [eval_rag.py](agent-service/scripts/eval/eval_rag.py) 已重构为 7 路单变量消融对比（`dense_only`、`hybrid_no_rerank`、`hybrid_rewrite_no_rerank`、`hybrid_rewrite_parallel_dedup`、`full_enhanced`、`hybrid_rerank_single`、`hyde_hybrid_rerank`），以便准确归因各组件贡献与 BM25/Rerank 影响。
  > 同受本地 Docker 未启动影响，新版 7 路单变量 RAG 消融暂未运行；仓库现有 [rag_eval.md](agent-service/reports/rag_eval.md) 为重构前历史快照，待启动 Docker 后复测更新。

### 4. 四阶段搜推漏斗与最新 12 产品实测报告

- **核心链路**（[app/recsys/](agent-service/app/recsys)）：多路召回 RRF 融合 → 年龄/职业/险种承保硬过滤 → 5 维可解释精排 → MMR 多样性重排 + 基于 `11/12` 官方费率表的预算分组背包优化。
- **最新实测对比**（30 条独立人工标注场景画像，12 款真实合同产品，报告见 [recsys_eval.md](agent-service/reports/recsys_eval.md)）：
  - 口径说明：本评测使用专家独立预标注的相关性与承保红线作为离线合成基准（非线上真实点击日志）；「承保资格合规率」用于量化工程硬过滤门禁修复原版 SQL 漏过滤缺陷的效果。

| 对比方案 | Recall@3 (%) | NDCG@5 | MRR | 承保资格合规率 (%) | 组合预算满足率 (%) | 可覆盖险种完成率 (%) | P95 耗时 (ms) | 平均排序 Token |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. 原版 SQL 规则排序 (Baseline) | 48.80 | 0.6294 | 0.6278 | 20.00 | 76.67 | 100.00 | 0.01 | 0.0 |
| 2. 单路语义召回 + 硬过滤 | 67.00 | 0.8984 | 0.9667 | 100.00 | 96.67 | 98.33 | 3.77 | 0.0 |
| 3. 多路 RRF + 5 维精排 (贪心无背包) | 71.44 | 0.9204 | 0.9444 | 100.00 | 93.33 | 98.33 | 6.34 | 0.0 |
| **4. 完整四阶段漏斗 (+MMR+预算背包)** | **71.44** | **0.9204** | **0.9444** | **100.00** | **100.00** | **97.22** | **6.74** | **0.0** |
| 竞品. 纯 LLM Listwise 直排 | 63.11 | 0.8223 | 0.9833 | 100.00 | 83.33 | 98.33 | 1807.89 | 1288.6 |

- **量化结论**：完整四阶段漏斗较原版 SQL 基线将 Recall@3 从 `48.80%` 提升至 **`71.44%`**、NDCG@5 从 `0.6294` 提升至 **`0.9204`**、承保合规率从 `20.00%` 提升至 **`100.00%`**、组合预算满足率从 `76.67%` 提升至 **`100.00%`**；相比纯 LLM Listwise 直排，Recall@3 高 `+8.33 pp`、NDCG@5 高 `+0.0981`、预算满足率高 `+16.67 pp`，P95 延迟降低 **`99.63%`**（`1807.89 ms → 6.74 ms`）且排序阶段零 Token 消耗。

---

## 三、核心 REST 业务接口一览（共 31 个路由）

系统已对照全栈接口契约补齐前后端真实联调闭环（密码采用 Argon2 哈希，Refresh Token 使用 `HttpOnly` Cookie，受保护接口支持 JWT Bearer 鉴权）：

| 业务域 | 方法 | 路径 | 说明 |
| :--- | :---: | :--- | :--- |
| **系统状态** | `GET` | `/healthy` | 健康检查接口 |
| **用户认证** | `POST` | `/api/v1/auth/register` | 用户注册（签发 Access Token + Refresh Cookie） |
| | `POST` | `/api/v1/auth/login` | 用户登录（支持已有 `demo` 账号哈希透明升级） |
| | `POST` | `/api/v1/auth/refresh` | 刷新 Access Token |
| | `POST` | `/api/v1/auth/logout` | 退出登录并清除 Refresh Cookie |
| **产品中心** | `GET` | `/api/v1/products` | 分页查询 12 款真实保险产品 |
| | `GET` | `/api/v1/products/candidates` | 按险种与用户属性筛选候选产品 |
| | `GET` | `/api/v1/products/{product_id}` | 查询产品详情 |
| | `GET` | `/api/v1/products/{product_id}/pricing-schema` | 获取动态保费试算表单结构 |
| **保费试算** | `POST` | `/api/v1/premium/calculate` | 官方费率查表试算（ID 1 显式返回估算标识与声明） |
| **保障方案** | `GET` | `/api/v1/insurance-plans` | 列出当前用户历史保障方案 |
| | `GET` | `/api/v1/insurance-plans/current` | 获取当前生效保障方案及明细 |
| | `PUT` | `/api/v1/insurance-plans/current` | 保存/更新当前保障方案 |
| | `GET` | `/api/v1/insurance-plans/{plan_id}` | 获取指定方案详情 |
| | `PATCH` | `/api/v1/insurance-plans/{plan_id}/items/{item_id}/insured` | 标记方案子项已投保并生成 V14 完整保单快照 |
| **保单中心** | `GET` | `/api/v1/policies` | 查询当前用户全部真实保单 |
| | `GET` | `/api/v1/policies/{policy_id}` | 查询指定保单详情 |
| **理赔服务** | `GET` | `/api/v1/claims` | 查询理赔记录列表 |
| | `GET` | `/api/v1/claims/{claim_id}` | 查询理赔案件详情 |
| | `GET` | `/api/v1/claim-guides/{policy_id}` | 按保单查询理赔指引与材料清单 |
| | `GET` | `/api/v1/claim-guides/application-form` | 下载标准 PDF 理赔申请书（`claim-application-form.pdf`） |
| **投保订单** | `GET` | `/api/v1/orders` | 查询投保订单列表 |
| | `POST` | `/api/v1/orders` | 创建投保订单（支持幂等键校验与保费复核） |
| | `GET` | `/api/v1/orders/{order_id}` | 查询投保订单详情 |
| | `POST` | `/api/v1/orders/{order_id}/cancel` | 撤单（级联校验并同步更新保单状态） |
| **智能对话** | `POST` | `/api/v1/chat` | SSE 流式对话与 LangGraph HITL `Command(resume=...)` |
| | `POST` | `/api/v1/chat-threads` | 创建会话线程 |
| | `GET` | `/api/v1/chat-threads` | 列出用户会话线程 |
| | `PATCH` | `/api/v1/chat-threads/{thread_id}` | 重命名会话线程 |
| | `DELETE` | `/api/v1/chat-threads/{thread_id}` | 删除会话线程 |
| | `GET` | `/api/v1/chat-threads/{thread_id}/messages` | 获取消息历史与挂起 `interrupt` 恢复状态 |

---

## 四、项目目录结构与工程规范

```text
hm-insurance/
├── .github/workflows/ci.yml       # GitHub Actions：后端 26 项 pytest + 前端 tsc/vite 构建
├── LICENSE                        # MIT 开源协议
├── deploy/                        # Docker Compose 基础设施（Postgres 16 / Milvus 3.0 / MinIO / Attu）
│   ├── docker-compose.yml         # 强制通过 deploy/.env 注入密钥，不保留明文默认密码
│   ├── .env.example               # 基础设施密码/端口环境变量模板
│   ├── milvus/                    # Milvus 用户配置模板（user.yaml.example）
│   └── postgres/init/             # 数据库建表（含 V5/V6/V14 列、订单/理赔/保单/费率表）与演示种子 SQL
├── agent-service/                 # Python 后端核心服务（含 Dockerfile 与 .dockerignore）
│   ├── app/
│   │   ├── agents/                # 主图编排、顾问混合子图、确定性护栏、理赔子图与工具集
│   │   ├── rag/                   # 保险术语改写、并行定额混合检索、去重与重排管道
│   │   ├── recsys/                # 四阶段搜推漏斗、分组背包预算优化与官方费率表定价引擎
│   │   ├── modules/               # auth / product / premium / insurance_plan / policy / claim / order / chat / chat_thread
│   │   ├── core/ & infra/ & common/
│   │   └── main.py                # FastAPI + AsyncPostgresSaver 应用入口
│   ├── data/
│   │   ├── parsed/rate_tables.json # 11/12 款产品结构化官方费率表（原始涉版权 PDF 已移出 Git 追踪）
│   │   └── assets/                # 真实理赔申请表 PDF 模板
│   ├── scripts/                   # parse_rate_tables / seed_catalog / ingest_product_catalog / eval/*
│   ├── reports/                   # recsys_eval.{md,json}（12产品已更新）及待复测的 rag/agent 快照
│   └── tests/                     # 26 条纯离线单测与全业务接口端到端测试（pytest + MemorySaver）
└── frontend/                      # React + TypeScript + Ant Design 前端（登录注册 / 智能顾问 / 试算下单闭环）
```

> **仓库脱敏说明**：原始商业保险条款与费率表 PDF（`agent-service/data/raw/`）因体积与版权原因已通过 `git rm --cached` 移出版本库追踪并纳入 `.gitignore`，仅保留脱敏后的结构化 JSON、Markdown 解析产物与标准申请表模板；容器配置已移除全部明文默认密码，改为通过 `deploy/.env` 注入。

---

## 五、本地快速启动指南

### 1. 环境配置

```bash
# 1) 复制后端环境变量模板并填入 DASHSCOPE_API_KEY、MINERU_TOKEN、DB_PASSWORD 等配置
cp agent-service/.env.example agent-service/.env

# 2) 复制基础设施环境变量模板并填入 POSTGRES_PASSWORD 与 MINIO_ROOT_PASSWORD
cp deploy/.env.example deploy/.env

# 3) 复制 Milvus 配置模板并填入对应的 dashscope_apikey
cp deploy/milvus/user.yaml.example deploy/milvus/user.yaml
```

### 2. 启动基础中间件（Docker Compose）

```bash
cd deploy
docker compose up -d
```

服务端口说明：PostgreSQL `5432`、Milvus `19530`、MinIO `9000/9001`、Attu `3000`。

### 3. 初始化后端数据与启动服务

```bash
cd agent-service
uv sync --dev

# 写入 12 款真实产品、清理历史虚构款、同步 11/12 官方费率表并升级演示账号哈希
uv run python scripts/seed_catalog.py

# 构建 Milvus 12 款产品检索向量索引（insurance_product_catalog）
uv run python scripts/ingest_product_catalog.py

# （如需重新导入条款知识库，请先将本地条款 PDF 放入 agent-service/data/raw/kb 后执行）
# uv run python scripts/ingest_rag.py

# 启动 FastAPI 服务（默认监听 http://127.0.0.1:8001）
uv run python -m app.main
```

### 4. 启动前端开发服务器

```bash
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173`（Vite 已默认配置 `/api` 反向代理至 `http://127.0.0.1:8001`）。

---

## 六、离线测试与实验复现

```bash
cd agent-service

# 1. 运行全部 26 条纯离线单测与全业务 REST 接口端到端测试（无需 Docker，0 Token）
uv run pytest -v

# 2. 复现 12 产品四阶段搜推消融（离线特征 + 费率表，无需 Docker；--with-llm-listwise 将请求通义千问）
uv run python scripts/eval/eval_recsys.py

# 3. 启动 Docker（Postgres + Milvus）后，复现 RAG 7 路单变量消融与 Agent 异构裁判评测
uv run python scripts/eval/eval_rag.py
uv run python scripts/eval/eval_agent.py
```

---

## 七、演进里程碑

- **`v0.1-baseline`**：基线仓库初始化与敏感密钥隔离规范落地。
- **`v0.2-p0-foundation-recsys`**：状态持久化/接口契约修复、保费估算与四阶段搜推漏斗原型验证。
- **`v0.3-p1-rag-and-agent`**：跨产品并行混合 RAG、LangGraph 混合顾问子图与确定性护栏、HITL 审批落库。
- **`v1.0-polished-demo`**：基础全链路文档化与前端交互验收。
- **`v1.1-real-catalog-and-full-api`**：全面剔除 38 款虚构产品收敛至 12 款真实合同、接入 11/12 官方 PDF 费率表、修复引用护栏与无依据兜底漏洞、升级理赔子图并接入护栏、按全量接口文档打通 JWT 认证/试算/方案保单/理赔指南/下单撤单前后端闭环、新增 MIT License / CI / Dockerfile 与容器脱敏规范。
