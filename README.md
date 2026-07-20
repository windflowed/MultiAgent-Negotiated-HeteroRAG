# MultiAgent-Negotiated-HeteroRAG

基于多Agent协商机制的异构数据源融合RAG问答系统

## 项目简介

企业内部数据普遍分散在结构化数据库（SQL）、非结构化文档、知识图谱三类异构载体中。传统RAG系统通常只支持单一数据源检索，当查询需要跨数据源综合回答时，存在**模态鸿沟**、**信息冲突**、**检索策略僵化**三大核心痛点。

本系统通过6个专用Agent分工协作，搭配**置信度加权表决**的协商融合算法，实现跨源信息的语义统一、冲突自动消解、检索策略动态调度，弥补传统单源RAG在复杂查询场景下的能力短板。

### 核心特性

- **异构多源检索**：同时对接SQLite结构化数据库、ChromaDB向量文档库、NetworkX内存知识图谱三类异构数据源
- **动态路由调度**：基于LLM零样本分类的查询意图识别，按需激活数据源，降低无效检索
- **协商融合消解**：事实三元组统一抽取 + 数值类冲突检测 + 置信度加权表决（SQL=0.5, Doc=0.3, KG=0.2）
- **答案可溯源**：每处事实陈述标注对应数据源编号（如`[SQL]`、`[DOC]`、`[KG]`），支持追溯到具体数据类型
- **KG实体消歧**：中英文实体名自动对齐（如"詹姆斯·卡梅隆"→"James Cameron"），支持1跳、2跳、共同邻居三类图谱推理

## 技术栈

| 类别 | 技术组件 | 版本 |
|------|----------|------|
| 核心框架 | LangChain | 0.3.0 |
| 编排框架 | LangGraph | 0.2.0 |
| 社区工具 | langchain-community | 0.3.0 |
| 向量数据库 | ChromaDB | 0.5.0 |
| 结构化数据库 | SQLite (SQLAlchemy) | 2.0.30 |
| 图计算库 | NetworkX | 3.3 |
| 中文嵌入模型 | BGE-small-zh-v1.5 | - |
| 关键词检索 | rank-bm25 | 0.2.2 |
| Wikipedia抓取 | wikipedia-api | 0.6.0+ |
| HTTP请求 | requests | 2.31.0+ |
| 配置管理 | python-dotenv | 1.0.1 |
| 前端框架 | Gradio | 4.44.0 |
| 开发语言 | Python | 3.11+ |

## 系统架构

```
用户查询
  ↓
┌─────────────────────────────────────────────┐
│  路由Agent（LLM零样本意图分类）              │
│  → 判断需要激活的数据源列表                   │
└─────────────────┬───────────────────────────┘
                  ↓ 条件边动态路由
    ┌─────────────┼─────────────┐
    ↓             ↓             ↓
┌────────┐  ┌────────┐  ┌──────────┐
│SQL Agent│  │文档Agent│  │KG Agent  │
│自然语言 │  │BM25+向量│  │实体链接+ │
│→SQL     │  │混合检索 │  │图谱推理  │
└────┬───┘  └────┬───┘  └────┬─────┘
     └───────────┼───────────┘
                 ↓ 并行汇合
┌─────────────────────────────────────────────┐
│  协商融合Agent                               │
│  三元组统一抽取 → 冲突检测 → 置信度加权消解   │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  答案生成Agent                               │
│  基于融合上下文生成答案，标注来源编号          │
└─────────────────┬───────────────────────────┘
                  ↓
             最终答案 + 来源
```

### LangGraph状态流转

```python
# AgentState 核心字段
AgentState = {
    "query": str,                 # 用户查询
    "routed_sources": List[str],  # 激活的数据源 ["sql", "doc", "kg"]
    "sql_results": dict,          # SQL检索结果
    "doc_results": dict,          # 文档检索结果
    "kg_results": dict,           # KG检索结果
    "fused_context": str,         # 融合后的统一上下文
    "triples": List[dict],        # 抽取的事实三元组
    "conflicts": List[dict],      # 检测到的冲突
    "answer": str,                # 最终答案
    "sources": List[str],         # 引用来源列表
    "final_confidence": float     # 最终置信度
}
```

## 快速开始

### 环境要求

- Python 3.11+
- DeepSeek API Key（注册 https://platform.deepseek.com 获取）

### 安装与配置

```bash
# 1. 克隆仓库
git clone https://github.com/windflowed/MultiAgent-Negotiated-HeteroRAG.git
cd MultiAgent-Negotiated-HeteroRAG

# 2. 创建虚拟环境
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量（在项目根目录创建 .env 文件，填入 DeepSeek API Key）
```

### .env 配置项

```env
LLM_API_KEY=your_deepseek_api_key
LLM_API_BASE=https://api.deepseek.com
LLM_MODEL_NAME=deepseek-v4-flash
TMDB_API_KEY=your_tmdb_api_key
DATA_DIR=./data
CHROMA_DB_DIR=./chroma_db
SQLITE_DB_PATH=./data/sql/movies.db
MODEL_DIR=./models
WIKI_LANGUAGE=zh
```

### 数据构建

```bash
# 一键构建三类数据源（SQLite + ChromaDB向量库 + NetworkX知识图谱）
python data/preprocess.py
```

构建内容：
- SQLite数据库：TMDB 5000电影数据（含中文标题、评分、票房、演员、导演）
- ChromaDB向量库：中文维基百科电影词条（BGE-small-zh-v1.5向量化）
- NetworkX知识图谱：电影-演员-导演-类型关系图（约26000节点、111000条边）

### 运行系统

```bash
# 方式一：命令行交互
python -m src.app

# 方式二：Gradio Web界面
python -m src.app_gradio
# 浏览器访问 http://localhost:7860
```

### 示例查询

| 查询类型 | 示例问题 |
|----------|----------|
| 纯SQL | 评分最高的5部电影是什么？ |
| 纯文档 | 阿凡达的剧情介绍 |
| 纯KG | 詹姆斯·卡梅隆导演了哪些电影？ |
| 多源融合 | 泰坦尼克号的导演还拍过哪些电影？ |

## 核心模块说明

### KG实体消歧机制

知识图谱中的演员和导演节点使用英文名存储（如"James Cameron"），而用户查询通常使用中文名（如"詹姆斯·卡梅隆"）。系统通过`_disambiguate_entities()`方法实现中英文实体对齐：

1. **LLM实体提取**：从用户查询中提取实体名称及英文名
2. **精确匹配**：优先在KG图谱中查找精确匹配的英文节点名
3. **大小写不敏感匹配**：对英文名进行大小写归一化后匹配
4. **直接匹配**：尝试直接用中文名在图谱中查找（适用于电影节点）

### BGE中文嵌入模型

系统使用`BAAI/bge-small-zh-v1.5`作为中文向量化模型，特点：
- 专为中文优化的512维嵌入模型
- 通过`sentence-transformers`库加载，本地CPU即可运行
- 支持中文语义相似度计算，检索效果优于通用多语言模型
- 集成于ChromaDB向量库，通过`SentenceTransformerEmbeddingFunction`接口接入

### 协商融合算法

1. **三元组统一抽取**：将SQL数值结果、文档文本片段、KG三元组统一转化为`(主体, 属性, 值)`标准格式
2. **冲突检测**：主体语义相同 + 属性语义相同 + 数值偏差>5% → 判定为冲突
3. **置信度加权表决**：`最终置信度 = 数据源基础权重 × 单条结果置信度`
   - SQL结构化数据：0.5（事实性最强）
   - 非结构化文档：0.3（描述性信息）
   - 知识图谱：0.2（推理结果）
4. **消解规则**：冲突时取置信度最高的结果；差值<0.1则保留多种说法

## 目录结构

```
MultiAgent-Negotiated-HeteroRAG/
├── src/                        # 核心源码
│   ├── agents/                 # 6个Agent实现
│   │   ├── sql_agent.py        # SQL查询Agent（自然语言→SQL）
│   │   ├── doc_retriever.py    # 文档检索Agent（BM25+向量混合检索）
│   │   ├── kg_agent.py         # 知识图谱Agent（实体链接+图谱推理）
│   │   ├── router_agent.py     # 查询路由Agent（LLM意图分类）
│   │   ├── fusion_agent.py     # 协商融合Agent（三元组抽取+冲突消解）
│   │   └── answer_generator.py # 答案生成Agent
│   ├── graph/                  # LangGraph工作流编排
│   │   ├── state_schema.py     # AgentState全局状态定义
│   │   └── workflow.py         # 工作流图构建与编译
│   ├── utils/                  # 通用工具
│   │   ├── prompts.py          # 所有Prompt模板集中管理
│   │   ├── common.py           # SQLite封装、配置加载、单例管理
│   │   ├── kg_utils.py         # NetworkX知识图谱构建与查询
│   │   ├── chroma_utils.py     # ChromaDB向量库封装
│   │   └── bm25_utils.py       # BM25检索器封装
│   ├── app.py                  # 命令行交互入口
│   └── app_gradio.py           # Gradio Web界面入口
├── data/                       # 数据与预处理
│   ├── sql/                    # SQLite数据库文件
│   ├── documents/              # 中文Wikipedia电影词条
│   ├── kg/                     # 知识图谱pickle文件
│   ├── raw/                    # 原始TMDB数据集
│   └── preprocess.py           # 一键数据预处理入口
├── eval/                       # 评测模块
│   ├── test_dataset.json       # 50题评测集
│   ├── baseline.py             # 三组基线系统
│   └── metrics.py              # 评价指标计算
├── config/
│   └── settings.example.py     # 配置示例文件
├── docs/                       # 项目文档
│   ├── PRD.md                  # 产品需求文档
│   ├── TASK.md                 # 开发任务清单
│   ├── SUMMARY.md              # 项目总体信息
│   └── AGENTS.md               # Agent角色定义
├── requirements.txt            # 固定版本依赖清单
├── .env                        # 环境变量配置（不提交Git）
├── .gitignore
├── LICENSE                     # MIT License
└── README.md
```

## 实验评估

### 评测集

50题标准评测集，覆盖5类查询场景：

| 场景 | 数量 | 考察能力 | 示例 |
|------|------|----------|------|
| 纯SQL可答 | 10 | 结构化数据查询 | 评分最高的5部电影 |
| 纯文档可答 | 10 | 非结构化文本检索 | 阿凡达的剧情介绍 |
| 纯KG可答 | 10 | 关系推理 | 詹姆斯·卡梅隆导演了哪些电影 |
| SQL+文档联合 | 10 | 双源融合 | 恐怖类票房最高的电影剧情简介是什么 |
| 三者联合 | 10 | 全源融合+冲突消解 | 列举某导演所有高评分电影及主演和剧情 |

### 评价指标

- **答案准确率（Factuality Score）**：事实性陈述的正确比例
- **信息完整度（Coverage Rate）**：标准答案关键信息点的覆盖比例
- **冲突消解正确率**：植入冲突用例中正确消解的比例
- **平均响应时延**：端到端查询的平均耗时

### 对比基线

- **基线1**：单模态RAG（仅文档检索）
- **基线2**：简单拼接式多源RAG（三路结果直接拼接，无协商融合）
- **基线3**：无动态路由的多Agent RAG（每次激活全部数据源）

## 后续优化

- 接入更多异构数据源（如PDF文档、网页内容）
- 优化复杂语义冲突检测（当前仅支持数值类冲突）
- 增量数据实时更新支持
- 引入Reranker微调提升重排序效果
- 支持多轮对话与上下文记忆
- 分布式部署与云端推理

## 参考资料

- LangChain 1.0 官方文档：https://python.langchain.com
- LangGraph 官方文档：https://langchain-ai.github.io/langgraph/
- BGE-small-zh-v1.5：https://huggingface.co/BAAI/bge-small-zh-v1.5
- BGE-Reranker-v2-m3：https://huggingface.co/BAAI/bge-reranker-v2-m3
- ChromaDB：https://docs.trychroma.com
- NetworkX：https://networkx.org
- 《Building Hierarchical Agentic RAG Systems》(InfoQ, 2026)
- 《A Collaborative Multi-Agent Approach to RAG Across Diverse Data Sources》(arXiv, 2024)

## License

[MIT License](LICENSE)
