# Agno RAG System

基于Agno框架开发的RAG（检索增强生成）系统，提供高效的文档检索和智能问答功能。

## 🚀 功能特性

- **智能文档处理**: 支持多种文档格式的自动解析和分块
- **语义搜索**: 基于向量数据库的高效语义检索
- **上下文增强**: 结合检索结果和语言模型的智能回答生成
- **异步架构**: 基于Agno框架的高性能异步处理
- **可扩展设计**: 模块化架构，易于扩展和定制
- **完整测试**: 包含单元测试和集成测试

## 📋 系统要求

- Python 3.9+
- Agno框架 0.1.0+
- 支持的操作系统: Windows, macOS, Linux

## 🛠️ 安装

### 1. 克隆项目

```bash
git clone https://github.com/your-org/agno-rag-system.git
cd agno-rag-system
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
# 生产环境：只安装核心依赖
pip install -e .

# 开发环境：安装核心依赖和开发工具
pip install -e ".[dev]"

# 测试环境：安装核心依赖和测试工具
pip install -e ".[test]"
```

### 4. 配置环境变量

创建 `.env` 文件并配置必要的环境变量：

```bash
# 数据库配置
VECTOR_DB_URL=localhost:6333
VECTOR_DB_COLLECTION=rag_documents

# 模型配置
EMBEDDING_MODEL=text-embedding-ada-002
LLM_MODEL=gpt-3.5-turbo

# RAG配置
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5

# 日志配置
LOG_LEVEL=INFO
```

## 🚀 快速开始

### 基本使用

```python
import asyncio
from src.agents.rag_agent import RAGAgent
from src.utils.config import Config

async def main():
    """RAG系统基本使用示例"""
    
    # 初始化配置
    Config.validate()
    
    # 创建RAG代理
    agent = RAGAgent()
    
    # 处理用户查询
    query = "什么是机器学习？"
    response = await agent.process_query(query)
    
    print(f"查询: {query}")
    print(f"回答: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 文档处理

```python
from src.services.document_processor import DocumentProcessor

async def process_documents():
    """文档处理示例"""
    
    processor = DocumentProcessor(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # 处理单个文档
    chunks = await processor.process_document("path/to/document.pdf")
    
    # 批量处理文档
    document_paths = ["doc1.pdf", "doc2.txt", "doc3.docx"]
    all_chunks = await processor.process_documents(document_paths)
    
    return all_chunks
```

### 自定义工具

```python
from src.tools.document_search import DocumentSearchTool
from agno import Tool

class CustomSearchTool(Tool):
    """自定义搜索工具"""
    
    name = "custom_search"
    description = "自定义文档搜索功能"
    
    async def execute(self, query: str, **kwargs):
        """执行自定义搜索逻辑"""
        # 实现自定义搜索逻辑
        return {"results": [], "metadata": {}}
```

## 📁 项目结构

```
agno-rag-system/
├── src/                          # 源代码目录
│   ├── agents/                   # 代理模块
│   │   ├── __init__.py
│   │   ├── rag_agent.py         # 主要RAG代理
│   │   └── search_agent.py      # 搜索代理
│   ├── tools/                    # 工具模块
│   │   ├── __init__.py
│   │   ├── document_search.py   # 文档搜索工具
│   │   └── text_generation.py   # 文本生成工具
│   ├── models/                   # 数据模型
│   │   ├── __init__.py
│   │   ├── document.py          # 文档模型
│   │   └── query.py             # 查询模型
│   ├── services/                 # 服务层
│   │   ├── __init__.py
│   │   ├── vector_store.py      # 向量存储服务
│   │   └── document_processor.py # 文档处理服务
│   └── utils/                    # 工具函数
│       ├── __init__.py
│       ├── config.py             # 配置管理
│       └── logging.py            # 日志工具
├── tests/                        # 测试目录
│   ├── test_agents/              # 代理测试
│   ├── test_tools/               # 工具测试
│   └── test_services/            # 服务测试
├── docs/                         # 文档目录
│   ├── api.md                    # API文档
│   └── deployment.md             # 部署指南
├── pyproject.toml                # 项目配置和依赖管理
├── pyproject.toml                # 项目配置
├── README.md                     # 项目说明
└── .cursorrules                  # Cursor规则
```

## 🧪 测试

### 运行所有测试

```bash
pytest
```

### 运行特定测试

```bash
# 运行单元测试
pytest tests/ -m "unit"

# 运行集成测试
pytest tests/ -m "integration"

# 运行特定模块测试
pytest tests/test_agents/
```

### 代码质量检查

```bash
# 代码格式化
black src/ tests/

# 代码检查
flake8 src/ tests/

# 类型检查
mypy src/

# 导入排序
isort src/ tests/
```

## 📚 API文档

详细的API文档请参考 [docs/api.md](docs/api.md)

### 核心组件

#### RAGAgent

主要的RAG代理类，负责协调文档检索和回答生成。

```python
from src.agents.rag_agent import RAGAgent

agent = RAGAgent()
response = await agent.process_query("你的问题")
```

#### DocumentProcessor

文档处理服务，负责文档的解析、分块和向量化。

```python
from src.services.document_processor import DocumentProcessor

processor = DocumentProcessor()
chunks = await processor.process_document("document.pdf")
```

#### VectorStore

向量存储服务，提供高效的语义搜索功能。

```python
from src.services.vector_store import VectorStore

store = VectorStore()
results = await store.similarity_search(query_vector, k=5)
```

## 🔧 配置

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `VECTOR_DB_URL` | `localhost:6333` | 向量数据库地址 |
| `VECTOR_DB_COLLECTION` | `rag_documents` | 向量数据库集合名 |
| `EMBEDDING_MODEL` | `text-embedding-ada-002` | 嵌入模型名称 |
| `LLM_MODEL` | `gpt-3.5-turbo` | 语言模型名称 |
| `CHUNK_SIZE` | `1000` | 文档块大小 |
| `CHUNK_OVERLAP` | `200` | 块重叠大小 |
| `TOP_K_RESULTS` | `5` | 检索结果数量 |

### 日志配置

```python
import logging
from src.utils.logging import setup_logging

# 设置日志级别
setup_logging(level="INFO", log_file="rag_system.log")
```

## 🚀 部署

### Docker部署

```bash
# 构建镜像
docker build -t agno-rag-system .

# 运行容器
docker run -p 8000:8000 agno-rag-system
```

### 生产环境配置

1. 设置环境变量
2. 配置数据库连接
3. 设置日志记录
4. 配置监控和告警

详细部署指南请参考 [docs/deployment.md](docs/deployment.md)

## 🤝 贡献

我们欢迎所有形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与项目开发。

### 开发流程

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 代码规范

- 遵循 PEP 8 代码风格
- 使用类型注解
- 编写完整的文档字符串
- 添加单元测试
- 确保测试覆盖率不低于80%

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Agno框架](https://docs.agno.com/) - 提供强大的AI代理框架
- [Python社区](https://www.python.org/) - 优秀的编程语言和生态系统
- 所有贡献者和用户

## 📞 联系我们

- 项目主页: https://github.com/your-org/agno-rag-system
- 问题反馈: https://github.com/your-org/agno-rag-system/issues
- 邮箱: team@example.com

---

**注意**: 这是一个基于Agno框架的RAG系统实现，遵循Agno社区最佳实践和Python规范。所有代码都包含详细的注释，便于理解和维护。 