# 医疗问答系统 Medical QA System

基于 PyTorch 的医疗问答系统，使用内置医疗知识库，支持知识图谱和持续学习。

A PyTorch-based medical question-answering system using built-in medical knowledge base, with knowledge graph and continuous learning support.

## 功能特性 Features

- 基于内置医疗知识库的问答 (Medical QA based on built-in knowledge base)
- 支持 Neo4j 知识图谱 (Neo4j knowledge graph support)
- 多轮对话上下文管理 (Multi-turn conversation context management)
- 用户反馈收集和模型评分 (User feedback collection and model scoring)
- 持续学习循环 (Continuous learning loop)
- 前后端分离架构 (Frontend-backend separation)
- Docker 和 Docker Compose 部署支持 (Docker and Docker Compose deployment)
- CI/CD 友好 (CI/CD friendly)

## 技术栈 Tech Stack

- Python/Flask (后端 Backend)
- PyTorch (AI 框架 AI Framework)
- Neo4j (知识图谱 Knowledge Graph)
- HTML/CSS/JavaScript (前端 Frontend)
- Docker/Docker Compose (部署 Deployment)

## 项目结构 Project Structure

```
.
├── api
│   ├── config         # 配置文件 Configuration files
│   ├── data           # 医疗数据 Medical data
│   ├── handlers       # API 处理程序 API handlers
│   ├── models         # AI 模型 AI models
│   ├── services       # 服务层 Services
│   └── utils          # 工具函数 Utilities
├── static             # 静态文件 Static files
├── templates          # 模板文件 Template files
├── tests              # 测试文件 Test files
├── Dockerfile         # Docker 配置 Docker configuration
├── docker-compose.yml # Docker Compose 配置 Docker Compose configuration
├── requirements.txt   # Python 依赖 Python dependencies
└── app.py             # 应用入口 Application entry point
```

## 快速开始 Quick Start

### 使用 Docker Compose 运行 (Recommended)

```bash
# 克隆项目 Clone the project
git clone <repository-url>
cd <project-directory>

# 启动服务 Start services
docker-compose up -d

# 访问应用 Access the application
# 前端: http://localhost:5000
# Neo4j: http://localhost:7474
```

### 本地运行 Local Development

```bash
# 创建虚拟环境 Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或者 on Windows
venv\Scripts\activate

# 安装依赖 Install dependencies
pip install -r requirements.txt

# 运行应用 Run application
python app.py
```

## API 接口 API Endpoints

### 问答接口 QA Endpoint

```
POST /api/qa/ask
```

请求体 Request Body:
```json
{
  "question": "问题内容",
  "session_id": "会话ID (可选)",
  "expected_keywords": ["关键词1", "关键词2"]  // 可选，用于评分
}
```

响应 Response:
```json
{
  "question": "问题内容",
  "answer": "回答内容",
  "score": 0.85,
  "session_id": "会话ID"
}
```

### 反馈接口 Feedback Endpoint

```
POST /api/qa/feedback
```

请求体 Request Body:
```json
{
  "question": "问题内容",
  "answer": "回答内容",
  "score": 4,  // 评分 1-5
  "session_id": "会话ID"
}
```

## 配置 Configuration

环境变量 Environment Variables:

| 变量名 | 默认值 | 描述 |
|--------|--------|------|
| NEO4J_URI | bolt://localhost:7687 | Neo4j 数据库地址 |
| NEO4J_USERNAME | neo4j | Neo4j 用户名 |
| NEO4J_PASSWORD | password | Neo4j 密码 |
| MODEL_NAME | custom_medical_model | 模型名称 |

## 持续学习 Continuous Learning

系统通过以下方式实现持续学习:

1. 收集用户反馈数据
2. 存储到知识图谱中
3. 可用于后续模型微调

The system implements continuous learning through:

1. Collecting user feedback
2. Storing in knowledge graph
3. Available for future model fine-tuning

## 开发 Development

### 单元测试 Unit Tests

```bash
# 运行测试 Run tests
python -m pytest tests/
```

## 部署 Deployment

### Docker 部署 Docker Deployment

构建镜像 Build image:
```bash
docker build -t medical-qa-app .
```

运行容器 Run container:
```bash
docker run -p 5000:5000 medical-qa-app
```

### Docker Compose 部署 Docker Compose Deployment

```bash
docker-compose up -d
```

## 许可证 License

MIT License