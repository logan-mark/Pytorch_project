# 🌐 AI 深度研究员 (AI Deep Researcher)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent-orange.svg)

本项目是一个企业级的前后端解耦 **多模态 Agentic RAG 系统**。旨在解决传统大模型无法理解 PDF 复杂图表与非结构化文档的痛点。通过引入视觉大模型 (VLM) 提取图表语义，并结合 LangGraph 状态机赋予 LLM 工具调用能力，实现了从“被动检索”到“主动探究”的智能化升级。

## ✨ 核心特性

- **👁️ 多模态特征融合 (Vision RAG)**：集成前置视觉解析流水线，自动提取文档插图与图表，转化为结构化语义并绑定 Metadata 并入向量库。
- **🧠 智能体自主路由 (Agent Tool Calling)**：基于 LangGraph 的 ReAct 架构，Agent 可根据问题复杂度自主决定是否触发本地知识库检索，有效收敛工具调用环路。
- **💾 多租户长期记忆 (Stateful Memory)**：基于 SQLite 异步 Checkpoint 机制，实现 Thread ID 级别的会话隔离与漫游，支持无状态前端随时拉取历史上下文。
- **🛡️ 工业级微服务网关 (Security API)**：基于 FastAPI 构建 RESTful API，内建纯手工“令牌桶 (Token Bucket)”算法防 DDoS 限流器与 X-API-Key 鉴权拦截。

## 💻 环境与硬件要求

本项目专为消费级硬件进行了极高标准的内存优化，可在以下环境中流畅运行：
- **GPU 算力**：完美适配 RTX 4060 笔记本（8GB 显存）。系统采用显存分时复用策略，在向量化（nomic-embed-text）、视觉解析（Moondream）与推理生成（Qwen2.5:7b）阶段动态释放闲置模型，避免 OOM。
- **依赖管理**：推荐使用虚拟环境安装 `requirements.txt` 中的依赖。
- **本地模型服务**：需提前安装 Ollama 服务端。

## 📁 系统架构与目录说明
```text
├── api_server.py      # FastAPI 后端服务：生命周期管理、路由定义、限流与鉴权
├── agent_core.py      # Agent 核心组装：LLM 初始化、向量数据库入库、工具绑定与 LangGraph 编排
├── vision_parser.py   # 视觉解析器：PDF 图像拦截与 VLM 语义重构
├── web_app.py         # Streamlit 前端：多租户会话管理、历史记忆拉取、UI 交互呈现
└── checkpoints.db     # (运行时生成) SQLite 数据库，用于存储全局 Agent 对话状态快照
