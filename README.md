# 🤖 AI Research Agent (DeepSeek + LangGraph)

本项目是一个基于 **LangGraph** 构建的自主科研调研智能体。它能够联网搜索、撰写报告、自我审校并多次迭代优化，直到产出专业级的研究报告。

## 🌟 核心特性
- **Agentic Workflow**: 采用有状态图架构，实现“搜索-撰写-审核-修正”的闭环迭代。
- **自反思机制**: 引入 Critic-Actor 模式，通过独立审核节点强制提升报告质量。
- **实时联网**: 集成 Tavily Search 获取 2024-2025 最新行业动态。
- **流式 UI**: 基于 Streamlit 构建交互式网页，支持任务全过程可视化。

## 🛠️ 技术栈
- **核心框架**: LangGraph, LangChain
- **大模型**: DeepSeek-V3 / R1
- **搜索引擎**: Tavily AI
- **展示层**: Streamlit

## 🚀 快速开始
1. 克隆项目：`git clone https://github.com/你的用户名/ai_research_agent.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 配置 `.env`: 填入 `DEEPSEEK_API_KEY` 和 `TAVILY_API_KEY`
4. 启动网页：`streamlit run app.py`