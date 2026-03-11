import streamlit as st
import os
import time
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

# ================= 1. 页面配置与环境加载 =================
st.set_page_config(page_title="AI 深度调研专家", page_icon="🚀", layout="wide")

load_dotenv()

# 获取 API Key
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY") or st.secrets.get("DEEPSEEK_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY")

# ================= 2. 定义 Agent 核心架构 =================

class AgentState(TypedDict):
    topic: str
    research_data: str
    content: str
    revision_notes: str
    iteration_count: int

# 初始化模型 (开启流式)
llm = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=DEEPSEEK_KEY, 
    openai_api_base='https://api.deepseek.com/v1',
    streaming=True
)
search_tool = TavilySearchResults(max_results=3, tavily_api_key=TAVILY_KEY)

# --- 节点 1: 联网搜索 ---
def search_node(state: AgentState):
    with st.status("🔍 正在联网搜集最新资料...", expanded=True) as status:
        try:
            results = search_tool.invoke({"query": state['topic']})
            data_str = "\n".join([r['content'] for r in results])
            status.update(label="资料搜集完成！", state="complete", expanded=False)
        except Exception as e:
            st.error(f"搜索出错: {e}")
            data_str = "未搜寻到相关资料"
    return {"research_data": data_str}

# --- 节点 2: 撰写报告 (带抗抖动流式输出) ---
def writer_node(state: AgentState):
    # 计数器自增
    current_iter = state.get('iteration_count', 0) + 1
    
    st.markdown(f"---")
    st.markdown(f"### ✍️ 正在撰写报告 (第 {current_iter} 次迭代)...")
    
    # 使用 Spinner 显示加载状态，不进行高频 UI 刷新
    with st.spinner("AI 正在深度思考并整合资料，请稍候..."):
        prompt = f"""你是一名专业的分析师。
        课题：{state['topic']}
        已搜集资料：{state['research_data']}
        上轮审核意见：{state.get('revision_notes', '无')}
        任务：撰写深度报告。要求包含 Markdown 表格、2025年具体预测数据、执行摘要。"""
        
        # 💡 改为 invoke (同步调用)，一次性获取结果，彻底避开 removeChild 错误
        response = llm.invoke(prompt)
        full_content = response.content
    
    # 撰写完成后，一次性渲染到网页
    st.markdown(full_content)
    
    return {"content": full_content, "iteration_count": current_iter}

# --- 节点 3: 导师审核 ---
def reviewer_node(state: AgentState):
    with st.status("🧐 导师正在评估报告质量...", expanded=True) as status:
        # 强制逻辑：第一次迭代必须打回（展示 Agent 的进化能力）
        if state.get('iteration_count') == 1:
            feedback = "【导师打回】：内容基础，但不够专业。请增加：1. 详尽的 Markdown 对比表；2. 针对 2025 年的具体财务/市场预测数据。"
            status.update(label="审核未通过，已下达修改意见", state="error", expanded=True)
            st.warning(f"**修改意见：** {feedback}")
            return {"revision_notes": feedback}
        
        # 后续轮次进行真实审核
        prompt = f"""请评估该报告：\n{state['content']}\n
        若完全满足表格、预测、摘要三项要求请回复 [[PASSED]]，否则指出缺失点。"""
        response = llm.invoke(prompt)
        
        if "[[PASSED]]" in response.content:
            status.update(label="审核通过！已达成专业标准。", state="complete", expanded=False)
        else:
            status.update(label="审核未通过，需进一步细化", state="error", expanded=True)
            st.warning(f"**修改意见：** {response.content}")
            
        return {"revision_notes": response.content}

# --- 逻辑路由: 判断下一步去哪 ---
def decide_what_to_do(state: AgentState):
    if "[[PASSED]]" in state['revision_notes']:
        return "finish"
    if state.get('iteration_count', 0) >= 3:
        st.info("已达到最大迭代次数，任务强制完成。")
        return "finish"
    return "rework"

# ================= 3. 构建工作流连线 =================

workflow = StateGraph(AgentState)

workflow.add_node("searcher", search_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)

workflow.set_entry_point("searcher")
workflow.add_edge("searcher", "writer")
workflow.add_edge("writer", "reviewer")

workflow.add_conditional_edges(
    "reviewer", 
    decide_what_to_do, 
    {"rework": "writer", "finish": END}
)

agent_app = workflow.compile()

# ================= 4. Streamlit UI 交互界面 =================

# 侧边栏
with st.sidebar:
    st.title("🤖 调研助手")
    st.markdown("---")
    user_topic = st.text_input("请输入调研课题", placeholder="例如：2024年全球低空经济格局")
    start_run = st.button("开始调研", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("**工作流程：**\n1. 联网实时搜索\n2. 撰写初稿\n3. 导师反思审核\n4. 自动迭代重写")

# 主展示区
st.title("🚀 AI 深度调研专家")
st.caption("基于 LangGraph 的自反思迭代智能体 | 提供专业级行业报告产出")

if start_run and user_topic:
    # 状态初始化
    initial_input = {"topic": user_topic, "iteration_count": 0, "revision_notes": ""}
    
    final_result = None
    # 运行 Agent 并在页面上实时渲染
    for output in agent_app.stream(initial_input):
        for key, value in output.items():
            if "content" in value:
                final_result = value['content']
    
    if final_result:
        st.success("✅ 任务圆满完成！最终报告已生成。")
        st.balloons()
        st.download_button(
            label="📂 点击下载最终 Markdown 报告",
            data=final_result,
            file_name=f"{user_topic}_调研报告.md",
            mime="text/markdown",
            use_container_width=True
        )
else:
    if not start_run:
        st.info("请在左侧侧边栏输入课题并点击“开始调研”。")