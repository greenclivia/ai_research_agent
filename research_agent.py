import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

# 1. 加载配置
load_dotenv()

# 定义状态
class AgentState(TypedDict):
    topic: str
    research_data: str
    content: str
    revision_notes: str
    iteration_count: int

# 2. 初始化模型 (确保地址正确)
llm = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
    openai_api_base='https://api.deepseek.com/v1',
    streaming=True  # <--- 开启流式输出
)
search_tool = TavilySearchResults(max_results=3)

# 3. 定义节点函数

def search_node(state: AgentState):
    print("\n--- [步骤1] 正在联网搜集最新资料 ---")
    results = search_tool.invoke({"query": state['topic']})
    data_str = "\n".join([r['content'] for r in results])
    # 这里不再进行 +1，只返回数据
    return {"research_data": data_str}

def writer_node(state: AgentState):
    # 在这里获取当前次数并 +1
    current_iter = state.get('iteration_count', 0) + 1
    
    print(f"\n--- [步骤2] 正在进行第 {current_iter} 次撰写报告 (实时生成中...) ---")
    
    prompt = f"""你是一名专业的分析师。课题：{state['topic']}
    资料：{state['research_data']}
    上轮审核意见：{state.get('revision_notes', '无')}
    请根据意见修订报告。"""
    
    full_content = ""
    print("\n【报告正文】：\n" + "-"*30)
    for chunk in llm.stream(prompt):
        content = chunk.content
        print(content, end="", flush=True)
        full_content += content
    print("\n" + "-"*30 + "\n")
    
    # 关键：在这里把更新后的次数返回给 State
    return {"content": full_content, "iteration_count": current_iter}

def reviewer_node(state: AgentState):
    print(f"--- [步骤3] 导师正在进行第 {state.get('iteration_count')} 次审核 ---")
    
    # 获取当前的字数，用来刁难 Agent
    current_length = len(state['content'])
    
    # 强制逻辑：如果是第一次，无论写多好都必须打回
    if state.get('iteration_count') == 1:
        feedback = (
            "初稿虽然内容详实，但逻辑架构过于常规。作为顶级咨询报告，我要求你：\n"
            "1. 必须在报告开头增加一个‘执行摘要（Executive Summary）’章节。\n"
            "2. 必须在表格中增加‘技术路线对比’这一列。\n"
            "3. 现在的字数还不够厚重，请针对每个竞争对手增加至少 200 字的深度点评。\n"
            "请重新思考并重写，这次不要让我失望。"
        )
        print(f"【导师反馈】：初稿一律打回，要求增加深度点评和执行摘要。")
        return {"revision_notes": feedback}

    # 如果是第二次及以后，再进入真正的逻辑判断
    prompt = f"""请评估这份经过修订的报告：
    {state['content']}
    
    【最终通过标准】：
    1. 必须包含‘执行摘要’。
    2. 表格必须包含‘技术路线对比’。
    3. 必须包含对 2025 年、2027 年的详细预测。
    4. 只有完全满足且包含标记 [[PASSED]] 才能结束。
    
    否则，请继续指出不足。"""
    
    response = llm.invoke(prompt)
    print(f"【导师反馈】：\n{response.content[:100]}...")
    return {"revision_notes": response.content}

# 4. 定义判断逻辑 (必须放在构建工作流之前！)

def decide_what_to_do(state: AgentState):
    print("--- [决策中心] 正在判断是否需要打回重写 ---")
    if "[[PASSED]]" in state['revision_notes']:
        print("--- ✅ 审核通过，任务圆满结束！ ---")
        return "finish"
    if state.get('iteration_count', 0) >= 3:
        print("--- ❌ 已达3次尝试上限，强制结束 ---")
        return "finish"
    print(f"--- 🔄 报告不合格，准备第 {state.get('iteration_count') + 1} 次迭代 ---")
    return "rework"

# 5. 构建工作流 (顺序非常重要)

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("searcher", search_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)

# 设置连线
workflow.set_entry_point("searcher")
workflow.add_edge("searcher", "writer")
workflow.add_edge("writer", "reviewer")

# 设置条件连线
workflow.add_conditional_edges(
    "reviewer",
    decide_what_to_do,
    {
        "rework": "writer", # 不合格就回去重写
        "finish": END      # 合格就结束
    }
)

# 编译
app = workflow.compile()

# 6. 运行入口
if __name__ == "__main__":
    print("🚀 Agent 启动，开始全自动深度调研...")
    initial_input = {
        "topic": "2024年全球大模型(LLM)行业竞争格局", 
        "iteration_count": 0,
        "revision_notes": ""
    }
    
    # 用来存储最终报告的变量
    final_report = ""

    # 运行工作流
    for output in app.stream(initial_input):
        for key, value in output.items():
            print(f"== 节点 {key} 执行完毕 ==")
            # 只要节点输出了 content，我们就记录下来，这样最后拿到的肯定是最新版
            if "content" in value:
                final_report = value["content"]

    # --- 核心：保存文件的逻辑 ---
    if final_report:
        with open("深度调研报告.md", "w", encoding="utf-8") as f:
            f.write(final_report)
        print("\n" + "="*30)
        print("✅ 任务全部完成！")
        print("📂 最终报告已保存在：深度调研报告.md")
        print("="*30)
    else:
        print("\n❌ 警告：未找到报告内容，保存失败。")