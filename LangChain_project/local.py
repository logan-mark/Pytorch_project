import os
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# ==========================================
# 1. 环境变量设置
# ==========================================
# ⚠️ 把这里替换成你刚刚在网页上申请的 API Key
os.environ["TAVILY_API_KEY"] = ""

# ==========================================
# 2. 初始化本地大模型
# ==========================================
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,  # 执行任务时保持严谨
    base_url="http://127.0.0.1:11434"  # 稳定直连本地
)

# ==========================================
# 3. 装备工具箱 (Tools)
# ==========================================
# max_results=3 表示每次搜索最多抓取 3 个网页的内容
search_tool = TavilySearchResults(max_results=3)

# 你可以往这个列表里塞入无数个工具（比如后面的本地 PDF 读取工具）
tools = [search_tool]

# ==========================================
# 4. 构建 Agent 工作流 (LangGraph)
# ==========================================
# create_react_agent 会自动帮我们把 LLM 和 Tools 绑定成一个循环状态机
app = create_react_agent(llm, tools)

# ==========================================
# 5. 运行测试
# ==========================================
if __name__ == "__main__":
    print("🚀 启动 AI 深度研究员 (联网模式)...")

    # 我们故意问一个模型原本绝对不知道的“实时问题”
    user_input = "帮我查一下最新的科技新闻，关于英伟达(NVIDIA)的最新动态是什么？"

    print(f"\n[用户提问]: {user_input}")
    print("\n[Agent 正在思考并执行搜索，请稍候...]")

    # 传入初始状态，启动 Agent
    result = app.invoke({"messages": [HumanMessage(content=user_input)]})

    # 打印最终生成的总结报告
    print("\n[最终研究报告]:")
    print(result["messages"][-1].content)
