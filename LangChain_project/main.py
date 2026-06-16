import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# 🌟 1. 新增导入：LangGraph 的记忆检查点模块
from langgraph.checkpoint.memory import MemorySaver

# 环境变量设置
os.environ["TAVILY_API_KEY"] = ""

# 初始化本地大模型
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
    base_url="http://127.0.0.1:11434"
)

# 构建本地向量知识库 (保持不变)
print("正在构建本地向量知识库...")
loader = TextLoader("secret_knowledge.txt", encoding="utf-8")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
#这以上是处理文本分词


vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OllamaEmbeddings(model="nomic-embed-text", base_url="http://127.0.0.1:11434")
)
retriever = vectorstore.as_retriever()


# 手搓本地数据库工具 (保持不变)
@tool
def local_secret_database(query: str) -> str:
    """这是一个本地最高机密数据库。当用户询问关于'2026年实验室'、'机密代号'、'密码'或'李雷'相关信息时，必须优先使用此工具进行检索。"""
    found_docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in found_docs])


# 装备工具箱
search_tool = TavilySearchResults(max_results=3)
tools = [search_tool, local_secret_database]

# ==========================================
# 🔥 记忆体升级核心代码
# ==========================================
# 2. 实例化一个基于内存的检查点保存器 (MemorySaver)
conn=sqlite3.connect("checkpoiont.db",check_same_thread=False)

memory=SqliteSaver(conn)


# 3. 将记忆体作为 checkpointer 传入 Agent
app = create_react_agent(llm, tools, checkpointer=memory)

# ==========================================
# 运行测试：多轮连贯对话测试
# ==========================================
if __name__ == "__main__":
    print("\n🚀 AI 深度研究员已启动 (已激活长期记忆模块)")
    print("输入 'quit'、'exit' 或 'q' 即可退出对话。\n")

    # 只要 thread_id 不变，记忆就会一直累积
    config = {"configurable": {"thread_id": "user_session_001"}}

    while True:
        # 1. 获取用户输入
        user_input = input("👤 用户: ")

        # 2. 设置退出条件
        if user_input.lower() in ["quit", "exit", "q"]:
            print("再见！对话存档已保存。")
            break

        # 3. 调用 Agent
        # 注意：这里我们只传入当前的 user_input，LangGraph 会自动从 memory 中找回之前的历史
        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )

        # 4. 打印 AI 回答
        # 我们获取消息列表中的最后一个内容
        response = result['messages'][-1].content
        print(f"🤖 AI: {response}\n")
