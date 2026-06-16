import os
import sqlite3
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver


# ==========================================
# 1. 数据库辅助函数
# ==========================================
def get_all_thread_ids(db_path="checkpoints.db"):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints';")
        if not cursor.fetchone():
            conn.close()
            return []
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC")
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in rows]
    except Exception as e:
        print(f"⚠️ 数据库读取错误: {e}")
        return []


# ==========================================
# 2. 核心智能体初始化工厂函数
# ==========================================
def build_agent():
    # ⚠️ 替换为你自己的 API KEY
    os.environ["TAVILY_API_KEY"] = ""

    # 1. 初始化模型
    llm = ChatOllama(model="qwen2.5:7b", temperature=0, base_url="http://127.0.0.1:11434")

    # 2. 加载与解析 PDF
    # 请确保你的项目目录下有研究论文的 PDF 文件
    loader = PyPDFLoader("research_paper.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 3. 向量化存入 Chroma
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model="nomic-embed-text", base_url="http://127.0.0.1:11434")
    )
    retriever = vectorstore.as_retriever()

    # 4. 封装本地 PDF 检索工具
    @tool
    def local_pdf_database(query: str) -> str:
        """这是一个本地专业文献数据库。当用户询问关于[这里填入你PDF的主题]的相关信息时，必须优先使用此工具进行深度检索。"""
        found_docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in found_docs])

    tools = [TavilySearchResults(max_results=3), local_pdf_database]

    # 5. 连接 SQLite 持久化记忆
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    # 返回组装好的 Agent
    return create_react_agent(llm, tools, checkpointer=memory)
