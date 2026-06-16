import os
import sqlite3
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from vision_parser import extract_and_describe_images
from langchain_community.document_loaders import PyMuPDFLoader


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
def build_agent(memory):
 
    os.environ["TAVILY_API_KEY"] = ""

    # 1. 初始化大模型
    llm = ChatOllama(model="qwen2.5:7b", temperature=0, base_url="http://127.0.0.1:11434")

    # 2. 加载与解析 PDF
    pdf_file = "textpdf.pdf"

    # 【分支 A：提取纯文本并打标签】
    print("⏳ 正在提取 PDF 纯文本...")
    loader = PyMuPDFLoader(pdf_file)
    text_docs = loader.load()
    for doc in text_docs:
        # 给纯文字打上明确的身份证标签
        doc.metadata["type"] = "pure_text"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_splits = text_splitter.split_documents(text_docs)

    # 【分支 B：提取视觉特征】
    print("⏳ 正在召唤 VLM 提取视觉图表特征...")
    image_docs = extract_and_describe_images(pdf_file)

    # 【💥 大合流】：把文本碎片和图表描述拼在一个列表里
    all_splits = text_splits + image_docs

    # 3. 混合向量化存入 Chroma 数据库
    print(f"⏳ 正在将总计 {len(all_splits)} 个多模态数据块进行向量化入库...")
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=OllamaEmbeddings(model="nomic-embed-text", base_url="http://127.0.0.1:11434")
    )

    # 调大 K 值，防止多模态数据互相挤占名额
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 4. 封装本地 多模态 检索工具
    @tool
    def local_pdf_database(query: str) -> str:
        """这是一个多模态本地知识库。当用户询问文档内的正文细节、图表数据、图片内容时，必须优先使用此工具进行检索。"""
        found_docs = retriever.invoke(query)

        # 组装返回给大模型的内容，顺便把 metadata 里的页码也带上，防止大模型幻觉
        result = []
        for doc in found_docs:
            source_type = "【图表/图片】" if doc.metadata.get("type") == "visual_chart" else "【纯文本】"
            page_num = doc.metadata.get("page", "未知")
            result.append(f"{source_type} (第{page_num}页): {doc.page_content}")

        #return "\n\n".join(result)
        final_context = "\n\n".join(result)
        return final_context + "\n\n【系统强制指令】：以上就是全部资料，请立即停止调用工具，直接根据上述资料用中文回答用户！"

    # ==========================================
    # 3. 组装与返回多模态 Agent
    # ==========================================
    # 🌟 修复点 2：显式定义 tools 列表
    # 这里我把联网搜索和你的多模态本地检索双剑合璧了！
    tools = [local_pdf_database]#TavilySearchResults(max_results=3),

    # 返回组装好的智能体
    return create_react_agent(llm, tools, checkpointer=memory)
    # system_prompt = """你是一个严谨的 AI 深度研究员。
    # 1. 当用户提问时，你必须且只能使用 `local_pdf_database` 工具去检索资料。
    # 2. 只要工具返回了结果，你必须立即停止调用工具，并根据结果直接用中文回答用户的问题。
    # 3. 绝对不要重复调用同一个工具。"""
    # return create_react_agent(
    #     llm,
    #     tools,
    #     checkpointer=memory,
    #     state_modifier=system_prompt  # 注入灵魂护城河
    # )
