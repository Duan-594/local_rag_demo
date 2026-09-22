from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import gradio as gr
import os
import shutil

# --------------------------配置区---------------------------
CHROMA_PATH = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
EMBED_MODEL_NAME = "./all-MiniLM-L6-v2"

RAG_PROMPT_TEMPLATE = """
你是知识库问答助手，请严格使用下面【参考上下文】的内容回答用户问题。
如果参考上下文里面没有答案，直接回答："知识库中没有找到相关信息"，不要编造内容。

【参考上下文】
{context}

【用户问题】
{question}
"""

# ---------------------------初始化---------------------------
embedding_func = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

# 如果数据库文件夹存在就加载，不存在就新建
vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_func
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

llm = ChatOllama(
    model="modelscope.cn/Qwen/Qwen1.5-7B-Chat-GGUF:latest",
    base_url="http://localhost:11434",
    temperature=0
)

prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

retriever = vector_db.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# -------------------------------工具函数-----------------------------
def upload_pdf_file(pdf_file):
    if pdf_file is None:
        return "请选择PDF文件上传"
    loader = PyPDFLoader(pdf_file.name)
    pages = loader.load()
    split_docs = text_splitter.split_documents(pages)
    vector_db.add_documents(split_docs)
    return f"✅PDF处理完成！一共切分 {len(split_docs)} 个文本块，可以开始提问。"


def chat_answer(question):
    if not question.strip():
        return "请输入你的问题"
    answer = rag_chain.invoke(question)
    return answer


def clear_knowledge_base():
    """清空向量数据库，新增功能"""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    # 删除后重新初始化空库
    global vector_db
    vector_db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_func
    )
    return "🗑️知识库已经全部清空，可以上传新PDF"

# -------------------------------Gradio网页-----------------------------
with gr.Blocks(title="本地RAG知识库问答系统") as demo:
    gr.Markdown("# 📖RAG本地PDF知识库问答系统")
    with gr.Row():
        pdf_input = gr.File(label="上传PDF知识库文件", file_types=[".pdf"])
        upload_msg = gr.Textbox(label="上传状态")
    with gr.Row():
        upload_btn = gr.Button("解析PDF并导入知识库", variant="primary")
        clear_btn = gr.Button("🗑️清空整个知识库", variant="stop")

    upload_btn.click(upload_pdf_file, inputs=[pdf_input], outputs=[upload_msg])
    clear_btn.click(clear_knowledge_base, outputs=[upload_msg])

    user_q = gr.Textbox(label="你的问题", placeholder="向刚刚上传的PDF提问")
    answer_out = gr.Textbox(label="AI回答", lines=8)
    ask_btn = gr.Button("提问", variant="primary")
    ask_btn.click(chat_answer, inputs=[user_q], outputs=[answer_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
