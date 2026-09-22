# local_rag_demo
本地PDF文档RAG问答Demo，基于LangChain + Gradio，本地Embedding模型，私有PDF知识库问答。

✨ 项目特点
- 完全本地运行，PDF数据不上传到第三方云端
- PDF文档解析、文本切块、向量库检索
- Gradio 简易Web交互界面
- 使用 all-MiniLM-L6-v2 开源Embedding模型

📦 安装依赖
```bash
pip install langchain gradio pypdf chromadb sentence-transformers
```
🚀 启动
```bush
python rag_app.py
```
运行后访问本地 Web 页面，上传 PDF，就可以基于文档提问。

📁 项目文件

- `rag_app.py`：RAG 主程序
- `.gitignore`：忽略模型缓存、PyCharm 配置、临时文件
