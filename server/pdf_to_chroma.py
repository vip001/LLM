"""
使用 LangChain 解析工程根目录下 pdf 文件夹的 PDF，存入向量库（持久化到 chromastore）。

- PDF 目录：项目根目录/pdf
- 向量库目录：项目根目录/chromastore（持久化）
- 嵌入模型：Ollama 本地模型（需先 pull 一个 embedding 模型，如 bge-m3）

运行前：
  pip install -r requirements.txt
  ollama pull bge-m3   # 或其它支持 embed 的模型
  在项目根目录下创建 pdf 文件夹并放入 PDF 文件
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_db import VectorDB

# 路径：相对项目根目录（脚本建议在根目录运行）
PROJECT_ROOT = Path(__file__).resolve().parent
PDF_DIR = PROJECT_ROOT / "pdf"
STORE_DIR = PROJECT_ROOT / "chromastore"

# 分块与嵌入
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "bge-m3"  # 需在 Ollama 中已拉取


def main() -> None:
    if not PDF_DIR.is_dir():
        raise SystemExit(f"PDF 目录不存在，请先创建: {PDF_DIR}")

    # 1. 初始化向量数据库（force_new=True 表示每次运行替换旧数据）
    vector_db = VectorDB(store_path=STORE_DIR, embed_model=EMBED_MODEL, force_new=True)

    # 2. 加载 pdf 目录下所有 PDF
    loader = PyPDFDirectoryLoader(
        path=str(PDF_DIR),
        glob="**/*.pdf",
        recursive=True,
        silent_errors=False,
    )
    documents = loader.load()
    if not documents:
        print("未在 pdf 目录下找到任何 PDF 文件，退出。")
        return
    print(f"已加载 {len(documents)} 个文档片段（页）。")

    # 3. 分块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"分块后共 {len(chunks)} 个 chunk。")

    # 4. 新增到向量库并持久化
    vector_db.add(chunks)
    vector_db.save()
    print(f"已写入向量库，持久化目录: {STORE_DIR}")


if __name__ == "__main__":
    main()
