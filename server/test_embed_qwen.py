"""
测试 Qwen3-VL-Embedding-2B 作为向量库 embedding 与语义分块是否可用。

用法（在项目根目录）：
  .venv/bin/python server/test_embed_qwen.py

需先：ollama pull fervent_mcclintock/Qwen3-VL-Embedding-2B:F16
若 Ollama 不在 localhost:11434，可设置环境变量 OLLAMA_BASE_URL 或改脚本内 OLLAMA_BASE_URL。

为何请求会发到 127.0.0.1:xxxxx/embedding？
  已改为使用 DirectOllamaEmbeddings（仅 httpx 直连 base_url/api/embed），
  不再经过 ollama 包，请求会发往你配置的 base_url（默认 http://localhost:11434）。
"""
from pathlib import Path

# 保证从 server 目录可导入
sys_path = Path(__file__).resolve().parent
if str(sys_path) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(sys_path))

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from embedding_provider import EmbeddingProvider, EMBED_MODEL_QWEN
from vector_db import VectorDB, DEFAULT_BASE_URL

TEST_STORE = Path(__file__).resolve().parent / "chromastore_test_embed"
# 直连 Ollama 的地址；避免用 OLLAMA_HOST（可能被 IDE 设为代理）
OLLAMA_BASE_URL = DEFAULT_BASE_URL


def main() -> None:
    import os
    # 临时清除可能被 IDE 注入的代理变量，确保直连 Ollama（embedding_provider 已用 trust_env=False）
    for key in ("OLLAMA_HOST", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(key, None)
    base_url = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL) or OLLAMA_BASE_URL
    if not base_url.startswith("http"):
        base_url = f"http://{base_url}"
    print(f"使用 Ollama base_url: {base_url}")
    print(f"默认模型: {EMBED_MODEL_QWEN}\n")

    provider = EmbeddingProvider(base_url=base_url)
    emb = provider.get_embeddings()

    # 1. 测试 embed_query / embed_documents
    print("1. 测试 embed_query ...")
    vec = emb.embed_query("测试文本：向量检索与语义分块")
    print(f"   embed_query OK, 维度: {len(vec)}, 前 3 维: {vec[:3]}")
    print("2. 测试 embed_documents ...")
    vecs = emb.embed_documents(["第一段文本", "第二段较长一些的文本"])
    print(f"   embed_documents OK, 数量: {len(vecs)}, 维度: {len(vecs[0])}")

    # 3. 测试语义分块
    print("3. 测试 SemanticChunker（语义相关分块）...")
    semantic_splitter = SemanticChunker(
        emb,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=1.5,
    )
    long_text = (
        "机器学习是人工智能的一个分支。深度学习使用神经网络。"
        "自然语言处理处理文本和语言。向量数据库用于相似度检索。"
        "嵌入模型将文本转换为向量。RAG 结合检索与生成。"
    )
    doc = Document(page_content=long_text, metadata={"source": "test"})
    chunks = semantic_splitter.split_documents([doc])
    print(f"   语义分块 OK, 得到 {len(chunks)} 个 chunk")

    # 4. 测试向量库写入与检索
    print("4. 测试向量库 add + search ...")
    if TEST_STORE.exists():
        import shutil
        shutil.rmtree(TEST_STORE)
    db = VectorDB(store_path=TEST_STORE, embeddings=emb, force_new=True)
    db.add([Document(page_content="苹果是一种水果。", metadata={"source": "test1"})])
    db.add([Document(page_content="向量检索用于找到相似文档。", metadata={"source": "test2"})])
    db.save()
    results = db.search("水果", k=2)
    print(f"   检索 '水果' 得到 {len(results)} 条，第一条内容前 20 字: {results[0].page_content[:20] if results else 'N/A'}...")
    if TEST_STORE.exists():
        import shutil
        shutil.rmtree(TEST_STORE)
    print("\n全部测试通过，该模型可用于向量库 embedding 与语义分块。")


if __name__ == "__main__":
    main()
