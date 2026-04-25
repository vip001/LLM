"""
测试 DashScope 多模态向量模型（tongyi-embedding-vision-plus）是否可用。

用法（在项目根目录）：
  .venv/bin/python server/test_embed_dashscope.py

需在 .env 中配置 DASHSCOPE_API_KEY。
"""
from pathlib import Path

sys_path = Path(__file__).resolve().parent
if str(sys_path) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(sys_path))

from embedding_provider import DashScopeMultimodalEmbeddings, EMBED_MODEL_DASHSCOPE


def main() -> None:
    print(f"测试 DashScope 多模态嵌入: {EMBED_MODEL_DASHSCOPE}\n")

    emb = DashScopeMultimodalEmbeddings()

    print("1. embed_query ...")
    vec = emb.embed_query("测试文本：向量检索与语义分块")
    print(f"   OK, 维度: {len(vec)}, 前 3 维: {vec[:3]}")

    print("2. embed_documents ...")
    vecs = emb.embed_documents(["第一段文本", "第二段较长一些的文本"])
    print(f"   OK, 数量: {len(vecs)}, 维度: {len(vecs[0])}")

    print("\nDashScope 多模态嵌入测试通过，可以正常使用。")


if __name__ == "__main__":
    main()
