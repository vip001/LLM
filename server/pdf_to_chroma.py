"""
使用 LangChain 解析工程根目录下 pdf 文件夹的 PDF，存入向量库（持久化到 chromastore）。

- PDF 目录：项目根目录/pdf
- 向量库目录：项目根目录/chromastore（持久化）
- 嵌入模型：通过 EmbeddingProvider 获取（默认 tongyi-embedding-vision-plus 多模态，支持图文向量化）
- 分块策略：混合分块（递归粗分 + 长块语义细分）
- 图片：从每页提取图片，携带该页文字作为上下文，用多模态模型做「图文融合」向量后入库，检索时可同时命中文本与图片

运行前：
  pip install -r requirements.txt
  在 .env 中配置 DASHSCOPE_API_KEY（使用 DashScope 多模态时）
  在项目根目录下创建 pdf 文件夹并放入 PDF 文件
"""
import base64
import io
import json
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf._page import PageObject

from embedding_provider import (
    EmbeddingProvider,
    DashScopeMultimodalEmbeddings,
)
from vector_db import VectorDB

# 路径：相对项目根目录（脚本建议在根目录运行）
PROJECT_ROOT = Path(__file__).resolve().parent
PDF_DIR = PROJECT_ROOT / "pdf"
STORE_DIR = PROJECT_ROOT / "chromastore"
MANIFEST_PATH = STORE_DIR / "pdf_manifest.json"

# 分块与嵌入（使用 EmbeddingProvider 默认模型，见 embedding_provider.EMBED_MODEL_QWEN）
# 递归粗分参数
RECURSIVE_CHUNK_SIZE = 1200
RECURSIVE_CHUNK_OVERLAP = 200
# 仅对超过此长度的递归块做语义二次切分
SEMANTIC_SPLIT_MIN_LEN = 800

# DashScope 单张图片不超过 3MB，超过则跳过
MAX_IMAGE_BYTES = 3 * 1024 * 1024


def _store_ready() -> bool:
    """判断向量库持久化文件是否已存在。"""
    return (STORE_DIR / "index.faiss").exists() and (STORE_DIR / "index.pkl").exists()


def _build_pdf_snapshot() -> list[dict[str, int | str]]:
    """构建当前 PDF 目录快照，用于判断是否需要重建向量库。"""
    snapshot: list[dict[str, int | str]] = []
    for pdf_path in sorted(PDF_DIR.rglob("*.pdf")):
        if not pdf_path.is_file():
            continue
        stat = pdf_path.stat()
        snapshot.append(
            {
                "path": pdf_path.relative_to(PDF_DIR).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return snapshot


def _load_manifest() -> list[dict[str, int | str]] | None:
    """读取上次构建时保存的 PDF 快照。"""
    if not MANIFEST_PATH.exists():
        return None
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    files = data.get("files")
    return files if isinstance(files, list) else None


def _save_manifest(snapshot: list[dict[str, int | str]]) -> None:
    """保存本次构建对应的 PDF 快照。"""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps({"files": snapshot}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_rebuild_reason(
    snapshot: list[dict[str, int | str]],
) -> str | None:
    """返回需要重建的原因；无变化则返回 None。"""
    if not _store_ready():
        return "向量库文件不存在"
    previous_snapshot = _load_manifest()
    if previous_snapshot is None:
        return "缺少 PDF 清单文件"
    if previous_snapshot != snapshot:
        return "检测到 PDF 文件新增、删除或变更"
    return None


def _page_text_map(documents: list[Document]) -> dict[tuple[str, int], str]:
    """从 Loader 得到的按页 Document 列表，构建 (source_path, page_number) -> page_content 的映射。"""
    out: dict[tuple[str, int], str] = {}
    for doc in documents:
        meta = doc.metadata or {}
        source = meta.get("source") or ""
        if not source:
            continue
        path = str(Path(source).resolve())
        page = int(meta.get("page", 0))
        out[(path, page)] = doc.page_content or ""
    return out


def _extract_pdf_images_with_context(
    pdf_path: Path,
    page_text_map: dict[tuple[str, int], str],
) -> list[tuple[str, str, dict]]:
    """
    从单个 PDF 中提取所有图片，每张图片附带当页文字作为上下文。
    返回 [(context_text, image_data_uri, metadata), ...]
    """
    result: list[tuple[str, str, dict]] = []
    first_extract_error: str | None = None
    path_str = str(pdf_path.resolve())
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        print(f"  跳过 PDF（无法打开）: {pdf_path} - {e}")
        return result
    for page_idx, page in enumerate[PageObject](reader.pages):
        context = page_text_map.get((path_str, page_idx), "")
        try:
            images_iter = getattr(page, "images", None)
            if images_iter is None:
                continue
            for img_idx, img_obj in enumerate[Any](images_iter):
                try:
                    pil_img = getattr(img_obj, "image", None)
                    if pil_img is None:
                        continue
                    buf = io.BytesIO()
                    fmt = getattr(pil_img, "format", None) or "PNG"
                    if fmt.upper() not in ("PNG", "JPEG", "JPG", "BMP"):
                        fmt = "PNG"
                    pil_img.save(buf, format=fmt)
                    raw = buf.getvalue()
                    if len(raw) > MAX_IMAGE_BYTES:
                        continue
                    b64 = base64.b64encode(raw).decode("ascii")
                    mime = "png" if fmt.upper() == "PNG" else "jpeg" if fmt.upper() in ("JPEG", "JPG") else "bmp"
                    data_uri = f"data:image/{mime};base64,{b64}"
                    meta = {
                        "source": path_str,
                        "page": page_idx,
                        "image_index": img_idx,
                        "type": "image",
                        "image_data": data_uri,
                    }
                    result.append((context, data_uri, meta))
                except Exception as e:
                    if first_extract_error is None:
                        first_extract_error = (
                            f"{type(e).__name__}: {e} (page={page_idx}, image_index={img_idx})"
                        )
                    continue
        except Exception:
            continue
    if not result and first_extract_error is not None:
        print(f"  PDF 图片提取失败: {pdf_path.name} - {first_extract_error}")
    return result


def _collect_all_pdf_image_items(
    documents: list[Document],
) -> tuple[list[tuple[str, str]], list[dict], list[str]]:
    """从所有 PDF 中提取图片及上下文，返回 (items, metadatas, ids)。"""
    page_text_map = _page_text_map(documents)
    seen_sources: set[str] = set()
    all_items: list[tuple[str, str]] = []
    all_metadatas: list[dict] = []
    all_ids: list[str] = []
    for doc in documents:
        source = (doc.metadata or {}).get("source")
        if not source:
            continue
        path = Path(source).resolve()
        if not path.is_file() or str(path) in seen_sources:
            continue
        seen_sources.add(str(path))
        for ctx, data_uri, meta in _extract_pdf_images_with_context(path, page_text_map):
            all_items.append((ctx, data_uri))
            all_metadatas.append(meta)
            pid = len(all_ids)
            all_ids.append(f"img:{path.name}:{meta['page']}:{meta['image_index']}:{pid}")
    return all_items, all_metadatas, all_ids


def _hybrid_chunk_document(
    doc: Document,
    recursive_splitter: RecursiveCharacterTextSplitter,
    semantic_splitter: SemanticChunker,
) -> tuple[list[Document], int]:
    """混合分块：先递归粗分，再对长块做语义细分。返回 (Document 列表, 走语义分块的块数)。"""
    if not doc.page_content or not doc.page_content.strip():
        return [], 0
    recursive_docs = recursive_splitter.split_documents([doc])
    final_docs: list[Document] = []
    semantic_count = 0
    for d in recursive_docs:
        if len(d.page_content) > SEMANTIC_SPLIT_MIN_LEN:
            semantic_count += 1
            try:
                semantic_docs = semantic_splitter.split_documents([d])
                final_docs.extend(semantic_docs)
            except (IndexError, ValueError):
                # 语义分块对极短或单句可能失败，回退为整块保留
                final_docs.append(d)
        else:
            final_docs.append(d)
    return final_docs, semantic_count


def main() -> None:
    if not PDF_DIR.is_dir():
        raise SystemExit(f"PDF 目录不存在，请先创建: {PDF_DIR}")

    snapshot = _build_pdf_snapshot()
    rebuild_reason = _get_rebuild_reason(snapshot)
    if rebuild_reason is None:
        print(f"PDF 未变化，复用现有向量库: {STORE_DIR}")
        return
    print(f"准备重建向量库，原因: {rebuild_reason}")

    # 1. 嵌入模型（供语义分块与向量库共用，通过 EmbeddingProvider 获取）
    provider = EmbeddingProvider()
    embedding_model = provider.get_embeddings()

    # 2. 初始化向量数据库（force_new=True 表示每次运行替换旧数据）
    vector_db = VectorDB(store_path=STORE_DIR, embeddings=embedding_model, force_new=True)

    # 3. 递归粗分
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
        length_function=len,
    )
    # 4. 语义分块（细粒度，校准语义）
    semantic_splitter = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=1.5,
    )

    # 5. 加载 pdf 目录下所有 PDF
    loader = PyPDFDirectoryLoader(
        path=str(PDF_DIR),
        glob="**/*.pdf",
        recursive=True,
        silent_errors=False,
    )
    documents = loader.load()
    if not documents:
        print("未在 pdf 目录下找到任何 PDF 文件，将保存空向量库。")
        vector_db.save()
        _save_manifest(snapshot)
        print(f"已写入空向量库，持久化目录: {STORE_DIR}")
        return
    print(f"已加载 {len(documents)} 个文档片段（页）。")

    # 6. 混合分块（内部用 split_document，直接得到 Document 列表）
    chunks: list[Document] = []
    total_semantic = 0
    for doc in documents:
        doc_chunks, n_semantic = _hybrid_chunk_document(
            doc, recursive_splitter, semantic_splitter
        )
        chunks.extend(doc_chunks)
        total_semantic += n_semantic
    chunks = [c for c in chunks if c.page_content.strip()]
    print(f"混合分块后共 {len(chunks)} 个 chunk。")
    print(f"其中走语义分块的长块数: {total_semantic}。")

    # 7. 新增文本 chunk 到向量库
    vector_db.add(chunks)

    # 8. 提取 PDF 内图片并做多模态向量化（仅当使用 DashScope 多模态模型时）
    image_items, image_metadatas, image_ids = _collect_all_pdf_image_items(documents)
    if image_items and isinstance(embedding_model, DashScopeMultimodalEmbeddings):
        print(f"正在对 {len(image_items)} 张 PDF 内图片做多模态向量化（含文字上下文）...")
        image_embeddings = embedding_model.embed_images_with_context(image_items)
        # 检索时展示的文字：该页上下文，便于 RAG 同时返回文本与图片相关信息
        text_embed_pairs = [((ctx.strip() or "（本页无文字）")[:2000], vec) for (ctx, _), vec in zip(image_items, image_embeddings)]
        vector_db.add_embeddings(text_embed_pairs, metadatas=image_metadatas, ids=image_ids)
        print(f"已写入 {len(image_items)} 条图片向量。")
    elif image_items:
        print(
            "检测到 "
            f"{len(image_items)} 张 PDF 内图片，但当前嵌入模型不是 DashScopeMultimodalEmbeddings "
            f"（实际类型: {type(embedding_model).__name__}），已跳过图片向量化。"
        )
    else:
        print("未从 PDF 中提取到可向量化图片：可能 PDF 本身无嵌入图片，或图片提取失败。")

    vector_db.save()
    _save_manifest(snapshot)
    print(f"已写入向量库，持久化目录: {STORE_DIR}")


if __name__ == "__main__":
    main()
