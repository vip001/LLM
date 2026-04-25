"""
诊断本机 Ollama 上可用于向量查询的 embed 模型。

用法（在项目根目录）：
  .venv/bin/python server/check_ollama_embed.py
  .venv/bin/python server/check_ollama_embed.py "fervent_mcclintock/Qwen3-VL-Embedding-2B:F16"
  .venv/bin/python server/check_ollama_embed.py "http://localhost:11434" "fervent_mcclintock/Qwen3-VL-Embedding-2B:F16"

会列出本机模型、尝试 /api/embed，并打印失败时的完整响应以便排查。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx

DEFAULT_BASE = "http://localhost:11434"
MODEL_TO_TEST = "fervent_mcclintock/Qwen3-VL-Embedding-2B:F16"


def main() -> None:
    if len(sys.argv) >= 3:
        base, model = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 2:
        a = sys.argv[1]
        base, model = (a, MODEL_TO_TEST) if a.startswith("http") else (DEFAULT_BASE, a)
    else:
        base, model = DEFAULT_BASE, MODEL_TO_TEST
    if not base.startswith("http"):
        base = f"http://{base}"
    base = base.rstrip("/")

    print(f"Ollama base: {base}")
    print(f"测试 embed 模型: {model}\n")

    with httpx.Client(timeout=30.0, trust_env=False) as client:
        # 1. 列出本机模型
        print("1. 本机已拉取的模型 (GET /api/tags):")
        try:
            r = client.get(f"{base}/api/tags")
            r.raise_for_status()
            data = r.json()
            for m in data.get("models", []):
                name = m.get("name", "")
                print(f"   - {name}")
            if not data.get("models"):
                print("   (无)")
        except Exception as e:
            print(f"   请求失败: {e}")
            return

        # 2. 尝试 embed
        print(f"\n2. 尝试 POST /api/embed model={model!r} ...")
        try:
            r = client.post(
                f"{base}/api/embed",
                json={
                    "model": model,
                    "input": "hello",
                    "truncate": True,
                },
            )
            if r.status_code == 200:
                d = r.json()
                embs = d.get("embeddings", [])
                dim = len(embs[0]) if embs else 0
                print(f"   成功. 向量数={len(embs)}, 维度={dim}")
                return
            print(f"   失败 status={r.status_code}")
            print(f"   响应体: {r.text}")
            try:
                print("   解析为 JSON:", json.dumps(r.json(), indent=2, ensure_ascii=False))
            except Exception:
                pass
        except Exception as e:
            print(f"   异常: {e}")

        print("\n3. 排查建议:")
        print("   - 若错误中出现 127.0.0.1:xxxxx/embedding：说明 Ollama 服务端在把 embed 请求转发到代理。")
        print("     请在不带 OLLAMA_HOST/代理 的环境下重启 Ollama，或在系统自带终端（非 Cursor 内）运行本脚本。")
        print("   - 在终端执行: ollama list  查看模型名是否与上面一致（含大小写、tag）")
        print("   - 若名称不同，用列表里的完整名称作为第二个参数再试。")
        print("   - 部分 VL 模型在 Ollama 中仅支持 /api/chat 不支持 /api/embed，可改用纯文本 embedding 模型:")
        print("     ollama pull nomic-ai/text-embed-v1.5  或  ollama pull qwen3-embedding")
        print("     并在 embedding_provider 中把 EMBED_MODEL_QWEN 改为上述模型名。")


if __name__ == "__main__":
    main()
