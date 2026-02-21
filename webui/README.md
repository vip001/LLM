# RAG 问答前端 (Next.js)

基于 Next.js 的 RAG 问答界面：在页面输入问题，请求后端 `/ask` 接口并展示返回结果。

## 运行方式

1. **先启动 Flask 后端**（在项目根目录 `llm` 下）：
   ```bash
   cd /path/to/llm
   .venv/bin/python server/ollama_qwen.py serve
   ```
   后端默认运行在 `http://127.0.0.1:5000`。

2. **再启动前端**：
   ```bash
   cd webui
   npm install   # 首次需要
   npm run dev
   ```
   浏览器打开 [http://localhost:3000](http://localhost:3000)。

3. 在页面输入框中输入问题（如「MMKV的用法」），点击「提问」，即可看到 RAG 返回的回答与原始 JSON。

## 接口说明

- 前端通过 **POST `/api/ask`** 提交 `{ "query": "你的问题" }`。
- Next.js 会将该请求代理到 Flask 的 `http://127.0.0.1:5000/ask`（可通过环境变量 `FLASK_ASK_URL` 修改）。

## 技术栈

- Next.js 16 (App Router)
- React 19
- Tailwind CSS 4
- TypeScript
