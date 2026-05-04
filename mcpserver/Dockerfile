# MCP server (FastMCP): streamable-http on 0.0.0.0:8001, path /mcp
#
# Build from repository root (needs sibling llm_common for file:../llm_common):
#   docker build -f mcpserver/Dockerfile -t llm-mcpserver .
#
# Run (set PG_*, DASHSCOPE_API_KEY, etc. as needed):
#   docker run --rm -p 8001:8001 -e PGHOST=... -e PGDATABASE=... llm-mcpserver

FROM docker.m.daocloud.io/library/python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY llm_common/ /app/llm_common/
COPY mcpserver/ /app/mcpserver/

WORKDIR /app/mcpserver

RUN pip install -e .

EXPOSE 8001

CMD ["llm-mcp"]
