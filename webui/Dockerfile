# Build from repository root: docker build -f webui/Dockerfile -t webui .
# If metadata pull times out to docker.io, retry or add a registry mirror. Pin: --build-arg NODE_IMAGE=node:25.1.0-bookworm-slim
ARG NODE_IMAGE=docker.m.daocloud.io/library/node:25-bookworm-slim
FROM ${NODE_IMAGE} AS builder

WORKDIR /app

# Install dependencies (cache layer when lockfile unchanged)
COPY webui/package.json webui/package-lock.json ./
RUN npm ci

# 与 compose 中 Flask 服务名一致；compose 可覆盖。未设置时 route.ts 在 /.dockerenv 下也会回退到该地址
ARG FLASK_ASK_URL=http://server:5000/ask
ENV FLASK_ASK_URL=${FLASK_ASK_URL}
COPY webui/ ./
RUN npm run build

# Production runtime for Next standalone server (includes /api routes)
FROM ${NODE_IMAGE} AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0
ENV FLASK_ASK_URL=http://server:5000/ask

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
