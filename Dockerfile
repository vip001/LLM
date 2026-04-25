# Build from repository root: docker build -t webui .
# If metadata pull times out to docker.io, retry or add a registry mirror. Pin: --build-arg NODE_IMAGE=node:25.1.0-bookworm-slim
ARG NODE_IMAGE=node:25-bookworm-slim
FROM ${NODE_IMAGE} AS builder

WORKDIR /app

# Install dependencies (cache layer when lockfile unchanged)
COPY webui/package.json webui/package-lock.json ./
RUN npm ci

# Application source
COPY webui/ ./

RUN npm run build

ENV NODE_ENV=production
ENV PORT=3000
EXPOSE 3000

# Listen on all interfaces inside the container
CMD ["npm", "run", "start", "--", "-H", "0.0.0.0", "-p", "3000"]
