#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ACR_REGISTRY=registry.cn-hangzhou.aliyuncs.com \
#   ACR_NAMESPACE=your-namespace \
#   IMAGE_TAG=v1.0.0 \
#   # optional: deploy directly to aliyun ecs after push
#   ALIYUN_HOST=1.2.3.4 \
#   ALIYUN_USER=root \
#   ALIYUN_PORT=22 \
#   ALIYUN_SSH_KEY=~/.ssh/id_ed25519_aliyun   # optional, if not using default keys
#   DOCKER_PLATFORM=linux/amd64 \             # default; required when ECS is x86 and you build on Apple Silicon
#   REMOTE_APP_DIR=/opt/llm \
#   ./deploy-aliyun.sh
  #IMAGE_TAG=v1.0.0 
#   # optional: deploy directly to aliyun ecs after push
#   ALIYUN_HOST=1.2.3.4 \
#   ALIYUN_USER=root \
#   ALIYUN_PORT=22 \
#   REMOTE_APP_DIR=/opt/llm \
  ACR_REGISTRY=crpi-et6d0bbajol75ec4.cn-hangzhou.personal.cr.aliyuncs.com
  ACR_NAMESPACE=llmai
  ALIYUN_HOST=120.26.236.143
  ALIYUN_USER=root
  ALIYUN_SSH_KEY=~/.ssh/login.pem  # optional, if not using default keys

if [[ -z "${ACR_REGISTRY:-}" || -z "${ACR_NAMESPACE:-}" ]]; then
  echo "ACR_REGISTRY and ACR_NAMESPACE are required."
  exit 1
fi

IMAGE_TAG="${IMAGE_TAG:-latest}"
ALIYUN_PORT="${ALIYUN_PORT:-22}"
REMOTE_APP_DIR="${REMOTE_APP_DIR:-/opt/llm}"

# Optional: path to private key for ssh/scp (expand ~ for -i which does not expand it)
SSH_IDENTITY_ARGS=()
if [[ -n "${ALIYUN_SSH_KEY:-}" ]]; then
  _key="${ALIYUN_SSH_KEY/#\~/${HOME}}"
  SSH_IDENTITY_ARGS=(-i "${_key}")
fi

WEBUI_IMAGE="${ACR_REGISTRY}/${ACR_NAMESPACE}/llm-webui:${IMAGE_TAG}"
SERVER_IMAGE="${ACR_REGISTRY}/${ACR_NAMESPACE}/llm-server:${IMAGE_TAG}"
LOGIN_IMAGE="${ACR_REGISTRY}/${ACR_NAMESPACE}/llm-loginserver:${IMAGE_TAG}"
MCP_SERVER_IMAGE="${ACR_REGISTRY}/${ACR_NAMESPACE}/llm-mcpserver:${IMAGE_TAG}"
NGINX_IMAGE="${ACR_REGISTRY}/${ACR_NAMESPACE}/llm-nginx:${IMAGE_TAG}"

# ECS is usually linux/amd64; Docker on Apple Silicon defaults to arm64 without --platform.
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

echo "Building ${WEBUI_IMAGE} (${DOCKER_PLATFORM})"
docker build --platform "${DOCKER_PLATFORM}" -f webui/Dockerfile -t "${WEBUI_IMAGE}" .

echo "Building ${SERVER_IMAGE} (${DOCKER_PLATFORM})"
docker build --platform "${DOCKER_PLATFORM}" -f server/Dockerfile -t "${SERVER_IMAGE}" .

echo "Building ${LOGIN_IMAGE} (${DOCKER_PLATFORM})"
docker build --platform "${DOCKER_PLATFORM}" -f loginserver/Dockerfile -t "${LOGIN_IMAGE}" .

echo "Building ${MCP_SERVER_IMAGE} (${DOCKER_PLATFORM})"
docker build --platform "${DOCKER_PLATFORM}" -f mcpserver/Dockerfile -t "${MCP_SERVER_IMAGE}" .

echo "Building ${NGINX_IMAGE} (${DOCKER_PLATFORM})"
docker build --platform "${DOCKER_PLATFORM}" -f nginx/Dockerfile -t "${NGINX_IMAGE}" .

echo "Pushing ${WEBUI_IMAGE}"
docker push "${WEBUI_IMAGE}"

echo "Pushing ${SERVER_IMAGE}"
docker push "${SERVER_IMAGE}"

echo "Pushing ${LOGIN_IMAGE}"
docker push "${LOGIN_IMAGE}"

echo "Pushing ${MCP_SERVER_IMAGE}"
docker push "${MCP_SERVER_IMAGE}"

echo "Pushing ${NGINX_IMAGE}"
docker push "${NGINX_IMAGE}"

if [[ -n "${ALIYUN_HOST:-}" ]]; then
  if [[ -z "${ALIYUN_USER:-}" ]]; then
    echo "ALIYUN_USER is required when ALIYUN_HOST is set."
    exit 1
  fi

  REMOTE="${ALIYUN_USER}@${ALIYUN_HOST}"
  echo "Deploying to ${REMOTE}:${REMOTE_APP_DIR}"
  ssh "${SSH_IDENTITY_ARGS[@]}" -p "${ALIYUN_PORT}" "${REMOTE}" \
    "mkdir -p '${REMOTE_APP_DIR}/nginx' '${REMOTE_APP_DIR}/nginx/logs'"
  scp "${SSH_IDENTITY_ARGS[@]}" -P "${ALIYUN_PORT}" docker-compose.aliyun.yml "${REMOTE}:${REMOTE_APP_DIR}/docker-compose.yml"
  scp "${SSH_IDENTITY_ARGS[@]}" -P "${ALIYUN_PORT}" nginx/default.conf "${REMOTE}:${REMOTE_APP_DIR}/nginx/default.conf"
  ssh "${SSH_IDENTITY_ARGS[@]}" -p "${ALIYUN_PORT}" "${REMOTE}" \
    "cd '${REMOTE_APP_DIR}' && \
     ACR_REGISTRY='${ACR_REGISTRY}' ACR_NAMESPACE='${ACR_NAMESPACE}' IMAGE_TAG='${IMAGE_TAG}' \
     docker compose pull && \
     ACR_REGISTRY='${ACR_REGISTRY}' ACR_NAMESPACE='${ACR_NAMESPACE}' IMAGE_TAG='${IMAGE_TAG}' \
     docker compose up -d"

  echo "Remote deploy complete."
else
  echo "Push complete. On aliyun server run:"
  echo "  mkdir -p ${REMOTE_APP_DIR}"
  echo "  # upload docker-compose.aliyun.yml to ${REMOTE_APP_DIR}/docker-compose.yml"
  echo "  cd ${REMOTE_APP_DIR}"
  echo "  export ACR_REGISTRY=${ACR_REGISTRY}"
  echo "  export ACR_NAMESPACE=${ACR_NAMESPACE}"
  echo "  export IMAGE_TAG=${IMAGE_TAG}"
  echo "  docker compose pull"
  echo "  docker compose up -d"
fi
