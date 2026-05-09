#!/bin/sh
set -e
cd "$(dirname "$0")"
python grpc_ask_server.py &
exec gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 4 --timeout 600 ollama_qwen:app
