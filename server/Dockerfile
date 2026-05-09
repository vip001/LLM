FROM docker.m.daocloud.io/library/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY llm_common/ ./llm_common/
COPY server/requirements.txt ./server/requirements.txt
WORKDIR /app/server
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -r requirements.txt

WORKDIR /app
COPY server/ ./server/
COPY proto/ ./proto/

WORKDIR /app/server

RUN chmod +x run_services.sh

EXPOSE 5000 50051
CMD ["./run_services.sh"]
