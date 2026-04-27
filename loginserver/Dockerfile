FROM docker.m.daocloud.io/library/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY loginserver/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -r requirements.txt

COPY loginserver/ ./loginserver/

WORKDIR /app/loginserver

EXPOSE 8000
CMD ["uvicorn", "loginserver:app", "--host", "0.0.0.0", "--port", "8000"]
