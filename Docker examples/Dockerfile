FROM python:3.7.8-slim

RUN mkdir -p /app
WORKDIR /app

COPY ccap/azure_functions_local_execution_v3/requirements.txt requirements.txt
RUN pip install -r requirements.txt
