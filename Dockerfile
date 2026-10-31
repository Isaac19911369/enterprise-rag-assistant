FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .
COPY corpus ./corpus

RUN useradd --create-home --uid 10001 app
USER 10001
EXPOSE 8080
CMD ["uvicorn", "rag_assistant.api:app", "--host", "0.0.0.0", "--port", "8080"]
