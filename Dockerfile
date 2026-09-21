FROM node:24-bookworm-slim AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 CONTOUR_ROOT=/app CONTOUR_DATA_DIR=/app/data CONTOUR_ARTIFACT_DIR=/app/artifacts
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.lock pyproject.toml ./
RUN pip install -r requirements.lock
COPY src/ ./src/
RUN pip install --no-deps . && useradd --uid 10001 --create-home contour
COPY --from=web /web/dist ./web/dist/
RUN mkdir -p /app/data/runtime /app/artifacts && chown -R contour:contour /app
USER contour
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=20s --start-period=40s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/ready',timeout=15)"
CMD ["uvicorn", "moscollector.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
