FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api-base.txt requirements-api.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-api.txt


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FACEGUARD_MODEL_ROOT=/models/insightface \
    FACEGUARD_DEEPFAKE_MODEL_PATH=/models/deepfake/efficientnet_b4.onnx

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api-base.txt requirements-api.txt ./
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels -r requirements-api.txt \
    && rm -rf /wheels

COPY faceguard_api ./faceguard_api

RUN mkdir -p /models/insightface /models/deepfake \
    && useradd --create-home --uid 10001 faceguard \
    && chown -R faceguard:faceguard /app /models

USER faceguard
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "faceguard_api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
