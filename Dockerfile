FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir -U pip && \
    python -m pip install --no-cache-dir .

CMD ["python", "-m", "pytest", "-q"]
