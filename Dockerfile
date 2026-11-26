FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY support_bot ./support_bot
COPY scripts ./scripts

RUN useradd --create-home bot && mkdir -p /app/logs && chown bot /app/logs
USER bot

EXPOSE 8000
# One process on purpose: dialogue state lives in memory. Parallel users are served by asyncio.
CMD ["python", "-m", "support_bot.demo", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]
