FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    nodejs \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper base model so first request doesn't timeout
RUN python -c "import whisper; whisper.load_model('base')"

COPY . .

EXPOSE 8000
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8000} --timeout 300 --workers 1 app:app"]
