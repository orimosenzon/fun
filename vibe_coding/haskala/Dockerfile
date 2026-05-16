# Web/cloud image (Hugging Face Spaces — Docker SDK, port 7860).
# Desktop deps (pywebview/Qt) are intentionally excluded — the cloud
# build is the browser flow, not the native wrapper.
FROM python:3.11-slim

# HF Spaces runs the container as a non-root user (UID 1000).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt gunicorn

COPY --chown=user app.py .
COPY --chown=user templates/ templates/

EXPOSE 7860

# workers=1 is required: the OCR job store (JOBS) lives in process memory,
# so a second worker wouldn't see in-flight jobs. Concurrency (the SSE
# stream + the fetch that started it) is handled by threads. timeout 0
# disables the worker silence-kill — OCR pages take 15-45s each and the
# SSE response streams for the whole job.
CMD ["gunicorn", "--workers", "1", "--threads", "8", \
     "--worker-class", "gthread", "--timeout", "0", \
     "--bind", "0.0.0.0:7860", "app:app"]
