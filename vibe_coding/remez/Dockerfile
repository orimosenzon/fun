# Web/cloud image (Hugging Face Spaces — Docker SDK, port 7860).
# annotate.py / experiment.py are CLI tools, intentionally excluded —
# the Space serves the labeling UI only.
FROM python:3.11-slim

# HF Spaces runs the container as a non-root user (UID 1000).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt gunicorn

COPY --chown=user label_ui.py .
COPY --chown=user remez_lib.py .
COPY --chown=user templates/ templates/

EXPOSE 7860

# workers=1: uploads + annotation JSONs live on the container's local FS,
# so a second worker would race on shared state. Threads cover concurrent
# transcribe calls (each spends ~30s in the Anthropic API).
CMD ["gunicorn", "--workers", "1", "--threads", "8", \
     "--worker-class", "gthread", "--timeout", "0", \
     "--bind", "0.0.0.0:7860", "label_ui:app"]
