# "מה אמרו במועצה" ב-Hugging Face Space (CPU בחינם, 16GB).
FROM python:3.10-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user HF_HOME=/home/user/hf PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 PORT=7860 RAG_DEVICE=cpu RAG_INDEX=/home/user/app/index \
    RAG_QUERY_LOG=/tmp/queries.jsonl
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

USER user
# המודל נצרב לתוך האימג', כדי שהפעלה אחרי שינה לא תחכה להורדה של 2GB
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-m3')"

COPY --chown=user . .
EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "1", "--threads", "8", "--timeout", "120", "app:app"]
