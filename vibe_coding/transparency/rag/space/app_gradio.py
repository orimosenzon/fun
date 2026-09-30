"""נקודת הכניסה ב-Hugging Face Space מסוג Gradio (Docker עבר לתשלום ב-9/2026).

Gradio בנוי על FastAPI, ולכן אפשר להריץ את שרת ה-Flask שלנו (server.py, עותק של
rag/app.py) כמו שהוא, מתחת ל-FastAPI. ממשק Gradio מינימלי יושב ב-/gradio רק כדי
לעמוד בדרישות של סוג ה-Space. ה-deploy.sh מעתיק את הקובץ הזה ל-Space בשם app.py.
"""
import os

os.environ.setdefault("RAG_DEVICE", "cpu")
os.environ.setdefault("RAG_INDEX", os.path.join(os.path.dirname(os.path.abspath(__file__)), "index"))
os.environ.setdefault("RAG_QUERY_LOG", "/tmp/queries.jsonl")

import gradio as gr  # noqa: E402
import spaces  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.wsgi import WSGIMiddleware  # noqa: E402

from server import app as flask_app  # noqa: E402


@spaces.GPU
def _unused():
    """ZeroGPU דורש לפחות פונקציה אחת כזו. החיפוש רץ על CPU ולא קורא לה."""
    return None


with gr.Blocks() as demo:
    gr.Markdown("הכלי עצמו נמצא בכתובת הראשית של ה-Space.")

api = FastAPI()
api = gr.mount_gradio_app(api, demo, path="/gradio")
api.mount("/", WSGIMiddleware(flask_app))

if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=7860)
