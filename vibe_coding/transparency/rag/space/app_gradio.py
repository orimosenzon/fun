"""נקודת הכניסה ב-Hugging Face Space מסוג Gradio (Docker עבר לתשלום ב-9/2026).

Gradio מפעיל את השרת בדרך הרגילה (demo.launch). זה גם מה שמדווח ל-ZeroGPU על פונקציית
ה-GPU; הרצה עצמאית דרך uvicorn נפלה פעמיים (30/9): פעם על "No @spaces.GPU function
detected", ופעם על פורט 7860 שכבר תפוס. אחרי ההפעלה מכניסים לראש טבלת הנתיבים של
Gradio את הדף שלנו ואת /api/*, שמטופלים על ידי שרת ה-Flask (server.py, עותק של
rag/app.py) כמו שהוא. ה-deploy.sh מעתיק את הקובץ הזה ל-Space בשם app.py.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("RAG_DEVICE", "cpu")
os.environ.setdefault("RAG_INDEX", os.path.join(HERE, "index"))
os.environ.setdefault("RAG_QUERY_LOG", "/tmp/queries.jsonl")

import spaces  # noqa: E402  חייב לבוא לפני כל ספרייה שנוגעת ב-CUDA
import gradio as gr  # noqa: E402
from starlette.middleware.wsgi import WSGIMiddleware  # noqa: E402
from starlette.responses import FileResponse  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402

from server import app as flask_app  # noqa: E402


@spaces.GPU
def _unused():
    """ZeroGPU דורש לפחות פונקציה אחת כזו. החיפוש רץ על CPU ולא קורא לה."""
    return None


def flask_under_mount(environ, start_response):
    """Mount("/api") מעביר ל-Flask את הנתיב בלי /api; מחזירים אותו, כי הנתיבים שם הם /api/..."""
    environ["PATH_INFO"] = environ.get("SCRIPT_NAME", "") + environ.get("PATH_INFO", "")
    environ["SCRIPT_NAME"] = ""
    return flask_app(environ, start_response)


async def index(request):
    return FileResponse(os.path.join(HERE, "index.html"))


with gr.Blocks() as demo:
    gr.Markdown("החיפוש נמצא בכתובת הראשית של האתר.")

if __name__ == "__main__":
    # ssr_mode=False: ב-HF, Gradio 6 מריץ שרת רינדור (Node) על 7860 ומעביר את הפייתון
    # ל-7861, ואז הנתיבים שלנו לא מקבלים בקשות בכלל (30/9)
    server_app, _, _ = demo.launch(server_name="0.0.0.0", server_port=7860,
                                   prevent_thread_lock=True, ssr_mode=False)
    for route in reversed([Route("/", index), Mount("/api", WSGIMiddleware(flask_under_mount))]):
        server_app.router.routes.insert(0, route)
    demo.block_thread()
