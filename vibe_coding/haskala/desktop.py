"""Native desktop wrapper for the השכלה OCR tester.

Runs the existing Flask app in a background thread and shows it in a
pywebview window, so Avishai gets a real double-click app with native
file open/save dialogs that remember the last folder — no browser, no
terminal, 100% of the OCR logic reused unchanged.

Run:  python desktop.py
"""

import os
import threading
import time
import urllib.request

import webview

# Must be set before importing app: it enables the local file-path
# endpoints (and skips Basic Auth) for this trusted single-user window.
os.environ["HASKALA_DESKTOP"] = "1"

from app import app  # noqa: E402

HOST = "127.0.0.1"
PORT = 5050
BASE_URL = f"http://{HOST}:{PORT}"

# Remember the last-used directory across all three dialogs, so each pick
# opens where the previous one left off (the user's main UX request).
_STATE_DIR = os.path.expanduser("~/.haskala")
_LAST_DIR_FILE = os.path.join(_STATE_DIR, "last_dir")


def _load_last_dir() -> str:
    try:
        with open(_LAST_DIR_FILE, encoding="utf-8") as f:
            d = f.read().strip()
        if d and os.path.isdir(d):
            return d
    except OSError:
        pass
    return os.path.expanduser("~")


def _save_last_dir(path: str) -> None:
    try:
        os.makedirs(_STATE_DIR, exist_ok=True)
        with open(_LAST_DIR_FILE, "w", encoding="utf-8") as f:
            f.write(os.path.dirname(path))
    except OSError:
        pass


class Api:
    """Exposed to the page as window.pywebview.api.*"""

    def __init__(self):
        self.last_dir = _load_last_dir()
        self._window = None

    def _remember(self, path: str) -> None:
        self.last_dir = os.path.dirname(path)
        _save_last_dir(path)

    def pick_pdf(self):
        """Native open dialog for an exercise file. Returns a path or None."""
        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            directory=self.last_dir,
            allow_multiple=False,
            file_types=(
                "קבצי תרגיל (*.pdf;*.jpg;*.jpeg;*.png)",
                "כל הקבצים (*.*)",
            ),
        )
        if not result:
            return None
        path = result[0] if isinstance(result, (list, tuple)) else result
        self._remember(path)
        return path

    def pick_json(self):
        """Native open dialog filtered to JSON. Returns a path or None."""
        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            directory=self.last_dir,
            allow_multiple=False,
            file_types=("קבצים מפוענחים (*.json)", "כל הקבצים (*.*)"),
        )
        if not result:
            return None
        path = result[0] if isinstance(result, (list, tuple)) else result
        self._remember(path)
        return path

    def save_json(self, default_name: str, content: str):
        """Native save dialog. Writes `content` to the chosen path.
        Returns the saved path or None if cancelled."""
        result = self._window.create_file_dialog(
            webview.SAVE_DIALOG,
            directory=self.last_dir,
            save_filename=default_name,
            file_types=("קבצים מפוענחים (*.json)",),
        )
        if not result:
            return None
        path = result[0] if isinstance(result, (list, tuple)) else result
        if not path.lower().endswith(".json"):
            path += ".json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        self._remember(path)
        return path

    def save_docx(self, default_name: str, content_b64: str):
        """Native save dialog for the evaluation .docx. The browser already
        has the bytes (it called /evaluation/docx and base64'd the result),
        so we just place them — no separate Python build path."""
        result = self._window.create_file_dialog(
            webview.SAVE_DIALOG,
            directory=self.last_dir,
            save_filename=default_name,
            file_types=("קובץ Word (*.docx)",),
        )
        if not result:
            return None
        path = result[0] if isinstance(result, (list, tuple)) else result
        if not path.lower().endswith(".docx"):
            path += ".docx"
        import base64 as _b64
        with open(path, "wb") as f:
            f.write(_b64.b64decode(content_b64))
        self._remember(path)
        return path


def _run_flask():
    app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)


def _wait_for_server(timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(BASE_URL, timeout=1)
            return
        except Exception:
            time.sleep(0.2)


def main():
    threading.Thread(target=_run_flask, daemon=True).start()
    _wait_for_server()

    api = Api()
    window = webview.create_window(
        "השכלה — מערכת אוטומטית לבדיקת תרגילים",
        BASE_URL,
        js_api=api,
        width=1200,
        height=900,
        text_select=True,
    )
    api._window = window
    webview.start()


if __name__ == "__main__":
    main()
