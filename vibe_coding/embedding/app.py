import os
import threading
from flask import Flask, render_template, jsonify, request, abort
from analogy_engine import AnalogyEngine

app = Flask(__name__)
engine = None
load_status = {"ready": False, "progress": 0, "message": "Starting up..."}


def _load_engine():
    global engine, load_status
    try:
        def on_progress(progress, message):
            load_status["progress"] = progress
            load_status["message"] = message
        engine = AnalogyEngine(on_progress=on_progress)
        load_status.update({"ready": True, "progress": 100, "message": "Ready!"})
    except Exception as e:
        load_status.update({"ready": False, "progress": -1, "message": f"Error: {e}"})


threading.Thread(target=_load_engine, daemon=True).start()


def _check_ready():
    if not load_status["ready"]:
        abort(503, description="Model is still loading")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify(load_status)


@app.route("/api/explore", methods=["POST"])
def explore():
    _check_ready()
    data = request.json
    a = data.get("a", "").strip()
    b = data.get("b", "").strip()
    c = data.get("c", "").strip() or None
    if not a or not b:
        return jsonify({"error": "Please enter both words"})
    return jsonify(engine.explore_analogy(a, b, c))


@app.route("/api/nearest", methods=["POST"])
def nearest():
    _check_ready()
    word = request.json.get("word", "").strip()
    if not word:
        return jsonify({"error": "Please enter a word"})
    results, error = engine.nearest_words(word)
    if error:
        return jsonify({"error": error})
    return jsonify({"results": [{"word": w, "score": s} for w, s in results]})


if __name__ == "__main__":
    import socket, os
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        port = 5000
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("localhost", port)) != 0:
                    break
            port += 1
        os.environ["FLASK_PORT"] = str(port)
        print(f"\n  Word Analogy Explorer → http://localhost:{port}\n")
    else:
        port = int(os.environ.get("FLASK_PORT", 5000))
    app.run(debug=True, port=port)
