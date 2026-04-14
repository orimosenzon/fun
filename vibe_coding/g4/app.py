#!/usr/bin/env python3
"""g4 — Gemma 4 E2B web chatbot. Run: python3 app.py"""

from flask import Flask, render_template, request, Response, stream_with_context
import requests
import json

app = Flask(__name__)

MODEL = "gemma4:e2b"
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = (
    "You are a helpful, conversational AI assistant running locally. "
    "Respond concisely and naturally. "
    "You can answer in the same language the user writes in."
)


@app.route("/")
def index():
    return render_template("index.html", model=MODEL)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    messages = data.get("messages", [])

    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "stream": True,
        "options": {"num_ctx": 2048},
    }

    def generate():
        try:
            resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield f"data: {json.dumps({'text': content})}\n\n"
                if chunk.get("done"):
                    yield "data: [DONE]\n\n"
                    break
        except requests.exceptions.ConnectionError:
            yield f"data: {json.dumps({'error': 'Cannot connect to Ollama'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


if __name__ == "__main__":
    print(f"=== g4 — Gemma 4 E2B Web Chatbot ===")
    print(f"Open: http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
