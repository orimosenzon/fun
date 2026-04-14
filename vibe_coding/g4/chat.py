#!/usr/bin/env python3
"""g4 — Local Gemma 4 (E2B) chatbot via Ollama. Run: python3 chat.py"""

import requests
import json
import sys
import readline  # arrow-key history support
from bidi.algorithm import get_display

MODEL = "gemma4:e2b"
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = (
    "You are a helpful, conversational AI assistant running locally. "
    "Respond concisely and naturally. "
    "You can answer in the same language the user writes in."
)


def fix_rtl(text: str) -> str:
    """Apply bidi algorithm line by line so Hebrew displays correctly in LTR terminals."""
    return "\n".join(get_display(line) if line.strip() else line for line in text.splitlines())


def chat(messages: list) -> str:
    payload = {"model": MODEL, "messages": messages, "stream": True, "options": {"num_ctx": 2048}}

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to Ollama. Is it running? Try: ollama serve")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] {e}")
        return ""

    print("Gemma: ...", end="\r", flush=True)

    full_response = []
    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = chunk.get("message", {}).get("content", "")
        if content:
            full_response.append(content)
        if chunk.get("done"):
            break

    result = "".join(full_response)
    print(f"Gemma: {fix_rtl(result)}\n")
    return result


def check_ollama_running() -> bool:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


def check_model_available() -> bool:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(MODEL in m for m in models)
    except Exception:
        return False


def main():
    print("=== g4 — Gemma 4 E2B Chatbot (local) ===")
    print(f"Model: {MODEL}")
    print("Type 'exit' or 'quit' to stop | Ctrl+C to interrupt\n")

    if not check_ollama_running():
        print("[ERROR] Ollama is not running.")
        print("  Install: curl -fsSL https://ollama.com/install.sh | sh")
        print("  Then:    ollama serve  (or it starts automatically)")
        sys.exit(1)

    if not check_model_available():
        print(f"[ERROR] Model '{MODEL}' not found locally.")
        print(f"  Run: ollama pull {MODEL}")
        sys.exit(1)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Bye!")
            break

        messages.append({"role": "user", "content": user_input})
        reply = chat(messages)

        if reply:
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
