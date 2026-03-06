import os
from flask import Flask, render_template, jsonify, request
from analogy_engine import AnalogyEngine

app = Flask(__name__)
engine = AnalogyEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/quiz/question")
def quiz_question():
    return jsonify(engine.generate_question())


@app.route("/api/free/words")
def free_words():
    return jsonify({"words": engine.get_word_pool(24)})


@app.route("/api/free/ask", methods=["POST"])
def free_ask():
    data = request.json
    a = data.get("a", "").strip()
    b = data.get("b", "").strip()
    c = data.get("c", "").strip()
    if not all([a, b, c]):
        return jsonify({"error": "Please fill all three slots"})
    results, error = engine.answer_analogy(a, b, c)
    if error:
        return jsonify({"error": error})
    return jsonify({"results": [{"word": w, "score": round(s, 3)} for w, s in results]})


@app.route("/api/visualize", methods=["POST"])
def visualize():
    data = request.json
    n = min(max(int(data.get("n", 20)), 5), 60)
    method = data.get("method", "pca")
    category = data.get("category") or None
    points = engine.get_2d_projection(n, method, category)
    return jsonify({"points": points})


@app.route("/api/visualize/categories")
def visualize_categories():
    return jsonify({"categories": list(engine.categories.keys())})


@app.route("/api/nearest", methods=["POST"])
def nearest():
    word = request.json.get("word", "").strip()
    if not word:
        return jsonify({"error": "Please enter a word"})
    results, error = engine.nearest_words(word)
    if error:
        return jsonify({"error": error})
    return jsonify({"results": [{"word": w, "score": s} for w, s in results]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
