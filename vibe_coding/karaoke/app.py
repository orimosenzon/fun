# Copyright (C) 2026 Ori Mosenzon and Claude (Anthropic AI)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# See the LICENSE file for details.

import os
import time
from flask import Flask, render_template, jsonify, request
from transcriber import process_url, url_id, STATIC_DIR, search_songs

app = Flask(__name__)

_jobs = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process():
    url = request.json.get("url", "").strip()
    title = request.json.get("title", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    job_id = url_id(url)
    _jobs[job_id] = {"stage": "starting", "started_at": time.time()}

    def on_stage(stage):
        _jobs[job_id]["stage"] = stage
        _jobs[job_id]["elapsed"] = round(time.time() - _jobs[job_id]["started_at"], 1)

    try:
        data = process_url(url, title, on_stage)
        _jobs[job_id]["stage"] = "done"
        return jsonify(data)
    except Exception as e:
        _jobs[job_id]["stage"] = "error"
        return jsonify({"error": str(e)}), 500


@app.route("/api/job_id", methods=["POST"])
def get_job_id():
    url = request.json.get("url", "").strip()
    return jsonify({"id": url_id(url)})


@app.route("/api/status/<job_id>")
def job_status(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"stage": "unknown"})
    elapsed = round(time.time() - job["started_at"], 1)
    return jsonify({"stage": job["stage"], "elapsed": elapsed})


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"results": []})
    results = search_songs(q)
    return jsonify({"results": results})



if __name__ == "__main__":
    app.run(debug=True, port=5001, threaded=True)
