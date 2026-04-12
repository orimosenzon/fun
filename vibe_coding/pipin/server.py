#!/usr/bin/env python3
# server.py - Flask dev server with error logging for pipin

import json
import os
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.')
LOG_FILE = 'pipin_errors.log'

def write_log(entry: dict):
    entry['server_time'] = datetime.now().isoformat()
    line = json.dumps(entry, ensure_ascii=False)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    # Also print to terminal for live visibility
    level = entry.get('level', 'ERROR').upper()
    msg = entry.get('message', '')
    src = entry.get('source', '')
    print(f"[{entry['server_time']}] [{level}] {msg}" + (f"  ({src})" if src else ''))

@app.route('/log', methods=['POST'])
def log_endpoint():
    try:
        data = request.get_json(force=True, silent=True) or {}
        write_log(data)
        return jsonify({'ok': True})
    except Exception as e:
        print(f'[LOG ENDPOINT ERROR] {e}')
        return jsonify({'ok': False}), 500

@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    print(f'Pipin dev server running on http://localhost:8765')
    print(f'Errors logged to: {os.path.abspath(LOG_FILE)}')
    app.run(port=8765, debug=False)
