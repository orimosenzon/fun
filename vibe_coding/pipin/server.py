#!/usr/bin/env python3
# server.py - Flask dev server with error logging for pipin

import json
import os
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.')
LOG_FILE = 'pipin_errors.log'
DB_FILE = 'pipin_world.db'


# ── DB ──────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS locations (
            location_id  TEXT PRIMARY KEY,
            player_id    TEXT,
            created_at   TEXT,
            narrative    TEXT,
            image_data   TEXT
        )''')


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

# ── World API ───────────────────────────────────────────────────────────

@app.route('/api/world/location/<location_id>', methods=['GET'])
def get_location(location_id):
    with get_db() as conn:
        row = conn.execute(
            'SELECT * FROM locations WHERE location_id=?', (location_id,)
        ).fetchone()
    if not row:
        return jsonify({'found': False}), 404
    return jsonify({
        'found': True,
        'narrative': row['narrative'],
        'image_data': row['image_data'],
        'player_id': row['player_id'],
        'created_at': row['created_at'],
    })


@app.route('/api/world/location/<location_id>', methods=['POST'])
def save_location(location_id):
    with get_db() as conn:
        existing = conn.execute(
            'SELECT location_id FROM locations WHERE location_id=?', (location_id,)
        ).fetchone()
        if existing:
            return jsonify({'saved': False, 'reason': 'already_canonized'})
        data = request.get_json(force=True, silent=True) or {}
        conn.execute(
            'INSERT INTO locations (location_id, player_id, created_at, narrative, image_data) VALUES (?,?,?,?,?)',
            (location_id, data.get('player_id'), datetime.now().isoformat(),
             data.get('narrative'), data.get('image_data'))
        )
    return jsonify({'saved': True})


# ── Logging ─────────────────────────────────────────────────────────────

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
    init_db()
    print(f'Pipin dev server running on http://localhost:8765')
    print(f'Errors logged to: {os.path.abspath(LOG_FILE)}')
    print(f'World DB: {os.path.abspath(DB_FILE)}')
    app.run(port=8765, debug=False)
