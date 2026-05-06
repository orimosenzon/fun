#!/usr/bin/env python3
# server.py - Flask dev server with error logging for pipin

import json
import os
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)
LOG_FILE = 'pipin_errors.log'
DB_FILE = os.environ.get('DB_FILE', 'pipin_world.db')


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

@app.route('/api/world/player/<player_id>/stats', methods=['GET'])
def get_player_stats(player_id):
    with get_db() as conn:
        rows = conn.execute(
            'SELECT location_id FROM locations WHERE player_id=? ORDER BY created_at',
            (player_id,)
        ).fetchall()
    return jsonify({
        'player_id': player_id,
        'locations_canonized': len(rows),
        'location_ids': [r['location_id'] for r in rows],
    })


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
    """Idempotent canon save. Narrative and image arrive in separate calls
    (text is fast, image generation is slow), so we let either field fill an
    empty slot — but never overwrite a value that's already canonized."""
    data = request.get_json(force=True, silent=True) or {}
    new_narrative = data.get('narrative')
    new_image = data.get('image_data')
    with get_db() as conn:
        row = conn.execute(
            'SELECT player_id, narrative, image_data FROM locations WHERE location_id=?',
            (location_id,)
        ).fetchone()
        if row is None:
            conn.execute(
                'INSERT INTO locations (location_id, player_id, created_at, narrative, image_data) VALUES (?,?,?,?,?)',
                (location_id, data.get('player_id'), datetime.now().isoformat(),
                 new_narrative, new_image)
            )
            return jsonify({'saved': True, 'created': True})

        updates, params = [], []
        if new_narrative and not row['narrative']:
            updates.append('narrative=?'); params.append(new_narrative)
        if new_image and not row['image_data']:
            updates.append('image_data=?'); params.append(new_image)
        if not updates:
            return jsonify({'saved': False, 'reason': 'already_canonized'})
        params.append(location_id)
        conn.execute(f"UPDATE locations SET {', '.join(updates)} WHERE location_id=?", params)
    return jsonify({'saved': True, 'updated': updates})


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

init_db()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8765))
    print(f'Pipin server running on port {port}')
    app.run(host='0.0.0.0', port=port, debug=False)
