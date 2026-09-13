#!/usr/bin/env python3
"""זורע את ההכוונה הוויזואלית מ-generate_images.py לתוך מסד העולם.

הפרומפטים ההם נכתבו ביד לכל 30 המקומות בסשן קודם, כשהצינור עוד היה
Gemini. הם עדיין ההכוונה הוויזואלית הכי טובה שיש, ולכן במקום לזרוק אותם
הם נכנסים לעמודת visual_notes וממשיכים לשמש את מודל התמונות של Azure.

מריצים פעם אחת. לא דורס מקום שכבר יש בו הכוונה ידנית.
"""

import argparse
import ast
import pathlib
import sqlite3

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / 'generate_images.py'


def load_prompts() -> dict[str, str]:
    """קורא את הדיקט מהקובץ בלי לייבא אותו — הייבוא של google.genai בראשו
    היה מפיל את הסקריפט על מכונה בלי החבילה."""
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'LOCATION_PROMPTS':
            return ast.literal_eval(node.value)
    raise SystemExit(f'לא נמצא LOCATION_PROMPTS ב-{SOURCE}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=str(ROOT / 'pipin_world.db'))
    ap.add_argument('--force', action='store_true',
                    help='לדרוס גם מקומות שכבר יש בהם הכוונה ויזואלית')
    args = ap.parse_args()

    prompts = load_prompts()
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    cols = {r['name'] for r in conn.execute('PRAGMA table_info(locations)')}
    if 'visual_notes' not in cols:
        raise SystemExit('העמודה visual_notes לא קיימת. הרץ את server.py פעם אחת '
                         'כדי שההגירה תרוץ.')

    written = skipped = 0
    with conn:
        for loc_id, text in prompts.items():
            conn.execute('INSERT OR IGNORE INTO locations (location_id) VALUES (?)',
                         (loc_id,))
            row = conn.execute('SELECT visual_notes FROM locations WHERE location_id=?',
                               (loc_id,)).fetchone()
            if row['visual_notes'] and not args.force:
                skipped += 1
                continue
            conn.execute('UPDATE locations SET visual_notes=? WHERE location_id=?',
                         (text, loc_id))
            written += 1

    print(f'נכתבו {written} מקומות, דולגו {skipped}.')


if __name__ == '__main__':
    main()
