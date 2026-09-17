#!/usr/bin/env python3
"""Nightly scan of the shared Google Drive folder.

The council people (עירית רתם first of all) drop invitations to committee
meetings and other material into the shared folder, and until 14/9/2026 the
only way the wiki learned about a new file was a note on the bot's talk page.
This script makes the folder itself the notification: every file it has not
seen before is either handled or reported, and a short note on the bot's talk
page says which.

What it does with a new file:
  * an image inside the "וועדות המועצה" subfolder is a meeting poster. It is
    read with the Azure vision model, checked (the date must parse, lie ahead,
    and agree with the weekday printed on the poster), uploaded to the wiki
    as ועדה-<ועדה>-<יום>-<חודש>.jpg with the usual fair-use text, and added
    as a row to "ישיבות ועדה קרובות" on [[ועדות המועצה]]. The poster itself is
    in the row, so a reader can always check the machine's reading against it.
  * anything else (a docx roster, a file in another folder, a poster the model
    could not read consistently) is only reported, for a person to handle.

Reading the folder needs no credentials (see CLAUDE.md); writing to it is not
possible, which is why the seen-list lives here, in drive_seen.json.

Usage:
    python3 drive_scan.py [--dry-run] [--seed]
      --seed   mark everything currently in the folder as seen, without
               touching the wiki. Run once when installing.
"""

import argparse
import datetime as dt
import html
import io
import json
import pathlib
import re
import subprocess
import sys

import requests

from wiki_client import WikiClient, API_URL
import update_committees as uc

ROOT = '11wEMLKgdSESQuE3rVYwJ4LlLShXKLbjQ'
COMMITTEES_FOLDER = 'וועדות המועצה'
STATE = pathlib.Path(__file__).with_name('drive_seen.json')
TALK = 'שיחת משתמש:אורי מוסנזון בוט'
BOARD = uc.PAGE
UA = {'User-Agent': 'Pardespedia-Bot/1.0 (orimosenzon@gmail.com)'}
IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.webp')

WEEKDAYS = ['שני', 'שלישי', 'רביעי', 'חמישי', 'שישי', 'שבת', 'ראשון']  # date.weekday() order

VISION_PROMPT = (
    'זו כרזה של המועצה המקומית פרדס חנה-כרכור. קרא אותה והשב ב-JSON בלבד, בלי הסברים, '
    'עם המפתחות הבאים: committee (שם הוועדה בדיוק כפי שכתוב, בלי "של פרדס חנה כרכור"), '
    'date (התאריך בצורה YYYY-MM-DD; אם השנה לא כתובה, השתמש בשנה %d), '
    'weekday (יום בשבוע כפי שכתוב, למשל "רביעי"), time (השעה בצורה HH:MM), '
    'place (מקום הישיבה כפי שכתוב), organizer (מי חתום על ההזמנה, שם ותפקיד, אם כתוב), '
    'topic (נושא הישיבה, רק אם כתוב במפורש בכרזה). '
    'כל מפתח שאין לו מידע בכרזה יקבל null. אל תמציא דבר שלא כתוב.'
)


# ---------------------------------------------------------------- drive ----

def list_folder(folder_id: str):
    """[(file_id, name)] for one Drive folder, via the public embedded view."""
    r = requests.get('https://drive.google.com/embeddedfolderview',
                     params={'id': folder_id}, headers=UA, timeout=60)
    r.raise_for_status()
    out = []
    for m in re.finditer(r'<div class="flip-entry" id="entry-([^"]+)".*?'
                         r'<div class="flip-entry-title">(.*?)</div>', r.text, re.S):
        out.append((m.group(1), html.unescape(m.group(2)).strip()))
    return out


def walk(root: str):
    """Every file in the root and one level of subfolders: {id: (name, folder)}."""
    files = {}
    for fid, name in list_folder(root):
        # a subfolder has no extension; probe it, and if it lists, descend
        if '.' not in name:
            try:
                sub = list_folder(fid)
            except requests.RequestException:
                sub = []
            if sub:
                for sid, sname in sub:
                    files[sid] = (sname, name)
                continue
        files[fid] = (name, '')
    return files


def download(file_id: str) -> bytes:
    r = requests.get('https://drive.google.com/uc',
                     params={'export': 'download', 'id': file_id}, headers=UA, timeout=120)
    r.raise_for_status()
    return r.content


# --------------------------------------------------------------- poster ----

def read_poster(data: bytes):
    """Ask the vision model twice; accept only if both readings agree on the
    facts that go into the board row. Returns dict or None."""
    import ai_azure
    year = dt.date.today().year
    readings = []
    for _ in range(2):
        raw = ai_azure.vision(VISION_PROMPT % year, data, max_tokens=600, temperature=0)
        m = re.search(r'\{.*\}', raw, re.S)
        if not m:
            return None
        try:
            readings.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            return None
    a, b = readings
    for k in ('committee', 'date', 'time', 'place'):
        if not a.get(k) or a.get(k) != b.get(k):
            return None
    try:
        d = dt.date.fromisoformat(a['date'])
    except ValueError:
        return None
    if a.get('weekday') and a['weekday'].replace('יום ', '') != WEEKDAYS[d.weekday()]:
        return None                       # printed weekday disagrees with the date
    if d < dt.date.today():
        return None                       # a past meeting is not for the board
    a['_date'] = d
    return a


def short_committee(name: str) -> str:
    name = re.sub(r'\s+של\s+(פרדס חנה|המועצה).*$', '', name).strip()
    return name


def poster_filename(committee: str, d: dt.date) -> str:
    stem = re.sub(r'^(ועדת|ועדה ל|הוועדה ל|ועדה)\s*', '', committee).strip()
    stem = re.sub(r'[\\/:*?"<>|#\[\]{}]', '', stem)
    return f'ועדה-{stem}-{d.day}-{d.month}.jpg'


def to_jpeg(data: bytes) -> bytes:
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert('RGB')
    w, h = im.size
    if w > 700:
        im = im.resize((700, int(h * 700 / w)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=85)
    return buf.getvalue()


def upload_poster(client: WikiClient, fn: str, data: bytes, info: dict):
    desc = (f'כרזת הזמנה לישיבת {info["committee"]} של המועצה המקומית פרדס חנה-כרכור, '
            f'{info["_date"].strftime("%d.%m.%Y")} בשעה {info["time"]}, {info["place"]}.'
            + (f' מזמין: {info["organizer"]}.' if info.get('organizer') else '') + '\n\n'
            'מקור: חומר הסברה של המועצה המקומית פרדס חנה-כרכור, שהועבר לפרסום בוויקי '
            'דרך התיקייה המשותפת.\n\n'
            'התמונה מובאת בשימוש הוגן לצורך המחשה בלוח ישיבות ועדה אנציקלופדי, ברזולוציה נמוכה.\n')
    r = client.session.post(API_URL, data={
        'action': 'upload', 'filename': fn, 'text': desc,
        'comment': 'כרזת ישיבת ועדה מהתיקייה המשותפת (העלאה אוטומטית)',
        'token': client._csrf_token(), 'ignorewarnings': '1', 'format': 'json',
    }, files={'file': (fn, io.BytesIO(data), 'image/jpeg')})
    res = r.json()
    if res.get('upload', {}).get('result') != 'Success':
        raise RuntimeError(f'upload failed: {res}')


def add_board_row(client: WikiClient, info: dict, fn: str) -> bool:
    """Insert a row into the upcoming table. False if a row for that file exists."""
    text = client.get_page(BOARD)['wikitext']
    if f'קובץ:{fn}' in text:
        return False
    b = uc.section_bounds(text, uc.UPCOMING)
    if not b:
        raise RuntimeError('סעיף הישיבות הקרובות לא נמצא')
    sec = text[b[0]:b[1]]
    parts = uc.split_table(sec)
    if not parts:
        raise RuntimeError('טבלת הישיבות הקרובות לא נמצאה')
    d = info['_date']
    note = info.get('organizer') or ''
    if info.get('topic'):
        note = (note + '; ' if note else '') + info['topic']
    note = (note + ' ' if note else '') + "''(נוסף אוטומטית מהתיקייה המשותפת)''"
    row = (f'| data-sort-value="{d.isoformat()}" | יום {WEEKDAYS[d.weekday()]}, {d.day}.{d.month} '
           f'|| {info["time"]} || {short_committee(info["committee"])} || {info["place"]} '
           f'|| [[קובץ:{fn}|90px]] || {note}')
    rows = parts[2] + [row]
    rows.sort(key=lambda r: uc.DATE_RE.search(r).group(1) if uc.DATE_RE.search(r) else '9999')
    new = text[:b[0]] + uc.rebuild(sec, parts, rows, uc.EMPTY_UPCOMING) + text[b[1]:]
    client.edit_page(BOARD, new,
                     summary=f'ישיבת {short_committee(info["committee"])} {d.day}.{d.month} '
                             'נוספה ללוח לפי כרזה מהתיקייה המשותפת')
    return True


# ----------------------------------------------------------------- talk ----

def post_note(client: WikiClient, lines):
    sig = subprocess.check_output([sys.executable, 'wiki_sig.py'], text=True,
                                  cwd=pathlib.Path(__file__).parent).strip()
    today = dt.date.today().strftime('%d.%m.%Y')
    body = (f'\n\n== קבצים חדשים בתיקייה המשותפת ({today}) ==\n'
            + '\n'.join('* ' + l for l in lines)
            + f'\n[[משתמש:אורי מוסנזון בוט|אורי מוסנזון בוט]] ([[שיחת משתמש:אורי מוסנזון בוט|שיחה]]) {sig}\n')
    text = client.get_page(TALK)['wikitext'].rstrip('\n')
    client.edit_page(TALK, text + body, summary='סריקת לילה של התיקייה המשותפת: קבצים חדשים')


# ----------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--seed', action='store_true')
    args = ap.parse_args()

    seen = json.loads(STATE.read_text()) if STATE.exists() else {}
    files = walk(ROOT)
    new = {fid: v for fid, v in files.items() if fid not in seen}
    print(f'קבצים בתיקייה: {len(files)} | חדשים: {len(new)}')

    if args.seed:
        for fid, (name, folder) in new.items():
            seen[fid] = {'name': name, 'folder': folder, 'seen': dt.date.today().isoformat(),
                         'action': 'seeded'}
        STATE.write_text(json.dumps(seen, ensure_ascii=False, indent=1))
        print('נרשמו כנראו, בלי לגעת בוויקי.')
        return 0

    if not new:
        return 0

    client = WikiClient()
    if not args.dry_run:
        client.login()

    notes = []
    for fid, (name, folder) in sorted(new.items(), key=lambda kv: kv[1][0]):
        where = f'{folder}/{name}' if folder else name
        action = 'reported'
        try:
            if folder == COMMITTEES_FOLDER and name.lower().endswith(IMAGE_EXT):
                data = to_jpeg(download(fid))
                info = read_poster(data)
                if info is None:
                    notes.append(f'"{where}": כרזה שלא הצלחתי לקרוא בביטחון (או שתאריכה כבר עבר). '
                                 'מחכה לטיפול ידני.')
                    print(f'  {where}: קריאה לא בטוחה')
                else:
                    fn = poster_filename(info['committee'], info['_date'])
                    print(f'  {where}: {short_committee(info["committee"])} {info["_date"]} '
                          f'{info["time"]} @ {info["place"]} -> {fn}')
                    if not args.dry_run:
                        upload_poster(client, fn, data, info)
                        added = add_board_row(client, info, fn)
                        notes.append(f'"{where}": נקראה ונוספה ל[[ועדות המועצה]] כישיבת '
                                     f'{short_committee(info["committee"])} ב-{info["_date"].day}.{info["_date"].month} '
                                     f'בשעה {info["time"]}' + ('' if added else ' (השורה כבר הייתה שם)')
                                     + '. אם קראתי משהו לא נכון, תקנו בטבלה.')
                    action = 'board'
            else:
                notes.append(f'"{where}": קובץ חדש שאיני מטפל בו לבד. מחכה לטיפול ידני.')
                print(f'  {where}: דווח בלבד')
        except Exception as e:                       # one bad file must not stop the rest
            notes.append(f'"{where}": נכשל ({type(e).__name__}). מחכה לטיפול ידני.')
            print(f'  {where}: שגיאה: {e}', file=sys.stderr)
            action = 'error'
        seen[fid] = {'name': name, 'folder': folder, 'seen': dt.date.today().isoformat(),
                     'action': action}

    if args.dry_run:
        print('\n'.join(notes))
        return 0
    if notes:
        post_note(client, notes)
    STATE.write_text(json.dumps(seen, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
