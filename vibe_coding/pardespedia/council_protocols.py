#!/usr/bin/env python3
"""Keep [[ישיבות מועצה]] — the archive of council plenum sittings — current,
from the municipality's own protocol archive.

That page is the moshava's record of what the council did: one section per
sitting, with the video embedded and the agenda spelled out, back to 2018. It
was hand-maintained and had drifted five sittings behind (9/9/2026). This
script closes that gap and adds the one thing the page never had: what each
sitting actually *decided*.

What it writes, and only this:
  * for a sitting the page already covers — one AUTO-marked block appended to
    that section, holding the resolutions and links to the documents. The
    hand-written heading, video and agenda above it are never touched.
  * for a sitting the page is missing — a whole new section in the right year,
    with the video embedded (never merely linked), the agenda, and the same
    resolutions block.

The resolutions are not written by us. Every protocol states its own in a
fixed formula — a line opening "מליאת המועצה מאשרת/מחליטה/דוחה..." — and only
those lines are lifted, verbatim. A protocol with no such line (it happens,
mostly in "שלא מן המניין" sittings) gets a section that says so and links to
the documents, rather than an invented summary.

Two sittings often share one day, and the page's own convention is to cover
them in a single section. Blocks are therefore keyed by date, not by meeting
number, and a shared day lists both sittings inside one block.

Usage:
    python3 council_protocols.py [--limit 24] [--dry-run] [--refresh]
"""

import argparse
import datetime as dt
import difflib
import hashlib
import html
import os
import re
import subprocess
import sys
import unicodedata
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import requests

from wiki_client import WikiClient

PAGE = 'ישיבות מועצה'
SOURCE = 'https://www.pardes-hanna-karkur.muni.il/council/protocols/'
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'protocol_cache')

UA = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) Chrome/120 Pardespedia-Bot',
      'Referer': SOURCE}

# the municipality's page is a GeoDirectory listing; each sitting is one card
# whose fields carry these stable class names
FIELDS = {
    'title': 'post_title',
    'date': 'gd__protocols__date',
    'protocol': 'gd__protocols__file',
    'agenda': 'gd_seder_yom',
    'transcript': 'gd_protocols_timlul',
    'video': 'youtube_link',
}

BIDI = dict.fromkeys(map(ord, '‎‏‪‫‬‭‮⁦⁧⁨⁩'), None)

RESOLUTION_RE = re.compile(
    r'^מליאת המועצה\s+(?:מאשרת|מחליטה|דוחה|מסמיכה|ממנה|מתקנת|מקבלת|אינה)\b')

# Protocols spell out the national ID number of everyone whose signature
# rights or employment the council votes on. The municipality may publish
# that; we will not mirror it onto an indexed, permanent wiki page. Names and
# roles stay — those are the public part of a public decision.
ID_RE = re.compile(r'ת\.?\s*ז\.?\s*[:\-]?\s*\d{6,9}')

# pdftotext puts the item number at the *end* of an RTL line ("...תמיכות. 2.")
AGENDA_MARK = re.compile(r'^(.*?)\s*\.(\d{1,2})$')

HE_MONTHS = {1: 'ינואר', 2: 'פברואר', 3: 'מרץ', 4: 'אפריל', 5: 'מאי', 6: 'יוני',
             7: 'יולי', 8: 'אוגוסט', 9: 'ספטמבר', 10: 'אוקטובר',
             11: 'נובמבר', 12: 'דצמבר'}
HE_WEEKDAYS = {0: 'שני', 1: 'שלישי', 2: 'רביעי', 3: 'חמישי', 4: 'שישי',
               5: 'שבת', 6: 'ראשון'}
MONTH_BY_NAME = {v: k for k, v in HE_MONTHS.items()}

BLOCK_START = ('<!-- AUTO:DECISIONS:%s:START — נבנה אוטומטית מהפרוטוקולים, '
               'אל תערוך ידנית -->')
BLOCK_END = '<!-- AUTO:DECISIONS:%s:END -->'


# --- text helpers ----------------------------------------------------------

def clean(s: str) -> str:
    return re.sub(r'[ \t]+', ' ',
                  unicodedata.normalize('NFKC', s.translate(BIDI))).strip()


def strip_tags(s: str) -> str:
    return clean(html.unescape(re.sub(r'<[^>]+>', ' ', s)))


def detangle(s: str) -> str:
    """Undo the spacing pdftotext loses when it unwraps a right-to-left line."""
    s = ID_RE.sub('(מספר זהות הושמט)', s)
    s = re.sub(r'(\d)([א-ת])', r'\1 \2', s)
    s = re.sub(r'([א-ת])(\d)', r'\1 \2', s)
    # only after a percent sign: gershayim live *inside* Hebrew words
    # (תב"ר, מנכ"ל), and splitting on them mangles ordinary abbreviations
    s = re.sub(r'(%)([א-ת])', r'\1 \2', s)
    # a sentence-final period migrates in front of a trailing year
    s = re.sub(r'\s\.(\d{4})\s*$', r' \1.', s)
    # RTL unwrapping leaves the comma and the full stop detached from the word
    # they belong to (" ,חופש" / "היום) ."), which reads as a typo on the page
    s = re.sub(r'\s+,\s*', ', ', s)
    s = re.sub(r'\s+\.(?!\d)', '. ', s)
    return re.sub(r'\s+', ' ', s).strip(' .') + '.'


def wiki_escape(s: str) -> str:
    return (s or '').replace('|', '‖').replace('[', '(').replace(']', ')')


def encode_url(url: str) -> str:
    """Percent-encode the Hebrew filenames the municipality uploads."""
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, quote(unquote(p.path), safe='/'),
                       p.query, p.fragment))


def youtube_id(url: str) -> str:
    """The bare video id, from a watch, live or short-form URL."""
    for pat in (r'[?&]v=([\w-]{11})', r'/live/([\w-]{11})',
                r'youtu\.be/([\w-]{11})', r'/embed/([\w-]{11})'):
        m = re.search(pat, url or '')
        if m:
            return m.group(1)
    return ''


# --- the municipality's archive -------------------------------------------

def parse_cards(page_html: str) -> list:
    marks = []
    for key, cls in FIELDS.items():
        for m in re.finditer(r'geodir-field-%s\b' % re.escape(cls), page_html):
            marks.append((m.start(), key))
    marks.sort()

    cards, cur = [], None
    for pos, key in marks:
        if key == 'title' or cur is None:
            if cur:
                cards.append(cur)
            cur = {}
        chunk = page_html[pos:pos + 1500]
        href = re.search(r'href="([^"]+)"', chunk)
        cur[key] = {'url': href.group(1) if href else '',
                    'text': strip_tags(chunk.split('</div>')[0])}
    if cur:
        cards.append(cur)
    return cards


def card_date(card: dict):
    raw = (card.get('date') or {}).get('text', '')
    m = re.search(r'(\d{2})/(\d{2})/(\d{4})', raw)
    return dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1))) if m else None


def fetch_pdf_text(url: str, key: str, refresh: bool = False) -> str:
    os.makedirs(CACHE, exist_ok=True)
    txt_path = os.path.join(CACHE, key + '.txt')
    if os.path.exists(txt_path) and not refresh:
        return open(txt_path, encoding='utf-8', errors='replace').read()
    pdf_path = os.path.join(CACHE, key + '.pdf')
    try:
        r = requests.get(encode_url(url), headers=UA, timeout=90)
        if not r.ok or b'%PDF' not in r.content[:1024]:
            print('  לא הורד פרוטוקול (%s): %s' % (r.status_code, key),
                  file=sys.stderr)
            return ''
        with open(pdf_path, 'wb') as fh:
            fh.write(r.content)
        subprocess.run(['pdftotext', '-layout', pdf_path, txt_path],
                       check=True, capture_output=True)
    except Exception as exc:
        print('  כשל בקריאת הפרוטוקול %s: %s' % (key, exc), file=sys.stderr)
        return ''
    return open(txt_path, encoding='utf-8', errors='replace').read()


def resolutions(text: str) -> list:
    out = []
    lines = [clean(l) for l in text.split('\n')]
    for i, line in enumerate(lines):
        if not RESOLUTION_RE.match(line):
            continue
        parts = [line]
        for nxt in lines[i + 1:i + 4]:
            if not nxt or re.match(r'^(הצבעה|בעד|נגד|נמנע|מר |גב\'|עו"ד |ד"ר )', nxt):
                break
            parts.append(nxt)
        item = detangle(' '.join(parts).strip(' .'))
        if item not in out:
            out.append(item)
    return out


def agenda(text: str) -> list:
    lines = [clean(l) for l in text.split('\n')]
    try:
        start = next(i for i, l in enumerate(lines)
                     if l.startswith('על סדר היום'))
    except StopIteration:
        return []
    items, cur = {}, None
    for l in lines[start + 1:start + 40]:
        if not l or 'מועצה מקומית' in l or 'ישיבת מועצה' in l:
            continue
        if l.isdigit():          # a page number ends the agenda block
            if items:
                break
            continue
        m = AGENDA_MARK.match(l)
        if m:
            cur = int(m.group(2))
            items[cur] = m.group(1).strip()
        elif cur:
            items[cur] += ' ' + l
    return [detangle(items[k]) for k in sorted(items) if items[k].strip()]


def protocol_date(text: str, listed: dt.date):
    """Correct the listing's date from the protocol's own front matter.

    Neither source is trustworthy alone. The listing filed sitting 33 under
    06/05/2026 while its protocol is headed 07.05.2026 — and the council sits
    on Thursdays, so the protocol is right. But protocol 32 carries a typo of
    its own, a running header reading 05.03.2025 for a sitting held in 2026.

    So: take the dates the front matter states, and accept one only if it
    lands within a week of the listing. That is enough to fix a day's slip
    and not nearly enough to move a sitting into the wrong year.
    """
    head = ''.join(clean(l) + '\n' for l in text.split('\n')[:60])
    for dd, mm, yy in re.findall(r'\b(\d{2})\.(\d{2})\.(\d{4})\b', head):
        try:
            found = dt.date(int(yy), int(mm), int(dd))
        except ValueError:
            continue
        if abs((found - listed).days) <= 7:
            return found
    return None


def meeting_number(card: dict, text: str):
    for source in (card.get('title', {}).get('text', ''), text[:4000]):
        m = re.search(r"מס['׳]?\s*(\d{1,3})", clean(source))
        if m:
            return m.group(1)
    return None


# --- the wiki page ---------------------------------------------------------

def parse_sections(text: str) -> list:
    """Every "=== ... ===" sitting section: (start, end, heading, date, numbers)."""
    out = []
    heads = list(re.finditer(r'^===\s*(.+?)\s*===\s*$', text, re.M))
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        nxt_top = re.search(r'^==[^=]', text[m.end():], re.M)
        if nxt_top and m.end() + nxt_top.start() < end:
            end = m.end() + nxt_top.start()
        title = m.group(1)
        d = None
        dm = re.search(r'(\d{1,2})\s+ב([א-ת]+)\s+(\d{4})', title)
        if dm and dm.group(2) in MONTH_BY_NAME:
            try:
                d = dt.date(int(dm.group(3)), MONTH_BY_NAME[dm.group(2)],
                            int(dm.group(1)))
            except ValueError:
                d = None
        nums = set(re.findall(r"מס['׳]\s*(\d{1,3})", title))
        out.append({'start': m.start(), 'end': end, 'title': title,
                    'date': d, 'numbers': nums})
    return out


def year_bounds(text: str, year: int):
    m = re.search(r'^==\s*%d\s*==\s*$' % year, text, re.M)
    if not m:
        return None
    nxt = re.search(r'^==[^=]', text[m.end():], re.M)
    return m.end(), (m.end() + nxt.start() if nxt else len(text))


def build_block(group: dict) -> str:
    """The AUTO block for one day's sitting(s), or "" when none can be read.

    Protocols up to late 2025 are photocopier scans with no text layer
    (Producer: "Develop ineo+ 257i"), and pdftotext returns nothing for them.
    A sitting we could not read is not a sitting that decided nothing, so it
    gets no block at all rather than a block asserting the difference away.
    """
    key = group['date'].isoformat()
    readable = [s for s in group['sittings'] if s['readable']]
    if not readable:
        return ''
    lines = [BLOCK_START % key]
    for s in readable:
        label = ("ישיבה מס' %s" % s['number']) if s['number'] else 'הישיבה'
        lines.append("'''מה סוכם''' (%s):" % label)
        if s['resolutions']:
            lines += ['* ' + wiki_escape(x) for x in s['resolutions']]
        else:
            lines.append(': ''\'\'בפרוטוקול אין החלטה מנוסחת. '
                         'ראו את המסמכים המלאים.\'\'')
        docs = []
        for name, field in (('פרוטוקול', 'protocol'), ('תמלול', 'transcript'),
                            ('סדר יום', 'agenda')):
            url = (s['card'].get(field) or {}).get('url', '')
            if url:
                docs.append('[%s %s]' % (encode_url(url), name))
        if docs:
            lines.append("''מסמכים:'' " + ' · '.join(docs))
    lines.append(BLOCK_END % key)
    return '\n'.join(lines)


def build_section(group: dict) -> str:
    """A whole new sitting section, video embedded, for a day the page lacks."""
    d = group['date']
    nums = [s['number'] for s in group['sittings'] if s['number']]
    if len(nums) > 1:
        what = 'ישיבות מס\' %s' % ' ומס\' '.join(nums)
    elif nums:
        what = 'ישיבה מס\' %s' % nums[0]
    else:
        what = 'ישיבת מועצה'
    heading = '=== %d ב%s %d (יום %s) — %s ===' % (
        d.day, HE_MONTHS[d.month], d.year, HE_WEEKDAYS[d.weekday()], what)

    lines = [heading]
    vid = ''
    for s in group['sittings']:
        vid = youtube_id((s['card'].get('video') or {}).get('url', '')) or vid
    if vid:
        lines.append('<youtube width="320" height="180">%s</youtube>' % vid)

    combined = []
    for s in group['sittings']:
        for item in s['agenda']:
            if item not in combined:
                combined.append(item)
    if combined:
        lines.append("'''על סדר היום:'''")
        lines += ['# ' + wiki_escape(x) for x in combined]

    lines.append(build_block(group))
    return '\n'.join(lines)


def upsert(text: str, group: dict, sections: list) -> tuple:
    """Put the group's block in the page. Returns (text, what_happened)."""
    key = group['date'].isoformat()
    start_marker, end_marker = BLOCK_START % key, BLOCK_END % key
    block = build_block(group)

    if start_marker in text and end_marker in text:
        i = text.index(start_marker)
        j = text.index(end_marker) + len(end_marker)
        if not block:                       # nothing readable: take it back out
            return text[:i].rstrip('\n') + '\n' + text[j:].lstrip('\n'), 'removed'
        if text[i:j] == block:
            return text, 'unchanged'
        return text[:i] + block + text[j:], 'refreshed'
    if not block:
        return text, 'unreadable'

    nums = {s['number'] for s in group['sittings'] if s['number']}
    match = None
    for sec in sections:
        if nums and sec['numbers'] & nums:
            match = sec
            break
        if sec['date'] == group['date']:
            match = sec
            break

    if match:
        body = text[match['start']:match['end']].rstrip()
        return (text[:match['start']] + body + '\n' + block + '\n\n'
                + text[match['end']:]), 'appended'

    bounds = year_bounds(text, group['date'].year)
    if not bounds:
        return text, 'no-year'
    lo, hi = bounds
    # the page runs newest-first inside a year; slot in before the first
    # section that is older than this one
    insert_at = hi
    for sec in sections:
        if lo <= sec['start'] < hi and sec['date'] and sec['date'] < group['date']:
            insert_at = sec['start']
            break
    return text[:insert_at] + build_section(group) + '\n\n' + text[insert_at:], 'created'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=24,
                    help='how many recent sittings to consider (default: 24)')
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    r = requests.get(SOURCE, headers=UA, timeout=60)
    r.raise_for_status()
    dated = [(card_date(c), c) for c in parse_cards(r.text)]
    dated = sorted([(d, c) for d, c in dated if d], key=lambda x: x[0], reverse=True)
    print('נמצאו %d ישיבות מליאה בארכיון המועצה' % len(dated), file=sys.stderr)

    today = dt.date.today()
    groups = {}
    for d, card in dated:
        if d > today or len(groups) >= args.limit and d not in groups:
            continue
        url = (card.get('protocol') or {}).get('url', '')
        key = '%s-%s' % (d.isoformat(), hashlib.md5(url.encode()).hexdigest()[:8])
        body = fetch_pdf_text(url, key, args.refresh) if url else ''
        stated = protocol_date(body, d)
        if stated and stated != d:
            print('  תאריך הרשימה %s, הפרוטוקול אומר %s — לפי הפרוטוקול'
                  % (d.isoformat(), stated.isoformat()), file=sys.stderr)
            d = stated
        groups.setdefault(d, {'date': d, 'sittings': []})
        groups[d]['sittings'].append({
            'card': card, 'resolutions': resolutions(body),
            'agenda': agenda(body), 'number': meeting_number(card, body),
            # a real protocol runs to thousands of characters; anything less
            # means the PDF is a scan and pdftotext gave us nothing to work on
            'readable': len(body.strip()) > 500})

    client = WikiClient()
    text = original = client.get_page(PAGE)['wikitext']
    tally = {}
    for d in sorted(groups, reverse=True):
        group = groups[d]
        group['sittings'].sort(key=lambda s: int(s['number'] or 0))
        text, what = upsert(text, group, parse_sections(text))
        tally[what] = tally.get(what, 0) + 1
        print('  %s — %s — %d החלטות — %s'
              % (d.isoformat(),
                 '/'.join(s['number'] or '?' for s in group['sittings']),
                 sum(len(s['resolutions']) for s in group['sittings']), what),
              file=sys.stderr)

    print('סיכום: %s' % ', '.join('%s=%d' % kv for kv in sorted(tally.items())),
          file=sys.stderr)
    if text == original:
        print('אין שינוי בדף.', file=sys.stderr)
        return 0
    if args.dry_run:
        sys.stdout.writelines(difflib.unified_diff(
            original.splitlines(True), text.splitlines(True),
            fromfile='לפני', tofile='אחרי', n=2))
        return 0

    client.login()
    client.edit_page(PAGE, text,
                     summary='עדכון אוטומטי מפרוטוקולי המועצה: החלטות וישיבות חסרות')
    return 0


if __name__ == '__main__':
    sys.exit(main())
