#!/usr/bin/env python3
"""Bring the neighbourhood articles in line with hood_maps/hoods.json.

Point edits only, each on a fresh copy of the page:
  * the "שכונות גובלות" line of {{מידע שכונה}}
  * the automatic street list ("== רחובות בשכונה ==") where the article has one
    and the borders changed

    python3 hood_articles.py            # dry run: show what would change
    python3 hood_articles.py --write
"""
import difflib, re, sys
from wiki_client import WikiClient, API_URL
from hood_facts import zones, neighbours, streets

LABEL_LINK = {}          # zone name -> wikilink
RESTREET = {'מזרח כרכור', 'עומרים', 'רמז'}     # borders moved in 11/2026
SUMMARY = 'עדכון לפי חלוקת השכונות המעודכנת (נובמבר 2026)'


def link(z):
    a, n = z['article'], z['name']
    return '[[%s]]' % a if a == n else '[[%s|%s]]' % (a, n)


def main():
    write = '--write' in sys.argv
    c = WikiClient()
    if write: c.login()
    zs = [z for z in zones() if z['article']]
    by = {z['name']: z for z in zs}
    nb, st = neighbours(), streets()
    names = sorted({s for n in RESTREET for s in st[n]})
    r = c.session.get(API_URL, params={'action': 'query', 'format': 'json',
                                       'titles': '|'.join(n + ' (רחוב)' for n in names)[:6000]}).json()
    have = {p['title'] for p in r['query']['pages'].values() if 'missing' not in p}
    for z in zs:
        text = old = c.get_page(z['article'])['wikitext']
        val = ', '.join('%s (%s)' % (link(by[n]), d) for n, _, d, _ in nb[z['name']] if n in by)
        text = re.sub(r'(\|\s*שכונות גובלות\s*=)[^\n]*', lambda m: m.group(1) + ' ' + val, text, count=1)
        if z['name'] in RESTREET:
            m = re.search(r'(== רחובות בשכונה ==\n)([^\n]+)(\n\n<small>הרשימה הופקה אוטומטית)', text)
            if m:
                kept = dict(re.findall(r'(\[\[[^\]|]+\|([^\]]+)\]\])', m.group(2)))
                kept = {v: k for k, v in kept.items()}
                items = [kept.get(s) or ('[[%s (רחוב)|%s]]' % (s, s) if s + ' (רחוב)' in have else s)
                         for s in st[z['name']]]
                text = text[:m.start(2)] + ', '.join(items) + '.' + text[m.end(2):]
        if text == old:
            print('==', z['article'], ': אין שינוי'); continue
        print('==', z['article'])
        for l in difflib.unified_diff(old.split('\n'), text.split('\n'), lineterm='', n=0):
            if l[:1] in '+-' and l[:3] not in ('+++', '---'): print('  ', l[:400])
        if write:
            print('   ->', c.edit_page(z['article'], text, SUMMARY).get('edit', {}).get('result'))


if __name__ == '__main__':
    main()
