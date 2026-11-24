"""סורק את דפי התקציב באתר עיריית ירושלים ובונה src/manifest.json.

האתר חוסם בקשות שלא מדפדפן (403), ולכן הכול רץ בכרום אמיתי דרך Playwright.
לכל שנה: רשימת מדורים (כותרת האקורדיון), ובכל מדור שורות טבלה: טקסטים של התאים
וקישור לכל עמודה לפי כותרת העמודה (תקציב / דברי הסבר / תקציב ודברי הסבר).
"""
import json, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
YEARS = range(2017, 2027)

JS = r"""() => {
  const out = [];
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  document.querySelectorAll('table').forEach(tbl => {
    // section title: nearest preceding H2/H3
    let el = tbl, title = '';
    while (el && !title) {
      let s = el.previousElementSibling;
      while (s && !title) {
        const h = s.matches('h2,h3') ? s : s.querySelector && s.querySelector('h2,h3');
        if (h && clean(h.innerText)) title = clean(h.innerText);
        s = s.previousElementSibling;
      }
      el = el.parentElement;
    }
    const rows = [...tbl.querySelectorAll('tr')];
    if (!rows.length) return;
    const head = [...rows[0].children].map(c => clean(c.innerText));
    const body = rows.slice(1).map(tr => [...tr.children].map(c => ({
      text: clean(c.innerText),
      links: [...c.querySelectorAll('a[href]')].map(a => a.href).filter(h => h.includes('/media/')),
    })));
    out.push({ title, head, rows: body });
  });
  // links outside tables (some years use lists)
  const loose = [...document.querySelectorAll('main a[href*="/media/"]')]
    .filter(a => !a.closest('table'))
    .map(a => ({ text: clean(a.innerText || a.title), href: a.href }));
  return { tables: out, loose };
}"""

def main():
    years = [int(a) for a in sys.argv[1:]] or list(YEARS)
    path = ROOT / "src" / "manifest.json"
    man = json.loads(path.read_text()) if path.exists() else {}
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=False, args=["--window-position=2000,2000"])
        pg = b.new_page()
        for y in years:
            url = f"https://www.jerusalem.muni.il/he/city/budget/budget{y}/"
            pg.goto(url, wait_until="networkidle", timeout=90000)
            d = pg.evaluate(JS)
            d["url"] = url
            man[str(y)] = d
            nl = sum(len(c["links"]) for t in d["tables"] for r in t["rows"] for c in r)
            print(y, "tables", len(d["tables"]), "links in tables", nl, "loose", len(d["loose"]))
        b.close()
    path.write_text(json.dumps(man, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
