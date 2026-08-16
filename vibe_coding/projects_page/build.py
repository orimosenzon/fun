#!/usr/bin/env python3
"""Regenerate /home/ori/fun/vibe_coding/index.html from projects.json.

projects.json is the single source of truth. To change the landing page, edit
the JSON (or the site chrome in template.html) and run:

    python3 projects_page/build.py

Header stats (project count, live demos, AI-powered, categories) are computed,
never hand-maintained. `--check` audits the repo for projects missing from the
JSON instead of building.
"""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "projects.json"
TEMPLATE = HERE / "template.html"
OUTPUT = ROOT / "index.html"

# Matched as whole words against a project's tags. Set "ai": true/false on a
# project to override the guess.
AI_TAGS = {
    "claude", "anthropic", "gpt", "openai", "gemini", "imagen", "llm", "ai",
    "vision", "ocr", "whisper", "vosk", "porcupine", "tts", "kokoro", "gtts",
    "langchain", "langgraph", "mcp", "ollama", "gemma", "vllm", "glove",
    "word2vec", "embeddings", "sonnet", "haiku", "opus", "pytorch",
    "transformers", "huggingface", "diffusion", "rag", "stt",
}

IND = "        "  # card indentation inside a section grid


def is_ai(project: dict) -> bool:
    if project.get("ai") is not None:
        return bool(project["ai"])
    words = set(re.findall(r"[a-z0-9]+", " ".join(project.get("tags", [])).lower()))
    return bool(words & AI_TAGS)


def plural(n: int) -> str:
    return "פרויקט אחד" if n == 1 else f"{n} פרויקטים"


def render_links(links: list) -> str:
    out = []
    for l in links:
        target = ' target="_blank"' if l.get("external") else ""
        out.append(
            f'{IND}            <a href="{l["href"]}" class="btn-{l["style"]}"{target}>{l["label"]}</a>'
        )
    return "\n".join(out)


def badge_type(b) -> str:
    return b if isinstance(b, str) else b.get("type", "")


def render_badge(b) -> str:
    """A badge is {"type": "live"|"wip", "label": "..."}; bare strings are
    accepted as shorthand for the default label."""
    if isinstance(b, str):
        b = {"type": b, "label": "LIVE" if b == "live" else "⚙ בפיתוח"}
    dot = '<span class="live-dot"></span>' if b["type"] == "live" else ""
    return f'<span class="{b["type"]}-badge">{dot}{b["label"]}</span>'


def render_link_row(links: list, pad: str) -> str:
    """A project with nothing public to link to (unpublished source, no demo)
    renders without a button row rather than with an empty one."""
    if not links:
        return ""
    return f'\n{IND}{pad}<div class="card-links">\n{render_links(links)}\n{IND}{pad}</div>'


def render_card(p: dict) -> str:
    small = bool(p.get("small"))
    classes = ["project-card"]
    if small:
        classes.append("card-small")
    classes.append(p.get("theme", "t-indigo"))

    # Small cards have no illustration column, so they skip the wrapper <div>
    # that keeps the grid gap from separating header/desc/links.
    pad = "" if small else "    "

    badges = "".join(
        f'\n{IND}{pad}            {render_badge(b)}' for b in p.get("badges", [])
    )
    title_html = (
        f'\n{IND}{pad}            {p["title"]}{badges}\n{IND}{pad}        '
        if badges
        else p["title"]
    )

    tags = "".join(
        f'\n{IND}{pad}                <span class="tag">{t}</span>' for t in p.get("tags", [])
    )
    tags_html = (
        f'\n{IND}{pad}            <div class="card-tags">{tags}\n{IND}{pad}            </div>'
        if tags
        else ""
    )

    illus = ""
    if p.get("illus"):
        # dedent first so re-indenting is idempotent across rebuilds
        from textwrap import dedent, indent

        svg = dedent(p["illus"].replace("\t", "    ")).strip("\n")
        illus = (
            f'\n{IND}    <div class="card-illus">\n'
            f'{indent(svg, IND + "        ")}\n'
            f'{IND}    </div>'
        )

    body = f"""{IND}{pad}<div class="card-header">
{IND}{pad}    <div class="card-icon">{p.get('icon', '')}</div>
{IND}{pad}    <div>
{IND}{pad}        <div class="card-title">{title_html}</div>{tags_html}
{IND}{pad}    </div>
{IND}{pad}</div>
{IND}{pad}<div class="card-desc">{p['desc']}</div>{render_link_row(p['links'], pad)}"""

    if not small:
        body = f"{IND}    <div>\n{body}\n{IND}    </div>{illus}"

    return f"""{IND}<!-- {p['title']} -->
{IND}<div class="{' '.join(classes)}">
{body}
{IND}</div>"""


def render_section(s: dict, number: int) -> str:
    cards = "\n\n".join(render_card(p) for p in s["projects"])
    rule = "═" * 42
    return f"""    <!-- {rule}
         {number}. {s['title']}
    {rule} -->
    <div class="section-header">
        <span class="section-emoji">{s['emoji']}</span>
        <span class="section-title">{s['title']}</span>
        <span class="section-count">{plural(len(s['projects']))}</span>
    </div>
    <div class="{s.get('grid', 'projects')}">

{cards}

    </div><!-- /{s['title']} -->"""


def render_header(data: dict) -> str:
    projects = [p for s in data["sections"] for p in s["projects"]]
    live = sum(1 for p in projects if any(badge_type(b) == "live" for b in p.get("badges", [])))
    ai = sum(1 for p in projects if is_ai(p))
    meta = data.get("site", {})
    stats = [
        (len(projects), "פרויקטים"),
        (live, "Live Demos"),
        (ai, "AI-Powered"),
        (len(data["sections"]), "קטגוריות"),
    ]
    stat_html = "\n".join(
        f'            <div class="stat"><div class="stat-num">{n}</div>'
        f'<div class="stat-label">{label}</div></div>'
        for n, label in stats
    )
    return f"""    <header>
        <h1>{meta.get('title', '✦ Vibe Coding')}</h1>
        <p class="header-sub">{meta.get('subtitle', '')}</p>
        <p class="header-meta">{meta.get('meta', '')}</p>
        <div class="header-stats">
{stat_html}
        </div>
    </header>
"""


def assert_not_excluded(data: dict) -> None:
    """EXCLUDED projects must never reach the published page. Fail the build
    loudly rather than quietly publishing one that slipped back in."""
    from urllib.parse import unquote

    for s in data["sections"]:
        for p in s["projects"]:
            targets = {p.get("dir", "")}
            for l in p["links"]:
                path = unquote(l["href"]).replace(
                    "https://github.com/orimosenzon/fun/tree/master/vibe_coding/", ""
                )
                if not re.match(r"https?:", path):
                    targets.add(path.split("/")[0])
            hit = targets & EXCLUDED
            if hit:
                raise SystemExit(
                    f'הפרויקט "{p["title"]}" ({", ".join(sorted(hit))}) מוחרג מהדף '
                    f"לבקשת אורי (16.8.2026). הסר אותו מ-projects.json."
                )


def build() -> str:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    assert_not_excluded(data)
    sections = "\n\n".join(
        render_section(s, i + 1) for i, s in enumerate(data["sections"])
    )
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("{{HEADER}}", render_header(data).rstrip("\n"))
    html = html.replace("{{SECTIONS}}", sections)
    OUTPUT.write_text(html, encoding="utf-8")
    n = sum(len(s["projects"]) for s in data["sections"])
    return f"built {OUTPUT} — {len(data['sections'])} sections, {n} projects"


# ── audit ────────────────────────────────────────────────────────────────────
# Ori asked on 16.8.2026 that these never appear on the public landing page.
# Do not add cards for them, and do not report them as missing.
EXCLUDED = {
    "chidonait",          # חידתי — מצגת משקיעים
    "alon",               # הגיטרות של יצחק
    "כרמית",              # חידות לכרמית
    "vocab_optionA.html",  # תרגול אוצר מילים — תיכון שמיר
    "ירדן",               # screen capture — צילום מסך מהמגש
}

# Deliberately off the page: infrastructure, private work, and superseded
# iterations of a project that already has a card.
SKIP = EXCLUDED | {
    "projects_page", "tmp", "memory", "billing-check", "job-search", "factotum",
    "ori_android",
    # earlier iterations of "Smart Business Search" (ori/index.html)
    "map_search.html", "smart_map.html", "ms.html", "ms_es.html", "ms_mob.html",
    "wip.html", "tmp.html",
}


def git_tracked() -> set:
    """Paths git knows about, relative to vibe_coding/. Anything outside this
    set is not published by GitHub Pages, so a link to it will 404."""
    import subprocess

    # -z keeps Hebrew paths literal; without it git escapes them into "..."
    out = subprocess.run(
        ["git", "ls-files", "-z", "vibe_coding"],
        cwd=ROOT.parent, capture_output=True, text=True,
    )
    if out.returncode != 0:
        return set()
    return {
        p[len("vibe_coding/"):]
        for p in out.stdout.split("\0")
        if p.startswith("vibe_coding/")
    }


def check() -> int:
    """Audit projects.json against the repo: projects with no card, and links
    that will 404 on the live site."""
    data = json.loads(DATA.read_text(encoding="utf-8"))
    projects = [p for s in data["sections"] for p in s["projects"]]
    hrefs = [l["href"] for p in projects for l in p["links"]]
    tracked = git_tracked()
    tracked_tops = {h.split("/")[0] for h in tracked}

    # 1. things in the directory that no card claims.
    # A card claims a directory via its "dir" field, or implicitly via any link
    # whose path starts with that directory.
    claimed = {p["dir"] for p in projects if p.get("dir")}
    for href in hrefs:
        path = re.sub(r"^https://github\.com/orimosenzon/fun/tree/master/vibe_coding/", "", href)
        if not re.match(r"https?:", path):
            claimed.add(path.split("/")[0])

    missing = []
    for entry in sorted(ROOT.iterdir()):
        if entry.name.startswith(".") or entry.name in SKIP or entry.name in claimed:
            continue
        if entry.is_dir():
            missing.append((entry.name + "/", entry.name in tracked_tops))
        elif entry.suffix == ".html" and entry.name != "index.html":
            missing.append((entry.name, entry.name in tracked))

    # 2. links that point nowhere useful
    broken, unpublished = [], []
    for p in projects:
        if p.get("private"):
            continue
        for l in p["links"]:
            href = l["href"]
            fun_src = re.match(
                r"https://github\.com/orimosenzon/fun/tree/master/vibe_coding/(.+)", href
            )
            if fun_src:
                # a source link into this repo only resolves if git tracks it
                from urllib.parse import unquote

                path = unquote(fun_src.group(1)).rstrip("/")
                if path not in tracked_tops and not any(
                    t.startswith(path + "/") for t in tracked
                ):
                    unpublished.append((p["title"], href))
                continue
            if re.match(r"https?:|mailto:", href):
                continue
            path = href.rstrip("/")
            if not (ROOT / path).exists():
                broken.append((p["title"], href))
            elif path not in tracked and not any(t.startswith(path + "/") for t in tracked):
                unpublished.append((p["title"], href))

    pending = [p["title"] for p in projects if p.get("pending_publish")]

    clean = not (missing or broken or unpublished)
    if pending:
        print("כרטיסים בלי קישור ציבורי (הקוד עוד לא פורסם):")
        for t in pending:
            print(f"  • {t}")
        print()
    if clean:
        print("✓ הדף מסונכרן: כל פרויקט מיוצג וכל הקישורים המקומיים יעבדו בשידור חי")
        return 0

    listed = [m for m, t in missing if t]
    if listed:
        print("פרויקטים בלי כרטיס בדף (מוכנים לפרסום):")
        for m in listed:
            print(f"  • {m}")
    unlisted = [m for m, t in missing if not t]
    if unlisted:
        print("\nפרויקטים בלי כרטיס שגם לא נמצאים ב-git (צריך קומיט לפני שיהיו חיים):")
        for m in unlisted:
            print(f"  • {m}")
    if broken:
        print("\nקישורים לקבצים שלא קיימים:")
        for t, h in broken:
            print(f"  • {t} → {h}")
    if unpublished:
        print("\nקישורים לקבצים שקיימים מקומית אבל לא ב-git (יחזירו 404 בשידור חי):")
        for t, h in unpublished:
            print(f"  • {t} → {h}")
    print("\nעדכן את projects.json והרץ build.py, או הוסף ל-SKIP אם זה בכוונה מחוץ לדף.")
    return 1


LIVE_BASE = "https://orimosenzon.github.io/fun/vibe_coding/"


def check_links() -> int:
    """Fetch every link on the published page and report what does not answer.
    Needs network; slower than --check, so it is a separate flag."""
    import subprocess
    from urllib.parse import quote

    data = json.loads(DATA.read_text(encoding="utf-8"))
    seen, bad = {}, []
    for s in data["sections"]:
        for p in s["projects"]:
            for l in p["links"]:
                seen.setdefault(l["href"], p["title"])

    for href, title in seen.items():
        url = href if href.startswith("http") else LIVE_BASE + quote(href)
        code = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "20", "-L", url],
            capture_output=True, text=True,
        ).stdout.strip()
        # 403 is what some hosts answer a bare curl; not a broken link
        if code not in ("200", "403"):
            bad.append((code, title, href))

    print(f"נבדקו {len(seen)} קישורים")
    if not bad:
        print("✓ כולם עונים")
        return 0
    for code, title, href in bad:
        note = "  (401 = Space פרטי)" if code == "401" else ""
        print(f"  {code}  {title} → {href}{note}")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="audit the data against the repo")
    ap.add_argument("--links", action="store_true", help="fetch every link on the live page")
    args = ap.parse_args()
    if args.links:
        raise SystemExit(check_links())
    raise SystemExit(check() if args.check else print(build()) or 0)
