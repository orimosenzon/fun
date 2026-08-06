#!/usr/bin/env python3
"""גרפי סטטיסטיקה לפרדספדיה — נתונים חיים מה-API, לא הערכות.

מייצר שלושה קבצי PNG לערך "פרדספדיה":
  1. גידול מספר הערכים לאורך זמן (מצטבר)
  2. עריכות לפי חודש, בפילוח אדם / בוט
  3. התורמים המובילים

הרצה:  python3 stats_charts.py [outdir]
"""
import collections
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import MultipleLocator
from bidi.algorithm import get_display

from wiki_client import WikiClient, API_URL

HEB_FONT = "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
BOT_USER = "אורי מוסנזון בוט"

# פלטה מאומתת (dataviz skill, light mode, surface #fcfcfb)
SURFACE = "#fcfcfb"
SERIES_1 = "#2a78d6"   # כחול — תושבים
SERIES_2 = "#eb6834"   # כתום — בוט
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3de"


def he(s):
    """טקסט עברי מוכן לרינדור ב-matplotlib (bidi)."""
    return get_display(str(s))


def setup_font():
    fm.fontManager.addfont(HEB_FONT)
    name = fm.FontProperties(fname=HEB_FONT).get_name()
    plt.rcParams.update({
        "font.family": name,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.edgecolor": GRID,
        "axes.titlesize": 15,
        "font.size": 10,
    })


# ---------------------------------------------------------------- collect data

def fetch_creations(client):
    """יצירות דפים במרחב הראשי, לפי חודש."""
    per_month = collections.Counter()
    cont = {}
    while True:
        params = {"action": "query", "list": "logevents", "letype": "create",
                  "lelimit": "500", "ledir": "newer",
                  "leprop": "timestamp|title|user", "format": "json"}
        params.update(cont)
        data = client.session.get(API_URL, params=params).json()
        for ev in data.get("query", {}).get("logevents", []):
            if ev.get("ns") == 0:
                per_month[ev["timestamp"][:7]] += 1
        if "continue" in data:
            cont = data["continue"]
        else:
            return per_month


def fetch_revisions(client):
    """כל העריכות: לפי חודש (בפילוח בוט/אדם) ולפי משתמש."""
    per_month_human = collections.Counter()
    per_month_bot = collections.Counter()
    per_user = collections.Counter()
    cont = {}
    while True:
        params = {"action": "query", "list": "allrevisions", "arvlimit": "500",
                  "arvdir": "newer", "arvprop": "timestamp|user", "format": "json"}
        params.update(cont)
        data = client.session.get(API_URL, params=params).json()
        for page in data.get("query", {}).get("allrevisions", []):
            for rev in page["revisions"]:
                month = rev["timestamp"][:7]
                user = rev.get("user", "?")
                per_user[user] += 1
                if user == BOT_USER:
                    per_month_bot[month] += 1
                else:
                    per_month_human[month] += 1
        if "continue" in data:
            cont = data["continue"]
        else:
            return per_month_human, per_month_bot, per_user


def month_range(first, last):
    """כל החודשים בין שני מפתחות 'YYYY-MM' ועד בכלל."""
    y, m = (int(x) for x in first.split("-"))
    ly, lm = (int(x) for x in last.split("-"))
    out = []
    while (y, m) <= (ly, lm):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


# ------------------------------------------------------------------- chart #1

def chart_growth(per_month, path):
    months = month_range(min(per_month), max(per_month))
    total, cumulative = 0, []
    for month in months:
        total += per_month.get(month, 0)
        cumulative.append(total)
    x = range(len(months))

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    ax.fill_between(x, cumulative, color=SERIES_1, alpha=0.13, linewidth=0)
    ax.plot(x, cumulative, color=SERIES_1, linewidth=2, solid_capstyle="round")

    # תוויות ישירות בשלוש נקודות המפנה בלבד — לא על כל נקודה
    highlights = [("2019-03", "השקה: 2019"), ("2024-12", "התעוררות: 2024"),
                  (months[-1], f"{cumulative[-1]} דפים")]
    for month, label in highlights:
        if month not in months:
            continue
        i = months.index(month)
        ax.plot([i], [cumulative[i]], "o", color=SERIES_1, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        ax.annotate(he(label), (i, cumulative[i]),
                    textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=10, color=INK, fontweight="bold")

    year_ticks = [i for i, m in enumerate(months) if m.endswith("-01")]
    ax.set_xticks(year_ticks)
    ax.set_xticklabels([months[i][:4] for i in year_ticks], fontsize=9)
    ax.set_xlim(-1, len(months) + 3)
    ax.set_ylim(0, max(cumulative) * 1.22)
    ax.set_title(he("גידול פרדספדיה: דפים שנוצרו במרחב הראשי, 2019–2026"),
                 fontweight="bold", pad=26)
    # יחידת המדידה בשורה אופקית — תווית ציר מסובבת בעברית נקראת רע
    ax.text(0, 1.04, he("דפים (מצטבר, כולל הפניות)"), transform=ax.transAxes,
            fontsize=9.5, color=INK_2, ha="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return cumulative[-1]


# ------------------------------------------------------------------- chart #2

def chart_edits(human, bot, path):
    months = month_range("2019-01", max(list(human) + list(bot)))
    h = [human.get(m, 0) for m in months]
    b = [bot.get(m, 0) for m in months]
    x = range(len(months))

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    # רווח 2px בין המקטעים המוערמים, בצבע המשטח
    ax.bar(x, h, color=SERIES_1, width=0.8, label=he("תושבים"),
           edgecolor=SURFACE, linewidth=0.6)
    ax.bar(x, b, bottom=h, color=SERIES_2, width=0.8, label=he("הבוט"),
           edgecolor=SURFACE, linewidth=1.4)

    year_ticks = [i for i, m in enumerate(months) if m.endswith("-01")]
    ax.set_xticks(year_ticks)
    ax.set_xticklabels([months[i][:4] for i in year_ticks], fontsize=9)
    ax.set_xlim(-1, len(months))
    ax.set_title(he("קצב העריכה בפרדספדיה — תושבים לעומת הבוט"),
                 fontweight="bold", pad=34)
    ax.text(0, 1.04, he("עריכות בחודש"), transform=ax.transAxes,
            fontsize=9.5, color=INK_2, ha="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MultipleLocator(500))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    legend = ax.legend(frameon=False, fontsize=10, ncol=2,
                       loc="lower right", bbox_to_anchor=(1.0, 1.005),
                       handlelength=1.1, handleheight=1.1, columnspacing=1.4)
    for text in legend.get_texts():
        text.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- chart #3

def chart_contributors(per_user, path, top=10):
    pairs = [(u, n) for u, n in per_user.most_common(top * 2) if u != "?"][:top]
    pairs.reverse()          # הגדול למעלה
    names = [p[0] for p in pairs]
    counts = [p[1] for p in pairs]
    colors = [SERIES_2 if n == BOT_USER else SERIES_1 for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.barh(range(len(names)), counts, color=colors, height=0.62)
    ax.set_xlim(max(counts) * 1.13, 0)   # הפוך = RTL, הבר גדל שמאלה, עם מרווח לתוויות

    # תווית ערך אחידה מחוץ לקצה הבר. ha='right' עם ציר הפוך = הטקסט נמשך שמאלה
    for i, count in enumerate(counts):
        ax.text(count + max(counts) * 0.015, i, str(count), va="center",
                ha="right", fontsize=9.5, color=INK_2)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([he(n) for n in names], fontsize=10, color=INK)
    ax.yaxis.tick_right()
    ax.set_xlabel(he("עריכות"), fontsize=10)
    ax.set_title(he("עשרת התורמים המובילים בפרדספדיה"),
                 fontweight="bold", pad=16)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.01, 0.02,
             he("כתום: חשבון הבוט. כחול: עורכים תושבים."),
             ha="left", fontsize=8, color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    setup_font()
    client = WikiClient()

    print("שולף יומני יצירה...")
    creations = fetch_creations(client)
    print("שולף היסטוריית עריכות...")
    human, bot, per_user = fetch_revisions(client)

    articles = chart_growth(creations, f"{outdir}/pardespedia_growth.png")
    chart_edits(human, bot, f"{outdir}/pardespedia_edits.png")
    chart_contributors(per_user, f"{outdir}/pardespedia_contributors.png")

    total_h, total_b = sum(human.values()), sum(bot.values())
    print(f"\nדפים שנוצרו במרחב הראשי: {articles}")
    print(f"עריכות: {total_h + total_b} (תושבים {total_h} / בוט {total_b}"
          f" = {100 * total_b / (total_h + total_b):.0f}% בוט)")
    print(f"תורמים ייחודיים: {len(per_user)}")
    print("שלושה גרפים נשמרו ב-" + outdir)


if __name__ == "__main__":
    main()
