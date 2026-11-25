# Pardespedia all-articles improvement sweep

(Written for the first sweep, 23/11/2026, by 31 agents. Since 24/11/2026 it is also the brief for the
recurring sweep every 14 days, `cron_sweep.sh`; that run's prompt gives the batch file, work dir and report path.)

Ori (the wiki's owner) asked: go over every article on pardespedia.info, check the web for
whether it can be updated, and improve how it looks and reads. He trusts us; quality over speed.
You own ONE batch of articles (a JSON list of titles). Work through every title in it.

## Setup
- Work dir: `~/fun/vibe_coding/pardespedia`. Run every `python3` from there (imports break otherwise).
- Read `~/fun/vibe_coding/pardespedia/CLAUDE.md` first. It is binding. Key points are repeated below.
- WebSearch / WebFetch are deferred tools: load them with ToolSearch (`select:WebSearch,WebFetch`).
- Scratch files: `<WORK>` (the work dir given in your prompt; create it). Not /tmp.

## The edit tool (use ONLY this for article edits)
```
python3 sweep_edit.py get  "<title>" <WORK>/<n>.wiki
# edit that same file (Edit tool), then:
python3 sweep_edit.py diff "<title>" <same file>      # review your change
python3 sweep_edit.py put  "<title>" <same file> "<short Hebrew summary>"
```
- `get` prints SKIP-RECENT-HUMAN if a person edited the page in the last 6 hours → skip that page, note it.
- `put` uses the base revision, so if someone edits meanwhile it fails with an edit conflict → re-get, re-apply.
- `put` prefixes the summary with "סבב שיפור: ". Summary examples: "עדכון: העסק נסגר (מקור)", "תמונה, מקורות והגהה".
- One `put` per article (two at most). Never write a page from scratch; change the text you got.

## What to do for each article
1. `get`, read it fully. Understand what it is (person / business / street / place / event / history...).
2. **Research on the web** (2–4 searches, then fetch the best primary pages). Look for:
   - **Is it still true?** Business still open? moved? new owner? hours/phone/site changed?
     Person: new role, book, album, award, death? Institution: new principal, renamed, closed?
     Event/project: did it happen, what came of it? Planned construction: was it built?
   - Facts that are clearly missing and well sourced (founding year, who it's named after, etc.).
   - A good image (if the article has none) and a genuinely relevant YouTube video.
3. **Improve the article**, only where there is a real gain:
   - Update stale facts, with the date: "נכון לנובמבר 2026...". A closed business: do NOT delete content;
     add a dated sentence at the top ("העסק נסגר ב-... [מקור]") and change present tense where needed.
     If closure is only suspected (site down, no posts) and not confirmed → do not state it; put it in the report.
   - Add sources: each new fact gets a source, either `<ref>` (then make sure there is a `<references />`
     under `== הערות שוליים ==`) or a line in the article's existing `== מקורות ==` list. Follow the
     article's existing style. Sources must be pages you actually opened, never a guessed URL.
   - Readability / UX: a clear bold lead sentence that says what the subject is and links
     `[[פרדס חנה-כרכור]]` (or the neighborhood); split a wall of text into sections with headings;
     a `== פרטים ==` list for businesses/places (address with a Google Maps link that includes `hl=iw`,
     hours, website); `== ראו גם ==` with internal links to related EXISTING articles; internal links on
     first mention of existing articles (check they exist); fix broken wikitext (tables, unclosed `'''`,
     `[[`, refs, bare URLs → `[url label]`), spelling and grammar, wrong/duplicate/missing categories
     (use only categories that already exist; check with the API).
   - Image: see rules below. Place it `[[קובץ:X|ממוזער|שמאל|250px|כיתוב]]` near the top.
   - Video: `<youtube width="300" height="180">VIDEOID</youtube>` with a one-line caption, only after
     confirming (title/description/channel) that the clip really is about the subject.
4. **If nothing is worth changing, don't edit.** An honest "checked, nothing new" is a fine result.

## Hard rules (from CLAUDE.md and Ori's standing feedback)
- **Never invent facts.** WebSearch summaries hallucinate: verify names/dates on the source page.
  Watch for namesakes (a different person/business with the same name, in another town).
  If a source contradicts the article, don't silently pick: write both with sources, or put it in the report.
- **Don't remove human-written content.** Rewording for clarity is fine; deleting facts is not
  (unless clearly wrong, and then say why in the summary).
- **Hebrew style:** no em-dash/en-dash "—" in prose you write (use comma, colon, or a new sentence).
  Natural Hebrew, no AI-style filler ("חשוב לציין", "לסיכום", superlatives). Don't copy phrasing from sources.
- **Only edit articles in your batch.** Need a change elsewhere (a hub table, another article,
  a new article that should exist)? Put it in the report under "הצעות לדפים אחרים". Exception: uploading files.
- **Never edit templates (`תבנית:`), `מדיה ויקי:`, user pages, or talk pages.** Don't change template calls'
  parameters except to fix an obvious factual error.
- **Privacy:** no home address or phone of private people. Public business contact info is fine.
  A person who left a business: no contact block for that business in the person's article.
  Living private individuals mentioned in passing: keep as the article has it, don't add more.
  Minors: never add names/photos.
- `~~~~` must never appear literally. The spam filter blocks URL shorteners (bit.ly, goo.gl, forms.gle,
  linktr.ee, l.facebook.com): open them and use the full URL.
- No ParserFunctions (`{{#if:}}` doesn't work), no `<style>`/`<script>`.

## Images
- First check the wiki already has one: `list=allimages` with `aiprefix`, or `list=search` in namespace 6
  (srnamespace=6, srwhat=text). Reusing an existing wiki file is best.
- Priority: free image (Wikimedia Commons, `iiurlwidth=1200` thumburl, check `imagerepository`, keep author+license)
  > fair use from the subject's own site / press / social (Ori approved this explicitly), low-res, with source.
- **Always look at the image (Read tool) before uploading.** Wrong subject, logo-only or broken → don't.
  In group photos never guess who is who. Apply `ImageOps.exif_transpose`. Blur faces of uninvolved
  private people and license plates if visible (PIL GaussianBlur on the region).
- Upload: `python3 upload_fairuse.py <local> "<Hebrew filename>.jpg" <descfile>` where the desc file is the
  full file page wikitext: description, `== מקור ==` with the URL, `== רישיון ==` (fair-use rationale:
  "שימוש הוגן: תמונה בגודל מוקטן להמחשת נושא הערך, מתוך <מקור>"), and a category if one fits.
  For Commons images write author + license + Commons link. For a Commons file use the same script with a
  description that says so (or look at `upload_commons.py` for the pattern).
- Never write "נמסר באמצעות חוה אהרון" or any intermediary credit. Never burn text/URLs into an image.
- Filename = the subject's Hebrew name (`<שם הערך>.jpg`, then `<שם הערך> 2.jpg`).

## Special cases
- **Streets:** research who/what the street is named after (Wikipedia is fine as a source for the namesake),
  when it was named if findable, what's on it (existing wiki articles: schools, parks, businesses on it),
  the neighborhood. A short, correct, linked article beats a long padded one.
- **Timeline / decade pages and `תקופה` pages** belong to חוה אהרון: conservative edits only
  (typos, broken markup, a clearly sourced missing fact). Don't restructure.
- **Pages with `תבנית:אתר לשימור` / `ציטוט נספח השימור` / `ציטוט סקר טבע`:** the quote templates are
  verbatim from official documents. Don't touch them. You may add current info (e.g. "נכון ל-2026 המבנה...").
- **Rare plants:** never publish exact locations.
- **Articles about politicians / council members / disputes:** every claim attributed with a source; neutral.
- Hub tables like `דמויות בפרדס חנה-כרכור`: don't edit, report.

## Report (required)
Write `<REPORT>` (the report path given in your prompt) incrementally (append after each article,
so a crash leaves a partial report). Hebrew. Per article one line:
`* [[כותרת]]: <מה נעשה / "נבדק, אין חדש"> (מקורות: ...)`
Then sections:
- `## לבירור אצל אורי`: suspected closures, contradictions, anything sensitive you didn't publish.
- `## הצעות לדפים אחרים`: concrete proposed edits outside the batch.
- `## רעיונות לערכים חדשים`: with a source each. **Before proposing a new article, make sure it doesn't
  already exist under another name**: `list=search` with `srwhat=text` (body text, not only titles) for the
  name and its variants, and read the hits. If an existing article covers the subject, propose a redirect
  or an addition to that article instead.
- `## בעיות שחוזרות`: patterns you saw (so the next agents can fix them better).
Final message to the coordinator: counts (edited / checked-no-change / skipped) + the 5 most important findings.

## Lessons from earlier batches
- The 24/11/2026 smoke test proposed a new article "חצר השוק", but that is the name of the compound in
  the existing article [[השוק הישן]] (it got a redirect instead). A title search alone misses such cases.
- history-1 published a "moved to Kfar Saba" claim taken from a search summary; the article itself didn't say it.
  **Write a fact only after you opened the page and saw it there.**
- Conservation-appendix buildings: the web rarely has post-2011 status. Don't spend more than 2 searches on that.
- The no-dash rule is for text YOU write. Don't edit a page just to swap dashes in existing human text
  (fine to do it incidentally inside a sentence you are already rewriting).
- Old council-site links of the form `.../articles/item/N` now redirect to the home page. Replace with a
  web.archive.org copy you actually opened, or the current page if you find it.
- Sensitive biographical facts (legal cases, divorces, permits disputes) found on Wikipedia: don't add, report.
- Do NOT spoof crawler User-Agents (e.g. facebookexternalhit) or otherwise get around a site's blocking.
  If Facebook/Instagram won't open normally, use what search results and other sites show, or report "לא אומת".
- Small local businesses often have no web footprint: don't state a closure; list them under "לבירור אצל אורי"
  as "לא נמצאו עקבות ברשת, לבדוק בשטח".
- The council's site is **pardes-hanna-karkur.muni.il** (pardes-hanna.org.il doesn't respond).
- Domains that expired and now redirect to unrelated/gambling sites: replace with an archived copy, note it.
