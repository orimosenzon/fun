"""
Pardespedia MediaWiki API client.
Handles authentication, reading, editing, and creating pages.
"""

import json
import os
import sys
import requests

API_URL = "https://pardespedia.info/api.php"
CREDS_FILE = os.path.join(os.path.dirname(__file__), "credentials.json")


def load_credentials():
    if not os.path.exists(CREDS_FILE):
        print(f"ERROR: credentials.json not found. Copy credentials.json.example to credentials.json and fill in your details.", file=sys.stderr)
        sys.exit(1)
    with open(CREDS_FILE) as f:
        return json.load(f)


class WikiClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Pardespedia-Bot/1.0 (orimosenzon@gmail.com)"})
        self.logged_in = False

    def login(self):
        creds = load_credentials()

        # Step 1: get login token
        r = self.session.get(API_URL, params={
            "action": "query",
            "meta": "tokens",
            "type": "login",
            "format": "json",
        })
        r.raise_for_status()
        login_token = r.json()["query"]["tokens"]["logintoken"]

        # Step 2: login
        r = self.session.post(API_URL, data={
            "action": "login",
            "lgname": creds["username"],
            "lgpassword": creds["password"],
            "lgtoken": login_token,
            "format": "json",
        })
        r.raise_for_status()
        result = r.json()
        if result["login"]["result"] != "Success":
            raise RuntimeError(f"Login failed: {result['login']['reason']}")

        self.logged_in = True
        print(f"Logged in as: {result['login']['lgusername']}")
        return self

    def _csrf_token(self):
        r = self.session.get(API_URL, params={
            "action": "query",
            "meta": "tokens",
            "format": "json",
        })
        r.raise_for_status()
        return r.json()["query"]["tokens"]["csrftoken"]

    def get_page(self, title: str) -> dict:
        """Return page info: wikitext, exists, url."""
        r = self.session.get(API_URL, params={
            "action": "query",
            "titles": title,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "format": "json",
        })
        r.raise_for_status()
        pages = r.json()["query"]["pages"]
        page = next(iter(pages.values()))

        exists = "missing" not in page
        wikitext = ""
        if exists:
            wikitext = page["revisions"][0]["slots"]["main"]["*"]

        return {
            "title": page.get("title", title),
            "exists": exists,
            "wikitext": wikitext,
            "url": f"https://pardespedia.info/wiki/{title.replace(' ', '_')}",
        }

    def edit_page(self, title: str, content: str, summary: str = "", minor: bool = False, create_only: bool = False) -> dict:
        """Edit or create a page. Returns API result.

        create_only=True refuses to touch an existing page (API createonly flag).
        """
        if not self.logged_in:
            raise RuntimeError("Must be logged in to edit.")

        token = self._csrf_token()
        data = {
            "action": "edit",
            "title": title,
            "text": content,
            "summary": summary,
            "token": token,
            "format": "json",
        }
        if minor:
            data["minor"] = "1"
        if create_only:
            data["createonly"] = "1"

        r = self.session.post(API_URL, data=data)
        r.raise_for_status()
        result = r.json()

        if "error" in result:
            if result["error"].get("code") == "articleexists":
                raise RuntimeError(f"Page already exists: {title}. Use 'edit' to modify it.")
            raise RuntimeError(f"Edit failed: {result['error']['info']}")

        edit_result = result["edit"]
        print(f"{'Created' if edit_result.get('new') else 'Edited'}: {title} (rev {edit_result.get('newrevid')})")
        return edit_result

    def move_page(self, from_title: str, to_title: str, reason: str = "", leave_redirect: bool = True) -> dict:
        """Rename/move a page. Returns API result."""
        if not self.logged_in:
            raise RuntimeError("Must be logged in to move.")

        token = self._csrf_token()
        data = {
            "action": "move",
            "from": from_title,
            "to": to_title,
            "reason": reason,
            "token": token,
            "format": "json",
        }
        if not leave_redirect:
            data["noredirect"] = "1"

        r = self.session.post(API_URL, data=data)
        r.raise_for_status()
        result = r.json()

        if "error" in result:
            raise RuntimeError(f"Move failed: {result['error']['info']}")

        print(f"Moved: {from_title} -> {to_title}")
        return result["move"]

    def search(self, query: str, limit: int = 10) -> list:
        """Search pages. Returns list of {title, snippet, url}."""
        r = self.session.get(API_URL, params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        })
        r.raise_for_status()
        results = r.json()["query"]["search"]
        return [
            {
                "title": item["title"],
                "snippet": item["snippet"],
                "url": f"https://pardespedia.info/wiki/{item['title'].replace(' ', '_')}",
            }
            for item in results
        ]

    def list_pages(self, prefix: str = "", limit: int = 0, namespace: int = 0,
                   redirects: str = "all") -> list:
        """List pages, optionally filtered by prefix.

        limit=0 (default) fetches all pages, following apcontinue. A positive
        limit caps the total number of results returned.

        redirects: "all" (default), "redirects", or "nonredirects". Callers
        that treat every title as an article want "nonredirects" — a redirect
        has no content of its own, so it reaches such a caller as a page with
        no description.
        """
        results = []
        apcontinue = None
        while True:
            batch_size = 500
            if limit:
                remaining = limit - len(results)
                if remaining <= 0:
                    break
                batch_size = min(batch_size, remaining)
            params = {
                "action": "query",
                "list": "allpages",
                "aplimit": batch_size,
                "apnamespace": namespace,
                "apfilterredir": redirects,
                "format": "json",
            }
            if prefix:
                params["apprefix"] = prefix
            if apcontinue:
                params["apcontinue"] = apcontinue

            r = self.session.get(API_URL, params=params)
            r.raise_for_status()
            data = r.json()
            for p in data["query"]["allpages"]:
                results.append({
                    "title": p["title"],
                    "url": f"https://pardespedia.info/wiki/{p['title'].replace(' ', '_')}",
                })

            cont = data.get("continue")
            if not cont:
                break
            apcontinue = cont.get("apcontinue")

        return results

    def get_categories(self, title: str) -> list:
        """Return list of categories for a page."""
        r = self.session.get(API_URL, params={
            "action": "query",
            "titles": title,
            "prop": "categories",
            "format": "json",
        })
        r.raise_for_status()
        pages = r.json()["query"]["pages"]
        page = next(iter(pages.values()))
        cats = page.get("categories", [])
        return [c["title"] for c in cats]
