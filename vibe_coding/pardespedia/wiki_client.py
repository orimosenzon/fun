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

    def edit_page(self, title: str, content: str, summary: str = "", minor: bool = False) -> dict:
        """Edit or create a page. Returns API result."""
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

        r = self.session.post(API_URL, data=data)
        r.raise_for_status()
        result = r.json()

        if "error" in result:
            raise RuntimeError(f"Edit failed: {result['error']['info']}")

        edit_result = result["edit"]
        print(f"{'Created' if edit_result.get('new') else 'Edited'}: {title} (rev {edit_result.get('newrevid')})")
        return edit_result

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

    def list_pages(self, prefix: str = "", limit: int = 50, namespace: int = 0) -> list:
        """List pages, optionally filtered by prefix."""
        params = {
            "action": "query",
            "list": "allpages",
            "aplimit": limit,
            "apnamespace": namespace,
            "format": "json",
        }
        if prefix:
            params["apprefix"] = prefix

        r = self.session.get(API_URL, params=params)
        r.raise_for_status()
        pages = r.json()["query"]["allpages"]
        return [
            {
                "title": p["title"],
                "url": f"https://pardespedia.info/wiki/{p['title'].replace(' ', '_')}",
            }
            for p in pages
        ]

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
