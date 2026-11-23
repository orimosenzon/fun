"""Creates the four categories Carmit started with (23/11/2026) in Firestore, collection `categories`.

Only creates: a category that already exists is left exactly as it is (so a rename in the admin page survives),
and nothing is ever deleted. A category Carmit deleted on purpose would come back if this is run again, so there is
no reason to run it after the first time. "כללי" is not here: it is built into the admin page, for every riddle
without a category.

Writes go through the REST API with Ori's gcloud token (a project owner is not bound by the rules).
Usage: python3 tools/seed_categories.py
"""
import json
import pathlib
import subprocess
import urllib.error
import urllib.request

PROJECT = "carmit-riddles"
GCLOUD = pathlib.Path.home() / "google-cloud-sdk/bin/gcloud"
BASE = f"https://firestore.googleapis.com/v1/projects/{PROJECT}/databases/(default)/documents/categories"

# (document id, name). The admin page lists categories by `created`, so the seconds below keep this order.
SEED = [
    ("yesh-li-shir", "יש לי שיר"),
    ("mi-ani", "מי? אני"),
    ("mechakim-laot", "מחכים לאות"),
    ("sfat-hakodesh", "שפת הקודש"),
]


def main():
    token = subprocess.check_output(
        [str(GCLOUD), "auth", "print-access-token", "--account=orimosenzon@gmail.com"], text=True).strip()
    headers = {"Authorization": f"Bearer {token}", "x-goog-user-project": PROJECT, "Content-Type": "application/json"}
    for i, (doc_id, name) in enumerate(SEED):
        fields = {
            "name": {"stringValue": name},
            "by": {"stringValue": "בוטי"},
            "created": {"timestampValue": f"2026-11-23T00:00:{i:02d}Z"},
        }
        # documentId on a POST creates and never overwrites: an existing document answers 409.
        req = urllib.request.Request(f"{BASE}?documentId={doc_id}", data=json.dumps({"fields": fields}).encode(),
                                     method="POST", headers=headers)
        try:
            urllib.request.urlopen(req).read()
            print(f"created  {doc_id}  {name}")
        except urllib.error.HTTPError as e:
            if e.code != 409:
                raise
            print(f"exists   {doc_id}  (left as it is)")


if __name__ == "__main__":
    main()
