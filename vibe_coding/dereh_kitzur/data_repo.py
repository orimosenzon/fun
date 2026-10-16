"""Put one JSON file in the data repo (orimosenzon/derech-kitzur-data), as the bot.

The app reads its layers from that repo's raw files, so a file published here
is live on every phone at once, with no build and no push of the site. One
file, one commit, through the GitHub API (`gh`), with no working copy to keep
clean. Nothing is written when only the `updated` stamp differs, so a script
that runs every night does not add a commit a night.
"""
import base64
import json
import subprocess

DATA_REPO = "orimosenzon/derech-kitzur-data"
BOT = "בוטי"
BOT_EMAIL = "orimosenzon@gmail.com"      # as on the bot's earlier commits there


def publish_json(path, doc, message, ignore=("updated",)):
    """Write `doc` to `path` in the data repo. Returns "פורסם" or "ללא שינוי"."""
    body = json.dumps(doc, ensure_ascii=False, indent=1)     # as save_json writes it
    api = f"repos/{DATA_REPO}/contents/{path}"
    sha = None
    try:
        got = json.loads(subprocess.run(["gh", "api", api], capture_output=True,
                                        text=True, check=True).stdout)
        sha = got.get("sha")
        if got.get("content"):
            old = json.loads(base64.b64decode(got["content"]).decode("utf-8"))
            strip = lambda d: {k: v for k, v in d.items() if k not in ignore}
            if strip(old) == strip(doc):
                return "ללא שינוי"
    except subprocess.CalledProcessError:
        pass                                            # first time: no file yet
    payload = {
        "message": message,
        "content": base64.b64encode(body.encode("utf-8")).decode("ascii"),
        "committer": {"name": BOT, "email": BOT_EMAIL},
        "author": {"name": BOT, "email": BOT_EMAIL},
    }
    if sha:
        payload["sha"] = sha
    subprocess.run(["gh", "api", "-X", "PUT", api, "--input", "-"],
                   input=json.dumps(payload), capture_output=True, text=True, check=True)
    return "פורסם"
