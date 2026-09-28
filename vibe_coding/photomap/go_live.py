#!/usr/bin/env python3
"""Put the photo map online: server + Cloudflare tunnel + a fixed link.

    python3 go_live.py

Starts serve.py (if it is not already up), opens a Cloudflare quick tunnel to
it, and publishes the tunnel's address to a secret gist. live.html on GitHub
Pages reads that gist, checks the map answers, and forwards the visitor. So
the link friends get never changes, while the tunnel address changes on every
start.

The photos and data/ never leave this computer; only the tunnel URL is
published. Runs in the foreground until the tunnel dies, then exits non-zero
so systemd (photomap-live.service) restarts it and a fresh URL is published.
"""
import datetime as dt
import json
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = 8797
CLOUDFLARED = Path.home() / ".local/bin/cloudflared"
GIST_ID = "635f409c8f70b93c2b0e37e32eacf356"
LOG = ROOT / "data" / "live.log"

children = []


def log(msg: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def server_up() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/ping", timeout=3) as r:
            return r.read() == b"ok"
    except OSError:
        return False


def publish(url) -> None:
    """Write the current address to the gist, retrying while the network comes up."""
    body = json.dumps({"url": url, "since": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
    payload = json.dumps({"files": {"live.json": {"content": body}}})
    for delay in (0, 5, 15, 30, 60, 120):
        time.sleep(delay)
        r = subprocess.run(["gh", "api", "-X", "PATCH", f"gists/{GIST_ID}", "--input", "-"],
                           input=payload, text=True, capture_output=True)
        if r.returncode == 0:
            log(f"published {url}")
            return
        log(f"publish failed ({r.stderr.strip()[:200]}), retrying")
    log("publish gave up")


def stop(*_):
    for p in children:
        p.terminate()
    sys.exit(0)


def main():
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    LOG.parent.mkdir(exist_ok=True)

    if not server_up():
        children.append(subprocess.Popen([sys.executable, str(ROOT / "serve.py"), str(PORT)], cwd=ROOT))
        for _ in range(30):
            time.sleep(1)
            if server_up():
                break
        else:
            log("serve.py did not come up")
            stop()
    log("server up")

    tunnel = subprocess.Popen([str(CLOUDFLARED), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
                              stderr=subprocess.PIPE, text=True)
    children.append(tunnel)
    url = None
    for line in tunnel.stderr:
        if not url:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m:
                url = m.group(0)
                publish(url)
    log(f"tunnel exited ({tunnel.wait()})")
    for p in children:
        p.terminate()
    sys.exit(1)


if __name__ == "__main__":
    main()
