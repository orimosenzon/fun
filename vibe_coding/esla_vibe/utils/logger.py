"""
Centralized logging configuration for esla_vibe.

Usage in any module:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("something happened")
    log.error("something broke", exc_info=True)

Log file: logs/esla_vibe.log  (rotating, max 1 MB × 3 files)
Console:  WARNING and above (so the Dash dev-server output stays clean)
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FILE = Path(__file__).parent.parent / "logs" / "esla_vibe.log"
_LOG_FILE.parent.mkdir(exist_ok=True)

_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s"
_DATE = "%H:%M:%S"

_configured = False


def _configure():
    global _configured
    if _configured:
        return

    root = logging.getLogger("esla")
    root.setLevel(logging.DEBUG)

    # ── File handler: DEBUG+, rotating ────────────────────────────────────────
    fh = RotatingFileHandler(_LOG_FILE, maxBytes=1_000_000, backupCount=3,
                             encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
    root.addHandler(fh)

    # ── Console handler: WARNING+ (keeps Dash terminal readable) ──────────────
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
    root.addHandler(ch)

    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'esla' root."""
    _configure()
    # Strip leading package path for cleaner names: "views.transform" etc.
    short = name.replace("esla_vibe.", "")
    return logging.getLogger(f"esla.{short}")
