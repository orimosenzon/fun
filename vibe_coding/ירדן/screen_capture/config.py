import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "screen-capture"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG = {
    "hotkey": None,
    "area_hotkey": None,
    "save_dir": str(Path.home() / "Pictures" / "Screenshots"),
}


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            data = json.load(f)
        return {**DEFAULT_CONFIG, **data}
    return dict(DEFAULT_CONFIG)


def save_config(config):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
