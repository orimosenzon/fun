from datetime import datetime
from pathlib import Path

import mss
import mss.tools


def _save_region(region, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / f"screenshot_{datetime.now():%Y-%m-%d_%H-%M-%S}.png"

    with mss.mss() as sct:
        shot = sct.grab(region)
        mss.tools.to_png(shot.rgb, shot.size, output=str(filename))

    return filename


def take_screenshot(save_dir):
    """Grab the full virtual screen and save it as a timestamped PNG. Returns the path."""
    with mss.mss() as sct:
        monitor = sct.monitors[0]
    return _save_region(monitor, save_dir)


def take_area_screenshot(save_dir, x, y, width, height):
    """Grab a specific screen region (absolute coordinates) and save it as a timestamped PNG."""
    region = {"left": x, "top": y, "width": max(1, width), "height": max(1, height)}
    return _save_region(region, save_dir)
