# screen capture

A small system-tray app for Linux/X11. Pick any keyboard key from the tray
menu, and pressing it anywhere takes a full-screen screenshot.

## Setup

```bash
cd "screen capture"
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python main.py
```

A tray icon appears. From its menu:

- **Set screenshot key...** — click it, then press any key; that key becomes
  the full-screen screenshot hotkey.
- **Take screenshot now** — capture the full screen immediately, no hotkey
  needed.
- **Set area-screenshot key...** — like above, but for area-select mode.
- **Select area now** — same as pressing the area key.
- **Open screenshots folder** — opens the save directory in your file manager.
- **Quit**.

Area-select mode dims the screen and shows a box centered on your mouse
cursor: scroll to resize it, left-click to capture whatever is inside it,
right-click or Esc to cancel.

Screenshots are saved to `~/Pictures/Screenshots` as timestamped PNGs
(`screenshot_2026-01-01_12-00-00.png`) and copied to the clipboard, ready to
paste. Both hotkeys are remembered in `~/.config/screen-capture/config.json`
between runs.

## Autostart

```bash
mkdir -p ~/.local/bin
ln -s "$(pwd)/scripts/launch.sh" ~/.local/bin/screen-capture-launch.sh
```

Then register `~/.local/bin/screen-capture-launch.sh` to run at login —
either via KDE's **System Settings → Startup and Shutdown → Autostart →
Add New → Add Login Script**, or by dropping a `.desktop` file in
`~/.config/autostart/` with `Exec=/home/rolo/.local/bin/screen-capture-launch.sh`.

`scripts/launch.sh` sets `PYTHONPATH` to the venv's `site-packages` and
calls the system `python3` directly, rather than invoking
`.venv/bin/python` (a symlink chain ending at `/usr/bin/python3`). That
symlink invocation works fine interactively, but under KDE Plasma 6's
systemd-based autostart it silently failed Python's venv auto-detection
(no `pyvenv.cfg` found relative to the resolved executable), falling back
to system site-packages and raising `ModuleNotFoundError: No module named
'PySide6'`. Setting `PYTHONPATH` explicitly sidesteps that detection
entirely, so it's independent of whatever launched it.

An earlier theory blamed a space in the `.venv` path itself (fixed via a
space-free symlink) — that turned out to be a red herring: the failure
persisted even through a space-free path once we could compare against a
GUI-registered Login Script entry, pointing at the venv-detection issue
above instead.

## Notes

- Requires an X11 session (global key listening and the tray icon both
  depend on it; this won't work under plain Wayland without XWayland).
- The tray icon is built with `PySide6-Essentials` (Qt). An earlier version
  used `pystray`'s plain-Xlib backend, but that rendered a blank/invisible
  icon under KDE Plasma's compositor — Qt's own tray implementation doesn't
  have that problem.
