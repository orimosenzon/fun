#!/bin/sh
# Launcher used for autostart/login-script mechanisms. Invoking
# .venv/bin/python (a symlink chain ending at /usr/bin/python3) let Python
# fail to detect the venv and its pyvenv.cfg under systemd's execution
# context, silently falling back to system site-packages and missing
# PySide6 - even though the exact same invocation works fine from an
# interactive shell. Setting PYTHONPATH directly and calling the system
# interpreter sidesteps that venv auto-detection entirely.
export PYTHONPATH="/home/rolo/rolo/screen capture/.venv/lib/python3.13/site-packages"
exec /usr/bin/python3 "/home/rolo/rolo/screen capture/main.py"
