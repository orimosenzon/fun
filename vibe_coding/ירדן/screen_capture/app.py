import subprocess
import sys

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon
from pynput import keyboard

from . import config as cfg
from .area_select import AreaSelectOverlay
from .key_capture import capture_next_key
from .keys import key_display_name, key_to_str, str_to_key
from .screenshot import take_area_screenshot, take_screenshot


def _make_icon():
    # Tray icons render as small as 16-22px, so use one bold, high-contrast
    # flat shape (white camera silhouette with a red lens) rather than
    # subtle multi-tone shading that disappears at that size.
    pix = QPixmap(64, 64)
    pix.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(255, 255, 255))
    painter.drawRoundedRect(2, 16, 60, 42, 10, 10)
    painter.drawRect(20, 6, 24, 12)
    painter.setBrush(QColor(220, 40, 40))
    painter.drawEllipse(20, 25, 24, 24)
    painter.end()
    return QIcon(pix)


class ScreenCaptureApp(QObject):
    # Emitted from the pynput listener thread; queued onto the Qt main
    # thread since tray/menu widgets, the clipboard, and creating the area
    # overlay window may only be touched there.
    key_captured = Signal(str)
    capture_requested = Signal()
    area_select_requested = Signal()

    def __init__(self):
        super().__init__()
        self.config = cfg.load_config()
        self.hotkey_listener = None
        self.area_hotkey_listener = None
        self.capture_listener = None
        self.listening_for_key = False
        self._overlay = None

        # Set while capture_next_key() is listening for a "set key" click,
        # so _on_key_captured() knows which config field/UI to update.
        self._pending_config_key = None
        self._pending_status_action = None
        self._pending_start_listener = None
        self._pending_label = None

        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)

        self.tray = QSystemTrayIcon(_make_icon())
        self.tray.setToolTip(self._title())

        self.status_action = QAction(self._status_text("Screenshot key", "hotkey"))
        self.status_action.setEnabled(False)

        self.area_status_action = QAction(self._status_text("Area key", "area_hotkey"))
        self.area_status_action.setEnabled(False)

        self.menu = QMenu()
        self.menu.addAction(self.status_action)
        self.menu.addAction("Set screenshot key...", self.on_set_key)
        self.menu.addAction("Take screenshot now", self.capture_requested.emit)
        self.menu.addSeparator()
        self.menu.addAction(self.area_status_action)
        self.menu.addAction("Set area-screenshot key...", self.on_set_area_key)
        self.menu.addAction("Select area now", self.area_select_requested.emit)
        self.menu.addSeparator()
        self.menu.addAction("Open screenshots folder", self.on_open_folder)
        self.menu.addSeparator()
        self.menu.addAction("Quit", self.app.quit)
        self.tray.setContextMenu(self.menu)

        self.key_captured.connect(self._on_key_captured, Qt.QueuedConnection)
        self.capture_requested.connect(self._capture, Qt.QueuedConnection)
        self.area_select_requested.connect(self._start_area_select, Qt.QueuedConnection)

        self.tray.setVisible(True)
        self._start_hotkey_listener()
        self._start_area_hotkey_listener()

    def _title(self):
        return (
            f"Screen Capture (key: {key_display_name(self.config.get('hotkey'))}, "
            f"area key: {key_display_name(self.config.get('area_hotkey'))})"
        )

    def _status_text(self, label, config_key):
        return f"{label}: {key_display_name(self.config.get(config_key))}"

    def _start_hotkey_listener(self):
        self.hotkey_listener = self._start_single_key_listener(
            self.hotkey_listener, "hotkey", self.capture_requested
        )

    def _start_area_hotkey_listener(self):
        self.area_hotkey_listener = self._start_single_key_listener(
            self.area_hotkey_listener, "area_hotkey", self.area_select_requested
        )

    def _start_single_key_listener(self, existing_listener, config_key, signal):
        if existing_listener:
            existing_listener.stop()

        hotkey_str = self.config.get(config_key)
        if not hotkey_str:
            return None

        target_key = str_to_key(hotkey_str)

        def on_press(key):
            if self.listening_for_key:
                return
            if key == target_key:
                signal.emit()

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener

    def _capture(self):
        path = take_screenshot(self.config["save_dir"])
        self.app.clipboard().setPixmap(QPixmap(str(path)))
        print(f"[screen-capture] Saved screenshot to {path} (copied to clipboard)")

    def _start_area_select(self):
        if self._overlay is not None:
            return
        self._overlay = AreaSelectOverlay(self._on_area_captured)
        self._overlay.destroyed.connect(self._on_overlay_closed)
        self._overlay.show()

    def _on_overlay_closed(self):
        self._overlay = None

    def _on_area_captured(self, x, y, width, height):
        path = take_area_screenshot(self.config["save_dir"], x, y, width, height)
        self.app.clipboard().setPixmap(QPixmap(str(path)))
        print(f"[screen-capture] Saved area screenshot to {path} (copied to clipboard)")

    def on_set_key(self):
        self._begin_set_key(
            "hotkey", self.status_action, self._start_hotkey_listener, "Screenshot key"
        )

    def on_set_area_key(self):
        self._begin_set_key(
            "area_hotkey",
            self.area_status_action,
            self._start_area_hotkey_listener,
            "Area key",
        )

    def _begin_set_key(self, config_key, status_action, start_listener, label):
        if self.listening_for_key:
            return
        self.listening_for_key = True
        self._pending_config_key = config_key
        self._pending_status_action = status_action
        self._pending_start_listener = start_listener
        self._pending_label = label

        status_action.setText("Press any key now...")
        self.tray.setToolTip("Screen Capture (press a key now...)")

        def on_captured(key):
            self.key_captured.emit(key_to_str(key))

        self.capture_listener = capture_next_key(on_captured)

    def _on_key_captured(self, key_str):
        self.listening_for_key = False
        self.config[self._pending_config_key] = key_str
        cfg.save_config(self.config)
        self._pending_status_action.setText(f"{self._pending_label}: {key_display_name(key_str)}")
        self.tray.setToolTip(self._title())
        self._pending_start_listener()
        print(f"[screen-capture] {self._pending_label} set to: {key_display_name(key_str)}")

    def on_open_folder(self):
        subprocess.Popen(["xdg-open", self.config["save_dir"]])

    def run(self):
        sys.exit(self.app.exec())
