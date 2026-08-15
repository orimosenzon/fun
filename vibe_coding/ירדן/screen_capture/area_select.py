from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

MIN_SIZE = 40
MAX_SIZE = 4000
DEFAULT_SIZE = QSize(240, 160)
SCROLL_STEP = 1.1


class AreaSelectOverlay(QWidget):
    """Full-screen overlay showing a box centered on the mouse cursor.

    Scroll to resize the box, left-click to capture whatever is under it,
    right-click or Esc to cancel. on_capture(x, y, width, height) is called
    with absolute screen coordinates once the user clicks.
    """

    def __init__(self, on_capture):
        super().__init__()
        self._on_capture = on_capture
        self._box_size = QSize(DEFAULT_SIZE)
        self._mouse_pos = QPoint(0, 0)

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.setFocusPolicy(Qt.StrongFocus)

        self.setGeometry(QApplication.primaryScreen().virtualGeometry())

    def showEvent(self, event):
        super().showEvent(event)
        self._mouse_pos = self.mapFromGlobal(QCursor.pos())
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.ActiveWindowFocusReason)
        self.update()

    def mouseMoveEvent(self, event):
        self._mouse_pos = event.position().toPoint()
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = SCROLL_STEP if delta > 0 else 1 / SCROLL_STEP
        w = min(MAX_SIZE, max(MIN_SIZE, round(self._box_size.width() * factor)))
        h = min(MAX_SIZE, max(MIN_SIZE, round(self._box_size.height() * factor)))
        self._box_size = QSize(w, h)
        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            self.close()
            return

        rect = self._current_rect()
        top_left = self.mapToGlobal(rect.topLeft())
        x, y, w, h = top_left.x(), top_left.y(), rect.width(), rect.height()
        self.close()
        # Give the window manager a moment to actually unmap this overlay
        # before grabbing the screen, or the dimmed background/box outline
        # would show up in the captured image.
        QTimer.singleShot(80, lambda: self._on_capture(x, y, w, h))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def _current_rect(self):
        screen = self.rect()
        w = min(self._box_size.width(), screen.width())
        h = min(self._box_size.height(), screen.height())
        x = self._mouse_pos.x() - w // 2
        y = self._mouse_pos.y() - h // 2
        x = max(0, min(x, screen.width() - w))
        y = max(0, min(y, screen.height() - h))
        return QRect(x, y, w, h)

    def paintEvent(self, event):
        painter = QPainter(self)

        # Qt doesn't automatically reset the backing buffer to transparent
        # before each paint even with WA_TranslucentBackground set; without
        # this, untouched regions (the box) can retain stale opaque content
        # instead of showing the desktop through.
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self._current_rect()
        screen = self.rect()
        dim = QColor(0, 0, 0, 100)

        # Dim only the four strips around the box; the box itself is left
        # untouched so it stays fully transparent (relying on
        # CompositionMode_Clear to "punch a hole" instead didn't render
        # correctly under this compositor).
        painter.fillRect(QRect(0, 0, screen.width(), rect.top()), dim)
        painter.fillRect(
            QRect(0, rect.bottom() + 1, screen.width(), screen.height() - rect.bottom() - 1),
            dim,
        )
        painter.fillRect(QRect(0, rect.top(), rect.left(), rect.height()), dim)
        painter.fillRect(
            QRect(rect.right() + 1, rect.top(), screen.width() - rect.right() - 1, rect.height()),
            dim,
        )

        painter.setPen(QPen(QColor(220, 40, 40), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            rect.x(), rect.y() - 8, f"{rect.width()} x {rect.height()}"
        )
