"""Draw the (black) PNG icons in a button's text colour, so they stay visible in the dark theme."""

from PyQt5.QtCore import QEvent, QObject, QSize
from PyQt5.QtGui import QColor, QIcon, QPainter, QPalette, QPixmap


def tinted_icon(icon: QIcon, color: QColor, size: QSize) -> QIcon:
    """``icon`` with every opaque pixel painted ``color`` (its alpha, i.e. its shape, is kept)."""
    pixmap = QPixmap(icon.pixmap(size * 2))  # 2x: sharp on HiDPI screens
    painter = QPainter(pixmap)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), color)
    painter.end()
    return QIcon(pixmap)


class IconTinter(QObject):
    """Keeps each button's icon in its current text colour. The colour comes from the theme QSS and changes with the
    theme and with the button's state (e.g. Start Record is grey while disabled, green once a config is loaded)."""

    def __init__(self, buttons, parent=None):
        super().__init__(parent)
        self._icons = {b: b.icon() for b in buttons if not b.icon().isNull()}  # the originals, tinted again each time
        for button in self._icons:
            button.installEventFilter(self)

    def retint(self, button):
        button.ensurePolished()
        button.setIcon(tinted_icon(self._icons[button], button.palette().color(QPalette.ButtonText), button.iconSize()))

    def retint_all(self):
        for button in self._icons:
            self.retint(button)

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.PaletteChange, QEvent.EnabledChange) and obj in self._icons:
            self.retint(obj)
        return False
