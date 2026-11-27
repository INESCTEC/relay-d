"""The sidebar icons are black PNGs: they are redrawn in the button's text colour, keeping their shape."""


def test_tinted_icon_keeps_shape_and_takes_the_colour(qapp):
    from PyQt5.QtCore import QSize, Qt
    from PyQt5.QtGui import QColor, QIcon, QPixmap
    from relay_d.acquisition.qt_app.utils.icon_tint import tinted_icon

    src = QPixmap(8, 8)
    src.fill(Qt.transparent)
    src_img = src.toImage()
    for x in range(4):  # left half opaque black, right half transparent
        for y in range(8):
            src_img.setPixelColor(x, y, QColor("black"))
    img = tinted_icon(QIcon(QPixmap.fromImage(src_img)), QColor("#86EFAC"), QSize(8, 8)).pixmap(QSize(8, 8)).toImage()

    assert img.pixelColor(1, 4).name() == "#86efac"
    assert img.pixelColor(6, 4).alpha() == 0


def test_icon_follows_the_button_state(qapp):
    from PyQt5.QtCore import QSize, Qt
    from PyQt5.QtGui import QIcon, QPixmap
    from PyQt5.QtWidgets import QPushButton
    from relay_d.acquisition.qt_app.utils.icon_tint import IconTinter

    src = QPixmap(8, 8)
    src.fill(Qt.black)
    button = QPushButton()
    button.setIcon(QIcon(src))
    button.setIconSize(QSize(8, 8))
    button.setStyleSheet("QPushButton { color: #00ff00; } QPushButton:disabled { color: #808080; }")
    button.setEnabled(False)
    tinter = IconTinter([button], parent=button)  # parented, as in the app (else it is garbage-collected)
    tinter.retint_all()

    def colour():
        return button.icon().pixmap(QSize(8, 8)).toImage().pixelColor(4, 4).name()

    assert colour() == "#808080"
    button.setEnabled(True)  # e.g. Start Record once a config is loaded
    assert colour() == "#00ff00"
