from PyQt5.QtWidgets import (
    QDialog,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QApplication,
)
from PyQt5.QtCore import Qt


class CustomMessageBox(QDialog):
    """
    Drop-in styled replacement for QMessageBox.
    Respects your app's QSS stylesheet fully.
    """

    INFO = "ℹ️  Information"
    WARNING = "⚠️  Warning"
    ERROR = "❌  Error"
    QUESTION = "❓  Question"

    def __init__(
        self,
        title: str,
        message: str,
        kind: str = INFO,
        buttons: list = None,
        parent=None,
    ):

        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setModal(True)
        self.setMinimumWidth(360)
        self.setObjectName("CustomMessageBox")

        app = QApplication.instance()
        if app:
            mw = app.activeWindow()
            if mw and mw.styleSheet():
                self.setStyleSheet(mw.styleSheet())
            elif app.styleSheet():
                self.setStyleSheet(app.styleSheet())

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 16)

        # Kind label (icon + type)
        kind_label = QLabel(kind)
        kind_label.setObjectName("msgKindLabel")

        if kind == CustomMessageBox.WARNING:
            kind_label.setStyleSheet("color: #D97706; font-weight: 600;")
        elif kind == CustomMessageBox.ERROR:
            kind_label.setStyleSheet("color: #DC2626; font-weight: 600;")
        elif kind == CustomMessageBox.QUESTION:
            kind_label.setStyleSheet("color: #7B3A9E; font-weight: 600;")
        else:
            kind_label.setStyleSheet("color: #1A7DC4; font-weight: 600;")

        layout.addWidget(kind_label)

        # Message
        msg_label = QLabel(message)
        msg_label.setObjectName("msgBodyLabel")
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        if buttons is None:
            buttons = ["OK"]

        for label in buttons:
            btn = QPushButton(label)
            btn.setFixedWidth(90)
            if label in ("OK", "Yes"):
                btn.setDefault(True)
                btn.clicked.connect(self.accept)
            else:
                btn.setDefault(False)
                btn.clicked.connect(self.reject)
            btn_row.addWidget(btn)

        layout.addLayout(btn_row)

    @staticmethod
    def info(message: str, title: str = "Information", parent=None):
        dlg = CustomMessageBox(title, message, CustomMessageBox.INFO, ["OK"], parent)
        dlg.exec_()

    @staticmethod
    def warning(message: str, title: str = "Warning", parent=None):
        dlg = CustomMessageBox(title, message, CustomMessageBox.WARNING, ["OK"], parent)
        dlg.exec_()

    @staticmethod
    def error(message: str, title: str = "Error", parent=None):
        dlg = CustomMessageBox(title, message, CustomMessageBox.ERROR, ["OK"], parent)
        dlg.exec_()

    @staticmethod
    def question(message: str, title: str = "Question", parent=None) -> bool:
        """Returns True if user clicked Yes, False if No."""
        dlg = CustomMessageBox(
            title, message, CustomMessageBox.QUESTION, ["Yes", "No"], parent
        )
        return dlg.exec_() == QDialog.Accepted
