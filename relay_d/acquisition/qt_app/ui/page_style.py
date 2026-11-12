"""The one layout every page of the main window follows: margins, a title, a thin separator, then the content.

Colours, fonts and the separator line come from the theme files (``ui/styles/relayd_*.qss``: ``QLabel#pageTitle``,
``QFrame#pageSeparator``, ``QGroupBox``, ``QTabBar``). Pages built in code use these helpers; the pages made in Qt Designer
(Post-process, Config) have the same margins and use the same QSS rules through their own object names.
"""

from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout

PAGE_MARGINS = (24, 20, 24, 20)
PAGE_SPACING = 12


def page_layout(page) -> QVBoxLayout:
    """The root layout of ``page`` with the shared margins and spacing."""
    layout = QVBoxLayout(page)
    layout.setContentsMargins(*PAGE_MARGINS)
    layout.setSpacing(PAGE_SPACING)
    return layout


def add_page_header(layout, title: str) -> QLabel:
    """Add the page title and the separator line under it to ``layout``."""
    label = QLabel(title)
    label.setObjectName("pageTitle")
    line = QFrame()
    line.setObjectName("pageSeparator")
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Plain)
    line.setMaximumHeight(1)
    layout.addWidget(label)
    layout.addWidget(line)
    return label
