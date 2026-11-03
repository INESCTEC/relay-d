"""Keep OpenCV's bundled Qt from hijacking PyQt5's platform plugins.

The full `opencv-python` wheel (often pulled in by other packages, e.g.
ultralytics) ships its own Qt and, on `import cv2`, points
QT_QPA_PLATFORM_PLUGIN_PATH at cv2/qt/plugins. PyQt5 then fails to load its
"xcb" platform plugin from there and aborts. Call `sanitize_qt_env()` after
importing cv2 and before creating a QApplication.
"""
import os

_VARS = ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH", "QT_QPA_FONTDIR")


def sanitize_qt_env() -> None:
    for var in _VARS:
        value = os.environ.get(var, "")
        if os.path.join("cv2", "qt") in value.replace("\\", "/").replace("/", os.sep):
            os.environ.pop(var, None)
