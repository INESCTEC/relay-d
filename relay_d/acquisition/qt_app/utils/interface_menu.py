# Contains functions for the sliding menu
"""
Sliding Menu Module
Contains all menu-related functionality for Qt applications
"""

from relay_d.utils.coloring_logger import logger
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve, QObject, pyqtSignal

class SlidingMenuManager(QObject):
    """
    Manages sliding menu functionality for any Qt widget
    """

    # Signals
    menuToggled = pyqtSignal(bool)  # Emitted when menu state changes
    menuExpanded = pyqtSignal()  # Emitted when menu expands
    menuCollapsed = pyqtSignal()  # Emitted when menu collapses

    def __init__(self, menu_widget, toggle_button=None):
        """
        Initialize the sliding menu manager

        Args:
            menu_widget: The widget to animate (e.g., leftMenuSubContainer)
            toggle_button: Optional button to connect toggle functionality
        """
        super().__init__()

        self.menu_widget = menu_widget
        self.toggle_button = toggle_button

        # Animation settings
        self.collapsed_width = 70
        self.expanded_width = 250
        self.animation_duration = 300  # milliseconds
        self.is_expanded = False

        # Setup animations
        self._setup_animations()

        # Connect toggle button if provided
        if self.toggle_button:
            self.toggle_button.clicked.connect(self.toggle_menu)

    def _setup_animations(self):
        """Setup the sliding animations"""
        # Main width animation
        self.animation = QPropertyAnimation(self.menu_widget, b"maximumWidth")
        self.animation.setDuration(self.animation_duration)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)

        # Minimum width animation for smooth resize
        self.animation_min = QPropertyAnimation(self.menu_widget, b"minimumWidth")
        self.animation_min.setDuration(self.animation_duration)
        self.animation_min.setEasingCurve(QEasingCurve.OutCubic)

        # Connect animation finished signals
        self.animation.finished.connect(self._on_animation_finished)

    def set_widths(self, collapsed_width, expanded_width):
        """
        Set custom widths for collapsed and expanded states

        Args:
            collapsed_width (int): Width when menu is collapsed
            expanded_width (int): Width when menu is expanded
        """
        self.collapsed_width = collapsed_width
        self.expanded_width = expanded_width

    def set_animation_duration(self, duration):
        """
        Set animation duration in milliseconds

        Args:
            duration (int): Animation duration in milliseconds
        """
        self.animation_duration = duration
        self.animation.setDuration(duration)
        self.animation_min.setDuration(duration)

    def set_easing_curve(self, curve):
        """
        Set animation easing curve

        Args:
            curve: QEasingCurve type (e.g., QEasingCurve.OutCubic)
        """
        self.animation.setEasingCurve(curve)
        self.animation_min.setEasingCurve(curve)

    def toggle_menu(self):
        """Toggle menu between collapsed and expanded states"""
        if self.is_expanded:
            self.collapse_menu()
        else:
            self.expand_menu()

    def expand_menu(self):
        """Expand the menu with animation"""
        if not self.is_expanded:
            self._animate_to_width(self.expanded_width)
            self.is_expanded = True
            self.menuExpanded.emit()
            self.menuToggled.emit(True)

    def collapse_menu(self):
        """Collapse the menu with animation"""
        if self.is_expanded:
            self._animate_to_width(self.collapsed_width)
            self.is_expanded = False
            self.menuCollapsed.emit()
            self.menuToggled.emit(False)

    def _animate_to_width(self, target_width):
        """
        Animate menu to specified width

        Args:
            target_width (int): Target width in pixels
        """
        current_width = self.menu_widget.width()

        # Set animation start and end values
        self.animation.setStartValue(current_width)
        self.animation.setEndValue(target_width)

        self.animation_min.setStartValue(self.menu_widget.minimumWidth())
        self.animation_min.setEndValue(target_width)

        # Start animations
        self.animation.start()
        self.animation_min.start()

    def _on_animation_finished(self):
        """Called when animation finishes"""
        # Ensure final width is set correctly
        target_width = self.expanded_width if self.is_expanded else self.collapsed_width
        self.menu_widget.setFixedWidth(target_width)

    def set_expanded_state(self, expanded, animate=True):
        """
        Set menu state programmatically

        Args:
            expanded (bool): True to expand, False to collapse
            animate (bool): Whether to animate the change
        """
        if expanded and not self.is_expanded:
            if animate:
                self.expand_menu()
            else:
                self._set_width_immediately(self.expanded_width)
                self.is_expanded = True
                self.menuToggled.emit(True)
        elif not expanded and self.is_expanded:
            if animate:
                self.collapse_menu()
            else:
                self._set_width_immediately(self.collapsed_width)
                self.is_expanded = False
                self.menuToggled.emit(False)

    def _set_width_immediately(self, width):
        """Set width immediately without animation"""
        self.menu_widget.setFixedWidth(width)
        self.menu_widget.setMinimumWidth(width)
        self.menu_widget.setMaximumWidth(width)

    def is_menu_expanded(self):
        """
        Check if menu is currently expanded

        Returns:
            bool: True if expanded, False if collapsed
        """
        return self.is_expanded

    def get_current_width(self):
        """
        Get current menu width

        Returns:
            int: Current width in pixels
        """
        return self.menu_widget.width()


def setup_button_widget_switching(ui_instance, stacked_widget_name="stackedWidget"):
    """
    Simple function to connect menu buttons to widget pages

    Args:
        ui_instance: Your UI class instance
        stacked_widget_name: Name of your stacked widget (default: "stackedWidget")
    """
    # Get the stacked widget
    stacked_widget = getattr(ui_instance, stacked_widget_name)

    # Connect each button to switch to a specific page
    if hasattr(ui_instance, "btn_start"):
        ui_instance.btn_start.clicked.connect(lambda: stacked_widget.setCurrentIndex(0))

    if hasattr(ui_instance, "btn_data_post_process"):
        ui_instance.btn_data_post_process.clicked.connect(
            lambda: stacked_widget.setCurrentIndex(1)
        )

    if hasattr(ui_instance, "btn_import"):
        ui_instance.btn_import.clicked.connect(
            lambda: stacked_widget.setCurrentIndex(2)
        )

    if hasattr(ui_instance, "btn_settings"):
        ui_instance.btn_settings.clicked.connect(
            lambda: stacked_widget.setCurrentIndex(4)
        )


# Utility functions for easy integration
def setup_sliding_menu(
    menu_widget, toggle_button=None, collapsed_width=70, expanded_width=250
):
    """
    Quick setup function for sliding menu

    Args:
        menu_widget: Widget to animate
        toggle_button: Optional toggle button
        collapsed_width: Width when collapsed
        expanded_width: Width when expanded

    Returns:
        SlidingMenuManager: Configured menu manager instance
    """
    manager = SlidingMenuManager(menu_widget, toggle_button)
    manager.set_widths(collapsed_width, expanded_width)
    return manager


def setup_menu_buttons(ui_instance, button_names=None):
    """
    Quick setup for menu buttons using default handlers

    Args:
        ui_instance: UI class instance containing the buttons
        button_names: List of button attribute names, defaults to standard names
    """
    if button_names is None:
        button_names = [
            "btn_start",
            "btn_data_post_process",
            "btn_import",
            "btn_settings",
        ]

    buttons = []
    for name in button_names:
        if hasattr(ui_instance, name):
            buttons.append(getattr(ui_instance, name))

    setup_button_widget_switching(ui_instance)
