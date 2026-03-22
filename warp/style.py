# warp/style.py
# SETS-matching dark theme for WARP and WARP CORE windows.
# Colors are taken directly from main.py Launcher.theme['defaults'].

from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QWidget

# ── Color constants (match SETS main.py theme['defaults']) ───────────────────
BG     = '#1a1a1a'   # background
MBG    = '#242424'   # medium background
LBG    = '#404040'   # light background
ACCENT = '#c59129'   # SETS gold accent
FG     = '#eeeeee'   # foreground text
MFG    = '#bbbbbb'   # secondary text / hints
BC     = '#888888'   # border color

# Semantic colors — used for review list item states and status labels.
# These intentionally differ from the UI chrome so users can read status at a glance.
C_CONFIRMED  = '#7effc8'   # green  — confirmed annotation / annotated file
C_ERROR      = '#ff5555'   # red    — unmatched / error
C_CONF_HIGH  = '#ffaaaa'   # soft red — high-confidence pending
C_CONF_MED   = '#ff8888'   # medium red — medium-confidence pending
C_WARNING    = '#e8c060'   # amber  — warnings / attention labels
C_SUCCESS    = '#7effc8'   # green  — training complete
C_FAILURE    = '#ff7e7e'   # red    — training failed

# ── Button style helpers ─────────────────────────────────────────────────────

def primary_btn_style() -> str:
    """Heavy action button — gold background, dark text (matches SETS heavy_button)."""
    return (
        f'QPushButton {{'
        f'background:{ACCENT};color:#1a1a1a;'
        f'border:1px solid {ACCENT};border-radius:3px;'
        f'padding:5px 14px;font-weight:bold;}}'
        f'QPushButton:hover{{background:#d4a030;}}'
        f'QPushButton:disabled{{background:{LBG};color:{BC};border-color:{LBG};}}'
    )

def secondary_btn_style(checked_border: bool = False) -> str:
    """Regular button — transparent background, gold border (matches SETS button)."""
    checked = f'QPushButton:checked{{background:{MBG};border:2px solid {ACCENT};}}' if checked_border else ''
    return (
        f'QPushButton {{'
        f'background:transparent;color:{FG};'
        f'border:1px solid {ACCENT};border-radius:3px;padding:4px 10px;}}'
        f'QPushButton:hover{{border-color:{BC};}}'
        f'QPushButton:disabled{{color:{BC};border-color:{LBG};}}'
        + checked
    )

def warning_btn_style(checked_border: bool = False) -> str:
    """Button with amber/warning accent — for Add BBox and similar attention actions."""
    checked = f'QPushButton:checked{{background:{MBG};border:2px solid {C_WARNING};}}' if checked_border else ''
    return (
        f'QPushButton {{'
        f'background:transparent;color:{C_WARNING};'
        f'border:1px solid {C_WARNING};border-radius:3px;padding:4px 8px;}}'
        f'QPushButton:hover{{background:{MBG};}}'
        + checked
    )

def danger_btn_style() -> str:
    """Destructive action button — red accent."""
    return (
        f'QPushButton {{'
        f'background:transparent;color:{C_CONF_HIGH};'
        f'border:1px solid #884040;border-radius:3px;padding:4px 8px;}}'
        f'QPushButton:hover{{background:#2a1a1a;}}'
    )

# ── Global stylesheet ────────────────────────────────────────────────────────

WARP_QSS = f"""
QWidget {{
    background-color: {BG};
    color: {FG};
    font-size: 11px;
}}
QMainWindow, QDialog {{
    background-color: {BG};
}}
QGroupBox {{
    color: {FG};
    border: 1px solid {BC};
    border-radius: 3px;
    margin-top: 8px;
    padding-top: 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    color: {MFG};
}}
QLabel {{
    background-color: transparent;
    color: {FG};
}}
QPushButton {{
    background-color: transparent;
    color: {FG};
    border: 1px solid {ACCENT};
    border-radius: 3px;
    padding: 4px 10px;
}}
QPushButton:hover {{
    border-color: {BC};
}}
QPushButton:disabled {{
    color: {BC};
    border-color: {LBG};
}}
QPushButton:checked {{
    background-color: {MBG};
    border: 2px solid {ACCENT};
}}
QListWidget {{
    background-color: {MBG};
    color: {FG};
    border: 1px solid {BC};
    border-radius: 2px;
    outline: 0;
}}
QListWidget::item:selected {{
    background-color: {LBG};
}}
QListWidget::item:hover {{
    background-color: {LBG};
}}
QLineEdit {{
    background-color: {MBG};
    color: {FG};
    border: 1px solid {BC};
    border-radius: 2px;
    padding: 2px 5px;
    selection-background-color: {LBG};
}}
QLineEdit:focus {{
    border-color: {ACCENT};
}}
QPlainTextEdit {{
    background-color: {MBG};
    color: {FG};
    border: 1px solid {BC};
    border-radius: 2px;
    padding: 3px;
    selection-background-color: {LBG};
}}
QPlainTextEdit:focus {{
    border-color: {ACCENT};
}}
QComboBox {{
    background-color: {BG};
    color: {FG};
    border: 1px solid {BC};
    border-radius: 2px;
    padding: 2px 5px;
}}
QComboBox QAbstractItemView {{
    background-color: {MBG};
    color: {FG};
    border: 1px solid {BC};
    selection-background-color: {LBG};
    outline: 0;
}}
QComboBox::drop-down {{
    border-style: none;
}}
QRadioButton {{
    color: {FG};
    spacing: 6px;
}}
QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BC};
    border-radius: 7px;
    background-color: {LBG};
}}
QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QRadioButton::indicator:hover {{
    border-color: {ACCENT};
}}
QCheckBox {{
    color: {FG};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BC};
    border-radius: 2px;
    background-color: {LBG};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox::indicator:hover {{
    border-color: {ACCENT};
}}
QDoubleSpinBox, QSpinBox {{
    background-color: {MBG};
    color: {FG};
    border: 1px solid {BC};
    border-radius: 2px;
    padding: 2px;
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {{
    background-color: {LBG};
    border: none;
    width: 16px;
}}
QScrollArea {{
    background-color: {BG};
    border: none;
}}
QScrollBar:vertical {{
    background: none;
    border: none;
    width: 8px;
    margin: 0;
}}
QScrollBar:horizontal {{
    background: none;
    border: none;
    height: 8px;
    margin: 0;
}}
QScrollBar::handle {{
    background-color: rgba(100,100,100,0.75);
    border-radius: 4px;
    border: none;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}
QProgressBar {{
    background-color: {LBG};
    color: {FG};
    border: none;
    border-radius: 3px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 3px;
}}
QSplitter::handle {{
    background-color: {LBG};
}}
QStatusBar {{
    background-color: {MBG};
    color: {MFG};
    border-top: 1px solid {BC};
}}
QToolBar {{
    background-color: {MBG};
    border-bottom: 1px solid {BC};
    spacing: 4px;
    padding: 2px;
}}
QMenuBar {{
    background-color: {MBG};
    color: {FG};
}}
QMenuBar::item:selected {{
    background-color: {LBG};
}}
QMenu {{
    background-color: {MBG};
    color: {FG};
    border: 1px solid {BC};
    border-radius: 2px;
}}
QMenu::item:selected {{
    background-color: {LBG};
    border: 1px solid {ACCENT};
}}
QFrame[frameShape="4"],
QFrame[frameShape="5"] {{
    color: {BC};
    background-color: {BC};
    border: none;
    max-height: 1px;
}}
QToolTip {{
    background-color: {MBG};
    color: {FG};
    border: 1px solid {LBG};
    padding: 2px;
}}
"""


def apply_dark_style(widget: QWidget) -> None:
    """Apply the SETS-matching dark theme to a widget and all its children."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(BG))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(FG))
    palette.setColor(QPalette.ColorRole.Base,            QColor(MBG))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(LBG))
    palette.setColor(QPalette.ColorRole.Text,            QColor(FG))
    palette.setColor(QPalette.ColorRole.BrightText,      QColor(FG))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(FG))
    palette.setColor(QPalette.ColorRole.Button,          QColor(MBG))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(LBG))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(FG))
    palette.setColor(QPalette.ColorRole.ToolTipBase,     QColor(MBG))
    palette.setColor(QPalette.ColorRole.ToolTipText,     QColor(FG))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(MFG))
    widget.setPalette(palette)
    widget.setStyleSheet(WARP_QSS)
