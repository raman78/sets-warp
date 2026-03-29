# warp/warp_button.py
# Injects WARP import button and WARP CORE trainer button into SETS menu bar.
#
# Integration: called once from app.py setup_main_layout() after the right_button_group
# is added. The function inject_warp_buttons() receives the SETS `self` and the
# menu_layout (GridLayout at row 0).
#
# Uses PySide6 (same as SETS).

from __future__ import annotations

import logging
from pathlib import Path
from PySide6.QtWidgets import QHBoxLayout, QPushButton
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QFont, QIcon, QPixmap

log = logging.getLogger(__name__)


def inject_warp_buttons(sets_app, menu_layout) -> None:
    """
    Adds WARP and WARP CORE buttons to the SETS top menu bar.
    Call this from setup_main_layout() after menu_layout is fully populated.

    Args:
        sets_app:    The SETS application instance (self in app.py)
        menu_layout: The GridLayout at row 0 of content_layout (has 3 columns)
    """
    warp_layout = QHBoxLayout()
    warp_layout.setSpacing(4)
    warp_layout.setContentsMargins(0, 0, 0, 0)

    # ── WARP import button ────────────────────────────────────────────────
    btn_warp = QPushButton("  WARP")
    btn_warp.setToolTip(
        "Import a build from STO screenshots\n"
        "(Folder must contain screenshots of ONE build)"
    )
    btn_warp.setFixedHeight(28)
    _warp_icon_path = Path(__file__).resolve().parent.parent / 'local' / 'warp.jpg'
    if _warp_icon_path.exists():
        _pix = QPixmap(str(_warp_icon_path)).scaled(
            40, 24, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        btn_warp.setIcon(QIcon(_pix))
        btn_warp.setIconSize(QSize(40, 24))
    _apply_warp_style(btn_warp, color='#eeeeee', bg='#242424', border='#c59129')
    btn_warp.clicked.connect(lambda: _open_warp_dialog(sets_app))

    # ── WARP CORE trainer button ──────────────────────────────────────────
    btn_core = QPushButton("  WARP CORE")
    btn_core.setToolTip(
        "Open the WARP CORE ML Trainer\n"
        "Mycelial Harmonic Matter-Antimatter Core")
    btn_core.setFixedHeight(28)
    btn_core.setIconSize(QSize(24, 24))
    # Load warp core icon from local/ folder (relative to SETS root)
    _icon_path = Path(__file__).resolve().parent.parent / 'local' / 'warp_core_icon.png'
    if _icon_path.exists():
        _pix = QPixmap(str(_icon_path)).scaled(
            24, 24, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        btn_core.setIcon(QIcon(_pix))
    _apply_warp_style(btn_core, color='#eeeeee', bg='#242424', border='#a07820')
    btn_core.clicked.connect(lambda: _open_warp_core(sets_app))

    warp_layout.addWidget(btn_warp)
    warp_layout.addWidget(btn_core)

    # Insert into column 3 (new column to the right of Settings / Export)
    menu_layout.addLayout(warp_layout, 0, 3,
                          Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    # Store refs to avoid GC
    sets_app.widgets.warp_btn      = btn_warp
    sets_app.widgets.warp_core_btn = btn_core

    log.info('WARP buttons injected into menu bar')

    # Schedule background update check — fires 3 s after app is ready
    from updater import schedule_update_check
    QTimer.singleShot(3000, lambda: schedule_update_check(sets_app))



# ──────────────────────────────────────────────────────────────────────────────
# Dialog launchers
# ──────────────────────────────────────────────────────────────────────────────

def _open_warp_dialog(sets_app) -> None:
    """Opens the WARP import dialog."""
    from warp.warp_dialog import WarpDialog
    dlg = WarpDialog(sets_app=sets_app, parent=sets_app.window)
    dlg.exec()


def _open_warp_core(sets_app) -> None:
    """Opens or raises the WARP CORE trainer window (singleton)."""
    from warp.trainer.trainer_window import WarpCoreWindow
    win = getattr(sets_app, '_warp_core_window', None)
    if win is None or not win.isVisible():
        win = WarpCoreWindow(sets_app=sets_app)
        sets_app._warp_core_window = win
        win.showMaximized()
    else:
        win.raise_()
        win.activateWindow()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _apply_warp_style(btn: QPushButton, color: str, bg: str, border: str) -> None:
    f = QFont()
    f.setBold(True)
    btn.setFont(f)
    btn.setStyleSheet(
        f"QPushButton {{"
        f"  background-color: {bg}; color: {color};"
        f"  border: 1px solid {border}; border-radius: 3px; padding: 2px 10px;"
        f"}}"
        f"QPushButton:hover {{ background-color: {_lighten(bg)}; }}"
        f"QPushButton:pressed {{ background-color: {_darken(bg)}; }}"
    )


def _lighten(hex_color: str) -> str:
    """Very simple hex color lightener (+40 to each channel)."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return '#{:02x}{:02x}{:02x}'.format(
        min(255, r + 40), min(255, g + 40), min(255, b + 40))


def _darken(hex_color: str) -> str:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return '#{:02x}{:02x}{:02x}'.format(
        max(0, r - 20), max(0, g - 20), max(0, b - 20))
