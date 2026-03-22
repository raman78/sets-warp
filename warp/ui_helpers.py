# warp/ui_helpers.py
# Shared UI helper utilities for WARP dialogs.

from __future__ import annotations

import time as _time_mod

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel


def time_spent_counter(
    parent,
    prefix: str = 'Time: ',
    style: str = 'color:#bbbbbb;font-size:10px;',
) -> tuple[QLabel, QTimer]:
    """
    Create a self-updating elapsed-time label and QTimer pair.

    Returns (label, timer).
      - Call timer.start(1000) to begin counting from now.
      - Call timer.stop()  to freeze the display.

    Customise appearance for the whole app by changing the defaults here:
      prefix — text before the M:SS value  (e.g. 'Time: ', 'Elapsed: ')
      style  — Qt stylesheet string applied to the label
    """
    _start = [_time_mod.monotonic()]

    lbl = QLabel(f'{prefix}0:00', parent)
    lbl.setStyleSheet(style)

    def _tick():
        elapsed = int(_time_mod.monotonic() - _start[0])
        m, s = divmod(elapsed, 60)
        lbl.setText(f'{prefix}{m}:{s:02d}')

    timer = QTimer(parent)
    timer.timeout.connect(_tick)
    return lbl, timer
