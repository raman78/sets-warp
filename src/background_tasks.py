# src/background_tasks.py
# Generic periodic background task runner.
#
# Upstream proposable — no WARP or SETS-specific dependencies.
#
# Usage:
#   btm = BackgroundTaskManager()
#   btm.register(my_fn, interval_ms=10 * 60 * 1000, startup_delay_ms=15_000)
#   btm.start()   # hooks QApplication.aboutToQuit for clean shutdown

from __future__ import annotations

import logging
from typing import Callable

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

log = logging.getLogger(__name__)


class BackgroundTaskManager:
    """
    Manages a set of periodic background tasks (callables on QTimers).

    Call register() to add tasks, then start() once to activate all timers.
    Timers registered after start() are started immediately.
    stop_all() is connected to QApplication.aboutToQuit automatically.
    """

    def __init__(self) -> None:
        self._timers: list[QTimer] = []
        self._stop_hooks: list[Callable] = []
        self._started = False

    def register(
        self,
        fn: Callable,
        interval_ms: int,
        startup_delay_ms: int = 0,
    ) -> None:
        """
        Register a periodic task.

        Args:
            fn:               Callable to invoke on each tick.
            interval_ms:      Repeat interval in milliseconds.
            startup_delay_ms: One-shot delay before the first call (0 = wait for first tick).
        """
        t = QTimer()
        t.setInterval(interval_ms)
        t.timeout.connect(fn)
        self._timers.append(t)
        if self._started:
            t.start()
        if startup_delay_ms > 0:
            QTimer.singleShot(startup_delay_ms, fn)

    def on_stop(self, fn: Callable) -> None:
        """Register a callable to be called when stop_all() fires (e.g. worker.wait())."""
        self._stop_hooks.append(fn)

    def start(self) -> None:
        """Start all registered timers and hook into app quit signal."""
        self._started = True
        for t in self._timers:
            t.start()
        app = QApplication.instance()
        if app:
            try:
                app.aboutToQuit.connect(self.stop_all)
            except Exception:
                pass
        log.debug(f'BackgroundTaskManager: started {len(self._timers)} task(s)')

    def stop_all(self) -> None:
        """Stop all timers and run stop hooks. Called automatically on app quit."""
        for t in self._timers:
            t.stop()
        for hook in self._stop_hooks:
            try:
                hook()
            except Exception:
                pass
        log.debug(f'BackgroundTaskManager: stopped {len(self._timers)} task(s)')
