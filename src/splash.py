"""
Splash screen helpers.

Progress and status text are updated via a QTimer polling a shared state dict —
no Qt widget is ever touched from a worker thread.

Layout of the splash screen (set up in app.py):
  row 1 — banner image
  row 2 — loading_label: phase text + ETA  e.g. "Downloading Icons (ETA: 12s)"
  row 3 — progress_bar  (centre column, 70% width)
  row 4 — progress_detail: file counter  e.g. "234 / 3 500"
"""

from time import monotonic
from PySide6.QtCore import QTimer


# Shared state written by worker thread (plain Python dict — GIL-safe)
_state = {
    'text': '',       # phase description, without counter/ETA
    'current': 0,
    'total': 0,
    'hidden': True,
    'failed_text': '',  # shown after sync if any files permanently failed
}

# Timer that polls _state and updates widgets — runs only in MainThread
_timer: QTimer | None = None

# ETA tracking — reset each time the bar becomes visible
_eta = {
    'start': 0.0,
    'start_current': 0,
    'last_total': 0,
}


def _format_eta(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f'{s}s'
    return f'{s // 60}m {s % 60:02d}s'


def _start_poll_timer(self):
    global _timer
    if _timer is not None:
        return
    _timer = QTimer()
    _timer.setInterval(50)
    _timer.timeout.connect(lambda: _poll(self))
    _timer.start()


def _poll(self):
    """Called every 50 ms in MainThread. Reads _state and updates widgets."""
    text    = _state['text']
    current = _state['current']
    total   = _state['total']
    hidden  = _state['hidden']

    bar    = self.widgets.progress_bar
    detail = self.widgets.progress_detail
    lbl    = self.widgets.loading_label

    if hidden or total <= 0:
        if bar is not None and bar.isVisible():
            bar.hide()
        if detail is not None and detail.isVisible():
            detail.hide()
        retry_frame = getattr(self.widgets, 'retry_frame', None)
        if retry_frame is not None and retry_frame.isVisible():
            retry_frame.hide()
        if lbl is not None and text and lbl.text() != text:
            lbl.setText(text)
        return

    # ── bar became visible or total changed → reset ETA baseline ─────────
    if total != _eta['last_total']:
        _eta['start'] = monotonic()
        _eta['start_current'] = current
        _eta['last_total'] = total

    # ── compute ETA ───────────────────────────────────────────────────────
    eta_str = ''
    done = current - _eta['start_current']
    remaining = total - current
    elapsed = monotonic() - _eta['start']
    if done > 0 and remaining > 0:
        rate = done / elapsed          # files per second
        eta_sec = remaining / rate
        eta_str = f' (ETA: {_format_eta(eta_sec)})'

    # ── update status label: phase text + ETA, no counter ────────────────
    label_text = f'{text}{eta_str}'
    if lbl is not None and lbl.text() != label_text:
        lbl.setText(label_text)

    # ── update progress bar ───────────────────────────────────────────────
    bar.setRange(0, total)
    bar.setValue(current)
    if not bar.isVisible():
        bar.show()

    # ── update file counter below bar ─────────────────────────────────────
    detail_text = f'{current:,} / {total:,}'
    if detail is not None:
        if detail.text() != detail_text:
            detail.setText(detail_text)
        if not detail.isVisible():
            detail.show()

    # ── failed summary box — shown if any files permanently failed ───────
    retry_lbl   = getattr(self.widgets, 'retry_label', None)
    retry_frame = getattr(self.widgets, 'retry_frame', None)
    failed_text = _state.get('failed_text', '')
    if retry_lbl is not None and retry_lbl.text() != failed_text:
        retry_lbl.setText(failed_text)
    if retry_frame is not None:
        if failed_text and not retry_frame.isVisible():
            retry_frame.show()
        elif not failed_text and retry_frame.isVisible():
            retry_frame.hide()


# ── public API (called from MainThread) ───────────────────────────────────────

def enter_splash(self):
    """Shows splash screen."""
    _state['text'] = 'Loading...'
    _state['hidden'] = True
    _state['total'] = 0
    _state['failed_text'] = ''
    _eta['last_total'] = 0
    self.widgets.loading_label.setText('Loading...')
    if self.widgets.progress_bar is not None:
        self.widgets.progress_bar.hide()
    if self.widgets.progress_detail is not None:
        self.widgets.progress_detail.hide()
    self.widgets.splash_tabber.setCurrentIndex(1)
    _start_poll_timer(self)


def exit_splash(self):
    """Leaves splash screen."""
    global _timer
    if _timer is not None:
        _timer.stop()
        _timer = None
    if self.widgets.progress_bar is not None:
        self.widgets.progress_bar.hide()
    if self.widgets.progress_detail is not None:
        self.widgets.progress_detail.hide()
    self.widgets.splash_tabber.setCurrentIndex(0)


# ── called from worker thread (write to _state only — no Qt) ──────────────────

def splash_text(self, new_text: str):
    """Update status text — safe to call from any thread."""
    _state['text'] = new_text


def splash_progress(self, current: int, total: int):
    """
    Update progress bar state — safe to call from any thread.
    Call (0, 0) to hide the bar.
    """
    if total <= 0:
        _state['hidden'] = True
        _state['total'] = 0
        _state['current'] = 0
    else:
        _state['hidden'] = False
        _state['total'] = total
        _state['current'] = current
