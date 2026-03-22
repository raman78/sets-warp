# warp/updater.py
#
# Background update checker for SETS-WARP.
#
# Checks GitHub Releases API once per session (in a daemon thread started 5 s
# after app launch).  If a newer release tag is found, shows a Qt dialog
# asking the user whether to update now or later.
#
# Two update paths (auto-detected):
#   git install  — runs `git pull` then restarts the process
#   zip install  — downloads the release zip, extracts it in-place, then restarts
#
# Cross-platform: Linux, macOS, Windows.

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
WARP_VERSION = '1.0b'
GITHUB_REPO  = 'raman78/sets-warp'
API_URL      = f'https://api.github.com/repos/{GITHUB_REPO}/releases/latest'
TIMEOUT      = 8   # seconds for the API request


# ── Public entry point ─────────────────────────────────────────────────────────

def schedule_update_check(sets_app) -> None:
    """
    Start a background update check.
    Called once from inject_warp_buttons() after the app is ready.
    The actual network call happens in a daemon thread so it never blocks the UI.
    """
    threading.Thread(
        target=_check_worker,
        args=(sets_app,),
        daemon=True,
        name='warp-update-check',
    ).start()


# ── Background worker ──────────────────────────────────────────────────────────

def _check_worker(sets_app) -> None:
    try:
        import json
        import urllib.request

        req = urllib.request.Request(
            API_URL,
            headers={
                'User-Agent': f'SETS-WARP/{WARP_VERSION}',
                'Accept': 'application/vnd.github+json',
            },
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        tag = data.get('tag_name', '').lstrip('v')
        if not tag:
            return
        if tag == WARP_VERSION:
            log.debug(f'WARP updater: up to date (v{WARP_VERSION})')
            return

        notes_raw = data.get('body', '') or ''
        notes = '\n'.join(notes_raw.splitlines()[:10]).strip()

        log.info(f'WARP updater: new release v{tag} available (current v{WARP_VERSION})')

        # Schedule dialog on the Qt main thread
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: _show_update_dialog(sets_app, tag, notes))

    except Exception as e:
        log.debug(f'WARP updater: check failed ({e})')


# ── Dialog ─────────────────────────────────────────────────────────────────────

def _show_update_dialog(sets_app, new_tag: str, notes: str) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QDialog, QDialogButtonBox, QLabel, QTextEdit, QVBoxLayout)

    parent = getattr(sets_app, 'window', None)
    dlg = QDialog(parent)
    dlg.setWindowTitle('SETS-WARP Update Available')
    dlg.setMinimumWidth(500)

    layout = QVBoxLayout(dlg)
    layout.setSpacing(10)

    layout.addWidget(QLabel(
        f'<b>New version available: v{new_tag}</b><br>'
        f'Current version: v{WARP_VERSION}'
    ))

    if notes:
        te = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText(notes)
        te.setFixedHeight(150)
        layout.addWidget(te)

    if _is_git_install():
        hint = 'The app will run <b>git pull</b> and restart automatically.'
    else:
        hint = 'The new release will be downloaded and extracted, then the app will restart.'
    layout.addWidget(QLabel(hint))

    bb = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    bb.button(QDialogButtonBox.StandardButton.Ok).setText('Update now')
    bb.button(QDialogButtonBox.StandardButton.Cancel).setText('Later')
    bb.accepted.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)
    layout.addWidget(bb)

    if dlg.exec() != QDialog.DialogCode.Accepted:
        return

    if _is_git_install():
        _do_git_update(sets_app)
    else:
        _do_zip_update(sets_app, new_tag)


# ── Update methods ─────────────────────────────────────────────────────────────

def _is_git_install() -> bool:
    return (_repo_root() / '.git').exists()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _do_git_update(sets_app) -> None:
    from PySide6.QtWidgets import QMessageBox
    parent = getattr(sets_app, 'window', None)
    try:
        result = subprocess.run(
            ['git', 'pull'],
            cwd=str(_repo_root()),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            QMessageBox.warning(
                parent, 'Update failed',
                f'git pull returned an error:\n\n{result.stderr.strip()}')
            return
    except FileNotFoundError:
        QMessageBox.warning(
            parent, 'Update failed',
            'git is not available in PATH.\n'
            'Please run "git pull" manually in the SETS-WARP folder.')
        return
    except Exception as e:
        QMessageBox.warning(parent, 'Update failed', str(e))
        return

    _restart()


def _do_zip_update(sets_app, new_tag: str) -> None:
    import shutil
    import tempfile
    import urllib.request
    import zipfile
    from PySide6.QtWidgets import QMessageBox

    parent = getattr(sets_app, 'window', None)
    zip_url = (f'https://github.com/{GITHUB_REPO}'
               f'/archive/refs/tags/v{new_tag}.zip')
    log.info(f'WARP updater: downloading {zip_url}')
    try:
        with urllib.request.urlopen(zip_url, timeout=120) as resp:
            tmp = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
            shutil.copyfileobj(resp, tmp)
            tmp.close()

        root = _repo_root()
        with zipfile.ZipFile(tmp.name) as zf:
            names = zf.namelist()
            # Top-level folder inside zip: e.g. "sets-warp-1.0b/"
            top = names[0].rstrip('/')
            prefix = top + '/'
            for member in names:
                if not member.startswith(prefix):
                    continue
                rel = member[len(prefix):]
                if not rel:
                    continue
                dest = root / rel
                if member.endswith('/'):
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(dest, 'wb') as dst:
                        shutil.copyfileobj(src, dst)

        os.unlink(tmp.name)
        log.info('WARP updater: extraction complete')

    except Exception as e:
        QMessageBox.warning(parent, 'Update failed', str(e))
        return

    _restart()


def _restart() -> None:
    """Restart the process in-place. Cross-platform."""
    log.info('WARP updater: restarting...')
    if sys.platform == 'win32':
        # os.execv can be unreliable on Windows — spawn new process and exit
        subprocess.Popen([sys.executable] + sys.argv)
        sys.exit(0)
    else:
        os.execv(sys.executable, [sys.executable] + sys.argv)
