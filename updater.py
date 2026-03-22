# updater.py
#
# Background update checker for SETS-WARP.
#
# Checks GitHub Releases for a newer version of SETS-WARP and offers to update
# in-place.  Current version is read from the nearest git tag (git describe),
# so after `git pull` the version advances automatically — no code change needed.
# VERSION constant is a fallback only (zip installs without git).
#
# Two update paths (auto-detected):
#   git install  — runs `git pull` then restarts the process
#   zip install  — downloads the release zip, extracts it in-place, then restarts
#
# User preferences (stored in QSettings):
#   warp_update/enabled          — bool, default True  (Autoupdate checkbox)
#   warp_update/snoozed_version  — str  (tag snoozed by "Don't remind me")
#
# Cross-platform: Linux, macOS, Windows.

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

try:
    from src.setsdebug import log
except Exception:
    log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
VERSION     = '1.2b'           # fallback for zip installs / no git
GITHUB_REPO = 'raman78/sets-warp'
API_URL      = f'https://api.github.com/repos/{GITHUB_REPO}/releases/latest'
TIMEOUT      = 8               # seconds for the API request


# ── Public entry point ─────────────────────────────────────────────────────────

def schedule_update_check(sets_app) -> None:
    """
    Start a background update check 8 s after app launch.
    Skips silently if autoupdate is disabled in Settings.
    """
    if not is_autoupdate_enabled(sets_app):
        log.debug('WARP updater: autoupdate disabled, skipping check')
        return
    threading.Thread(
        target=_check_worker,
        args=(sets_app,),
        daemon=True,
        name='warp-update-check',
    ).start()


def is_autoupdate_enabled(sets_app) -> bool:
    """Return True if autoupdate is enabled (default: True)."""
    try:
        return bool(sets_app.settings.value('warp_update/enabled', True, type=bool))
    except Exception:
        return True


# ── Version detection ──────────────────────────────────────────────────────────

def get_current_version() -> str:
    """
    Return the current installed version from the nearest git tag.
    Falls back to VERSION constant when git is unavailable.
    """
    try:
        result = subprocess.run(
            ['git', 'describe', '--tags', '--abbrev=0'],
            cwd=str(_repo_root()),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().lstrip('v')
    except Exception:
        pass
    return VERSION


# ── Background worker ──────────────────────────────────────────────────────────

def _check_worker(sets_app) -> None:
    try:
        import json
        import urllib.request

        current = get_current_version()

        req = urllib.request.Request(
            API_URL,
            headers={
                'User-Agent': f'SETS-WARP/{current}',
                'Accept': 'application/vnd.github+json',
            },
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        remote_tag = data.get('tag_name', '').lstrip('v')
        if not remote_tag:
            return
        if remote_tag == current:
            log.debug(f'WARP updater: up to date (v{current})')
            return

        # Check snooze
        try:
            snoozed = sets_app.settings.value('warp_update/snoozed_version', '')
            if snoozed == remote_tag:
                log.debug(f'WARP updater: v{remote_tag} snoozed by user, skipping')
                return
        except Exception:
            pass

        notes_raw = data.get('body', '') or ''
        notes = '\n'.join(notes_raw.splitlines()[:10]).strip()

        log.info(
            f'WARP updater: new release v{remote_tag} available (current v{current})')

        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        QTimer.singleShot(
            0, QApplication.instance(),
            lambda: _show_update_dialog(sets_app, current, remote_tag, notes))

    except Exception as e:
        log.warning(f'WARP updater: check failed ({e})')


# ── Dialog ─────────────────────────────────────────────────────────────────────

def _show_update_dialog(sets_app, current: str, new_tag: str, notes: str) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QCheckBox, QDialog, QDialogButtonBox, QLabel, QTextEdit, QVBoxLayout)

    parent = getattr(sets_app, 'window', None)
    dlg = QDialog(parent)
    dlg.setWindowTitle('SETS-WARP Update')
    dlg.setMinimumWidth(500)

    layout = QVBoxLayout(dlg)
    layout.setSpacing(10)

    layout.addWidget(QLabel(
        f'<b>New version available: v{new_tag}</b><br>'
        f'Current version: v{current}'
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
        hint = ('The new release will be downloaded and extracted, '
                'then the app will restart.')
    layout.addWidget(QLabel(hint))

    snooze_cb = QCheckBox(f"Don't remind me for v{new_tag}")
    layout.addWidget(snooze_cb)

    bb = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    bb.button(QDialogButtonBox.StandardButton.Ok).setText('Update now')
    bb.button(QDialogButtonBox.StandardButton.Cancel).setText('Later')
    bb.accepted.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)
    layout.addWidget(bb)

    accepted = dlg.exec() == QDialog.DialogCode.Accepted

    if not accepted:
        if snooze_cb.isChecked():
            try:
                sets_app.settings.setValue('warp_update/snoozed_version', new_tag)
                log.info(f'WARP updater: snoozed notifications for v{new_tag}')
            except Exception:
                pass
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
        subprocess.Popen([sys.executable] + sys.argv)
        sys.exit(0)
    else:
        os.execv(sys.executable, [sys.executable] + sys.argv)
