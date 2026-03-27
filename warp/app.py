import sys
import os
from pathlib import Path

from src.app import SETS
from src.constants import ALEFT, AVCENTER
from src.widgets import GridLayout
from warp.warp_button import inject_warp_buttons

# WARP — install mode (sets vs sets_warp)
_MODE_FILE = Path(__file__).parent.parent / '.config' / 'install_mode.txt'


def _get_install_mode() -> str:
    try:
        return _MODE_FILE.read_text().strip()
    except Exception:
        return 'sets_warp'


def _save_install_mode(mode: str) -> None:
    try:
        _MODE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _MODE_FILE.write_text(mode)
    except Exception:
        pass


_WARP_AVAILABLE = _get_install_mode() != 'sets'


class WarpSETS(SETS):
    # WARP-specific version strings
    version = '2026.03b070'
    __version__ = '2.4'

    def __init__(self, theme, args, path, config, versions):
        warp_versions = (self.__version__, self.version)
        super().__init__(theme, args, path, config, warp_versions)
        # Set Windows taskbar App User Model ID for correct grouping/icon
        if sys.platform == 'win32':
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('SETS.WARP.1')
            except Exception:
                pass

    def create_main_window(self, argv=[]):
        app, window = super().create_main_window(argv)
        app.setApplicationName('sets-warp')
        app.setOrganizationName('SETS-WARP')
        return app, window

    def setup_main_layout(self):
        super().setup_main_layout()
        if _WARP_AVAILABLE:
            inject_warp_buttons(self, self.widgets.menu_layout)

    def setup_settings_frame(self):
        super().setup_settings_frame()
        self._add_warp_settings_sections()

    def _add_warp_settings_sections(self):
        scroll_layout = self.widgets.settings_scroll_layout
        isp = self.theme['defaults']['isp'] * self.config['ui_scale']

        # WARP Updates section
        try:
            from updater import get_current_version
            sep_wu = self.create_frame()
            sep_wu.setFixedHeight(isp)
            scroll_layout.addWidget(sep_wu)
            wu_header = self.create_label('SETS-WARP Updates:', 'label_heading')
            scroll_layout.addWidget(wu_header, alignment=ALEFT)
            sec_wu = GridLayout(spacing=isp)
            sec_wu.setColumnMinimumWidth(1, 3 * isp)
            sec_wu.setColumnStretch(3, 1)
            autoupdate_label = self.create_label('Check for updates automatically')
            sec_wu.addWidget(autoupdate_label, 0, 0, alignment=ALEFT)
            autoupdate_cb = self.create_checkbox()
            autoupdate_cb.setChecked(
                bool(self.settings.value('warp_update/enabled', True, type=bool)))
            autoupdate_cb.checkStateChanged.connect(
                lambda state: self.settings.setValue(
                    'warp_update/enabled',
                    state == autoupdate_cb.checkState().Checked))
            sec_wu.addWidget(autoupdate_cb, 0, 2, alignment=ALEFT | AVCENTER)
            current_ver = get_current_version()
            ver_label = self.create_label(
                f'Installed version: v{current_ver}', 'hint_label')
            sec_wu.addWidget(ver_label, 1, 0, 1, 3, alignment=ALEFT)
            scroll_layout.addLayout(sec_wu)
        except Exception:
            pass

        # Installation section
        sep_inst = self.create_frame()
        sep_inst.setFixedHeight(isp)
        scroll_layout.addWidget(sep_inst)
        inst_header = self.create_label('Installation:', 'label_heading')
        scroll_layout.addWidget(inst_header, alignment=ALEFT)
        sec_inst = GridLayout(spacing=isp)
        sec_inst.setColumnMinimumWidth(1, 3 * isp)
        sec_inst.setColumnStretch(3, 1)
        warp_inst_label = self.create_label('SETS + WARP (screenshot recognition & ML training)')
        sec_inst.addWidget(warp_inst_label, 0, 0, alignment=ALEFT)
        warp_inst_cb = self.create_checkbox()
        warp_inst_cb.setChecked(_get_install_mode() == 'sets_warp')
        warp_inst_hint = self.create_label(
            'Adds ~10 GB of ML dependencies. Uncheck to switch to SETS-only (~3 GB). '
            'Requires restart — the installer runs automatically on next launch.',
            'hint_label')
        warp_inst_hint.setWordWrap(True)
        sec_inst.addWidget(warp_inst_cb, 0, 2, alignment=ALEFT | AVCENTER)
        sec_inst.addWidget(warp_inst_hint, 1, 0, 1, 3, alignment=ALEFT)

        def _on_warp_install_toggle(state):
            from PySide6.QtWidgets import QMessageBox
            import subprocess
            new_mode = 'sets_warp' if warp_inst_cb.isChecked() else 'sets'
            current_mode = _get_install_mode()
            if new_mode == current_mode:
                return
            if new_mode == 'sets_warp':
                title = 'Install SETS + WARP?'
                msg = (
                    'This will switch to the full SETS + WARP installation.\n\n'
                    'On restart the installer will download ~10 GB of ML dependencies '
                    '(PyTorch, EasyOCR, OpenCV).\n\n'
                    'Continue and restart now?'
                )
            else:
                title = 'Switch to SETS-only?'
                msg = (
                    'This will switch to SETS-only mode.\n\n'
                    'WARP packages (~7 GB) will be removed from the virtual environment '
                    'automatically on restart.\n\n'
                    'Continue and restart now?'
                )
            reply = QMessageBox.question(
                self.window, title, msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                warp_inst_cb.blockSignals(True)
                warp_inst_cb.setChecked(current_mode == 'sets_warp')
                warp_inst_cb.blockSignals(False)
                return
            _save_install_mode(new_mode)
            _bootstrap = str(Path(__file__).resolve().parent.parent / 'bootstrap.py')
            if sys.platform == 'win32':
                subprocess.Popen([sys.executable, _bootstrap])
                sys.exit(0)
            else:
                os.execv(sys.executable, [sys.executable, _bootstrap])

        warp_inst_cb.checkStateChanged.connect(_on_warp_install_toggle)
        scroll_layout.addLayout(sec_inst)

        # Uninstall section
        sep = self.create_frame()
        sep.setFixedHeight(isp)
        scroll_layout.addWidget(sep)
        uninstall_header = self.create_label('Uninstall:', 'label_heading')
        scroll_layout.addWidget(uninstall_header, alignment=ALEFT)
        sec_uninst = GridLayout(spacing=isp)
        sec_uninst.setColumnMinimumWidth(1, 3 * isp)
        sec_uninst.setColumnStretch(3, 1)
        uninstall_button = self.create_button('Uninstall SETS-WARP')
        uninstall_button.clicked.connect(self._on_uninstall)
        sec_uninst.addWidget(uninstall_button, 0, 0, alignment=ALEFT)
        uninstall_label = self.create_label(
            'Removes the desktop entry and permanently deletes this installation folder.',
            'hint_label')
        uninstall_label.setWordWrap(True)
        sec_uninst.addWidget(uninstall_label, 0, 2, alignment=ALEFT)
        scroll_layout.addLayout(sec_uninst)

        # Force the scroll frame to recalculate its size after all sections are added
        self.widgets.settings_scroll_frame.adjustSize()

    def _on_uninstall(self):
        """Confirm then schedule full uninstall: desktop entry + installation directory."""
        from PySide6.QtWidgets import QMessageBox
        install_dir = str(Path(__file__).resolve().parent.parent)
        reply = QMessageBox.warning(
            self.window,
            'Uninstall SETS-WARP',
            f'This will permanently delete:\n\n'
            f'  {install_dir}\n\n'
            f'and remove the desktop entry.\n\n'
            f'This cannot be undone. Continue?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._run_uninstall(install_dir)

    @staticmethod
    def _run_uninstall(install_dir: str):
        """Write a cleanup shell script to /tmp and exec it, then exit the app."""
        import tempfile, stat
        apps_dir = Path.home() / '.local' / 'share' / 'applications'
        desktop_files = []
        for df in apps_dir.glob('sets-warp*.desktop'):
            try:
                if install_dir in df.read_text():
                    desktop_files.append(str(df))
            except Exception:
                pass

        remove_desktops = '\n'.join(f'rm -f "{d}"' for d in desktop_files)

        script = f"""#!/bin/sh
# SETS-WARP auto-uninstall — generated, runs once
sleep 1
{remove_desktops}
# Refresh DE caches
gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
rm -rf "{install_dir}"
rm -- "$0"
"""
        fd, path = tempfile.mkstemp(suffix='.sh', prefix='sets_warp_uninstall_')
        try:
            os.write(fd, script.encode())
            os.close(fd)
            os.chmod(path, stat.S_IRWXU)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(None, 'Uninstall error', f'Could not write uninstall script:\n{e}')
            return

        import subprocess
        if sys.platform != 'win32':
            subprocess.Popen(['/bin/sh', path], close_fds=True, start_new_session=True)
        sys.exit(0)
