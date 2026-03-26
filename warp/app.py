import sys
import os
from pathlib import Path
from PySide6.QtWidgets import QApplication

from src.app import SETS
from warp.warp_button import inject_warp_buttons

# WARP — Screenshot-to-Build importer (optional; gracefully disabled if deps missing)
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

# Determine if WARP features are available based on install mode
_WARP_AVAILABLE = _get_install_mode() != 'sets'

class WarpSETS(SETS):
    # WARP Specific Versions
    version = '2026.03b070'
    __version__ = '2.4'

    def __init__(self, theme, args, path, config, versions):
        # Override versions with WARP values before calling super
        warp_versions = (self.__version__, self.version)
        super().__init__(theme, args, path, config, warp_versions)
        
        # Override application name and organization for WARP
        if sys.platform == 'win32':
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('SETS.WARP.1')
            except Exception:
                pass
        
        q_app = QApplication.instance()
        if q_app:
            q_app.setApplicationName('sets-warp')
            q_app.setOrganizationName('SETS-WARP')

    def setup_main_layout(self):
        super().setup_main_layout()
        # WARP import + WARP CORE trainer buttons
        # Note: We will need to make sure menu_layout is accessible. 
        # In this phase, we are just establishing the override.
        if _WARP_AVAILABLE:
            # This is a placeholder for actual injection which happens in Phase 2
            pass

    def setup_settings_frame(self):
        super().setup_settings_frame()
        # WARP settings sections will be moved here in Phase 2
