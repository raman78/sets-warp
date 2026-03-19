# warp/warp_dialog.py
# Multi-step WARP import dialog.
# Fully integrated with SETS cache, build data model, and UI patterns.
#
# Uses PySide6 (same as SETS).
# Accesses sets_app.cache.equipment, sets_app.cache.traits,
#           sets_app.cache.starship_traits, sets_app.cache.images
# Writes to  sets_app.build   via slot_equipment_item / slot_trait_item
#            sets_app.widgets via the same functions
# Then calls sets_app.autosave()

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QButtonGroup, QRadioButton, QGroupBox, QFileDialog,
    QProgressBar, QStackedWidget, QWidget, QScrollArea,
    QMessageBox, QSizePolicy, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QFont, QPixmap, QImage

from warp.warp_importer import WarpImporter, ImportResult, RecognisedItem

log = logging.getLogger(__name__)

# ── SETS build key mappings ────────────────────────────────────────────────────
# Maps WARP slot names → (SETS build_key, environment, is_equipment)
# Derived from SETS empty_build() and EQUIPMENT_TYPES in constants.py
SLOT_MAP = {
    # SPACE equipment
    'Fore Weapons':          ('fore_weapons',   'space', True),
    'Aft Weapons':           ('aft_weapons',    'space', True),
    'Experimental Weapon':   ('experimental',   'space', True),
    'Devices':               ('devices',        'space', True),
    'Hangars':               ('hangars',        'space', True),
    'Deflector':             ('deflector',      'space', True),
    'Sec-Def':               ('sec_def',        'space', True),
    'Engines':               ('engines',        'space', True),
    'Warp Core':             ('core',           'space', True),
    'Shield':                ('shield',         'space', True),
    'Universal Consoles':    ('uni_consoles',   'space', True),
    'Engineering Consoles':  ('eng_consoles',   'space', True),
    'Science Consoles':      ('sci_consoles',   'space', True),
    'Tactical Consoles':     ('tac_consoles',   'space', True),
    # SPACE traits
    'Personal Space Traits': ('traits',         'space', False),
    'Starship Traits':       ('starship_traits','space', False),
    'Reputation Traits':     ('rep_traits',     'space', False),
    'Active Rep Traits':     ('active_rep_traits','space',False),
    # GROUND equipment
    'Body Armor':            ('armor',          'ground', True),
    'EV Suit':               ('ev_suit',        'ground', True),
    'Personal Shield':       ('personal_shield','ground', True),
    'Weapons':               ('weapons',        'ground', True),
    'Kit':                   ('kit',            'ground', True),
    'Kit Modules':           ('kit_modules',    'ground', True),
    'Ground Devices':        ('ground_devices', 'ground', True),
    # GROUND traits
    'Personal Ground Traits':('traits',         'ground', False),
    'Ground Rep Traits':     ('rep_traits',     'ground', False),
}

BUILD_TYPES = [
    ('SPACE',         'Space Build',
     'Equipment, consoles, boffs and traits for a space loadout'),
    ('GROUND',        'Ground Build',
     'Equipment, boffs and traits for a ground loadout'),
    ('SPACE_SKILLS',  'Space Skills',
     'Space skill tree point allocation'),
    ('GROUND_SKILLS', 'Ground Skills',
     'Ground skill tree point allocation'),
]

_SETTINGS_KEY_LAST_DIR = 'warp/last_import_dir'

SCREENSHOT_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


# ── Background worker ──────────────────────────────────────────────────────────

class _ImportWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(object)   # ImportResult
    error    = Signal(str)

    def __init__(self, folder: Path, build_type: str, sets_app):
        super().__init__()
        self._folder     = folder
        self._build_type = build_type
        self._sets_app   = sets_app

    def run(self):
        try:
            importer = WarpImporter(
                build_type=self._build_type,
                sets_app=self._sets_app,
                progress_callback=lambda p, m: self.progress.emit(p, m),
            )
            importer.set_interrupt_check(self.isInterruptionRequested)
            result = importer.process_folder(self._folder)
            if not self.isInterruptionRequested():
                self.finished.emit(result)
        except Exception as exc:
            log.exception('WARP import worker error')
            self.error.emit(str(exc))


# ── Main dialog ────────────────────────────────────────────────────────────────

class WarpDialog(QDialog):
    """
    3-page import dialog:
      Page 0 — build type + folder selection
      Page 1 — progress
      Page 2 — review / per-item accept/reject
    """

    def __init__(self, sets_app, parent=None):
        super().__init__(parent)
        self._sets = sets_app
        self._settings = QSettings()
        last = self._settings.value(_SETTINGS_KEY_LAST_DIR, '')
        self._folder: Path | None = Path(last) if last and Path(last).is_dir() else None
        self._build_type = 'SPACE'
        self._worker: _ImportWorker | None = None
        self._import_result: ImportResult | None = None

        self.setWindowTitle('WARP — Screenshot Import')
        self.setMinimumWidth(580)
        self.setMinimumHeight(420)
        self.setModal(True)
        self._build_ui()
        if self._folder:
            self._folder_label.setText(str(self._folder))
            self._folder_label.setStyleSheet(
                'color:#eee;background:#111;border:1px solid #3a6a9c;'
                'border-radius:3px;padding:3px 7px;')

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 14)

        title = QLabel('⚡  WARP — Build Importer')
        title.setFont(QFont('', 13, QFont.Weight.Bold))
        title.setStyleSheet('color: #7ec8e3;')
        root.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet('color: #3a6a9c;')
        root.addWidget(sep)

        self._stack = QStackedWidget()
        root.addWidget(self._stack, 1)

        self._stack.addWidget(self._make_page_select())
        self._stack.addWidget(self._make_page_progress())

        # Footer
        footer = QHBoxLayout()
        self._btn_next   = QPushButton('Analyse & Import →')
        self._btn_cancel = QPushButton('Cancel')
        self._btn_next.setStyleSheet(
            'QPushButton {background:#1a5c3a;color:#7effc8;'
            'border:1px solid #3aac6a;border-radius:3px;padding:5px 14px;font-weight:bold;}'
            'QPushButton:hover{background:#2a8c5a;}')
        footer.addStretch()
        footer.addWidget(self._btn_cancel)
        footer.addWidget(self._btn_next)
        root.addLayout(footer)

        self._btn_cancel.clicked.connect(self._on_cancel)
        self._btn_next.clicked.connect(self._on_next)

    def _make_page_select(self) -> QWidget:
        page = QWidget()
        lay  = QVBoxLayout(page)
        lay.setSpacing(12)

        # Build type
        g1 = QGroupBox('1.  Select Build Type')
        g1_lay = QVBoxLayout(g1)
        self._type_bg = QButtonGroup(self)
        for i, (key, label, tip) in enumerate(BUILD_TYPES):
            rb = QRadioButton(label)
            rb.setToolTip(tip)
            rb.setProperty('build_key', key)
            if i == 0:
                rb.setChecked(True)
            g1_lay.addWidget(rb)
            self._type_bg.addButton(rb, i)
        lay.addWidget(g1)

        # Folder
        g2 = QGroupBox('2.  Select Screenshots Folder')
        g2_lay = QVBoxLayout(g2)
        notice = QLabel(
            '⚠  The folder should contain screenshots of ONE build only.\n'
            '   Multiple builds in the same folder will produce incorrect results.')
        notice.setWordWrap(True)
        notice.setStyleSheet('color:#e8c060;font-size:11px;')
        g2_lay.addWidget(notice)

        row = QHBoxLayout()
        self._folder_label = QLabel('No folder selected')
        self._folder_label.setStyleSheet(
            'color:#aaa;background:#111;border:1px solid #333;'
            'border-radius:3px;padding:3px 7px;')
        self._folder_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        btn_browse = QPushButton('Browse…')
        btn_browse.setFixedWidth(88)
        btn_browse.clicked.connect(self._browse_folder)
        row.addWidget(self._folder_label, 1)
        row.addWidget(btn_browse)
        g2_lay.addLayout(row)
        lay.addWidget(g2)
        lay.addStretch()
        return page

    def _make_page_progress(self) -> QWidget:
        page = QWidget()
        lay  = QVBoxLayout(page)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label = QLabel('Preparing…')
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setWordWrap(True)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        lay.addStretch()
        lay.addWidget(self._progress_label)
        lay.addSpacing(10)
        lay.addWidget(self._progress_bar)
        lay.addStretch()
        return page


    def _on_next(self):
        if self._stack.currentIndex() == 0:
            if not self._folder or not self._folder.is_dir():
                QMessageBox.warning(self, 'WARP', 'Please select a valid screenshots folder.')
                return
            checked = self._type_bg.checkedButton()
            self._build_type = checked.property('build_key') if checked else 'SPACE'
            self._start_import()

    # ── Import pipeline ────────────────────────────────────────────────────

    def _start_import(self):
        self._stack.setCurrentIndex(1)
        self._btn_next.setEnabled(False)
        self._progress_bar.setValue(0)
        self._progress_label.setText('Starting…')

        self._worker = _ImportWorker(self._folder, self._build_type, self._sets)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_import_done)
        self._worker.error.connect(self._on_import_error)
        self._worker.start()

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(2000)
        self.reject()

    def _on_progress(self, pct: int, msg: str):
        self._progress_bar.setValue(pct)
        self._progress_label.setText(msg)

    def _on_import_done(self, result: ImportResult):
        self._import_result = result
        self._apply_to_sets()
        n_detected   = len(result.items)
        n_imported   = sum(1 for i in result.items if i.name)
        n_unmatched  = n_detected - n_imported
        ship_info    = result.ship_name or 'Unknown ship'
        msg = (
            f"Detected {n_detected} item slot(s)\n"
            f"Identified and imported: {n_imported}\n"
            f"Unmatched (skipped): {n_unmatched}\n\n"
            f"Ship: {ship_info} {result.ship_type} {result.ship_tier}"
        )
        QMessageBox.information(self, "WARP — Import Complete", msg)
        self.accept()

    def _on_import_error(self, msg: str):
        self._stack.setCurrentIndex(0)
        self._btn_next.setEnabled(True)
        QMessageBox.critical(self, 'WARP — Analysis Error', msg)


    # ── Apply to SETS ──────────────────────────────────────────────────────

    def _apply_to_sets(self):
        """
        Writes all recognised items directly into the SETS build (no review step).
        Only items with a non-empty name are imported.
        """
        from src.buildupdater import slot_equipment_item, slot_trait_item

        r = self._import_result
        if r is None:
            return

        # Set ship info if recognised
        # Set ship name, ship selection and tier from recognised result
        if r.ship_name or r.ship_type or r.ship_tier:
            try:
                build   = self._sets.build['space']
                widgets = self._sets.widgets
                # Set free-text ship name (U.S.S. Barbarossa)
                if r.ship_name and not build.get('ship_name'):
                    build['ship_name'] = r.ship_name
                    widgets.ship['name'].setText(r.ship_name)
                # Try to select ship class from cache using ship_type string
                # cache.ships is keyed by wiki Page name
                if r.ship_type and build.get('ship', '<Pick Ship>') == '<Pick Ship>':
                    from difflib import get_close_matches
                    ships = getattr(self._sets.cache, 'ships', {})
                    candidates = list(ships.keys())
                    # Try exact then fuzzy match
                    match = None
                    if r.ship_type in ships:
                        match = r.ship_type
                    else:
                        hits = get_close_matches(r.ship_type, candidates, n=1, cutoff=0.72)
                        if hits:
                            match = hits[0]
                    if match:
                        log.info(f'WARP: auto-selecting ship {match!r} from {r.ship_type!r}')
                        from src.callbacks import select_ship as _sel
                        # Simulate ship selection without opening picker
                        from src.callbacks import _save_session_slots, align_space_frame
                        from src.iofunc import exec_in_thread
                        ship_data = ships[match]
                        widgets.ship['button'].setText(match)
                        build['space'] = build.get('space', {})
                        build['ship']  = match
                        tier = ship_data.get('tier', 6)
                        widgets.ship['tier'].clear()
                        if tier == 6:
                            widgets.ship['tier'].addItems(('T6', 'T6-X', 'T6-X2'))
                        elif tier == 5:
                            widgets.ship['tier'].addItems(('T5', 'T5-U', 'T5-X', 'T5-X2'))
                        else:
                            widgets.ship['tier'].addItem(f'T{tier}')
                        # Set tier from recognised result
                        if r.ship_tier:
                            idx = widgets.ship['tier'].findText(r.ship_tier)
                            if idx >= 0:
                                widgets.ship['tier'].setCurrentIndex(idx)
                        build['tier'] = r.ship_tier or f'T{tier}'
                    else:
                        log.info(f'WARP: ship {r.ship_type!r} not found in cache')
            except Exception as _e:
                log.debug(f'WARP: ship info widget update failed: {_e}')

        for ri in r.items:
            if not ri.name:
                continue
            slot_info = SLOT_MAP.get(ri.slot)
            if not slot_info:
                log.warning(f'WARP: Unknown slot "{ri.slot}" — skipping')
                continue

            build_key, env, is_equipment = slot_info
            idx = ri.slot_index

            try:
                if is_equipment:
                    item_data = self._make_equipment_item(ri, build_key, env)
                    if item_data:
                        slot_equipment_item(self._sets, item_data, env, build_key, idx)
                else:
                    item_data = {'item': ri.name}
                    slot_trait_item(self._sets, item_data, env, build_key, idx)
            except Exception as e:
                log.warning(f'WARP: Failed to import "{ri.name}" into {ri.slot}[{idx}]: {e}')

        # Switch to the correct build tab
        tab_map = {
            'SPACE':         0,
            'GROUND':        1,
            'SPACE_SKILLS':  2,
            'GROUND_SKILLS': 3,
        }
        tab_idx = tab_map.get(self._build_type, 0)
        self._sets.switch_main_tab(tab_idx)
        self._sets.autosave()

    def _make_equipment_item(
        self, ri: RecognisedItem, build_key: str, env: str
    ) -> dict | None:
        """
        Build an equipment item dict compatible with slot_equipment_item().
        Format: {'item': name, 'rarity': rarity, 'modifiers': [...]}
        Uses SETS default_rarity and default_mark from settings.
        """
        eq_cache = self._sets.cache.equipment.get(build_key, {})
        entry    = eq_cache.get(ri.name)
        if not entry:
            log.warning(f'WARP: Item "{ri.name}" not in cache[{build_key}] — skipping')
            return None

        default_rarity = self._sets.settings.value('default_rarity', 'Epic')
        default_mark   = self._sets.settings.value('default_mark', 'XV')

        rarity = entry.get('rarity') or default_rarity
        return {
            'item':      ri.name,
            'rarity':    rarity,
            'mark':      default_mark,
            'modifiers': [None] * 4,
        }

    # ── Folder picker ──────────────────────────────────────────────────────

    def _browse_folder(self):
        last = self._settings.value(_SETTINGS_KEY_LAST_DIR, '')
        dlg  = QFileDialog(self)
        dlg.setWindowTitle('Select Screenshots Folder')
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        # Show image files for context but keep directory-only selection
        dlg.setNameFilter('Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)')
        if last and Path(last).is_dir():
            dlg.setDirectory(last)
        # Make file list read-only: user can see images but only dirs are selectable
        from PySide6.QtWidgets import QListView, QTreeView
        for view in dlg.findChildren(QListView) + dlg.findChildren(QTreeView):
            view.setSelectionMode(view.SelectionMode.NoSelection)
        if dlg.exec():
            files = dlg.selectedFiles()
            if files:
                self._folder = Path(files[0])
                self._folder_label.setText(str(self._folder))
                self._folder_label.setStyleSheet(
                    'color:#eee;background:#111;border:1px solid #3a6a9c;'
                    'border-radius:3px;padding:3px 7px;')
                self._settings.setValue(_SETTINGS_KEY_LAST_DIR, str(self._folder))
